from torch.utils.data import Dataset
import os
import pickle
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel
from torch import nn
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from typing import NamedTuple
import json
from problems import PDTRP
from torch.utils.data import DataLoader
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# functions for beam search

def torch_lexsort(keys, dim=-1):
    if keys[0].is_cuda:
        return _torch_lexsort_cuda(keys, dim)
    else:
        # Use numpy lex sort
        return torch.from_numpy(np.lexsort([k.numpy() for k in keys], axis=dim))


def _torch_lexsort_cuda(keys, dim=-1):
    """
    Function calculates a lexicographical sort order on GPU, similar to np.lexsort
    Relies heavily on undocumented behavior of torch.sort, namely that when sorting more than
    2048 entries in the sorting dim, it performs a sort using Thrust and it uses a stable sort
    https://github.com/pytorch/pytorch/blob/695fd981924bd805704ecb5ccd67de17c56d7308/aten/src/THC/generic/THCTensorSort.cu#L330
    """

    MIN_NUMEL_STABLE_SORT = 2049  # Minimum number of elements for stable sort

    # Swap axis such that sort dim is last and reshape all other dims to a single (batch) dimension
    reordered_keys = tuple(key.transpose(dim, -1).contiguous() for key in keys)
    flat_keys = tuple(key.view(-1) for key in keys)
    d = keys[0].size(dim)  # Sort dimension size
    numel = flat_keys[0].numel()
    batch_size = numel // d
    batch_key = torch.arange(batch_size, dtype=torch.int64, device=keys[0].device)[:, None].repeat(1, d).view(-1)

    flat_keys = flat_keys + (batch_key,)

    # We rely on undocumented behavior that the sort is stable provided that
    if numel < MIN_NUMEL_STABLE_SORT:
        n_rep = (MIN_NUMEL_STABLE_SORT + numel - 1) // numel  # Ceil
        rep_key = torch.arange(n_rep, dtype=torch.int64, device=keys[0].device)[:, None].repeat(1, numel).view(-1)
        flat_keys = tuple(k.repeat(n_rep) for k in flat_keys) + (rep_key,)

    idx = None  # Identity sorting initially
    for k in flat_keys:
        if idx is None:
            _, idx = k.sort(-1)
        else:
            # Order data according to idx and then apply
            # found ordering to current idx (so permutation of permutation)
            # such that we can order the next key according to the current sorting order
            _, idx_ = k[idx].sort(-1)
            idx = idx[idx_]

    # In the end gather only numel and strip of extra sort key
    if numel < MIN_NUMEL_STABLE_SORT:
        idx = idx[:numel]

    # Get only numel (if we have replicated), swap axis back and shape results
    return idx[:numel].view(*reordered_keys[0].size()).transpose(dim, -1) % d

def beam_search(*args, **kwargs):
    beams, final_state = _beam_search(*args, **kwargs)
    return get_beam_search_results(beams, final_state)


def get_beam_search_results(beams, final_state):
    beam = beams[-1]  # Final beam
    if final_state is None:
        return None, None, None, None, beam.batch_size

    # First state has no actions/parents and should be omitted when backtracking
    actions = [beam.action for beam in beams[1:]]
    parents = [beam.parent for beam in beams[1:]]

    solutions = final_state.construct_solutions(backtrack(parents, actions))
    return beam.score, solutions, final_state.get_final_cost()[:, 0], final_state.ids.view(-1), beam.batch_size


def _beam_search(state, beam_size, propose_expansions=None,
                keep_states=False):

    beam = BatchBeam.initialize(state)

    # Initial state
    beams = [beam if keep_states else beam.clear_state()]

    # Perform decoding steps
    while not beam.all_finished():

        # Use the model to propose and score expansions
        parent, action, score = beam.propose_expansions() if propose_expansions is None else propose_expansions(beam)
        if parent is None:
            return beams, None

        # Expand and update the state according to the selected actions
        beam = beam.expand(parent, action, score=score)

        # Get topk
        beam = beam.topk(beam_size)

        # Collect output of step
        beams.append(beam if keep_states else beam.clear_state())

    # Return the final state separately since beams may not keep state
    return beams, beam.state


class BatchBeam(NamedTuple):
    """
    Class that keeps track of a beam for beam search in batch mode.
    Since the beam size of different entries in the batch may vary, the tensors are not (batch_size, beam_size, ...)
    but rather (sum_i beam_size_i, ...), i.e. flattened. This makes some operations a bit cumbersome.
    """
    score: torch.Tensor  # Current heuristic score of each entry in beam (used to select most promising)
    state: None  # To track the state
    parent: torch.Tensor
    action: torch.Tensor
    batch_size: int  # Can be used for optimizations if batch_size = 1
    device: None  # Track on which device

    # Indicates for each row to which batch it belongs (0, 0, 0, 1, 1, 2, ...), managed by state
    @property
    def ids(self):
        return self.state.ids.view(-1)  # Need to flat as state has steps dimension

    def __getitem__(self, key):
            assert torch.is_tensor(key) or isinstance(key, slice)
            return self._replace(
                # ids=self.ids[key],
                score=self.score[key] if self.score is not None else None,
                state=self.state[key],
                parent=self.parent[key] if self.parent is not None else None,
                action=self.action[key] if self.action is not None else None
            )

    # Do not use __len__ since this is used by namedtuple internally and should be number of fields
    # def __len__(self):
    #     return len(self.ids)

    @staticmethod
    def initialize(state):
        batch_size = len(state.ids)
        device = state.ids.device
        return BatchBeam(
            score=torch.zeros(batch_size, dtype=torch.float, device=device),
            state=state,
            parent=None,
            action=None,
            batch_size=batch_size,
            device=device
        )

    def propose_expansions(self):
        mask = self.state.get_mask()
        # Mask always contains a feasible action
        expansions = torch.nonzero(mask[:, 0, :] == 0)
        parent, action = torch.unbind(expansions, -1)
        return parent, action, None

    def expand(self, parent, action, score=None):
        return self._replace(
            score=score,  # The score is cleared upon expanding as it is no longer valid, or it must be provided
            state=self.state[parent].update(action),  # Pass ids since we replicated state
            parent=parent,
            action=action
        )

    def topk(self, k):
        idx_topk = segment_topk_idx(self.score, k, self.ids)
        return self[idx_topk]

    def all_finished(self):
        return self.state.all_finished()

    def cpu(self):
        return self.to(torch.device('cpu'))

    def to(self, device):
        if device == self.device:
            return self
        return self._replace(
            score=self.score.to(device) if self.score is not None else None,
            state=self.state.to(device),
            parent=self.parent.to(device) if self.parent is not None else None,
            action=self.action.to(device) if self.action is not None else None
        )

    def clear_state(self):
        return self._replace(state=None)

    def size(self):
        return self.state.ids.size(0)


def segment_topk_idx(x, k, ids):
    """
    Finds the topk per segment of data x given segment ids (0, 0, 0, 1, 1, 2, ...).
    Note that there may be fewer than k elements in a segment so the returned length index can vary.
    x[result], ids[result] gives the sorted elements per segment as well as corresponding segment ids after sorting.
    :param x:
    :param k:
    :param ids:
    :return:
    """
    assert x.dim() == 1
    assert ids.dim() == 1

    # Since we may have varying beam size per batch entry we cannot reshape to (batch_size, beam_size)
    # And use default topk along dim -1, so we have to be creative
    # Now we have to get the topk per segment which is really annoying :(
    # we use lexsort on (ids, score), create array with offset per id
    # offsets[ids] then gives offsets repeated and only keep for which arange(len) < offsets + k
    splits_ = torch.nonzero(ids[1:] - ids[:-1])

    if len(splits_) == 0:  # Only one group
        _, idx_topk = x.topk(min(k, x.size(0)))
        return idx_topk

    splits = torch.cat((ids.new_tensor([0]), splits_[:, 0] + 1))
    # Make a new array in which we store for each id the offset (start) of the group
    # This way ids does not need to be increasing or adjacent, as long as each group is a single range
    group_offsets = splits.new_zeros((splits.max() + 1,))
    group_offsets[ids[splits]] = splits
    offsets = group_offsets[ids]  # Look up offsets based on ids, effectively repeating for the repetitions per id

    # We want topk so need to sort x descending so sort -x (be careful with unsigned data type!)
    idx_sorted = torch_lexsort((-(x if x.dtype != torch.uint8 else x.int()).detach(), ids))

    # This will filter first k per group (example k = 2)
    # ids     = [0, 0, 0, 1, 1, 1, 1, 2]
    # splits  = [0, 3, 7]
    # offsets = [0, 0, 0, 3, 3, 3, 3, 7]
    # offs+2  = [2, 2, 2, 5, 5, 5, 5, 9]
    # arange  = [0, 1, 2, 3, 4, 5, 6, 7]
    # filter  = [1, 1, 0, 1, 1, 0, 0, 1]
    # Use filter to get only topk of sorting idx
    return idx_sorted[torch.arange(ids.size(0), out=ids.new()) < offsets + k]


def backtrack(parents, actions):

    # Now backtrack to find aligned action sequences in reversed order
    cur_parent = parents[-1]
    reversed_aligned_sequences = [actions[-1]]
    for parent, sequence in reversed(list(zip(parents[:-1], actions[:-1]))):
        reversed_aligned_sequences.append(sequence.gather(-1, cur_parent))
        cur_parent = parent.gather(-1, cur_parent)

    return torch.stack(list(reversed(reversed_aligned_sequences)), -1)


class CachedLookup(object):

    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), "CachedLookup does not support slicing, " \
                                           "you can slice the result of an index operation instead"

        if torch.is_tensor(key):  # If tensor, idx all tensors by this tensor:

            if self.key is None:
                self.key = key
                self.current = self.orig[key]
            elif len(key) != len(self.key) or (key != self.key).any():
                self.key = key
                self.current = self.orig[key]

            return self.current

        return super(CachedLookup, self).__getitem__(key)
    
    # models

def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches
    :param n: the total number of elements, optional if it cannot be determined as args[0].size(0)
    :param f: The function that is computed, should take only tensors as arguments and return tensor or tuple of tensors
    :param calc_batch_size: The batch size to use when computing this function
    :param args: Tensor arguments with equally sized first batch dimension
    :return: f(*args), this should be one or multiple tensors with equally sized first batch dimension
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    # all_res = [f(*batch_args) for batch_args in zip(*[torch.chunk(arg, n_batches) for arg in args])]
    # We do not use torch.chunk such that it also works for other classes that support slicing
    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)

def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])

def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts

class GNNLayer(nn.Module):
    """Configurable GNN Layer

    Implements the Gated Graph ConvNet layer:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
        where Aggr. is an aggregation function: sum/mean/max.

    References:
        - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
        - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
    """

    def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super(GNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"
        
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)
        
    def forward(self, h, e, graph):
        """
        Args:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node and edge features
        """
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates)  # B x V x H

        # Normalize node features
        h = self.norm_h(
            h.view(batch_size*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
        
        # Normalize edge features
        e = self.norm_e(
            e.view(batch_size*num_nodes*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e

        return h, e

    def aggregate(self, Vh, graph, gates):
        """
        Args:
            Vh: Neighborhood features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            gates: Edge gates (B x V x V x H)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H
        
        # Enforce graph structure through masking
        Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0
        
        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / torch.sum(1-graph, dim=2).unsqueeze(-1).type_as(Vh)
        
        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]
        
        else:
            return torch.sum(Vh, dim=2)
        

class GNNEncoder(nn.Module):
    """Configurable GNN Encoder
    """
    
    def __init__(self, n_layers, hidden_dim, aggregation="sum", norm="layer", 
                 learn_norm=True, track_norm=False, gated=True, *args, **kwargs):
        super(GNNEncoder, self).__init__()

        self.init_embed_edges = nn.Embedding(2, hidden_dim)

        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
                for _ in range(n_layers)
        ])

    def forward(self, x, graph):
        """
        Args:
            x: Input node features (B x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node features (B x V x H)
        """
        # Embed edge features
        e = self.init_embed_edges(graph.type(torch.long))

        for layer in self.layers:
            x, e = layer(x, e, graph)

        return x
    
class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

# functions for loading in Joshi's model 

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    
    if 'knn_strat' not in args:
        args['knn_strat'] = None
        
    if 'aggregation_graph' not in args:
        args['aggregation_graph'] = "mean"
    
    return args

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=False)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('\nLoading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage, weights_only=False)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict

def load_problem(name):
    from problems import TSP, TSPSL
    problem = {
        'tsp': TSP,
        'tspsl': TSPSL
        # 'cvrp': CVRP,
        # 'sdvrp': SDVRP,
        # 'op': OP,
        # 'pctsp_det': PCTSPDet,
        # 'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

def load_model(path, epoch=None, extra_logging=False): 
    from nets.attention_model import AttentionModel  
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])
    
    model_class = AttentionModel
    assert model_class is not None, "Unknown model: {}".format(model_class)
    encoder_class = GNNEncoder
    assert encoder_class is not None, "Unknown encoder: {}".format(encoder_class)
    model = model_class(
        problem=problem,
        embedding_dim=args['embedding_dim'],
        encoder_class=encoder_class,
        n_encode_layers=args['n_encode_layers'],
        aggregation=args['aggregation'],
        aggregation_graph=args['aggregation_graph'],
        normalization=args['normalization'],
        learn_norm=args['learn_norm'],
        track_norm=args['track_norm'],
        gated=args['gated'],
        n_heads=args['n_heads'],
        tanh_clipping=args['tanh_clipping'],
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        checkpoint_encoder=args['checkpoint_encoder'],
        shrink_size=args['shrink_size'],
        extra_logging=extra_logging
    )    
    
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args

model_path = "pretrained/tsp_20-50/rl-ar-var-20pnn-gnn-max_20200313T002243"

filenames = ["new_data/pdtrp/pdtrp50dod0.8v4th8_testing_ortools.txt", "new_data/pdtrp/pdtrp50dod0.5v4th8_testing_ortools.txt", "new_data/pdtrp/pdtrp50dod0.2v4th8_testing_ortools.txt", "new_data/pdtrp/pdtrp100dod0.8v4th8_testing_ortools.txt", "new_data/pdtrp/pdtrp100dod0.5v4th8_testing_ortools.txt", "new_data/pdtrp/pdtrp100dod0.2v4th8_testing_ortools.txt"]

model, model_args = load_model(model_path)
model.to(device)
model.eval()
decode_strategy = 'greedy'  # 'greedy' or 'sampling' or 'bs'
softmax_temp = 1.0  # Temperature when using sampling, 1.0

model.set_decode_type(
    "greedy" if decode_strategy in ('bs', 'greedy') else "sampling",
    temp=softmax_temp
)

problem = PDTRP()

batch_size = 1
width = 128

for filename in filenames:
    dataset = problem.make_dataset(filename=filename, batch_size=batch_size)

    # Let's make the final version of the PDTRP solver that used the Joshi Model + beam search and resolviing


    SPEED = 4 / 60.0  # in distance units per minute

    rewards = []
    times = []

    for input in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        # Get an initial solution for time t = 0
        input = move_to(input, device)

        start = time.time()

        fixed_starting_sequence = [0]
        done = False

        # now we need to use this best solution to roll out the rest of the PDTRP problem
        current_time = 0.0

        arrival_times = input['arrival_times'][0]
        nodes_for_tsp = input['all_nodes'][0]
        graph_for_tsp = input['graph'][0]
        service_times = input['service_times'][0]
        distance_matrix = input['distance_matrix'][0]
        arrival_mask = arrival_times == 0.0

        n_original_customers = len(nodes_for_tsp)

        total_customers = len(nodes_for_tsp[arrival_mask])

        while not done:

            current_nodes = nodes_for_tsp[arrival_mask].unsqueeze(0)
            current_graph = graph_for_tsp[arrival_mask, :][:, arrival_mask].unsqueeze(0)
            input_to_get_times = {}
            input_to_get_times['arrival_times'] = arrival_times[arrival_mask].unsqueeze(0)
            input_to_get_times['service_times'] = service_times[arrival_mask].unsqueeze(0)
            input_to_get_times['distance_matrix'] = distance_matrix[arrival_mask,:][:, arrival_mask].unsqueeze(0)
            input_to_get_times['speed'] = SPEED

            cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                        current_nodes, current_graph, beam_size=width,
                        max_calc_batch_size=10000, fixed_starting_sequence=fixed_starting_sequence
                    )
            min_cost = float("inf")
            best_sequence = None
            for seq, cost in zip(sequences, costs):
                if cost < min_cost:
                    min_cost = cost
                    best_sequence = seq


            complete_route = torch.cat((torch.tensor(fixed_starting_sequence), best_sequence))

            visit_times, _ = problem.get_times(input_to_get_times, torch.tensor(complete_route).unsqueeze(0))

            visit_times = visit_times.squeeze(0)

            vt_wo_final_depot_return = visit_times[:-1]

            end_of_service_times = vt_wo_final_depot_return + input_to_get_times['service_times'][0, torch.tensor(complete_route)]

            min_next_arrival = torch.min(arrival_times[total_customers:])
            # if we end up in the case where the minimum next arrival time is greater than all times
            if (min_next_arrival > end_of_service_times).all():
                fixed_starting_sequence = complete_route.tolist()
                time_at_end_of_partial_route = min_next_arrival.item()
            else:
                stopping_idx = torch.where(end_of_service_times > min_next_arrival)[0][0].item()
                fixed_starting_sequence = complete_route[:stopping_idx+1].tolist()  # include the depot at the start of the route
                time_at_end_of_partial_route = end_of_service_times[stopping_idx].item()

            n_arrivals = torch.sum(arrival_times[total_customers:] <= time_at_end_of_partial_route).item()

            # Now we need to find how many arrivals we need to account for in the route

            arrival_mask[:total_customers+n_arrivals] = True

            total_customers += n_arrivals

            done = arrival_mask.all()

        current_nodes = nodes_for_tsp[arrival_mask].unsqueeze(0)
        current_graph = graph_for_tsp[arrival_mask, :][:, arrival_mask].unsqueeze(0)
        input_to_get_times = {}
        input_to_get_times['arrival_times'] = arrival_times[arrival_mask].unsqueeze(0)
        input_to_get_times['service_times'] = service_times[arrival_mask].unsqueeze(0)
        input_to_get_times['distance_matrix'] = distance_matrix[arrival_mask,:][:, arrival_mask].unsqueeze(0)
        input_to_get_times['speed'] = SPEED

        cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    current_nodes, current_graph, beam_size=width,
                    max_calc_batch_size=10000, fixed_starting_sequence=fixed_starting_sequence
                )
        min_cost = float("inf")
        best_sequence = None
        for seq, cost in zip(sequences, costs):
            if cost < min_cost:
                min_cost = cost
                best_sequence = seq
        
        complete_route = torch.cat((torch.tensor(fixed_starting_sequence), best_sequence))

        duration = time.time() - start

        visit_times, _ = problem.get_times(input_to_get_times, torch.tensor(complete_route).unsqueeze(0))

        sorted_route = sorted(complete_route.tolist())
        assert sorted_route == list(range(n_original_customers)), "Error: route does not include all customers"

        rewards.append(min_cost)
        times.append(duration)
    print("filename: ", filename)
    print("Average reward: ", np.mean(rewards))
    print("Average time: ", np.mean(times))

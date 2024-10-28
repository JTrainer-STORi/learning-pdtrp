import math
import numpy as np
from typing import NamedTuple
import resource

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel

from utils.tensor_functions import compute_in_batches
from utils import move_to

from scipy.spatial.distance import pdist, squareform


def nearest_neighbor_distances_from_graph(nodes_b, graph_b):

    nodes_b_cpu = nodes_b.cpu().numpy()

    W_val = torch.tensor([squareform(pdist(nodes, metric='euclidean')) for nodes in nodes_b_cpu], dtype=torch.float32, device=nodes_b.device)

    W_val[graph_b == 0] = 0

    return W_val

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

class DynamicAttentionModel(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 encoder_class,
                 n_encode_layers,
                 aggregation="sum",
                 aggregation_graph="mean",
                 normalization="layer",
                 learn_norm=True,
                 track_norm=False,
                 gated=True,
                 n_heads=8,
                 tanh_clipping=10.0,
                 mask_inner=True,
                 mask_logits=True,
                 mask_graph=False,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 extra_logging=False,
                 set_start=False,
                 edge_features='distance',
                 *args, **kwargs):
        """
        Models with a GNN/Transformer/MLP encoder and the Autoregressive decoder using attention mechanism

        Args:
            problem: TSP/TSPSL, to identify the learning paradigm
            embedding_dim: Hidden dimension for encoder/decoder
            encoder_class: GNN/Transformer/MLP encoder
            n_encode_layers: Number of layers for encoder
            aggregation: Aggregation function for GNN encoder
            aggregation_graph: Graph aggregation function
            normalization: Normalization scheme ('batch'/'layer'/'none')
            learn_norm: Flag for enabling learnt affine transformation during normalization
            track_norm: Flag to enable tracking training dataset stats instead of using batch stats during normalization
            gated: Flag to enbale anisotropic GNN aggregation
            n_heads: Number of attention heads for Transformer encoder/MHA in decoder
            tanh_clipping: Constant value to clip decoder logits with tanh
            mask_inner: Flag to use visited mask during inner function of decoder
            mask_logits: Flag to use visited mask during log computation of decoder
            mask_graph: Flag to use graph mask during decoding
            checkpoint_encoder: Whether to use checkpoints for encoder embeddings
            shrink_size: N/A
            extra_logging: Flag to perform extra logging, used for plotting histograms of embeddings

        References:
            - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
            - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
        """

        super(DynamicAttentionModel, self).__init__()
        
        self.problem = problem
        self.embedding_dim = embedding_dim
        self.encoder_class = encoder_class
        self.n_encode_layers = n_encode_layers
        self.aggregation = aggregation
        self.aggregation_graph = aggregation_graph
        self.normalization = normalization
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.mask_graph = mask_graph
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.set_start = set_start
        self.edge_features = edge_features
        
        # Extra logging updates self variables with batch statistics (without returning them)
        self.extra_logging = extra_logging
        
        self.decode_type = None
        self.temp = 1.0


        assert problem.NAME in ("pdtsp", "pdtsp-rd"), "Unsupported problem: {}".format(problem.NAME)

        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        node_dim = 2  # x, y
        edge_dim  = 1 # Distance

        if not self.set_start:
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        
        # Node embedding layer
        self.node_embed = nn.Linear(node_dim, embedding_dim, bias=True)  

        # Edge embedding layer
        self.edge_embed = nn.Linear(edge_dim, embedding_dim, bias=True)
        
        # Encoder model
        self.embedder = self.encoder_class(n_layers=n_encode_layers, 
                                           n_heads=n_heads,
                                           hidden_dim=embedding_dim, 
                                           aggregation=aggregation, 
                                           norm=normalization, 
                                           learn_norm=learn_norm,
                                           track_norm=track_norm,
                                           gated=gated)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        if self.edge_features == 'adjacency':
            self.init_embed_edges = nn.Embedding(2, embedding_dim)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, nodes, graph, times, new_nodes, return_pi=False):
        """
        Args:
            nodes: Input graph nodes (B x V x 2)
            graph: Graph as **NEGATIVE** adjacency matrices (B x V x V)
            supervised: Toggles SL training, teacher forcing and NLL loss computation
            targets: Targets for teacher forcing and NLL loss
            return_pi: Toggles returning the output sequences 
                       (Not compatible with DataParallel as the results
                        may be of different lengths on different GPUs)
        """
        
        # Reinforcement learning or inference
        # Run inner function

        log_p, pi = self._inner(nodes, graph, times, new_nodes)    
        
        # Get predicted costs
        _, num_new_nodes, _ = new_nodes.shape

        if num_new_nodes > 0:
            total_nodes = torch.cat((nodes, new_nodes), dim=1)
        else:
            total_nodes = nodes

        # total_nodes = torch.cat((nodes, new_nodes), dim=1)

        cost, mask = self.problem.get_costs(move_to(total_nodes, device=nodes.device), move_to(pi, device=nodes.device))

        if mask is not None:
            mask = mask.bool()
        
        # Log likelihood is calculated within the model since 
        # returning it per action does not work well with DataParallel 
        # (since sequences can be of different lengths)
        ll = self._calc_log_likelihood(move_to(log_p, device=nodes.device), mask)

        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _calc_log_likelihood(self, log_p, mask):

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, nodes, graph, times, new_nodes):
        # Create lists to track the outputs of the policy and the sequences of node selections 
        outputs = []
        sequences = []
        
        # Create problem state for masking (tracks which nodes have been visited)
        state = self.problem.make_state(nodes, graph)

        # Embed input batch of graph using GNN (B x V x H)
        if self.edge_features == 'adjacency':
            edge_features = self.init_embed_edges(graph.long())
        elif self.edge_features == 'distance':             
            if self.problem.NAME == 'pdtsp':
                edge_features = nearest_neighbor_distances_from_graph(nodes, graph)
                edge_features = edge_features.unsqueeze(-1)
            elif self.problem.NAME == 'pdtsp-rd':
                nearest_neighbor_distances = state.distance_matrices
                nearest_neighbor_distances[move_to(state.graph, device=nodes.device)==0] = 0
                edge_features = nearest_neighbor_distances
                edge_features = edge_features.unsqueeze(-1)

        if self.checkpoint_encoder:
            embeddings = checkpoint(self.embedder, self.node_embed(nodes), self.edge_embed(edge_features), graph)
        else:
            embeddings = self.embedder(self.node_embed(nodes), self.edge_embed(edge_features), graph)
        
        if self.extra_logging:
            self.embeddings_batch = embeddings

        # Compute keys, values for the glimpse and keys for the logits for reuse
        fixed = self._precompute(embeddings)

        batch_size, num_nodes, _ = nodes.shape

        _, num_new_nodes, _ = new_nodes.shape

        # Perform decoding steps

        # initialize the tensor which keeps track of the number of arrivals in each environment
        n_arrivals = torch.zeros(batch_size, 1)
        # initialize the counter which tracks the number of embeddings generated - not counting the inital embedding
        embedding_counter = 0
        # initialize boolean tensor which tracks which environments are currently paused. A paused environment will not
        # make any new node selections until the embedding is up to date. 
        pause_queries = torch.zeros(batch_size, 1, dtype=torch.bool)
        # ts tracks the current timestep of each of the environments
        ts = torch.zeros(batch_size, 1, dtype=torch.int, device=times.device)

        # if set_start is true then we make the first selection the first node in the list of nodes and update the other necessary variables
        if self.set_start:
            ts = torch.ones_like(ts, dtype=torch.int, device=times.device)
            set_first_selection = [torch.tensor(0) for i in range(batch_size)]
            state = state.update(set_first_selection)
            sequences.append(set_first_selection) # not sure if this'll work, let's see :)
            outputs.append(torch.zeros((num_nodes, batch_size), dtype=torch.float, device=nodes.device)) # setting log_p of irrelevant actions should not affect the policy...

        while not state.all_finished():

            # Get log probabilities of next action
            log_p, mask = self._get_log_p(fixed, state)

            mask = mask.bool()

            # we query the policy for the entire batch of environments although some may be paused
            selected = self._select_node(
                    log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            
            # selection keeps track of which environments are paused and which are not
            selection = [None for i in range(batch_size)]

            # iterate over environments 
            for i in range(batch_size):
                # we do not select an action if the environment is paused, we do not query the policy for the return back to the starting node so we stop selections at time step = Ntotal
                if not pause_queries[i] and ts[i] < (num_nodes + num_new_nodes):
                    # Select an action, increment the time (could be done either way around) then check if an arrival happens
                    # this makes it so that if an arrival happens at ts = t' then it is available for selection in the state at ts = t'
                    selection[i] = selected[i]
                    ts[i] += 1

                    if num_new_nodes > 0 and ts[i] in times[i]:
                        # by using bincount here, we allow for the possibility of multiple arrivals per timestep
                        n_arrivals[i] += torch.bincount(times[i])[ts[i]].cpu() 
                        pause_queries[i] = True
                    
            # Update problem state
            state = state.update(selection)

            if torch.gt(n_arrivals, embedding_counter).all():
                # update the state to add all of the new arrivals
                state = state.arrivals(new_nodes[:, embedding_counter].unsqueeze(1))
                #  create a new embedding here
                if self.edge_features == 'adjacency':
                    edge_features = self.init_embed_edges(state.graph.long())
                elif self.edge_features == 'distance':
                    if self.problem.NAME == 'pdtsp':
                        edge_features = self.edge_embed(nearest_neighbor_distances_from_graph(move_to(state.loc, device=nodes.device), move_to(state.graph, device=nodes.device)).unsqueeze(-1))
                    elif self.problem.NAME == 'pdtsp-rd':
                        nearest_neighbor_distances = state.distance_matrices
                        nearest_neighbor_distances[move_to(state.graph, device=nodes.device)==0] = 0
                        edge_features = self.edge_embed(nearest_neighbor_distances.unsqueeze(-1))
                embeddings = self.embedder(self.node_embed(move_to(state.loc, device=nodes.device)), edge_features, move_to(state.graph, device=nodes.device))
                fixed = self._precompute(embeddings)
                # Increment the arrival counter
                embedding_counter += 1
                # only unpause the queries for which the number of arrivals is equal to the arrival counter
                for i in range(batch_size):
                    if n_arrivals[i] <= embedding_counter: 
                        pause_queries[i] = False

            # Collect output of step
            outputs.append(log_p[:, 0, :].permute(1,0))
            sequences.append(selection)

        # the purpose of this proceedure is to remove the paused steps for each of the environments so that the sequences of nodes and policy outputs form a contiguous tensor
        sequences_new = [[] for i in range(batch_size)]

        outputs_new = torch.zeros((batch_size, (num_nodes + num_new_nodes)))

        for i, selection in enumerate(sequences):
            for j in range(batch_size):
                if selection[j] is not None:
                    sequences_new[j].append(selection[j])
                    outputs_new[j, len(sequences_new[j])-1] = outputs[i][selection[j].item(), j]

        sequences_new_2 = torch.tensor(sequences_new)

        # Collected lists, return Tensor
        return outputs_new, sequences_new_2

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any NaNs"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        
        return selected

    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        if self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        elif self.aggregation_graph == "mean":
            graph_embed = embeddings.mean(1)
        else:  # Default: disable graph embedding
            graph_embed = embeddings.sum(1) * 0.0
        
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed, state, normalize=True):
        # Compute query = context node embedding, as far as I can tell this is not concatenation as described in the paper. Maybe it is equivalent though. 
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
        
        # Compute the mask, for masking next action based on previous actions
        mask = state.get_mask()

        mask = mask.bool()
        
        graph_mask = None
        if self.mask_graph:
            # Compute the graph mask, for masking next action based on graph structure 
            graph_mask = state.get_graph_mask()

            graph_mask = graph_mask.bool()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state):
        """
        Returns the context per step, optionally for multiple steps at once 
        (for efficient evaluation of the model)
        """
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()
    
        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.i.item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
        
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            # First step placeholder, cat in dim 1 (time steps)
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions to (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -1e10
            if self.mask_graph:
                compatibility[graph_mask[None, :, :, None, :].expand_as(compatibility)] = -1e10

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        
        # From the logits compute the probabilities by masking the graph, clipping, and masking visited
        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -1e10 
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -1e10

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

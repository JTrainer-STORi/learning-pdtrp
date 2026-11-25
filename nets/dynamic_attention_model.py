"""This file contains the code for the dynamic attention model with masked training.
"""

import math
import numpy as np
from typing import NamedTuple
import resource
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from utils.tensor_functions import compute_in_batches
from utils import move_to, scale_time
from problems.utils import nearest_neighbor_graph
from nets.encoders.gnn_encoder import TimeEncoding
from nets.encoders.tgnn_encoder import IncrementalUpdateEncoder
from utils.policy_rollout import avoidable_wait_minutes, completion_costs, revealed_state
from utils.time_window_features import action_lookahead_features, service_start_slack

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
    
class DecisionTrace(NamedTuple):
    actions: torch.Tensor
    active: torch.Tensor
    mask: torch.Tensor
    current_node: torch.Tensor
    time: torch.Tensor | None
    capacity: torch.Tensor | None
    slack: torch.Tensor | None
    lookahead: torch.Tensor | None = None


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
                 tanh_clipping=10.,
                 mask_inner= True,
                 mask_logits= True,
                 mask_graph= False,
                 edge_features='adjacency',
                 use_time_feature=False,
                 functional_time_encoding=False,
                 scale_times=False,
                 knn_strat='percentage',
                 neighbors=20,
                 use_arrival_lstm=False,
                 use_arrival_times=False,
                 recursively_remove_visited_nodes=False,
                 use_time_window_slack=False,
                 transformer_heads=2,
                 transformer_resblocks_before=2,
                 transformer_resblocks_after=2,
                 transformer_resblock_hidden=128,
                 *args, use_action_lookahead=False, use_capacity_aware_lookahead=False, **kwargs):
        """A node encoder followed by an autoregressive attention decoder.

        During training, the model encodes the entire graph of the partially dynamic routing problem
        and nodes which have been visited or have not yet arrived are masked. The decoder then decodes 
        the solution by attending over nodes in the graph. 

        During inference, because the entire graph is not known, the model must update the encoder when 
        new nodes arrive. The model then decodes the solution by attending over nodes in the graph.

        Args:
            problem: Which problem variant the model will be used for
            embedding_dim: Hidden Dimension for Encoder/Decoder 
            encoder_class: Which encoder class to use
            n_encoder_layers: Number of layers for the encoder
            aggregation: Aggregation function for GNN encoder
            aggregation_graph: Aggregation function for graph embedding
            normalization: batch or layer normalization
            learn_norm: Whether to learn normalization parameters (passed as argument to PyTorch normalization layers)
            track_norm: Whether to track normalization statistics (mean and var running estimates) (passed as argument to PyTorch normalization layers)
            gated: Whether to use gated skip connections in the encoder
            n_heads: Number of attention heads
            tanh_clipping: Clipping value for output logits
            mask_inner: Whether to mask nodes in the inner attention layer of the decoder
            mask_logits: Whether to mask logits in the final decoder step
            mask_graph: whether to mask logits based on graph structure (for problems in which this is a factor)
            edge_features: GNN edge distances or adjacency; unused by the transformer
        """
        super(DynamicAttentionModel, self).__init__()
        
        self.problem = problem
        self.embedding_dim = embedding_dim
        self.encoder = encoder_class
        self.uses_edge_features = getattr(encoder_class, 'uses_edge_features', True)
        self.n_encoder_layers = n_encode_layers
        self.aggregation = aggregation
        self.aggregation_graph = aggregation_graph
        self.normalization = normalization
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.edge_feature_type = edge_features
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.mask_graph = mask_graph
        self.use_time_feature = use_time_feature
        self.functional_time_encoding = functional_time_encoding
        self.knn_strat = knn_strat
        self.neighbors = neighbors
        self.scale_times = scale_times
        self.is_pdctrp = problem.NAME == 'pdctrp'
        self.is_pdtrp = problem.NAME == 'pdtrp'
        self.is_pdtrptw = problem.NAME == 'pdtrptw'
        self.is_pdctrptw = problem.NAME == 'pdctrptw'
        self.use_arrival_lstm = use_arrival_lstm
        self.use_arrival_times = use_arrival_times
        self.use_time_window_slack = use_time_window_slack
        self.use_capacity_aware_lookahead = use_capacity_aware_lookahead
        self.use_action_lookahead = use_action_lookahead or self.use_capacity_aware_lookahead
        if self.use_time_window_slack and not (self.is_pdtrptw or self.is_pdctrptw):
            raise ValueError('Time-window slack requires PDTRPTW or PDCTRPTW')
        if self.use_action_lookahead and not (self.is_pdtrptw or self.is_pdctrptw):
            raise ValueError('Action lookahead requires PDTRPTW or PDCTRPTW')
        if self.use_capacity_aware_lookahead and not self.is_pdctrptw:
            raise ValueError('Capacity-aware lookahead requires PDCTRPTW')
        self.recursively_remove_visited_nodes = recursively_remove_visited_nodes

        self.decode_type = None
        self.temp = 1.0

        step_context_dim = embedding_dim
        node_dim = 2
        edge_dim = 1

        if self.is_pdctrp:
            node_dim = 3 # x, y, demand
            step_context_dim += 1 # for the capacity
        if self.is_pdtrptw:
            if self.functional_time_encoding:
                node_dim = 2 + embedding_dim * 2
            else:
                node_dim = 4 # x,y, window_start, window_end
        if self.is_pdctrptw:
            if self.functional_time_encoding:
                node_dim = 3 + embedding_dim * 2 # x,y, window_start (fte), window_end (fte), demand
            else:
                node_dim = 5 # x,y, window_start, window_end, demand
            step_context_dim += 1 # for the capacity

        if self.use_time_feature and not self.functional_time_encoding:
            step_context_dim += 1

        if self.functional_time_encoding:
            self.time_encoding = TimeEncoding(embedding_dim)
            step_context_dim += embedding_dim  # Add time encoding to step context
        
        if self.use_arrival_times:
            if self.functional_time_encoding:
                node_dim += embedding_dim
            else:
                node_dim += 1

        if self.use_arrival_lstm:
            if self.functional_time_encoding:
                self.lstm = nn.LSTM(input_size=embedding_dim*2, hidden_size=embedding_dim, num_layers=1, batch_first=True)
            else:
                self.lstm = nn.LSTM(input_size=embedding_dim + 1, hidden_size=embedding_dim, num_layers=1, batch_first=True)
            self.lstm_hidden = None
            step_context_dim += embedding_dim  # Add the LSTM hidden state to the step context

        # Special embedding projection for the depot node
        self.init_embed_depot = nn.Linear(2, embedding_dim)

        # Node Embedding Layer
        self.node_embed = nn.Linear(node_dim, embedding_dim, bias=True)

        if self.uses_edge_features and self.edge_feature_type == 'adjacency':
            self.edge_embed = nn.Embedding(2, embedding_dim)
        elif self.uses_edge_features and self.edge_feature_type == 'distance':
            self.edge_embed = nn.Linear(1, embedding_dim, bias=True)

        # Encoder model
        self.embedder = self.encoder(n_layers=self.n_encoder_layers,
                                     n_heads=n_heads,
                                     hidden_dim=embedding_dim,
                                     aggregation=aggregation,
                                     norm=normalization,
                                     learn_norm=learn_norm,
                                     track_norm=track_norm,
                                     gated=gated,
                                     transformer_heads=transformer_heads,
                                     transformer_resblocks_before=transformer_resblocks_before,
                                     transformer_resblocks_after=transformer_resblocks_after,
                                     transformer_resblock_hidden=transformer_resblock_hidden)
        
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.use_time_window_slack:
            # Recompute a small per-customer projection at each decision while
            # retaining the cached encoder and its original training gradients.
            self.project_time_window_slack = nn.Linear(1, 3 * embedding_dim, bias=False)
        if self.use_action_lookahead:
            # A warm start keeps the original logits and sampling RNG.
            with torch.random.fork_rng(devices=[]):
                self.lookahead_bias = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
            nn.init.zeros_(self.lookahead_bias[-1].weight)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, return_times=False, mode='masked', pomo_batch_size=1,
                print_query_times=False, num_scenarios=10, record_decisions=False,
                return_baseline_features=False, rollout_candidates=0, decision_times=None,
                rollout_wait_penalty=0.0):
        """
            Args:
                input: Dictionary containing input data
                mode: Recursive modes evaluate exactly one instance and one
                    rollout per call. Evaluate datasets sequentially in these modes.
        """

        assert mode in ['masked', 'recursive', 'recursive_plus_removal', 'MSA'], "Mode must be either 'masked' or 'recursive' or 'recursive_plus_removal' or 'MSA'"

        if mode in ('recursive', 'recursive_plus_removal'):
            if input['all_nodes'].size(0) != 1 or pomo_batch_size != 1:
                raise ValueError(
                    "Recursive evaluation requires exactly one instance and one rollout "
                    "per call (batch_size=1, pomo_batch_size=1). "
                    "Evaluate multiple instances sequentially."
                )

        if rollout_candidates < 0:
            raise ValueError('rollout_candidates must be nonnegative')
        if not math.isfinite(rollout_wait_penalty) or rollout_wait_penalty < 0:
            raise ValueError('rollout_wait_penalty must be finite and nonnegative')
        if rollout_wait_penalty and (not rollout_candidates or not (self.is_pdtrptw or self.is_pdctrptw)):
            raise ValueError('The avoidable waiting penalty requires time-window policy rollouts')
        if rollout_candidates and (
            mode != 'recursive' or self.training or self.decode_type != 'greedy'
            or self.use_arrival_lstm or self.recursively_remove_visited_nodes
            or self.mask_graph or isinstance(self.embedder, IncrementalUpdateEncoder)
        ):
            raise ValueError('Policy rollouts require greedy recursive evaluation with cached embeddings, '
                             'without arrival LSTM, graph masking or visited-node removal')
        if decision_times is not None and (mode not in ('recursive', 'recursive_plus_removal')
                                           or isinstance(self.embedder, IncrementalUpdateEncoder)):
            raise ValueError('Decision timing requires recursive evaluation with cached embeddings')

        self.mode = mode
        if return_baseline_features and (mode != 'masked' or isinstance(self.embedder, IncrementalUpdateEncoder)):
            raise ValueError('Baseline features require a cached masked encoding')
        baseline_output = ()
        decisions = None
        if record_decisions:
            if (mode != 'masked' or pomo_batch_size != 1 or self.use_arrival_lstm or self.mask_graph
                    or isinstance(self.embedder, IncrementalUpdateEncoder)):
                raise ValueError('Decision recording requires masked decoding without POMO, LSTM, incremental encoding or graph masking')
            decisions = []

        if self.mode == 'recursive_plus_removal':
            self.recursively_remove_visited_nodes = True

        if not self.uses_edge_features:
            self.edge_features = None
        elif self.edge_feature_type == 'adjacency':
            self.edge_features = input['graph'].long() # just a placeholder for now, will eventually need to change for Real World Instances
        elif self.edge_feature_type == 'distance':
            self.edge_features = input['distance_matrix'].unsqueeze(-1)  # (batch_size, num_nodes, num_nodes, 1)

        if isinstance(self.embedder, IncrementalUpdateEncoder):

            log_p, pi, visit_times = self._inner(input, None, pomo_batch_size, print_query_times=print_query_times)

        elif self.mode == 'MSA':

            pi, visit_times = self._inner_msa(input, print_query_times=print_query_times, num_scenarios=num_scenarios)

        elif self.mode == 'masked':
            
            embeddings = self._encode(input)
            if return_baseline_features:
                baseline_output = (embeddings.detach().mean(dim=1),)

            log_p, pi, visit_times = self._inner(
                input, embeddings, pomo_batch_size, print_query_times=print_query_times, decisions=decisions)

        elif self.mode == 'recursive' or self.mode == 'recursive_plus_removal':

            pi, visit_times = self._inner_recursive(input, print_query_times=print_query_times,
                                                   rollout_candidates=rollout_candidates,
                                                   decision_times=decision_times,
                                                   rollout_wait_penalty=rollout_wait_penalty)

        visit_times = visit_times.detach().to(dtype=torch.float32)

        pi_w_depot = torch.cat((torch.zeros(pi.size(0), 1, dtype=torch.int64, device=pi.device), pi), dim=1)

        if self.is_pdtrptw or self.is_pdctrptw:
            cost, info = self.problem.get_costs(input, move_to(pi_w_depot, device=input['all_nodes'].device), move_to(visit_times, device=input['all_nodes'].device), gamma=move_to(input['gamma'], device=input['all_nodes'].device), theta=move_to(input['theta'], device=input['all_nodes'].device))
        else:
            cost, info = self.problem.get_costs(input, move_to(pi_w_depot, device=input['all_nodes'].device))

        if record_decisions:
            trace = DecisionTrace(*(torch.cat(steps, dim=1) if steps[0] is not None else None
                                    for steps in zip(*decisions)))
            return (cost, log_p, trace) + baseline_output
        if self.training:
            ll = self._calc_log_likelihood(log_p, mask=None)
        elif self.eval:
            ll = None

        output = (cost, ll, info)
        if return_pi:
            output += (pi_w_depot,)
        if return_times:
            output += (visit_times,)
        return output + baseline_output
        
    def _encode(self, input):
        """Embed edges only for encoders that consume them."""
        edge_embeddings = None
        if self.uses_edge_features:
            if self.edge_feature_type == 'adjacency':
                edge_features = input['graph'].long()
            else:
                edge_features = input['distance_matrix'].unsqueeze(-1)
            edge_embeddings = self.edge_embed(edge_features)
        return self.embedder(self._init_embed(input), edge_embeddings, input['graph'])

    def _calc_log_likelihood(self, log_p, mask):

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, input, embeddings, pomo_batch_size=1, print_query_times=False, decisions=None):
        # create lists to track the log probabilities and the selected nodes
        outputs = []
        sequences = []

        # create the problem state for tracking which nodes are masked
        state = self.problem.make_state(input)

        if embeddings is not None:
            fixed = self._precompute(embeddings)

        batch_size = int(input['all_nodes'].size(0) / pomo_batch_size)

        if self.use_arrival_lstm:
            rollout_batch_size = input['all_nodes'].size(0)
            self.lstm_hidden = (torch.zeros((self.lstm.num_layers, rollout_batch_size, self.lstm.hidden_size), device=input['all_nodes'].device),torch.zeros((self.lstm.num_layers, rollout_batch_size, self.lstm.hidden_size), device=input['all_nodes'].device))
            arrival_order = torch.argsort(input['arrival_times'], dim=1, stable=True)
            arrival_horizon = (
                input['time_horizon'][0] + input['latest_end'][0]
                if self.scale_times else None
            )

        while not state.all_finished():

            # If we are in a POMO setting, we select the start nodes for each instance in the batch

            if pomo_batch_size > 1 and state.i == 0:
                mask = state.get_mask()
                available_counts = (~mask[:, 0]).sum(dim=-1)
                if (available_counts == 0).any():
                    raise RuntimeError("A POMO rollout has no available starting customer")
                # A fully dynamic instance may reveal only one customer at its
                # first decision. Cycle through legal starts instead of forcing
                # some rollouts to visit customers that have not called yet.
                starts = torch.argsort(mask[:, 0].int(), dim=-1, stable=True)
                rollout = torch.arange(pomo_batch_size, device=input['all_nodes'].device).repeat(batch_size)
                selected = starts.gather(1, (rollout % available_counts)[:, None]).squeeze(1)
                log_p = torch.zeros((batch_size * pomo_batch_size, 1, input['all_nodes'].size(1)), device=input['all_nodes'].device)
            else:
                mask = state.get_mask()

                if isinstance(self.embedder, IncrementalUpdateEncoder):
                    temporal_mask = ~(state.not_arrived_[:, 0, :])
                    embeddings = self.embedder(self._init_embed(input), self.edge_embed(self.edge_features), input['graph'], temporal_mask)
                    fixed = self._precompute(embeddings)

                if print_query_times:
                    start_time = time.time()

                # get log probabilities of next action
                if decisions is None:
                    log_p, mask = self._get_log_p(fixed, state, mask)
                else:
                    current_node = state.get_current_node()
                    current_time = state.get_timestep(normalize=self.scale_times) if self.use_time_feature else None
                    capacity = state.get_remaining_capacity() if self.is_pdctrp or self.is_pdctrptw else None
                    slack = self._get_time_window_slack(state, mask) if self.use_time_window_slack else None
                    lookahead = self._get_action_lookahead(state, mask) if self.use_action_lookahead else None
                    context = self._step_context(fixed.node_embeddings, current_node, current_time, capacity)
                    log_p = self._decode_log_p(fixed, context, mask, slack, lookahead=lookahead)

                if print_query_times:
                    end_time = time.time()
                    print(f"Query time: {end_time - start_time:.4f} seconds")

                # we query the policy
                selected = self._select_node(
                    log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
                
            if decisions is not None:
                # A choice interrupted during waiting still counts. Depot
                # padding after finishing a batch row does not.
                decisions.append(DecisionTrace(selected[:, None], ~state.visited_.all(dim=-1),
                                               mask, current_node, current_time, capacity, slack, lookahead))

            # Track request revelation independently of action feasibility/capacity.
            if self.use_arrival_lstm:
                previously_pending = state.not_arrived_.clone()

            #update problem state
            state = state.update(selected)

            # if we're using the decoder LSTM, we need to input any arrivals that occurred to it
            if self.use_arrival_lstm:
                newly_revealed = previously_pending & ~state.not_arrived_
                arrival_sequences = self._batched_arrival_sequences(
                    embeddings, input['arrival_times'], newly_revealed[:, 0],
                    arrival_order, arrival_horizon,
                )
                self.update_lstm_with_arrivals_packed(arrival_sequences)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        all_sequences = torch.stack(sequences, 1)
        all_outputs = torch.stack(outputs, 1).gather(2, all_sequences.unsqueeze(2)).squeeze(-1)

        # Waiting choices contribute to policy gradients, but only executed
        # visits belong in the scored tour and its service-start timestamps.
        all_sequences, visit_times = state.get_solution(all_sequences)
        return all_outputs, all_sequences, visit_times

    def score_decisions(self, input, trace, return_baseline_features=False):
        """Score recorded actions with current weights and no simulator steps."""
        fixed = self._precompute(self._encode(input))
        scores = []
        # Chunking bounds the size of the per-step attention tensors.
        for start in range(0, trace.actions.size(1), 8):
            chunk = DecisionTrace(*(value[:, start:start + 8] if value is not None else None
                                    for value in trace))
            scores.append(checkpoint(self._score_decision_chunk, fixed, chunk, use_reentrant=False))
        log_p = torch.cat(scores, dim=1)
        if return_baseline_features:
            return log_p, fixed.node_embeddings.detach().mean(dim=1)
        return log_p

    def _score_decision_chunk(self, fixed, trace):
        context = self._step_context(fixed.node_embeddings, trace.current_node, trace.time, trace.capacity)
        log_p = self._decode_log_p(fixed, context, trace.mask, trace.slack, lookahead=trace.lookahead)
        return log_p.gather(2, trace.actions.unsqueeze(-1)).squeeze(-1)
    
    def _inner_recursive(self, input, print_query_times=False, rollout_candidates=0, decision_times=None,
                         rollout_wait_penalty=0.0):
        # If we want to run evaluations true to reality then we'll have to generate the graph and embeddings as customers arrive
        # note that due to it being annoying to do this for anything other than a single instance, this function is designed to only work for a single instance
        assert input['all_nodes'].size(0) == 1, "This forward method doesn't play nicely with batched inputs"

        batch_size = input['all_nodes'].size(0)

        # This bit is the same as the training inner method
        outputs = []
        sequences = []

        decision_started = time.perf_counter() if decision_times is not None else None
        state = self.problem.make_state(input)

        if self.recursively_remove_visited_nodes:
            arrival_mask = torch.logical_and(~state.not_arrived_[:,0,:], ~state.visited_[:,0,:])
            arrival_mask[0, 0] = True  # Ensure the depot is always included
        else:
            arrival_mask = ~state.not_arrived_[:,0,:]

        masked_input = self._mask_input(input, arrival_mask)
        embeddings = self._encode(masked_input)

        decoder_input = self._precompute(embeddings)

        if self.use_arrival_lstm:
            self.lstm_hidden = (torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), device=input['all_nodes'].device),torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), device=input['all_nodes'].device))

        prev_selected=None

        # This is the main loop for the decoder
        while not state.all_finished():
            if self.recursively_remove_visited_nodes:
                arrival_mask = torch.logical_and(~state.not_arrived_[:,0,:], ~state.visited_[:,0,:])
                arrival_mask[0, 0] = True  # Ensure the depot is always included
                mask = state.get_mask()[arrival_mask.unsqueeze(0)][None, None, :]
                self.current_indices = torch.arange(0, input['all_nodes'].size(1), device=input['all_nodes'].device)[arrival_mask[0]]
            else:
                mask = state.get_mask()[~state.not_arrived_][None, None, :]
                known_indices = torch.nonzero(arrival_mask[0], as_tuple=False).flatten()
                current_index = torch.nonzero(known_indices == state.prev_a.item()).item()
                prev_selected = embeddings[:, current_index:current_index + 1]

            if print_query_times:
                start_time = time.perf_counter()

            # get log probabilities of next action
            known_indices = torch.nonzero(arrival_mask[0], as_tuple=False).flatten()
            log_p, mask = self._get_log_p(
                decoder_input, state, mask, prev_selected=prev_selected,
                candidate_indices=known_indices,
            )

            if print_query_times:
                end_time = time.perf_counter()
                print(f"Query time: {end_time - start_time:.4f} seconds")

            if rollout_candidates:
                selected = self._select_rollout_action(
                    state, masked_input, known_indices, decoder_input, log_p, mask, rollout_candidates,
                    wait_penalty=rollout_wait_penalty)
            else:
                selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            true_selected = known_indices[selected]
            if decision_times is not None:
                if true_selected.is_cuda:
                    torch.cuda.synchronize(true_selected.device)
                now = time.perf_counter()
                decision_times.append(now - decision_started)
                decision_started = now
            previously_known = ~state.not_arrived_[:, 0, :]
            state = state.update(true_selected)
            outputs.append(log_p[:, 0, :])
            sequences.append(true_selected)

            # if an arrival has occurred, we need to update the embeddings and decoder input
            if self.recursively_remove_visited_nodes and not state.all_finished():
                arrival_mask = torch.logical_and(~state.not_arrived_[:,0,:], ~state.visited_[:,0,:])
                arrival_mask[0, 0] = True  # Ensure the depot is always included
                arrival_mask = arrival_mask.unsqueeze(0)
                masked_input = self._mask_input(input, arrival_mask[:,0,:])
                if not hasattr(state, 'last_action_executed') or state.last_action_executed.item():
                    prev_selected = embeddings[0, selected].unsqueeze(0)
                embeddings = self._encode(masked_input)
                decoder_input = self._precompute(embeddings)
            elif state.arrival_occured:
                arrival_mask = ~state.not_arrived_[:,0,:]
                masked_input = self._mask_input(input, ~state.not_arrived_[:,0,:])
                embeddings = self._encode(masked_input)
                decoder_input = self._precompute(embeddings, num_steps=1)
            if self.use_arrival_lstm and state.arrival_occured:
                arrival_sequences = []
                # If we're using the decoder LSTM, we need to input any arrivals that occurred to it
                newly_known = (~state.not_arrived_[:, 0, :]) & ~previously_known
                current_indices = torch.nonzero(arrival_mask.reshape(-1), as_tuple=False).flatten()
                arrival_indices = self._arrival_indices_by_time(
                    masked_input['arrival_times'][0], newly_known[0, current_indices]
                )
                if arrival_indices.numel() > 0:
                    if self.scale_times:
                        arrival_times = scale_time(masked_input['arrival_times'][0, arrival_indices][:, None], (masked_input['time_horizon'][0]+masked_input['latest_end'][0]))
                    else:
                        arrival_times = masked_input['arrival_times'][0, arrival_indices][:, None]
                    if self.functional_time_encoding:
                        arrival_times = self.time_encoding(arrival_times)
                    arrival_features = torch.cat([torch.gather(embeddings[0, :, :], 0, arrival_indices[:, None].expand(-1, embeddings.size(-1))), arrival_times], dim=-1)
                else:
                    arrival_features = torch.empty((0,))
                arrival_sequences.append(arrival_features)
            
                self.update_lstm_with_arrivals_packed(arrival_sequences)

        all_sequences = torch.stack(sequences, 1)

        all_sequences, visit_times = state.get_solution(all_sequences)
        return all_sequences, visit_times

    def _select_rollout_action(self, state, revealed_input, known_nodes, fixed, log_p, mask, width, wait_penalty=0.0):
        """Rank complete greedy continuations, keeping the ordinary choice on a tie."""
        scores = log_p[0, 0].masked_fill(mask[0, 0], -torch.inf)
        greedy = scores.argmax().view(1)
        customers = scores[1:].argsort(descending=True, stable=True)[:width] + 1
        candidates = torch.cat((customers, greedy.new_zeros(1)))
        candidates = candidates[~mask[0, 0, candidates] & (candidates != greedy)]
        if candidates.numel() == 0:
            return greedy
        candidates = torch.cat((greedy, candidates))
        planning = revealed_state(self.problem, state, revealed_input, known_nodes)
        branch_fixed = fixed[torch.zeros_like(candidates)]

        def choose_next(branches, finished):
            branch_mask = branches.get_mask()
            branch_mask[:, 0, 0] &= ~finished
            log_probs, _ = self._get_log_p(branch_fixed, branches, branch_mask)
            return log_probs[:, 0].masked_fill(branch_mask[:, 0], -torch.inf).argmax(dim=1)

        with_windows = self.is_pdtrptw or self.is_pdctrptw
        costs = completion_costs(
            planning, candidates, choose_next,
            gamma=revealed_input['gamma'] if with_windows else None,
            theta=revealed_input['theta'] if with_windows else None,
        )
        if wait_penalty:
            costs = costs + wait_penalty * avoidable_wait_minutes(planning, candidates)
        return candidates[costs.argmin().view(1)]
    
    def _inner_msa(self, input, print_query_times=False, num_scenarios=10):

        assert input['all_nodes'].size(0) == 1, "This forward method doesn't play nicely with batched inputs"

        batch_size = input['all_nodes'].size(0)

        sequences = []

        state = self.problem.make_state(input)

        arrival_mask = ~state.not_arrived_[:,0,:]

        masked_input = self._mask_input(input, arrival_mask)

        while not state.all_finished():

            # generate scenarios based on the state and masked input
            scenarios = self.problem.generate_scenarios(masked_input, state, num_scenarios=num_scenarios)

            scenario_actions = []
            for scenario in scenarios:
                embeddings = self._encode(scenario)
                decoder_input = self._precompute(embeddings)

                mask = state.get_mask()

                log_p, mask = self._get_log_p(decoder_input, state, mask)

                selected = self._select_node(
                    log_p.exp()[:, 0, :], mask[:, 0, :])

                scenario_actions.append(selected)

            # get the consensus action by majority vote
            scenario_actions = torch.stack(scenario_actions, dim=0)
            selected, _ = torch.mode(scenario_actions, dim=0)
            state = state.update(selected)
            sequences.append(selected)

            arrival_mask = ~state.not_arrived_[:,0,:]
            masked_input = self._mask_input(input, arrival_mask)

        all_sequences = torch.stack(sequences, 1)

        all_sequences, visit_times = state.get_solution(all_sequences)
        return all_sequences, visit_times


    def _mask_input(self, input, mask):
        """The job of this function is to mask the input dictionary prior to being passed to the encoder"""

        masked_input = {}
        # first step is to mask the coordinates, very simple
        assert input['all_nodes'].size(0) == 1, "Recursive input masking only supports batch size 1"
        indices = torch.nonzero(mask[0], as_tuple=False).squeeze(-1)
        masked_input['all_nodes'] = input['all_nodes'][:, indices]
        masked_input['arrival_times'] = input['arrival_times'][:, indices]
        masked_input['service_times'] = input['service_times'][:, indices]

        # Rebuild the graph over the currently known nodes, while preserving the
        # supplied distance metric (important for road-network/ORTEC instances).
        locs_on_cpu = masked_input['all_nodes'].squeeze(0).cpu().numpy()
        source_distances = input['distance_matrix'][0][indices][:, indices].cpu().numpy()
        nn_graph, distance_matrix = nearest_neighbor_graph(
            locs_on_cpu, self.neighbors, self.knn_strat, distance_matrix=source_distances
        )
        masked_input['graph'] = (~torch.as_tensor(
            nn_graph, dtype=torch.bool, device=input['all_nodes'].device
        )).unsqueeze(0)
        masked_input['distance_matrix'] = torch.as_tensor(
            distance_matrix, dtype=input['distance_matrix'].dtype,
            device=input['all_nodes'].device,
        ).unsqueeze(0)
        masked_input['time_horizon'] = input['time_horizon']
        masked_input['speed'] = input['speed']

        for key in (
            'latest_end',
            'service_times_mean',
            'service_times_std',
            'vehicle_capacity',
            'gamma',
            'theta',
        ):
            if key in input:
                masked_input[key] = input[key]
        for key in ('window_starts', 'window_ends', 'demand'):
            if key in input:
                masked_input[key] = input[key][:, indices]
        
        return masked_input

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any NaNs"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        
        return selected

    def _get_log_p(self, fixed, state, mask=None, normalize=True, prev_selected=None, candidate_indices=None):
        context = self._get_parallel_step_context(fixed.node_embeddings, state, prev_selected=prev_selected)
        slack = None
        lookahead = None
        if mask is None and (self.use_time_window_slack or self.use_action_lookahead):
            mask = state.get_mask()
        if self.use_time_window_slack:
            slack = self._get_time_window_slack(state, mask, candidate_indices)
        if self.use_action_lookahead:
            lookahead = self._get_action_lookahead(state, mask, candidate_indices)
        graph_mask = state.get_graph_mask().bool() if self.mask_graph else None
        return self._decode_log_p(fixed, context, mask, slack, graph_mask, normalize, lookahead), mask

    def _decode_log_p(self, fixed, context, mask, slack=None, graph_mask=None, normalize=True, lookahead=None):
        query = fixed.context_node_projected + self.project_step_context(context)
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
        if slack is not None:
            slack_K, slack_V, slack_logit = self.project_time_window_slack(slack).chunk(3, dim=-1)
            glimpse_K = glimpse_K + self._make_heads(slack_K)
            glimpse_V = glimpse_V + self._make_heads(slack_V)
            logit_K = logit_K + slack_logit

        bias = self.lookahead_bias(lookahead).squeeze(-1) if lookahead is not None else None
        log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask, bias)
        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()
        return log_p

    def _get_action_lookahead(self, state, mask, candidate_indices=None):
        features = action_lookahead_features(state, capacity_aware=self.use_capacity_aware_lookahead).unsqueeze(1)
        if candidate_indices is not None:
            features = features.index_select(-2, candidate_indices)
        return features.masked_fill(mask.unsqueeze(-1), 0)

    def _get_time_window_slack(self, state, mask, candidate_indices=None):
        """Align per-state slack with full or compact decoder candidate indices."""
        slack = service_start_slack(state, normalize=self.scale_times).unsqueeze(1)
        if candidate_indices is None:
            candidate_indices = torch.arange(slack.size(-1), device=slack.device)
        else:
            candidate_indices = candidate_indices.to(device=slack.device, dtype=torch.long)
            slack = slack.index_select(-1, candidate_indices)
        if slack.shape != mask.shape:
            raise ValueError('Slack features must match decoder candidates; provide their original node indices')
        # Zero input gives zero projection (no bias), including depot padding.
        # Late requests retain negative slack and remain legal soft-TW actions.
        unavailable = mask.bool() | (candidate_indices == 0)[None, None, :]
        return slack.masked_fill(unavailable, 0).unsqueeze(-1)

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
    
    def _get_parallel_step_context(self, embeddings, state, prev_selected=None):
        current_node = state.get_current_node() if prev_selected is None else None
        current_time = state.get_timestep(normalize=self.scale_times) if self.use_time_feature else None
        capacity = state.get_remaining_capacity() if self.is_pdctrp or self.is_pdctrptw else None
        context = self._step_context(embeddings, current_node, current_time, capacity, prev_selected)
        if self.use_arrival_lstm:
            h, _ = self.lstm_hidden
            context = torch.cat((context, h[-1].unsqueeze(1)), dim=-1)
        return context

    def _step_context(self, embeddings, current_node, current_time, capacity, prev_selected=None):
        if prev_selected is None:
            indices = current_node.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
            context = embeddings.gather(1, indices)
        else:
            context = prev_selected

        if current_time is not None:
            time_feature = current_time[:, :, None]
            if self.functional_time_encoding:
                time_feature = self.time_encoding(time_feature)
            context = torch.cat((context, time_feature), dim=-1)
        if capacity is not None:
            context = torch.cat((context, capacity[:, :, None]), dim=-1)
        return context

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None, bias=None):
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
        if bias is not None:
            logits = logits + bias
        
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
    
    def _init_embed(self, input):
        node_feats  = input['all_nodes'][:, 1:, :]
        if self.is_pdctrp or self.is_pdctrptw:
            node_feats = torch.cat((node_feats, input['demand'][:, 1:, None]), -1)
        if self.is_pdtrptw or self.is_pdctrptw:
            if self.scale_times:
                window_starts = scale_time(input['window_starts'][:, 1:, None], (input['time_horizon'][0]+input['latest_end'][0]))
                window_ends = scale_time(input['window_ends'][:, 1:, None], (input['time_horizon'][0]+input['latest_end'][0]))
            else:
                window_starts = input['window_starts'][:, 1:, None]
                window_ends = input['window_ends'][:, 1:, None]
            if self.functional_time_encoding:
                window_starts = self.time_encoding(window_starts)
                window_ends = self.time_encoding(window_ends)
            node_feats = torch.cat(
                (node_feats, window_starts, window_ends),
                -1
            )
        if self.use_arrival_times:        
            if self.scale_times:
                arrival_times = scale_time(input['arrival_times'][:, 1:, None], (input['time_horizon'][0]+input['latest_end'][0]))    
            else:
                arrival_times = input['arrival_times'][:, 1:, None]    
            if self.functional_time_encoding:
                node_feats = torch.cat(
                    (
                        node_feats,
                        self.time_encoding(arrival_times)
                    ),
                    -1
                )
            else:
                node_feats = torch.cat(
                    (
                        node_feats,
                        arrival_times
                    ),
                    -1
                )
        embed = torch.cat(
            (
                self.init_embed_depot(input['all_nodes'][:, 0, :])[:, None, :],
                self.node_embed(node_feats)
            ),
            1
            )
        return embed

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    @staticmethod
    def _arrival_indices_by_time(arrival_times, newly_revealed):
        """Order observed events chronologically without renumbering customers."""
        indices = torch.nonzero(newly_revealed, as_tuple=False).flatten()
        order = torch.argsort(arrival_times[indices], stable=True)
        return indices[order]

    def _batched_arrival_sequences(
        self, embeddings, arrival_times, newly_revealed, arrival_order, horizon,
    ):
        """Gather chronological arrivals with batch-wide GPU operations.

        The caller caches the stable time ordering for the rollout. Only the
        sequence lengths move to the CPU for splitting/packing; gathering each
        row separately would synchronize the GPU once per batch member.
        """
        lengths = newly_revealed.sum(dim=1).tolist()
        if not any(lengths):
            empty = embeddings.new_empty((0, self.lstm.input_size))
            return [empty] * len(lengths)
        ordered_mask = newly_revealed.gather(1, arrival_order)
        rows, positions = torch.nonzero(ordered_mask, as_tuple=True)
        nodes = arrival_order[rows, positions]
        times = arrival_times[rows, nodes, None]
        if self.scale_times:
            times = scale_time(times, horizon)
        if self.functional_time_encoding:
            times = self.time_encoding(times)
        features = torch.cat((embeddings[rows, nodes], times), dim=-1)
        return features.split(lengths)

    def update_lstm_with_arrivals_packed(self, arrival_sequences):
        """
        arrival_sequences: list of length batch_size
            each entry is a tensor [num_arrivals_i, input_dim]
            num_arrivals_i can be 0 if no arrivals for that env.
        """
        h, c = self.lstm_hidden  # h: [num_layers, batch, hidden_dim]

        # Collect only the sequences with length > 0
        nonempty_indices = [i for i, seq in enumerate(arrival_sequences) if seq.size(0) > 0]
        if len(nonempty_indices) == 0:
            # No arrivals in any env — do nothing
            return

        # Step 1: Build packed sequences with correct initial hidden states
        # Sort by sequence length (descending) for pack_sequence
        nonempty_sequences = [arrival_sequences[i] for i in nonempty_indices]
        lengths = [seq.size(0) for seq in nonempty_sequences]
        sorted_idx = sorted(range(len(nonempty_sequences)), key=lambda x: lengths[x], reverse=True)
        
        sorted_sequences = [nonempty_sequences[i] for i in sorted_idx]
        sorted_indices_in_batch = [nonempty_indices[i] for i in sorted_idx]
        batch_indices = torch.tensor(sorted_indices_in_batch, device=h.device)

        # Pack into [seq_total_len, input_dim] → PackedSequence
        packed = pack_sequence(sorted_sequences, enforce_sorted=True)

        # Extract initial states for only those envs with arrivals, sorted
        h_init = h.index_select(1, batch_indices)
        c_init = c.index_select(1, batch_indices)

        # Step 2: Run the LSTM
        _, (h_out, c_out) = self.lstm(packed, (h_init, c_init))

        # Step 3: Put updated states back into the full batch
        self.lstm_hidden = (
            h.index_copy(1, batch_indices, h_out),
            c.index_copy(1, batch_indices, c_out),
        )

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
                 *args, **kwargs):
        """This module is a GNN based Encoder, Attention Based Decoder Neural Network.

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
            edge_features: Whether to use edge distances as edge features in the encoder or adjacency matrix
        """
        super(DynamicAttentionModel, self).__init__()
        
        self.problem = problem
        self.embedding_dim = embedding_dim
        self.encoder = encoder_class
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
        self.is_pdcvrp = problem.NAME == 'pdcvrp'
        self.is_pdtrp = problem.NAME == 'pdtrp'
        self.is_pdtrptw = problem.NAME == 'pdtrptw'
        self.is_pdcvrptw = problem.NAME == 'pdcvrptw'
        self.use_arrival_lstm = use_arrival_lstm
        self.use_arrival_times = use_arrival_times
        self.recursively_remove_visited_nodes = recursively_remove_visited_nodes

        self.decode_type = None
        self.temp = 1.0

        step_context_dim = embedding_dim
        node_dim = 2
        edge_dim = 1

        if self.is_pdcvrp:
            node_dim = 3 # x, y, demand
            step_context_dim += 1 # for the capacity
        if self.is_pdtrptw:
            node_dim = 4 # x,y, window_start, window_end
        if self.is_pdcvrptw:
            if self.functional_time_encoding:
                node_dim = 4 + embedding_dim * 2
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

        if self.edge_feature_type == 'adjacency':
            self.edge_embed = nn.Embedding(2, embedding_dim)
        elif self.edge_feature_type == 'distance':
            self.edge_embed = nn.Linear(1, embedding_dim, bias=True)

        # Encoder model
        self.embedder = self.encoder(n_layers=self.n_encoder_layers,
                                     n_heads=n_heads,
                                     hidden_dim=embedding_dim,
                                     aggregation=aggregation,
                                     norm=normalization,
                                     learn_norm=learn_norm,
                                     track_norm=track_norm,
                                     gated=gated,)
        
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, return_times=False, mode='masked', pomo_batch_size=1, print_query_times=False):
        """
            Args:
                nodes: Coordinates of the nodes: static nodes first then dynamic nodes (batch_size, num_nodes, 2) dtype: torch.float32
                graph: Graph as a Boolean adjacency matrix (batch_size, num_nodes, num_nodes) dtype: torch.bool
                arrival_times: Arrival times of the nodes (batch_size, n_arrivals) dtype: torch.int64
                mode: 'masked' for embedding final graph and masking nodes which haven't arrived, 'recursive' for embedding graph as nodes arrive
        """

        assert mode in ['masked', 'recursive'], "Mode must be either 'masked' or 'recursive'"

        self.mode = mode

        if self.edge_feature_type == 'adjacency':
            self.edge_features = input['graph'].long() # just a placeholder for now, will eventually need to change for Real World Instances
        elif self.edge_feature_type == 'distance':
            self.edge_features = input['distance_matrix'].unsqueeze(-1)  # (batch_size, num_nodes, num_nodes, 1)

        if isinstance(self.embedder, IncrementalUpdateEncoder):

            log_p, pi, visit_times = self._inner(input, None, pomo_batch_size, print_query_times=print_query_times)

        elif self.mode == 'masked':
            
            embeddings = self.embedder(self._init_embed(input), self.edge_embed(self.edge_features), input['graph'])

            log_p, pi, visit_times = self._inner(input, embeddings, pomo_batch_size, print_query_times=print_query_times)

        elif self.mode == 'recursive':
            
            pi, visit_times = self._inner_recursive(input, print_query_times=print_query_times)

        visit_times = torch.tensor(visit_times, dtype=torch.float32, device=input['loc'].device)

        pi_w_depot = torch.cat((torch.zeros(pi.size(0), 1, dtype=torch.int64, device=pi.device), pi), dim=1)

        if self.is_pdtrptw or self.is_pdcvrptw:
            cost, info = self.problem.get_costs(input, move_to(pi_w_depot, device=input['loc'].device), move_to(visit_times, device=input['loc'].device), gamma=move_to(input['gamma'], device=input['loc'].device), theta=move_to(input['theta'], device=input['loc'].device))
        else:
            cost, info = self.problem.get_costs(input, move_to(pi_w_depot, device=input['loc'].device))

        if self.training:
            ll = self._calc_log_likelihood(log_p, mask=None)
        elif self.eval:
            ll = None

        if return_pi and return_times:
            return cost, ll, info, pi_w_depot, visit_times
        elif return_pi:
            return cost, ll, info, pi_w_depot
        elif return_times:
            return cost, ll, info, visit_times
        else:
            return cost, ll, info
        
    def _calc_log_likelihood(self, log_p, mask):

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, input, embeddings, pomo_batch_size=1, print_query_times=False):
        # create lists to track the log probabilities and the selected nodes
        outputs = []
        sequences = []

        # create the problem state for tracking which nodes are masked
        state = self.problem.make_state(input)

        if embeddings is not None:
            fixed = self._precompute(embeddings)

        batch_size = int(input['loc'].size(0) / pomo_batch_size)

        if self.use_arrival_lstm:
            self.lstm_hidden = (torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), device=input['loc'].device),torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), device=input['loc'].device))

        while not state.all_finished():

            # If we are in a POMO setting, we select the start nodes for each instance in the batch

            if pomo_batch_size > 1 and state.i == 0:
                selected = torch.arange(1, pomo_batch_size + 1, device=input['loc'].device).repeat(batch_size)
                log_p = torch.zeros((batch_size * pomo_batch_size, 1, input['loc'].size(1)), device=input['loc'].device)
            else:
                mask = state.get_mask()

                if isinstance(self.embedder, IncrementalUpdateEncoder):
                    temporal_mask = ~(state.not_arrived_[:, 0, :])
                    embeddings = self.embedder(self._init_embed(input), self.edge_embed(self.edge_features), input['graph'], temporal_mask)
                    fixed = self._precompute(embeddings)

                if print_query_times:
                    start_time = time.time()

                # get log probabilities of next action
                log_p, mask = self._get_log_p(fixed, state, mask)

                if print_query_times:
                    end_time = time.time()
                    print(f"Query time: {end_time - start_time:.4f} seconds")

                # we query the policy
                selected = self._select_node(
                    log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
                
            #update problem state
            state = state.update(selected)

            # if we're using the decoder LSTM, we need to input any arrivals that occurred to it
            if self.use_arrival_lstm:
                arrival_sequences = []
                for i in range(batch_size):
                    if state.arrival_occured[i]:
                        arrival_indices = torch.nonzero(torch.logical_and(~state.get_mask()[i, 0], mask[i, 0]), as_tuple=False).squeeze(-1).to(input['loc'].device)
                        if self.scale_times:
                            arrival_times = scale_time(input['arrival_times'][i, arrival_indices][:, None], input['time_horizon'][0])
                        else:
                            arrival_times = input['arrival_times'][i, arrival_indices][:, None]
                        if self.functional_time_encoding:
                            arrival_times = self.time_encoding(arrival_times)
                        arrival_features = torch.cat([torch.gather(embeddings[i, :, :], 0, arrival_indices[:, None].expand(-1, embeddings.size(-1))), arrival_times], dim=-1)
                    else: 
                        arrival_features = torch.empty((0,))
                    arrival_sequences.append(arrival_features)

                self.update_lstm_with_arrivals_packed(arrival_sequences)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        all_sequences = torch.stack(sequences, 1)
        all_outputs = torch.stack(outputs, 1).gather(2, all_sequences.unsqueeze(2)).squeeze(-1)

        return all_outputs, all_sequences, state.visit_times
    
    def _inner_recursive(self, input, print_query_times=False):
        # If we want to run evaluations true to reality then we'll have to generate the graph and embeddings as customers arrive
        # note that due to it being annoying to do this for anything other than a single instance, this function is designed to only work for a single instance
        assert input['loc'].size(0) == 1, "This forward method doesn't play nicely with batched inputs"

        batch_size = input['loc'].size(0)

        # This bit is the same as the training inner method
        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        if self.recursively_remove_visited_nodes:
            arrival_mask = torch.logical_and(~state.not_arrived_[:,0,:], ~state.visited_[:,0,:])
            arrival_mask[0, 0] = True  # Ensure the depot is always included
        else:
            arrival_mask = ~state.not_arrived_[:,0,:]

        masked_input = self._mask_input(input, arrival_mask)

        if self.edge_feature_type == 'adjacency':
            edge_features = masked_input['graph'].long()
        elif self.edge_feature_type == 'distance':
            edge_features = masked_input['distance_matrix'].unsqueeze(-1)

        embeddings = self.embedder(self._init_embed(masked_input), self.edge_embed(edge_features), masked_input['graph'])

        decoder_input = self._precompute(embeddings)

        if self.use_arrival_lstm:
            self.lstm_hidden = (torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), device=input['loc'].device),torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size), device=input['loc'].device))

        prev_selected=None

        # This is the main loop for the decoder
        while not state.all_finished():
            if self.recursively_remove_visited_nodes:
                arrival_mask = torch.logical_and(~state.not_arrived_[:,0,:], ~state.visited_[:,0,:])
                arrival_mask[0, 0] = True  # Ensure the depot is always included
                mask = state.get_mask()[arrival_mask.unsqueeze(0)][None, None, :]
                self.current_indices = torch.arange(0, input['loc'].size(1), device=input['loc'].device)[arrival_mask[0]]
            else:
                mask = state.get_mask()[~state.not_arrived_][None, None, :]

            if print_query_times:
                start_time = time.perf_counter()

            # get log probabilities of next action
            log_p, mask = self._get_log_p(decoder_input, state, mask, prev_selected=prev_selected)

            if print_query_times:
                end_time = time.perf_counter()
                print(f"Query time: {end_time - start_time:.4f} seconds")

            # we query the policy
            selected = self._select_node(
                log_p.exp()[:, 0, :], mask[:, 0, :])
            
            if self.recursively_remove_visited_nodes:
                #we need to correct selected to account for the visited nodes we've removed
                true_selected = self.current_indices[selected]
                state = state.update(true_selected)
                # Collect output of step
                outputs.append(log_p[:, 0, :])
                sequences.append(true_selected)
            else:
                state = state.update(selected)
                # Collect output of step
                outputs.append(log_p[:, 0, :])
                sequences.append(selected)

            # if an arrival has occurred, we need to update the embeddings and decoder input
            if self.recursively_remove_visited_nodes and not state.all_finished():
                arrival_mask = torch.logical_and(~state.not_arrived_[:,0,:], ~state.visited_[:,0,:])
                arrival_mask[0, 0] = True  # Ensure the depot is always included
                arrival_mask = arrival_mask.unsqueeze(0)
                masked_input = self._mask_input(input, arrival_mask[:,0,:])
                if self.edge_feature_type == 'adjacency':
                    edge_features = masked_input['graph'].long()
                elif self.edge_feature_type == 'distance':
                    edge_features = masked_input['distance_matrix'].unsqueeze(-1)
                prev_selected = embeddings[0, selected].unsqueeze(0)
                embeddings = self.embedder(self._init_embed(masked_input), self.edge_embed(edge_features), masked_input['graph'])
                decoder_input = self._precompute(embeddings)
            elif state.arrival_occured:
                masked_input = self._mask_input(input, ~state.not_arrived_[:,0,:])
                if self.edge_feature_type == 'adjacency':
                    edge_features = masked_input['graph'].long()
                elif self.edge_feature_type == 'distance':
                    edge_features = masked_input['distance_matrix'].unsqueeze(-1)
                embeddings = self.embedder(self._init_embed(masked_input), self.edge_embed(edge_features), masked_input['graph'])
                decoder_input = self._precompute(embeddings, num_steps=1)
            if self.use_arrival_lstm and state.arrival_occured:
                arrival_sequences = []
                # If we're using the decoder LSTM, we need to input any arrivals that occurred to it
                arrival_indices = torch.tensor(range(mask.size(-1), masked_input['loc'].size(1))).to(input['loc'].device)
                if arrival_indices.numel() > 0:
                    if self.scale_times:
                        arrival_times = scale_time(masked_input['arrival_times'][0, arrival_indices][:, None], masked_input['time_horizon'][0])
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

        return all_sequences, state.visit_times

    def _mask_input(self, input, mask):
        """The job of this function is to mask the input dictionary prior to being passed to the encoder"""

        masked_input = {}
        # first step is to mask the coordinates, very simple
        masked_input['loc'] = input['loc'][mask].unsqueeze(0)  # (batch_size, num_nodes, 2)
        masked_input['arrival_times'] = input['arrival_times'][mask].unsqueeze(0)  # (batch_size, num_nodes)
        # for now, going to do this the easy way and simply put the masked locations into the nearest neighbour function and get the new distance matrix and graph from that
        locs_on_cpu = masked_input['loc'].squeeze(0).cpu().numpy()
        nn_graph, distance_matrix = nearest_neighbor_graph(locs_on_cpu, self.neighbors, self.knn_strat)
        masked_input['graph'] = ~torch.BoolTensor(nn_graph).to(masked_input['loc'].device)
        masked_input['distance_matrix'] = torch.FloatTensor(distance_matrix).to(masked_input['loc'].device)
        masked_input['time_horizon'] = input['time_horizon']
        if 'window_starts' in input:
            masked_input['window_starts'] = input['window_starts'][mask].unsqueeze(0)
        if 'window_ends' in input:
            masked_input['window_ends'] = input['window_ends'][mask].unsqueeze(0)
        if 'demand' in input:
            masked_input['demand'] = input['demand'][mask].unsqueeze(0)
        
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

    def _get_log_p(self, fixed, state, mask=None, normalize=True, prev_selected=None):
        # Compute query = context node embedding, as far as I can tell this is not concatenation as described in the paper. Maybe it is equivalent though. 
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, prev_selected=prev_selected))


        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
        
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
        if prev_selected is None:
            current_node = state.get_current_node()
            batch_size, num_steps = current_node.size()
            context = torch.gather(
                embeddings,
                1,
                current_node.contiguous().view(batch_size, num_steps, 1)
                .expand(batch_size, num_steps, embeddings.size(-1))
            ).view(batch_size, num_steps, embeddings.size(-1))

        else:
            context = prev_selected

        if self.use_time_feature:
            if self.functional_time_encoding:
                context = torch.cat(
                    (
                        context,
                        self.time_encoding(state.get_timestep(normalize=self.scale_times)[:,:, None])
                    ),
                    -1
                )
            else:
                context = torch.cat((context, state.get_timestep(normalize=self.scale_times)[:,:, None]), -1)
        
        if self.is_pdcvrp or self.is_pdcvrptw:
            context = torch.cat(
                (
                    context,
                    state.get_remaining_capacity()[:, :, None]
                ),
                -1
            )

        if self.use_arrival_lstm: 
            h, _ = self.lstm_hidden
            h = h[-1].unsqueeze(1) # Get the last layer of the LSTM hidden state and unsqueeze to match context shape
            context = torch.cat(
                (
                    context,
                    h
                ),
                -1
            )

        return context

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
    
    def _init_embed(self, input):
        node_feats  = input['loc'][:, 1:, :]
        if self.is_pdcvrp or self.is_pdcvrptw:
            node_feats = torch.cat((node_feats, input['demand'][:, 1:, None]), -1)
        if self.is_pdtrptw or self.is_pdcvrptw:
            if self.scale_times:
                window_starts = scale_time(input['window_starts'][:, 1:, None], input['time_horizon'][0])
                window_ends = scale_time(input['window_ends'][:, 1:, None], input['time_horizon'][0])
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
                arrival_times = scale_time(input['arrival_times'][:, 1:, None], input['time_horizon'][0])    
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
                self.init_embed_depot(input['loc'][:, 0, :])[:, None, :],
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
        sorted_lengths = [lengths[i] for i in sorted_idx]

        # Pack into [seq_total_len, input_dim] → PackedSequence
        packed = pack_sequence(sorted_sequences, enforce_sorted=True)

        # Extract initial states for only those envs with arrivals, sorted
        h_init = h[:, sorted_indices_in_batch, :]  # [num_layers, sub_batch, hidden_dim]
        c_init = c[:, sorted_indices_in_batch, :]

        # Step 2: Run the LSTM
        _, (h_out, c_out) = self.lstm(packed, (h_init, c_init))

        # Step 3: Put updated states back into the full batch
        h_new, c_new = h.clone(), c.clone()
        for pos, env_idx in enumerate(sorted_indices_in_batch):
            h_new[:, env_idx:env_idx+1, :] = h_out[:, pos:pos+1, :]
            c_new[:, env_idx:env_idx+1, :] = c_out[:, pos:pos+1, :]

        self.lstm_hidden = (h_new, c_new)
# TGAT ENCODER

import torch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class TimeEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.freqs = nn.Parameter(torch.randn(dim - 1))
        self.phases = nn.Parameter(torch.randn(dim - 1))

    def forward(self, t):
        lin = self.linear(t)
        arg = t * self.freqs.unsqueeze(0) + self.phases.unsqueeze(0)
        sin = torch.sin(arg)
        return torch.cat([lin, sin], dim=-1)

class TGNNLayer(nn.Module):
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
        super(TGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"
        self.time_encoding = TimeEncoding(hidden_dim)

        in_dim = hidden_dim * 2
        
        self.U = nn.Linear(in_dim, hidden_dim, bias=True)
        self.V = nn.Linear(in_dim, hidden_dim, bias=True)
        self.A = nn.Linear(in_dim, hidden_dim, bias=True)
        self.B = nn.Linear(in_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)
        
    def forward(self, h, e, graph, mask, time):
        """
        Args:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            mask: Temporal mask indicating which nodes are present at query time
        Returns: 
            Updated node and edge features
        """
        batch_size, num_nodes, hidden_dim = h.shape
        e = e.detach()  # Detach edge features to avoid gradients through edges
        h = h.detach()  # Detach node features to avoid gradients through nodes
        h_in = h
        e_in = e

        time_embedding = self.time_encoding(time.unsqueeze(1).expand(-1, num_nodes, -1))

        h_time = torch.cat([h, time_embedding], dim=-1)

        # Linear transformations for node update
        Uh = self.U(h_time)  # B x V x H
        Vh = self.V(h_time).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h_time)  # B x V x H
        Bh = self.B(h_time)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates, mask)  # B x V x H

        # Mask out target nodes that are inactive
        if mask is not None:
            h = h * mask.unsqueeze(-1)  # zero updates where mask == False

        # Normalize
        if self.norm_h:
            h = self.norm_h(h.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)
        if self.norm_e:
            e = self.norm_e(e.view(batch_size * num_nodes * num_nodes, hidden_dim)).view(batch_size, num_nodes, num_nodes, hidden_dim)

        # Non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Residual connection
        h = h_in + h
        e = e_in + e

        return h, e

    def aggregate(self, Vh, graph, gates, mask):
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
        
        # Keep only edges where neighbor exists in the graph
        # and the neighbor node is active in the temporal mask
        if mask is not None:
            # mask_neighbors: (B x 1 x V) -> broadcast to (B x V x V)
            neighbor_mask = mask.unsqueeze(1).expand(-1, graph.size(1), -1)  # (B x V x V)
            combined_mask = graph.bool() & neighbor_mask.bool()
        else:
            combined_mask = graph.bool()

        # Zero out non-neighbor contributions
        Vh = Vh * combined_mask.unsqueeze(-1)  # B x V x V x H  

        # Degree for each node (how many valid neighbors)
        deg = combined_mask.sum(dim=2)  # (B x V)

        if self.aggregation == "mean":
            # Avoid division by zero: set to 1 where deg == 0
            deg_safe = deg.clamp(min=1).unsqueeze(-1)  # (B x V x 1)
            out = Vh.sum(dim=2) / deg_safe
        elif self.aggregation == "max":
            # Mask out invalid edges
            Vh_masked = Vh.masked_fill(~combined_mask.unsqueeze(-1), float('-inf'))
            out = Vh_masked.max(dim=2)[0]
            # Where deg == 0, set to zero vector
            out[deg == 0] = 0
            return out
        else:  # sum
            out = Vh.sum(dim=2)

        # If no neighbors, return zero
        out[deg == 0] = 0
        return out
        

class TGNNEncoder(nn.Module):
    """Configurable GNN Encoder
    """
    
    def __init__(self, n_layers, hidden_dim, aggregation="sum", norm="layer", 
                 learn_norm=True, track_norm=False, gated=True, *args, **kwargs):
        super(TGNNEncoder, self).__init__()

        self.layers = nn.ModuleList([
            SparseTGNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
                for _ in range(n_layers)
        ])

    def forward(self, x, e, graph, mask, time):
        """
        Args:
            x: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node features (B x V x H)
        """

        for layer in self.layers:
            x, e = layer(x, e, graph, mask, time)

        return x

class SparseTGNNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="sum", norm="layer",
                 learn_norm=True, track_norm=False, gated=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.gated = gated

        self.time_encoding = TimeEncoding(hidden_dim)

        in_dim = hidden_dim * 2

        self.U = nn.Linear(in_dim, hidden_dim, bias=True)
        self.V = nn.Linear(in_dim, hidden_dim, bias=True)
        self.A = nn.Linear(in_dim, hidden_dim, bias=True)
        self.B = nn.Linear(in_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(norm, None)

    def forward(self, h, e, edge_index, mask, time):
        """
        h: (B x V x H) node features
        edge_index: (2 x E) edge list
        e: (B x E x H) edge features
        mask: (B x V) boolean node mask
        time: (B x V x 1) or (B x 1 x 1)
        """
        B, V, H = h.shape
        h_in, e_in = h, e

        # Add time encoding
        t_enc = self.time_encoding(time.unsqueeze(1).expand(-1, V, -1))  # (B x V x H)
        h_time = torch.cat([h, t_enc], dim=-1)

        # Linear transformations for node update
        Uh = self.U(h_time)  # B x V x H
        Vh = self.V(h_time) # B x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h_time)  # B x V x H
        Bh = self.B(h_time)  # B x V x H
        Ce = self.C(e)  # B x E x H

        # Compute edge gates
        src, dst = edge_index
        e_msg = Ah[:, src, :] + Bh[:, dst, :] + Ce
        gates = torch.sigmoid(e_msg)

        # Gated neighbor features
        Vh_src = Vh[:, src, :] * gates  # (B x E x H)

        # Mask out edges from inactive nodes
        if mask is not None:
            valid_edges = mask[:, src] & mask[:, dst]  # (B x E)
            valid_edges = valid_edges.unsqueeze(-1)  # (B x E x 1)
            Vh_src = Vh_src * valid_edges

        # Aggregate messages
        h_agg = torch.zeros_like(h)  # (B x V x H)
        deg = torch.zeros(B, V, device=h.device, dtype=torch.long)

        for b in range(B):
            if self.aggregation == "mean" or self.aggregation == "sum":
                h_agg[b].index_add_(0, dst, Vh_src[b])
                deg[b].index_add_(0, dst, valid_edges[b, :, 0].long())
            elif self.aggregation == "max":
                # Initialize with -inf for max pooling
                max_buf = torch.full((V, H), float('-inf'), device=h.device)
                for i in range(dst.size(0)):
                    if valid_edges[b, i, 0]:
                        v = Vh_src[b, i]
                        max_buf[dst[i]] = torch.max(max_buf[dst[i]], v)
                        deg[b, dst[i]] += 1
                # Replace -inf with 0 for isolated nodes
                max_buf[max_buf == float('-inf')] = 0
                h_agg[b] = max_buf

        if self.aggregation == "mean":
            deg_safe = deg.clamp(min=1).unsqueeze(-1)
            h_agg = h_agg / deg_safe
        # If no neighbors, output zero
        h_agg[deg == 0] = 0

        # Node update
        h = Uh + h_agg
        if self.norm_h:
            h = self.norm_h(h.view(B * V, H)).view(B, V, H)
        h = F.relu(h)
        h = h_in + h

        # Edge update
        if self.norm_e:
            e_msg = self.norm_e(e_msg.view(B * e.size(1), H)).view_as(e_msg)
        e_msg = F.relu(e_msg)
        e = e_in + e_msg

        return h, e


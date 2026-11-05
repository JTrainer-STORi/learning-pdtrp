import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def aggregate_neighbor_messages(messages, graph, aggregation):
    """Reduce messages over positive adjacency, with zero for empty neighborhoods."""
    neighbors = graph.bool().unsqueeze(-1)
    degree = neighbors.sum(dim=2)
    if aggregation == "max":
        result = messages.masked_fill(~neighbors, float('-inf')).max(dim=2)[0]
        return torch.where(degree > 0, result, torch.zeros_like(result))

    result = messages.masked_fill(~neighbors, 0).sum(dim=2)
    if aggregation == "mean":
        result = result / degree.clamp(min=1).to(messages.dtype)
    return result


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

class GNNLayer(nn.Module):
    """Configurable GNN Layer

    Implements the Gated Graph ConvNet layer:
        h_i' = h_i + ReLU ( Norm( U*h_i + Aggr.( sigma_ij * V*h_j) ) ),
        sigma_ij = sigmoid( e_ij ),
        e_ij' = e_ij + ReLU ( Norm( A*h_j + B*h_i + C*e_ij ) ),
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

        # Both updates use the previous layer's features (Joshi et al., Eq. 2–3).
        gates = torch.sigmoid(e_in)  # B x V x V x H
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H

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
        return aggregate_neighbor_messages(gates * Vh, graph, self.aggregation)
        

class GNNEncoder(nn.Module):
    """Configurable GNN Encoder
    """
    
    def __init__(self, n_layers, hidden_dim, aggregation="sum", norm="layer", 
                 learn_norm=True, track_norm=False, gated=True, *args, **kwargs):
        super(GNNEncoder, self).__init__()

        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
                for _ in range(n_layers)
        ])

    def forward(self, x, e, graph):
        """
        Args:
            x: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node features (B x V x H)
        """

        for layer in self.layers:
            x, e = layer(x, e, graph)

        return x

# New encoder experimenting with a temporal graph neural network (TGNN) encoder

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class TGNNLayer(nn.Module):
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

    def forward(self, h, e, graph, mask):
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
        h_in, e_in = h, e

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
    

class IncrementalUpdateEncoder(nn.Module):
    """
    Incremental encoder for the *dense* TGNN setting.
    - Caches full-graph (h, e, mask, graph) from last call.
    - On subsequent calls, only recomputes for affected nodes (newly active + neighbors).
    - Merges updated subgraph results back into cached tensors.
    """
    def __init__(self, n_layers, hidden_dim, aggregation="sum", norm="layer",
                 learn_norm=True, track_norm=False, gated=True, neighborhood_hops=1, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TGNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
            for _ in range(n_layers)
        ])
        self.neighborhood_hops = neighborhood_hops  # expand affected set by K hops

        # Caches
        self._cached_h = None
        self._cached_e = None
        self._cached_mask = None
        self._cached_graph = None  # (B x V x V), positive adjacency (1=edge, 0=no edge)

    # ---------- public API ----------
    def reset_cache(self):
        self._cached_h = None
        self._cached_e = None
        self._cached_mask = None
        self._cached_graph = None

    @torch.no_grad()
    def _topology_changed(self, graph):
        if self._cached_graph is None:
            return True
        # If topology differs in any batch graph, consider it changed.
        return not torch.equal(graph, self._cached_graph)

    def _expand_affected(self, graph_b, seeds, hops):
        """
        graph_b: (V x V) positive adjacency (1=edge,0=no edge)
        seeds: 1D LongTensor of initial node indices
        hops: int, number of hop-expansions
        returns: sorted unique LongTensor of affected nodes
        """
        if seeds.numel() == 0:
            return seeds
        affected = seeds
        for _ in range(hops):
            # neighbors of current set: any j where graph[affected, j] == 1
            nbr_mask = (graph_b[affected] == 1).any(dim=0)  # (V,)
            affected = torch.unique(torch.cat([affected, nbr_mask.nonzero(as_tuple=True)[0]]))
        return affected

    def _slice_subgraph(self, x_b, e_b, g_b, m_b, idx):
        """
        Returns per-batch subgraph tensors for nodes 'idx'
        x_b:   (V x H)
        e_b:   (V x V x H)
        g_b:   (V x V)
        m_b:   (V,)
        """
        return (
            x_b[idx],                                # (v' x H)
            e_b[idx][:, idx, :],                     # (v' x v' x H)
            g_b[idx][:, idx],                        # (v' x v')
            m_b[idx],                                # (v',)
        )

    def _scatter_back(self, h_full_b, e_full_b, h_sub_b, e_sub_b, idx):
        """
        Writes subgraph outputs back into full-graph tensors in-place.
        """
        h_full_b[idx] = h_sub_b
        e_full_b[idx][:, idx, :] = e_sub_b

    def forward(self, x, e, graph, mask, active_nodes=None, force_full=False):
        """
        x:     (B x V x H)   node features
        e:     (B x V x V x H) edge features
        graph: (B x V x V)   positive adjacency (1=edge, 0=no edge)
        mask:  (B x V)       bool, True=active node
        active_nodes: Optional[List[LongTensor] or LongTensor] nodes to force-update per batch.
                      If a single 1D LongTensor is given, it's applied to all batches.
        force_full: bool, run a full pass regardless of cache
        returns:
            h: (B x V x H) updated node features
        """
        B, V, H = x.shape

        # First call or forced full or topology changes -> full forward on entire graph
        if force_full or self._cached_h is None or self._topology_changed(graph):
            h_full, e_full = x, e
            for layer in self.layers:
                h_full, e_full = layer(h_full, e_full, graph, mask)
            # cache
            self._cached_h = h_full.detach()
            self._cached_e = e_full.detach()
            self._cached_mask = mask.detach().clone()
            self._cached_graph = graph.detach().clone()
            return h_full

        # Determine seeds of change
        if active_nodes is None:
            # Newly active nodes since last call
            newly_active = mask & (~self._cached_mask)
        else:
            # Normalize active_nodes to per-batch list of 1D tensors
            if isinstance(active_nodes, torch.Tensor):
                active_nodes = [active_nodes for _ in range(B)]
            elif isinstance(active_nodes, list):
                assert len(active_nodes) == B, "active_nodes list must have length B"
            newly_active = [torch.zeros(V, dtype=torch.bool, device=x.device) for _ in range(B)]
            for b in range(B):
                if active_nodes[b].numel() > 0:
                    newly_active[b][active_nodes[b]] = True

        # If no newly active nodes, return cached state
        # Prepare mutable copies of cached states
        h_out = self._cached_h.clone()
        e_out = self._cached_e.clone()

        # Process each batch independently on its subgraph
        for b in range(B):
            seeds = None
            if active_nodes is None:
                seeds = newly_active[b].nonzero(as_tuple=True)[0]
            else:
                seeds = active_nodes[b]

            if seeds.numel() == 0:
                # Nothing to update in this batch
                continue

            # Expand to neighborhood
            affected_idx = self._expand_affected(graph[b], seeds, self.neighborhood_hops)

            # Slice subgraph tensors
            x_sub, e_sub, g_sub, m_sub = self._slice_subgraph(
                x[b], e[b], graph[b], mask[b], affected_idx
            )
            # Add batch dimension back (1 x v' x ...)
            x_sub = x_sub.unsqueeze(0)
            e_sub = e_sub.unsqueeze(0)
            g_sub = g_sub.unsqueeze(0)
            m_sub = m_sub.unsqueeze(0)

            # Run the stack on subgraph only
            h_sub, e_sub_out = x_sub, e_sub
            for layer in self.layers:
                h_sub, e_sub_out = layer(h_sub, e_sub_out, g_sub, m_sub)

            # Scatter back into cached full tensors
            self._scatter_back(h_out[b], e_out[b], h_sub[0], e_sub_out[0], affected_idx)

        # Refresh cache (detach to avoid backprop through time unless you want it)
        self._cached_h = h_out
        self._cached_e = e_out
        self._cached_mask = mask.clone()
        self._cached_graph = graph.clone()

        return h_out

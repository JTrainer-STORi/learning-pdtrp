"""AlphaStar-style set encoder adapted from deepgroebner_pytorch.

The source backbone is AlphaStarPairTransformer in network_components.py:
per-token residual MLPs, residual global self-attention, then residual MLPs.
Routing's existing node/depot projections supply the input embeddings. There
are no positional encodings, so customer ordering has no semantic meaning.
"""

import torch
from torch import nn
from torch.nn import functional as F


class TokenResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, node_mask):
        x = x + self.ff(F.relu(self.norm(x)))
        return x.masked_fill(~node_mask.unsqueeze(-1), 0)


class GlobalSelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.0, batch_first=True
        )

    def forward(self, x, node_mask, return_attention=False):
        attn_input = F.relu(self.norm(x))
        delta, weights = self.attn(
            attn_input, attn_input, attn_input,
            key_padding_mask=~node_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = (x + delta).masked_fill(~node_mask.unsqueeze(-1), 0)
        return (x, weights) if return_attention else x


class TransformerEncoder(nn.Module):
    """Global customer attention with the same output interface as GNNEncoder.

    ``node_mask`` marks tokens participating in attention. Revealed PPO uses
    the current request visibility here. Full-graph training includes every
    token and recursive inference passes an already restricted subgraph.
    Edge embeddings and KNN adjacency are unused.
    """

    uses_edge_features = False

    def __init__(self, n_layers=3, hidden_dim=128, transformer_heads=2,
                 transformer_resblocks_before=2, transformer_resblocks_after=2,
                 transformer_resblock_hidden=128, norm='layer', **kwargs):
        super().__init__()
        if norm != 'layer':
            raise ValueError('The AlphaStar-style transformer requires layer normalization')
        if hidden_dim <= 0 or transformer_heads <= 0 or hidden_dim % transformer_heads:
            raise ValueError('Transformer heads must be positive and divide embedding_dim')
        if n_layers < 1:
            raise ValueError('The transformer requires at least one attention layer')
        if min(transformer_resblocks_before, transformer_resblocks_after) < 0:
            raise ValueError('Transformer residual block counts must be nonnegative')
        if transformer_resblock_hidden <= 0:
            raise ValueError('Transformer residual hidden dimension must be positive')
        self.pre_blocks = nn.ModuleList([
            TokenResidualBlock(hidden_dim, transformer_resblock_hidden)
            for _ in range(transformer_resblocks_before)
        ])
        self.attn_blocks = nn.ModuleList([
            GlobalSelfAttentionBlock(hidden_dim, transformer_heads)
            for _ in range(n_layers)
        ])
        self.post_blocks = nn.ModuleList([
            TokenResidualBlock(hidden_dim, transformer_resblock_hidden)
            for _ in range(transformer_resblocks_after)
        ])

    def forward(self, x, e=None, graph=None, node_mask=None, return_attention=False):
        if x.dim() != 3 or x.size(1) == 0:
            raise ValueError('Expected a nonempty batch of node embeddings [B, N, E]')
        if node_mask is None:
            node_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            node_mask = node_mask.to(device=x.device, dtype=torch.bool)
        if node_mask.shape != x.shape[:2] or not node_mask.any(dim=-1).all():
            raise ValueError('Every graph must have at least one nonpadding node')
        x = x.masked_fill(~node_mask.unsqueeze(-1), 0)
        for block in self.pre_blocks:
            x = block(x, node_mask)
        attention_maps = {}
        for index, block in enumerate(self.attn_blocks):
            if return_attention:
                x, weights = block(x, node_mask, return_attention=True)
                attention_maps[f'encoder_{index}'] = {
                    'weights': weights, 'q_mask': node_mask, 'k_mask': node_mask,
                }
            else:
                x = block(x, node_mask)
        for block in self.post_blocks:
            x = block(x, node_mask)
        return (x, attention_maps) if return_attention else x

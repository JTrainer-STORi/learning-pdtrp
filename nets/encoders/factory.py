"""Resolve the same encoder architecture for training and checkpoint loading."""


def get_encoder_class(encoder='gnn', use_incremental_encoder=False):
    if encoder == 'gnn':
        if use_incremental_encoder:
            from nets.encoders.tgnn_encoder import IncrementalUpdateEncoder
            return IncrementalUpdateEncoder
        from nets.encoders.gnn_encoder import GNNEncoder
        return GNNEncoder
    if encoder == 'transformer':
        if use_incremental_encoder:
            raise ValueError('The transformer cannot use the incremental GNN encoder')
        from nets.encoders.transformer_encoder import TransformerEncoder
        return TransformerEncoder
    raise ValueError(f'Unknown encoder: {encoder}')

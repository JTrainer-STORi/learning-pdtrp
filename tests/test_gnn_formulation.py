import math

import pytest
import torch

from nets.encoders.gnn_encoder import GNNLayer
from nets.encoders.tgnn_encoder import TGNNLayer


@pytest.mark.parametrize("layer_class", [GNNLayer, TGNNLayer])
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize(
    "aggregation, expected",
    [("sum", [-2.5, -2.0, 0.0]), ("mean", [-1.25, -2.0, 0.0]),
     ("max", [-1.0, -2.0, 0.0])],
)
def test_neighbor_reduction_excludes_absent_edges_and_handles_isolated_nodes(
    layer_class, dtype, aggregation, expected
):
    layer = layer_class(1, aggregation=aggregation, norm=None)
    graph = torch.tensor([[[0, 1, 1], [1, 0, 0], [0, 0, 0]]], dtype=dtype)
    messages = torch.tensor([-4.0, -2.0, -3.0]).view(1, 1, 3, 1)
    messages = messages.expand(1, 3, 3, 1).clone().requires_grad_()
    gates = torch.full_like(messages, 0.5)
    if layer_class is TGNNLayer:
        actual = layer.aggregate(messages, graph, gates, mask=None)
    else:
        actual = layer.aggregate(messages, graph, gates)

    torch.testing.assert_close(actual.flatten(), torch.tensor(expected))
    actual.sum().backward()
    assert torch.isfinite(messages.grad).all()
    assert messages.grad[~graph.bool()].eq(0).all()


@pytest.mark.parametrize("aggregation", ["sum", "mean", "max"])
def test_temporal_aggregation_excludes_inactive_neighbors(aggregation):
    layer = TGNNLayer(1, aggregation=aggregation, norm=None)
    graph = ~torch.eye(3, dtype=torch.bool)[None]
    messages = torch.tensor([-4.0, -2.0, 100.0]).view(1, 1, 3, 1).expand(1, 3, 3, 1)
    mask = torch.tensor([[True, True, False]])
    result = layer.aggregate(messages, graph, torch.ones_like(messages), mask)
    third = -3.0 if aggregation == "mean" else (-2.0 if aggregation == "max" else -6.0)
    torch.testing.assert_close(result.flatten(), torch.tensor([-2.0, -4.0, third]))


@pytest.mark.parametrize("layer_class", [GNNLayer, TGNNLayer])
def test_node_and_edge_updates_use_previous_layer_states(layer_class):
    layer = layer_class(1, aggregation="sum", norm=None)
    with torch.no_grad():
        for name, weight in [("U", 0.0), ("V", 1.0), ("A", 1.0), ("B", 2.0), ("C", 3.0)]:
            linear = getattr(layer, name)
            linear.weight.fill_(weight)
            linear.bias.zero_()
    nodes = torch.tensor([[[1.0], [2.0]]])
    edges = torch.full((1, 2, 2, 1), math.log(3), requires_grad=True)
    graph = torch.tensor([[[False, True], [True, False]]])
    if layer_class is TGNNLayer:
        updated_nodes, updated_edges = layer(nodes, edges, graph, torch.ones(1, 2, dtype=torch.bool))
    else:
        updated_nodes, updated_edges = layer(nodes, edges, graph)

    # sigmoid(log(3)) = 3/4, independent of the affine edge update.
    torch.testing.assert_close(updated_nodes, torch.tensor([[[2.5], [2.75]]]))
    expected_edges = edges + torch.relu(
        nodes.unsqueeze(1) + 2 * nodes.unsqueeze(2) + 3 * edges
    )
    torch.testing.assert_close(updated_edges, expected_edges)
    updated_nodes.sum().backward()
    torch.testing.assert_close(edges.grad[0, :, :, 0], torch.tensor([[0.0, 0.375], [0.1875, 0.0]]))

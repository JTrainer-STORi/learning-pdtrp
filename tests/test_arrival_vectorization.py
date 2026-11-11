"""Numerical equivalence to the former per-instance arrival-history updates."""

import copy
from types import MethodType

import pytest
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems import PDTRPTW, PDCTRPTW
from utils.data_utils import dataset_to_input


def make_model(problem=PDTRPTW, scale_times=False, functional_time_encoding=False):
    model = DynamicAttentionModel(
        problem, embedding_dim=16, encoder_class=GNNEncoder, n_encode_layers=1,
        n_heads=4, edge_features="distance", use_arrival_lstm=True,
        scale_times=scale_times, functional_time_encoding=functional_time_encoding,
        knn_strat="fixed", neighbors=-1,
    )
    model.set_decode_type("greedy")
    return model


def rowwise_sequences(model, embeddings, arrival_times, newly_revealed, order, horizon):
    """Reference feature assembly used before batching the gather operations."""
    sequences = []
    for row in range(embeddings.size(0)):
        nodes = model._arrival_indices_by_time(arrival_times[row], newly_revealed[row])
        if nodes.numel():
            times = arrival_times[row, nodes, None]
            if model.scale_times:
                times = times / horizon
            if model.functional_time_encoding:
                times = model.time_encoding(times)
            features = torch.cat((embeddings[row, nodes], times), dim=-1)
        else:
            features = embeddings.new_empty((0, model.lstm.input_size))
        sequences.append(features)
    return sequences


def rowwise_update(model, sequences):
    """Reference packed LSTM with the former per-instance hidden-state writes."""
    h, c = model.lstm_hidden
    indices = [row for row, seq in enumerate(sequences) if len(seq)]
    if not indices:
        return
    indices.sort(key=lambda row: len(sequences[row]), reverse=True)
    packed = pack_sequence([sequences[row] for row in indices], enforce_sorted=True)
    _, (h_out, c_out) = model.lstm(packed, (h[:, indices], c[:, indices]))
    h_new, c_new = h.clone(), c.clone()
    for position, row in enumerate(indices):
        h_new[:, row:row+1] = h_out[:, position:position+1]
        c_new[:, row:row+1] = c_out[:, position:position+1]
    model.lstm_hidden = h_new, c_new


def assert_parameter_gradients_match(first, second):
    for (name, a), (other_name, b) in zip(first.named_parameters(), second.named_parameters()):
        assert name == other_name
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6, msg=name)


@pytest.mark.parametrize("scale_times", [False, True])
@pytest.mark.parametrize("functional_time_encoding", [False, True])
def test_batched_arrival_features_preserve_ties_empty_rows_and_gradients(
    scale_times, functional_time_encoding,
):
    model = make_model(scale_times=scale_times,
                       functional_time_encoding=functional_time_encoding).double()
    reference = copy.deepcopy(model)
    arrivals = torch.tensor([[0., 4., 2., 2., 8., 1.]] * 4, dtype=torch.float64, requires_grad=True)
    reference_arrivals = arrivals.detach().clone().requires_grad_()
    revealed = torch.tensor([[False, True, True, True, False, False],
                             [False] * 6,
                             [False, False, True, False, False, True],
                             [False, False, False, False, True, False]])
    embeddings = torch.randn(4, 6, 16, dtype=torch.float64, requires_grad=True)
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    order = arrivals.argsort(dim=1, stable=True)
    actual = model._batched_arrival_sequences(embeddings, arrivals, revealed, order, 600.)
    expected = rowwise_sequences(reference, reference_embeddings, reference_arrivals, revealed, order, 600.)
    assert [len(seq) for seq in actual] == [3, 0, 2, 1]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for sequences in (actual, expected):
        loss = sum((seq * torch.arange(1, len(seq)+1, dtype=seq.dtype)[:, None]).square().sum()
                   for seq in sequences)
        loss.backward()
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad)
    torch.testing.assert_close(arrivals.grad, reference_arrivals.grad)
    assert_parameter_gradients_match(model, reference)


def test_indexed_hidden_updates_preserve_multistep_gradients_and_unchanged_rows():
    model = make_model().double()
    model.lstm = torch.nn.LSTM(17, 16, num_layers=2, batch_first=True).double()
    reference = copy.deepcopy(model)
    initial = [torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    reference_initial = [value.detach().clone().requires_grad_() for value in initial]
    model.lstm_hidden = tuple(initial)
    reference.lstm_hidden = tuple(reference_initial)
    all_sequences, reference_sequences = [], []
    for lengths in ([1, 0, 3, 2], [0, 2, 0, 2], [0, 0, 0, 0]):
        sequences = [torch.randn(n, 17, dtype=torch.float64, requires_grad=True) for n in lengths]
        other = [value.detach().clone().requires_grad_() for value in sequences]
        before = model.lstm_hidden
        model.update_lstm_with_arrivals_packed(sequences)
        rowwise_update(reference, other)
        for actual, expected, previous in zip(model.lstm_hidden, reference.lstm_hidden, before):
            torch.testing.assert_close(actual, expected)
            for row, length in enumerate(lengths):
                if not length:
                    torch.testing.assert_close(actual[:, row], previous[:, row], rtol=0, atol=0)
        all_sequences.extend(sequences)
        reference_sequences.extend(other)
    for hidden in (model.lstm_hidden, reference.lstm_hidden):
        (hidden[0].square().sum() + hidden[1].sin().sum()).backward()
    for a, b in zip(initial + all_sequences, reference_initial + reference_sequences):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad)
    assert_parameter_gradients_match(model, reference)


@pytest.mark.parametrize("problem", [PDTRPTW, PDCTRPTW])
def test_complete_training_rollout_matches_rowwise_outputs_and_gradients(problem):
    dataset = problem.make_dataset(
        min_customers=5, max_customers=5, num_samples=3, batch_size=3,
        min_dod=.6, max_dod=.6, time_horizon_per_customer=.2,
        min_trips_required_lb=1, min_trips_required_ub=2,
    )
    inputs = dataset_to_input(next(iter(DataLoader(dataset, batch_size=3))), problem.NAME, "cpu")
    model = make_model(problem, scale_times=True).train()
    reference = copy.deepcopy(model)
    reference._batched_arrival_sequences = MethodType(rowwise_sequences, reference)
    reference.update_lstm_with_arrivals_packed = MethodType(rowwise_update, reference)
    outputs = [m(inputs, return_pi=True, return_times=True) for m in (model, reference)]
    for a, b in zip(outputs[0][:2] + outputs[0][3:], outputs[1][:2] + outputs[1][3:]):
        torch.testing.assert_close(a, b)
    for cost, likelihood, *_ in outputs:
        (cost.detach() * likelihood).mean().backward()
    assert_parameter_gradients_match(model, reference)

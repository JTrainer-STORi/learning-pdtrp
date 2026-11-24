"""Check rollout scoring, online information and recursive evaluation."""

import csv

import pytest
import torch

import evaluate_trained_policy
from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from utils.policy_rollout import completion_costs, revealed_state


def input_data():
    coords = torch.tensor([[[0., 0.], [10., 0.], [2., 0.], [3., 0.], [4., 0.]]])
    return dict(
        all_nodes=coords, distance_matrix=torch.cdist(coords, coords),
        graph=(~torch.eye(5, dtype=torch.bool))[None],
        arrival_times=torch.tensor([[0., 5., 0., 0., 0.]]),
        service_times=torch.tensor([[0., 7., 13., 13., 0.]]),
        service_times_mean=torch.tensor([[2.]]),
        window_starts=torch.tensor([[0., 5., 20., 0., 0.]]),
        window_ends=torch.tensor([[0., 35., 40., 12., 60.]]),
        demand=torch.tensor([[0., .25, .25, .5, .75]]),
        speed=torch.ones(1, 1), vehicle_capacity=torch.ones(1, 1),
        time_horizon=torch.tensor([[60.]]), latest_end=torch.tensor([[10.]]),
        gamma=torch.tensor([[2.]]), theta=torch.tensor([[3.]]),
    )


def make_model(problem):
    model = DynamicAttentionModel(
        problem, embedding_dim=16, encoder_class=GNNEncoder, n_encode_layers=1,
        n_heads=2, normalization="layer", neighbors=-1, knn_strat="fixed",
        use_time_feature=True, use_arrival_times=True, scale_times=True,
        use_time_window_slack=problem in (PDTRPTW, PDCTRPTW),
        use_capacity_aware_lookahead=problem is PDCTRPTW,
    ).eval()
    model.set_decode_type("greedy")
    return model


def test_rollout_scores_waiting_and_reload_using_only_the_online_prior():
    data = input_data()
    model = make_model(PDCTRPTW)
    live = PDCTRPTW.make_state(data).update(torch.tensor([4]))
    known = ~live.not_arrived_[:, 0]
    revealed_input = model._mask_input(data, known)
    known_nodes = known[0].nonzero().flatten()
    planning = revealed_state(PDCTRPTW, live, revealed_input, known_nodes)

    assert known_nodes.tolist() == [0, 2, 3, 4]
    assert planning.prev_a.item() == 3
    assert planning.used_cap.item() == .75
    assert planning.service_times.tolist() == [[0., 2., 2., 2.]]
    assert planning.get_mask()[0, 0].tolist() == [False, False, True, True]

    def earliest_deadline(state, finished):
        mask = state.get_mask()[:, 0]
        mask[:, 0] &= ~finished
        # The depot is legal before a reload. Prefer any available customer.
        priority = torch.tensor([0., 1., 2., 1.])
        return priority.masked_fill(mask, -torch.inf).argmax(dim=1)

    # Serving customer 2 now fills the vehicle and defers customer 3 to t=27.
    # Both options travel 10 units and the early depot return serves 3 on time.
    costs = completion_costs(planning, torch.tensor([1, 0]), earliest_deadline,
                             gamma=data["gamma"], theta=data["theta"])
    torch.testing.assert_close(costs, torch.tensor([60., 30.]))
    assert live.not_arrived_[0, 0, 1] and live.cur_times.item() == 4
    assert torch.equal(data["service_times"], input_data()["service_times"])

    # Give the policy a strong preference for waiting at customer 2.
    with torch.inference_mode():
        fixed = model._precompute(model._encode(revealed_input))
        chosen = model._select_rollout_action(
            live, revealed_input, known_nodes, fixed, torch.tensor([[[-10., 0., -100., -100.]]]),
            live.get_mask().index_select(-1, known_nodes), width=1)
    assert chosen.item() == 0


def test_completion_keeps_the_terminal_cost_when_another_branch_continues():
    data = input_data()
    data["demand"][:, 1:] = .25
    model = make_model(PDCTRP)
    live = PDCTRP.make_state(data).update(torch.tensor([4]))
    known = ~live.not_arrived_[:, 0]
    planning = revealed_state(PDCTRP, live, model._mask_input(data, known), known[0].nonzero().flatten())

    def choose_next(state, finished):
        mask = state.get_mask()[:, 0]
        mask[:, 0] &= ~finished
        return torch.tensor([0., 1., 2., 1.]).masked_fill(mask, -torch.inf).argmax(dim=1)

    # The depot-first branch adds a decision. The other branch already finished
    # at customer 3, with travel 3 and a 3-unit return included in its score.
    costs = completion_costs(planning, torch.tensor([1, 0]), choose_next)
    torch.testing.assert_close(costs, torch.tensor([6., 10.]))


@pytest.mark.parametrize("problem", (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW))
def test_rollout_executes_valid_dynamic_routes_for_all_variants(problem):
    torch.manual_seed(123)
    model = make_model(problem)
    data = input_data()
    greedy_times, search_times = [], []
    with torch.inference_mode():
        greedy = model(data, mode="recursive", return_pi=True)
        timed_greedy = model(data, mode="recursive", return_pi=True, decision_times=greedy_times)
        searched = model(data, mode="recursive", return_pi=True, decision_times=search_times, rollout_candidates=2)

    torch.testing.assert_close(greedy[0], timed_greedy[0], rtol=0, atol=0)
    torch.testing.assert_close(greedy[3], timed_greedy[3], rtol=0, atol=0)
    assert torch.isfinite(searched[0]).all()
    assert sorted(node for node in searched[3][0].tolist() if node) == [1, 2, 3, 4]
    for times, result in ((greedy_times, timed_greedy), (search_times, searched)):
        assert len(times) >= result[3].size(1) - 1 and all(t >= 0 for t in times)


def test_evaluation_writes_rollout_and_decision_metrics(tmp_path, monkeypatch):
    data = tmp_path / "data.txt"
    data.write_text("0 0 1 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
                    "window_starts 0 0 0 window_ends 0 1000 1000 demands 0 0.4 0.4 "
                    "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n")
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(evaluate_trained_policy, "load_model", lambda _: (make_model(PDCTRPTW), {"seed": 0}))
    output = tmp_path / "results"
    manifest = evaluate_trained_policy.main([
        "--problem", "pdctrptw", "--checkpoint", str(checkpoint), "--datasets", str(data),
        "--output-dir", str(output), "--device", "cpu", "--num-samples", "1",
        "--rollout-candidates", "2", "--record-decision-times",
    ])
    assert manifest["rollout_candidates"] == 2
    summary, = manifest["datasets"]
    with (output / "data.decisions.csv").open() as file:
        decisions = list(csv.DictReader(file))
    assert len(decisions) == summary["decisions"] >= 2
    assert [int(row["decision"]) for row in decisions] == list(range(len(decisions)))
    assert summary["p95_decision_seconds"] > 0
    assert summary["max_lateness_penalty"] == 0

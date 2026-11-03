"""Read-only diagnostic examples for the chapter 5/7 thesis review.

Run from the repository root: .venv/bin/python docs/thesis_environment_probes.py
These report current behavior; they are not assertions that it matches the thesis.
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from evaluate_online_ortools import online_scenario
from utils.ortools_baselines import scenario_from_nodes


def make_input(coordinates):
    coords = torch.tensor([coordinates], dtype=torch.float32)
    size = coords.shape[1]
    return {
        "all_nodes": coords,
        "distance_matrix": torch.cdist(coords, coords),
        "graph": (~torch.eye(size, dtype=torch.bool))[None],
        "arrival_times": torch.zeros(1, size),
        "service_times": torch.zeros(1, size),
        "speed": torch.tensor([1.0]),
        "time_horizon": torch.tensor([480.0]),
        "latest_end": torch.tensor([120.0]),
        "window_starts": torch.zeros(1, size),
        "window_ends": torch.full((1, size), 1000.0),
        "demand": torch.tensor([[0.0] + [0.1] * (size - 1)]),
        "vehicle_capacity": torch.tensor([[1.0]]),
        "gamma": torch.tensor([1.0]),
        "theta": torch.tensor([1.0]),
    }


def initial_dynamic_requests():
    actual = make_input([[0, 0], [1, 0], [0, 1]])
    actual["arrival_times"] = torch.tensor([[0.0, 10.0, 20.0]])
    return {
        problem.NAME: {
            "finished": bool(problem.make_state(actual).all_finished()),
            "has_legal_action": bool((~problem.make_state(actual).get_mask()).any()),
        }
        for problem in (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)
    }


def forced_wait():
    actual = make_input([[0, 0], [1, 0], [0.1, 0]])
    actual["arrival_times"] = torch.tensor([[0.0, 0.0, 10.0]])
    actual["window_starts"] = torch.tensor([[0.0, 100.0, 10.0]])
    actual["window_ends"] = torch.tensor([[0.0, 200.0, 12.0]])
    output = {}
    for problem in (PDTRPTW, PDCTRPTW):
        state = problem.make_state(actual)
        executed_route = [0]
        state = state.update(torch.tensor([1]))
        first_selection_executed = bool(state.visited_[0, 0, 1])
        if first_selection_executed:
            executed_route.append(1)
        current_time = float(state.cur_times.item())
        new_customer_revealed = not bool(state.not_arrived_[0, 0, 2])
        # A selection may be canceled by an arrival during the pre-departure
        # wait. Complete only the actual pending visits, rather than treating
        # that selection as an executed leg of the route.
        for customer in (2, 1):
            if not bool(state.visited_[0, 0, customer]):
                state = state.update(torch.tensor([customer]))
                executed = getattr(state, "last_action_executed", None)
                if executed is None or bool(executed[0]):
                    executed_route.append(customer)
        if hasattr(state, "routes"):
            executed_route = list(state.routes[0])
        observed_times = (
            state.get_visit_times() if hasattr(state, "get_visit_times")
            else state.visit_times
        )
        _, observed_info = problem.get_costs(
            actual, torch.tensor([executed_route]), torch.as_tensor(observed_times),
            actual["gamma"], actual["theta"],
        )
        committed = torch.tensor([[0, 1, 2]])
        committed_times, _ = problem.get_times(actual, committed)
        _, committed_info = problem.get_costs(
            actual, committed, committed_times, actual["gamma"], actual["theta"]
        )
        alternate = torch.tensor([[0, 2, 1]])
        alternate_times, _ = problem.get_times(actual, alternate)
        _, alternate_info = problem.get_costs(
            actual, alternate, alternate_times, actual["gamma"], actual["theta"]
        )
        output[problem.NAME] = {
            "first_next_decision_time": current_time,
            "new_customer_call_time": 10.0,
            "new_customer_revealed_at_next_decision": new_customer_revealed,
            "first_selection_executed": first_selection_executed,
            "executed_route": executed_route,
            "executed_route_customer_lateness": float(
                observed_info["lateness_penalty"].item()
            ),
            "committed_route_customer_lateness": float(
                committed_info["lateness_penalty"].item()
            ),
            "wait_for_call_then_serve_new_customer_lateness": float(
                alternate_info["lateness_penalty"].item()
            ),
        }
    return output


def service_time_information():
    actual = make_input([[0, 0], [1, 0], [0, 1]])
    actual["service_times"][0, 1] = 12.0
    actual["service_times_mean"] = torch.tensor([3.0])
    actual["service_times_std"] = torch.tensor([np.sqrt(5.0)])
    state = PDTRPTW.make_state(actual)
    scenario = online_scenario("pdtrptw", actual, state, 9)
    clairvoyant = scenario_from_nodes("pdtrptw", actual, 0, [1, 2], 0.0, 9)
    return {
        "actual_unserved_customer_service_time": 12.0,
        "service_time_given_to_online_solver": float(scenario["service"][1]),
        "service_time_given_to_clairvoyant_solver": float(clairvoyant["service"][1]),
        "thesis_default_expected_service_time": 3.0,
    }


@torch.inference_mode()
def future_information():
    torch.manual_seed(1234)
    model = DynamicAttentionModel(
        PDTRP, embedding_dim=32, encoder_class=GNNEncoder,
        n_encode_layers=1, n_heads=4, edge_features="distance",
    ).eval()
    output = {}
    for mode in ("masked", "recursive"):
        probabilities = []
        for future in ([1, 1], [50, 50]):
            actual = make_input([[0, 0], [1, 0], [0, 1], future])
            actual["arrival_times"][0, 3] = 100.0
            state = PDTRP.make_state(actual)
            mask = state.get_mask()
            if mode == "recursive":
                known = ~state.not_arrived_[:, 0, :]
                features = model._mask_input(actual, known)
                mask = mask[:, :, known[0]]
            else:
                features = actual
            edges = features["distance_matrix"].unsqueeze(-1)
            embeddings = model.embedder(
                model._init_embed(features), model.edge_embed(edges), features["graph"]
            )
            fixed = model._precompute(embeddings)
            log_probabilities, _ = model._get_log_p(fixed, state, mask)
            probabilities.append(log_probabilities[0, 0, 1:3].exp())
        output[mode] = {
            "known_customer_probabilities_original_future": probabilities[0].tolist(),
            "known_customer_probabilities_changed_future": probabilities[1].tolist(),
            "maximum_probability_change": float((probabilities[0] - probabilities[1]).abs().max()),
        }
    return output


def generated_windows():
    np.random.seed(1234)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        dataset = PDTRPTW.make_dataset(
            min_total=20, max_total=20, min_dod=0.5, max_dod=0.5,
            n_subregions=1, batch_size=1, num_samples=1,
        )
    item = dataset[0]
    dynamic = item["arrival_times"] > 0
    generated_delay = item["window_starts"][dynamic] - item["arrival_times"][dynamic]
    depot_travel = item["distance_matrix"][0, dynamic] / item["speed"].item()
    return {
        "generated_opening_minus_call_minutes": generated_delay.tolist(),
        "thesis_opening_minus_call_minutes_for_same_coordinates": depot_travel.tolist(),
    }


def main():
    torch.set_num_threads(1)
    result = {
        "initial_all_dynamic": initial_dynamic_requests(),
        "forced_wait": forced_wait(),
        "service_time_information": service_time_information(),
        "future_information": future_information(),
        "generated_windows": generated_windows(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

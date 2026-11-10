"""Online decisions must execute useful depot returns before capacity is full."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_msa_ortools as msa
import evaluate_online_ortools as online
from problems import PDCTRPTW
from utils.data_utils import dataset_to_input


def item():
    coords = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [1., 1.]])
    return dict(all_nodes=coords, distance_matrix=torch.cdist(coords, coords),
                graph=~torch.eye(4, dtype=torch.bool), speed=torch.tensor([1.]),
                time_horizon=torch.tensor([200.]), latest_end=torch.tensor([0.]),
                arrival_times=torch.zeros(4), service_times=torch.zeros(4),
                service_times_mean=torch.tensor([0.]), service_times_std=torch.tensor([0.]),
                demand=torch.tensor([0., .6, .5, .1]), vehicle_capacity=torch.tensor([1.]),
                window_starts=torch.tensor([0., 0., 0., 100.]),
                window_ends=torch.tensor([0., 2., 10., 200.]),
                gamma=torch.tensor([1.]), theta=torch.tensor([1.]))


def args():
    return SimpleNamespace(problem="pdctrptw", scenarios=1, workers=1, dod=0.,
                           min_trips=1, max_trips=9, n_subregions=1,
                           solver_time_ms=1, solver_no_improvement_ms=None,
                           online_time_ms=1, online_no_improvement_ms=None)


def sample_known_only(monkeypatch):
    monkeypatch.setattr(PDCTRPTW, "make_dataset", lambda **kwargs: [item()])


def test_msa_first_stage_includes_a_useful_return_but_not_a_redundant_one(monkeypatch):
    sample_known_only(monkeypatch)
    actual=dataset_to_input(item(), "pdctrptw", "cpu")
    at_depot=PDCTRPTW.make_state(actual)
    after_service=at_depot.update(torch.tensor([1]))
    assert ~after_service.get_mask()[0,0,0]
    assert ~after_service.get_mask()[0,0,3]  # Late-opening customer still fits.
    assert after_service.get_mask()[0,0,2]  # Urgent customer needs a reload.
    for state, include_depot in ((at_depot,False),(after_service,True)):
        scenario=msa.make_scenario(PDCTRPTW,"pdctrptw",actual,state,4,0.,1,9)
        actions=[scenario["mapping"][n] for n in scenario["first_nodes"]]
        assert (0 in actions) == include_depot
        assert None not in actions


@pytest.mark.parametrize("reload_only",[False,True])
def test_reload_first_seed_clears_load_and_serves_urgent_customer(monkeypatch,reload_only):
    sample_known_only(monkeypatch)
    actual=dataset_to_input(item(), "pdctrptw", "cpu")
    state=PDCTRPTW.make_state(actual).update(torch.tensor([1]))
    scenario=msa.make_scenario(PDCTRPTW,"pdctrptw",actual,state,4,0.,1,9)
    if reload_only:
        scenario["first_nodes"]=[min(scenario["reload_nodes"])]
    details=msa.solve_scenario(scenario,"pdctrptw",1,return_details=True)
    route=[scenario["mapping"][n] for n in details["route"]]
    assert route[0] == 0
    assert details["initial_route_method"].endswith("after_reload")
    assert details["solver_objective"] < 10
    for action in route:
        state=state.update(torch.tensor([action]))
    assert state.all_finished()
    assert state.visit_times[0][state.routes[0].index(2)] <= 10


@pytest.mark.parametrize("method",["msa","online"])
def test_online_policy_returns_before_waiting_for_late_opening_customer(monkeypatch,method):
    sample_known_only(monkeypatch)
    np.random.seed(0)
    parameters=args()
    if method == "msa":
        value,_=msa.evaluate_instance(parameters,PDCTRPTW,item())
    else:
        value,_,tour=online.evaluate_online(parameters,PDCTRPTW,
                                           dataset_to_input(item(),"pdctrptw","cpu"))
        # This is the returned plan's depot action, rather than a skipped return
        # followed by waiting until t=100 for the only customer still fitting.
        assert tour[:4] == [0,1,0,2]
    assert value < 10

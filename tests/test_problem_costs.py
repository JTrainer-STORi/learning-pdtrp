import pytest
import torch

from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW


@pytest.fixture
def square_input():
    nodes = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
    )
    return {
        "all_nodes": nodes,
        "distance_matrix": torch.cdist(nodes, nodes),
        "arrival_times": torch.zeros(1, 4),
        "service_times": torch.zeros(1, 4),
        "speed": torch.tensor([1.0]),
    }


@pytest.mark.parametrize("problem", [PDTRP, PDTRPTW])
def test_uncapacitated_cost_is_closed_tour_length(problem, square_input):
    tour = torch.tensor([[0, 1, 2, 3]])

    if problem is PDTRP:
        cost, info = problem.get_costs(square_input, tour)
    else:
        problem_input = {**square_input, "window_ends": torch.full((1, 4), 100.0)}
        visit_times = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
        cost, info = problem.get_costs(
            problem_input, tour, visit_times, torch.tensor([1.0]), torch.tensor([1.0])
        )

    torch.testing.assert_close(cost, torch.tensor([4.0]))
    if problem is PDTRPTW:
        torch.testing.assert_close(info["lateness_penalty"], torch.tensor([0.0]))


def test_time_window_cost_combines_distance_and_lateness(square_input):
    problem_input = {**square_input, "window_ends": torch.tensor([[0.0, 0.5, 1.0, 2.5]])}
    tour = torch.tensor([[0, 1, 2, 3]])
    visit_times = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])

    cost, info = PDTRPTW.get_costs(
        problem_input,
        tour,
        visit_times,
        gamma=torch.tensor([2.0]),
        theta=torch.tensor([3.0]),
    )

    # Lateness is 0.5 + 1.0 + 0.5; distance is 4.
    torch.testing.assert_close(info["lateness_penalty"], torch.tensor([2.0]))
    torch.testing.assert_close(info["distance_penalty"], torch.tensor([4.0]))
    torch.testing.assert_close(cost, torch.tensor([16.0]))


def test_time_window_cost_uses_per_instance_weights():
    distance_matrix = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    problem_input = {
        "distance_matrix": distance_matrix,
        "window_ends": torch.zeros(2, 2),
    }
    tour = torch.tensor([[0, 1], [0, 1]])
    visit_times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    cost, _ = PDTRPTW.get_costs(
        problem_input,
        tour,
        visit_times,
        gamma=torch.tensor([1.0, 10.0]),
        theta=torch.tensor([1.0, 2.0]),
    )

    torch.testing.assert_close(cost, torch.tensor([3.0, 14.0]))


def test_capacity_validation_uses_per_instance_capacity():
    problem_input = {
        "distance_matrix": torch.zeros(2, 2, 2),
        "demand": torch.tensor([[0.0, 0.75], [0.0, 0.75]]),
        "vehicle_capacity": torch.tensor([[1.0], [0.5]]),
    }
    tour = torch.tensor([[0, 1], [0, 1]])

    with pytest.raises(AssertionError, match="capacity"):
        PDCTRP.get_costs(problem_input, tour)


def test_capacitated_cost_accepts_depot_returns(square_input):
    problem_input = {
        **square_input,
        "demand": torch.tensor([[0.0, 0.6, 0.6, 0.4]]),
        "vehicle_capacity": torch.tensor([1.0]),
    }
    tour = torch.tensor([[0, 1, 0, 2, 3]])

    cost, _ = PDCTRP.get_costs(problem_input, tour)

    expected = 1 + 1 + 2**0.5 + 1 + 1
    torch.testing.assert_close(cost, torch.tensor([expected]))


def test_capacitated_cost_rejects_over_capacity_route(square_input):
    problem_input = {
        **square_input,
        "demand": torch.tensor([[0.0, 0.6, 0.6, 0.4]]),
        "vehicle_capacity": torch.tensor([1.0]),
    }

    with pytest.raises(AssertionError, match="capacity"):
        PDCTRP.get_costs(problem_input, torch.tensor([[0, 1, 2, 3]]))


def test_capacitated_time_window_cost_ignores_depot_lateness(square_input):
    problem_input = {
        **square_input,
        "demand": torch.tensor([[0.0, 0.6, 0.6, 0.4]]),
        "vehicle_capacity": torch.tensor([1.0]),
        "window_ends": torch.tensor([[0.0, 0.5, 1.5, 10.0]]),
    }
    tour = torch.tensor([[0, 1, 0, 2, 3]])
    visit_times = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

    cost, info = PDCTRPTW.get_costs(
        problem_input,
        tour,
        visit_times,
        gamma=torch.tensor([1.0]),
        theta=torch.tensor([1.0]),
    )

    torch.testing.assert_close(info["lateness_penalty"], torch.tensor([2.0]))
    torch.testing.assert_close(cost, info["distance_penalty"] + 2.0)


@pytest.mark.parametrize("problem", [PDTRP, PDCTRP, PDTRPTW, PDCTRPTW])
def test_visit_times_include_waiting_service_and_return(problem, square_input):
    problem_input = {
        **square_input,
        "arrival_times": torch.tensor([[0.0, 5.0, 0.0, 0.0]]),
        "service_times": torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
        "window_starts": torch.zeros(1, 4),
    }
    tour = torch.tensor([[0, 1, 2, 3]])

    visit_times, info = problem.get_times(problem_input, tour)

    torch.testing.assert_close(visit_times, torch.tensor([[0.0, 6.0, 9.0, 10.0, 11.0]]))
    torch.testing.assert_close(info["total_travel_times"], torch.tensor([3.0]))
    torch.testing.assert_close(info["total_service_times"], torch.tensor([2.0]))


@pytest.mark.parametrize("problem", [PDTRP, PDTRPTW])
def test_uncapacitated_cost_rejects_invalid_tour(problem, square_input):
    tour = torch.tensor([[0, 1, 1, 3]])

    with pytest.raises(AssertionError, match="Invalid tour"):
        if problem is PDTRP:
            problem.get_costs(square_input, tour)
        else:
            problem.get_costs(
                {**square_input, "window_ends": torch.ones(1, 4)},
                tour,
                torch.zeros(1, 5),
                torch.ones(1),
                torch.ones(1),
            )

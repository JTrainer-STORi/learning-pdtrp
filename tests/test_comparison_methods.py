import torch

from evaluate_model_free import first_come_first_served, nearest_neighbour


class _State:
    def __init__(self, masks):
        self._masks = masks
        self.actions = []
        self.prev_a = torch.tensor([[0]])

    def all_finished(self):
        return len(self.actions) == len(self._masks)

    def get_mask(self):
        return torch.tensor([[self._masks[len(self.actions)]]])

    def update(self, selected):
        self.actions.append(int(selected.item()))
        self.prev_a = selected[:, None]
        return self


class _Problem:
    def __init__(self, masks):
        self.state = _State(masks)

    def make_state(self, _):
        return self.state

    def get_times(self, _, tour):
        return torch.zeros_like(tour, dtype=torch.float), None


def _input():
    # From customer 1, the depot is closer than customer 2.
    return {
        "distance_matrix": torch.tensor(
            [[[0.0, 2.0, 3.0], [0.1, 0.0, 1.0], [3.0, 1.0, 0.0]]]
        )
    }


def test_nearest_neighbour_does_not_reload_while_customer_is_feasible():
    problem = _Problem(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    tour, _ = nearest_neighbour(problem, _input())

    assert tour.tolist() == [[0, 1, 2]]


def test_nearest_neighbour_reloads_when_no_customer_is_feasible():
    problem = _Problem(
        [
            [True, False, True],
            [False, True, True],
            [True, True, False],
        ]
    )

    tour, _ = nearest_neighbour(problem, _input())

    assert tour.tolist() == [[0, 1, 0, 2]]


class _FcfsState(_State):
    def __init__(self, masks, visited, not_arrived):
        super().__init__(masks)
        self._visited = visited
        self._not_arrived = not_arrived

    @property
    def visited_(self):
        return torch.tensor([[self._visited[len(self.actions)]]])

    @property
    def not_arrived_(self):
        return torch.tensor([[self._not_arrived[len(self.actions)]]])


class _FcfsProblem(_Problem):
    def __init__(self, masks, visited, not_arrived):
        self.state = _FcfsState(masks, visited, not_arrived)


def _fcfs_input():
    return {
        "arrival_times": torch.tensor([[0.0, 2.0, 1.0]]),
        "distance_matrix": _input()["distance_matrix"],
    }


def test_fcfs_uses_arrival_order_not_distance():
    problem = _FcfsProblem(
        masks=[
            [True, False, False],
            [True, False, True],
        ],
        visited=[
            [True, False, False],
            [True, False, True],
        ],
        not_arrived=[
            [False, False, False],
            [False, False, False],
        ],
    )

    tour, _ = first_come_first_served(problem, _fcfs_input())

    assert tour.tolist() == [[0, 2, 1]]


def test_fcfs_reloads_if_queue_head_does_not_fit():
    problem = _FcfsProblem(
        masks=[
            [False, False, True],
            [True, False, False],
            [True, False, True],
        ],
        visited=[
            [True, False, False],
            [True, False, False],
            [True, False, True],
        ],
        not_arrived=[
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ],
    )

    tour, _ = first_come_first_served(problem, _fcfs_input())

    assert tour.tolist() == [[0, 0, 2, 1]]

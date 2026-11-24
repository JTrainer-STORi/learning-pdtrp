"""Complete competing first actions on the currently revealed requests."""

import torch

from utils.service_times import DEFAULT_SERVICE_TIME_MEAN


def revealed_state(problem, state, revealed_input, known_nodes):
    """Project the live vehicle state onto the cached graph of revealed nodes."""
    inputs = dict(revealed_input)
    mean = inputs.get("service_times_mean", DEFAULT_SERVICE_TIME_MEAN)
    expected_service = torch.ones_like(inputs["arrival_times"]) * mean
    expected_service[:, 0] = 0
    inputs.update(service_times=expected_service, arrival_times=torch.zeros_like(expected_service))
    planning = problem.make_state(inputs)
    changes = dict(
        prev_a=torch.searchsorted(known_nodes, state.prev_a),
        cur_times=state.cur_times,
        cur_coords=state.cur_coords,
        visited_=state.visited_.index_select(-1, known_nodes),
    )
    if hasattr(state, "used_cap"):
        changes["used_cap"] = state.used_cap
    return planning._replace(**changes)


def completion_costs(state, first_actions, choose_next, *, gamma=None, theta=None):
    """Return remaining objective for each candidate on the same revealed backlog."""
    state = state[torch.zeros_like(first_actions)]
    actions = first_actions
    finished = torch.zeros_like(actions, dtype=torch.bool)
    distance = torch.zeros_like(actions, dtype=state.lengths.dtype)
    lateness = torch.zeros_like(distance)
    # A legal depot action between visits can occur at most once per customer.
    for _ in range(2 * state.coords.size(1)):
        state = state.update(actions)
        if gamma is not None:
            service_start = state.service_start_times.gather(1, state.visit_counts)[:, 0]
            deadline = state.window_ends[0, actions]
            late = (service_start - deadline).clamp_min(0)
            lateness += late.masked_fill(finished | (actions == 0), 0)
        now_finished = state.visited_.all(dim=-1)[:, 0]
        home = state.distance_matrix[0, state.prev_a[:, 0], 0]
        distance = torch.where(now_finished & ~finished, state.lengths[:, 0] + home, distance)
        finished = now_finished
        if finished.all():
            break
        actions = choose_next(state, finished)
    else:
        raise RuntimeError('Policy rollout could not clear the revealed requests')
    if gamma is None:
        return distance
    return theta.reshape(-1)[0] * distance + gamma.reshape(-1)[0] * lateness

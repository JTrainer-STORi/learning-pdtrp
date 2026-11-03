"""Shared departure, request-arrival, and service events for TW environments."""

import torch

from problems.request_events import reveal_and_wait_if_idle


def update_time_window_state(state, selected):
    """Commit at departure, interrupting a pre-departure wait on a new call.

    An arrival exactly at planned departure is revealed before committing.
    Calls received during travel/service are revealed at service completion.
    """
    selected = selected.reshape(-1, 1)
    finished = state.visited_.all(dim=-1)
    distances = state.distance_matrix[state.ids, state.prev_a, selected]
    travel = torch.round(distances / state.speed.reshape(-1, 1), decimals=4)
    departure = torch.maximum(
        state.cur_times, state.window_starts[state.ids, selected] - travel
    )
    future = torch.where(
        state.not_arrived_[:, 0], state.arrival_times,
        torch.full_like(state.arrival_times, float("inf")),
    ).min(dim=1, keepdim=True).values
    interrupted = (~finished) & (departure > state.cur_times) & (future <= departure)
    executed = (~finished) & (~interrupted)
    service_starts = torch.maximum(
        state.cur_times + travel, state.window_starts[state.ids, selected]
    )
    completion = service_starts + state.service_times[state.ids, selected]
    times = torch.where(interrupted, future, torch.where(executed, completion, state.cur_times))
    previous = torch.where(executed, selected, state.prev_a)
    visited = state.visited_.clone()
    records = [list(row) for row in state.visit_times]
    routes = [list(row) for row in state.routes]
    for row in range(len(times)):
        if executed[row, 0]:
            node = selected[row, 0].item()
            visited[row, 0, node] = True
            records[row].append(service_starts[row, 0].item())
            routes[row].append(node)
            if visited[row].all():
                return_distance = state.distance_matrix[row, node, 0]
                return_time = completion[row, 0] + torch.round(
                    return_distance / state.speed.reshape(-1)[row], decimals=4
                )
                records[row].append(return_time.item())
    changes = dict(
        prev_a=previous,
        cur_coords=state.coords[state.ids, previous],
        cur_times=times,
        lengths=state.lengths + torch.where(executed, distances, 0.0),
        visited_=visited,
        visit_times=records,
        routes=routes,
        last_action_executed=executed.squeeze(1),
        i=state.i + 1,
    )
    if hasattr(state, "used_cap"):
        used = (state.used_cap + state.demands[state.ids, selected]) * (selected != 0)
        changes["used_cap"] = torch.where(executed, used, state.used_cap)
    updated = state._replace(**changes)
    revealed = reveal_and_wait_if_idle(updated)
    return revealed


def solution_tensors(state):
    """Pack executed visits, excluding canceled choices and finished-row padding."""
    width = max(len(route) - 1 for route in state.routes)
    tours = [route[1:] + [0] * (width + 1 - len(route)) for route in state.routes]
    times = [row + [row[-1]] * (width + 2 - len(row)) for row in state.visit_times]
    return torch.tensor(tours, dtype=torch.long, device=state.coords.device), times


def slice_records(records, key):
    indices = torch.arange(len(records))[key.cpu() if torch.is_tensor(key) else key]
    return [list(records[index]) for index in indices.reshape(-1).tolist()]

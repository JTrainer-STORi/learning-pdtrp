"""Shared departure, request-arrival, and service events for TW environments."""

import torch

from problems.request_events import (
    depot_return_times, next_arrival, reveal_and_wait_if_idle, travel_time,
)


def update_time_window_state(state, selected):
    """Commit at departure, interrupting a pre-departure wait on a new call.

    An arrival exactly at planned departure is revealed before committing.
    Calls received during travel/service are revealed at service completion.
    """
    selected = selected.reshape(-1, 1)
    finished = state.visited_.all(dim=-1)
    distances = state.distance_matrix[state.ids, state.prev_a, selected]
    travel = travel_time(state, distances)
    departure = torch.maximum(
        state.cur_times, state.window_starts[state.ids, selected] - travel
    )
    arrival_times = state.arrival_times[state.ids[:, 0]]
    future = next_arrival(arrival_times, state.not_arrived_[:, 0])
    interrupted = (~finished) & (departure > state.cur_times) & (future <= departure)
    executed = (~finished) & (~interrupted)
    service_starts = torch.maximum(
        state.cur_times + travel, state.window_starts[state.ids, selected]
    )
    completion = service_starts + state.service_times[state.ids, selected]
    times = torch.where(interrupted, future, torch.where(executed, completion, state.cur_times))
    previous = torch.where(executed, selected, state.prev_a)
    visited = state.visited_.scatter(2, previous.unsqueeze(-1), True)
    visit_counts = state.visit_counts + executed
    # Keep executed visits contiguous. Interrupted or finished rows write only
    # to the new, unused column, beyond their recorded route.
    column = torch.where(executed, visit_counts, state.route_nodes.size(1))
    route_nodes = torch.cat((state.route_nodes, selected), dim=1)
    route_nodes.scatter_(1, column, selected)
    records = torch.cat((state.service_start_times, service_starts), dim=1)
    records.scatter_(1, column, service_starts.to(records.dtype))
    changes = dict(
        prev_a=previous,
        cur_coords=state.coords[state.ids, previous],
        cur_times=times,
        lengths=state.lengths + torch.where(executed, distances, 0.0),
        visited_=visited,
        service_start_times=records,
        route_nodes=route_nodes,
        visit_counts=visit_counts,
        last_action_executed=executed.squeeze(1),
        i=state.i + 1,
    )
    if hasattr(state, "used_cap"):
        used = (state.used_cap + state.demands[state.ids, selected]) * (selected != 0)
        changes["used_cap"] = torch.where(executed, used, state.used_cap)
    return reveal_and_wait_if_idle(state._replace(**changes), arrival_times=arrival_times)


def solution_tensors(state):
    """Pack executed visits, excluding canceled choices and finished-row padding."""
    width = state.visit_counts.max().item()
    positions = torch.arange(width + 2, device=state.coords.device)
    in_route = positions <= state.visit_counts
    tours = state.route_nodes[:, 1:width + 1].masked_fill(~in_route[:, 1:-1], 0)
    last_visit = state.service_start_times.gather(1, state.visit_counts)
    end_times = torch.where(state.visited_.all(dim=-1), depot_return_times(state), last_visit)
    times = torch.cat((state.service_start_times[:, :width + 1], end_times), dim=1)
    return tours, torch.where(in_route, times, end_times)


def route_list(state):
    lengths = state.visit_counts[:, 0].tolist()
    return [row[:length + 1] for row, length in zip(state.route_nodes.tolist(), lengths)]


def visit_time_list(state):
    lengths = state.visit_counts[:, 0] + 1 + state.visited_[:, 0].all(dim=1)
    _, times = solution_tensors(state)
    return [row[:length] for row, length in zip(times.tolist(), lengths.tolist())]

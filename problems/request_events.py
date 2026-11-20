"""Request revelation and idle waiting shared by all routing environments."""

import torch


def next_arrival(arrival_times, pending):
    return torch.where(pending, arrival_times, float("inf")).amin(dim=1, keepdim=True)


def reveal_and_wait_if_idle(state, *, arrival_times=None, earliest=None):
    """Reveal calls through the clock; an empty queue waits for its next call."""
    if arrival_times is None:
        arrival_times = state.arrival_times[state.ids[:, 0]]
    times = state.cur_times
    pending = state.not_arrived_[:, 0]
    revealed = pending & (arrival_times <= times)
    if earliest is not None:
        revealed &= arrival_times >= earliest
    pending = pending & ~revealed

    visited = state.visited_[:, 0]
    idle = (visited | pending).all(dim=1, keepdim=True) & ~visited.all(dim=1, keepdim=True)
    idle_times = torch.maximum(times, next_arrival(arrival_times, pending)).to(times.dtype)
    times = torch.where(idle, idle_times, times)
    revealed_while_idle = idle & pending & (arrival_times <= times)
    if earliest is not None:
        # The non-TW environments reveal calls at the jumped-to instant.
        revealed_while_idle &= arrival_times >= times
    return state._replace(
        cur_times=times,
        not_arrived_=(pending & ~revealed_while_idle).unsqueeze(1),
        arrival_occured=(revealed | revealed_while_idle).any(dim=1),
    )


def travel_time(state, distances):
    speed = torch.as_tensor(state.speed, device=distances.device).reshape(-1, 1)
    row_speed = speed.expand(state.coords.size(0), 1)[state.ids[:, 0]]
    return torch.round(distances / row_speed, decimals=4)


def depot_return_times(state):
    distance = state.distance_matrix[state.ids, state.prev_a, 0]
    return state.cur_times + travel_time(state, distance)


def update_state_without_time_windows(state, selected):
    selected = selected.reshape(-1, 1)
    distances = state.distance_matrix[state.ids, state.prev_a, selected]
    service_starts = state.cur_times + travel_time(state, distances)
    updated = state._replace(
        prev_a=selected,
        cur_coords=state.coords[state.ids, selected],
        cur_times=service_starts + state.service_times[state.ids, selected],
        lengths=state.lengths + distances,
        visited_=state.visited_.scatter(2, selected.unsqueeze(-1), True),
        service_start_times=torch.cat((state.service_start_times, service_starts), dim=1),
        i=state.i + 1,
    )
    updated = reveal_and_wait_if_idle(updated, earliest=state.cur_times)
    if hasattr(state, "used_cap"):
        used = (state.used_cap + state.demands[state.ids, selected]) * (selected != 0)
        return updated._replace(used_cap=used)
    return updated._replace(arrival_occured=updated.arrival_occured & ~updated.all_finished())


def visit_times_without_time_windows(state):
    times = state.service_start_times
    if state.all_finished():
        times = torch.cat((times, depot_return_times(state)), dim=1)
    return times.tolist()

"""Dynamic time-window features using the information in a decision state."""

import torch

from problems.request_events import travel_time
from utils.service_times import DEFAULT_SERVICE_TIME_MEAN


def _row_value(value, state, like):
    tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    return tensor.reshape(-1, 1).expand(state.coords.size(0), 1)[state.ids[:, 0]]


def service_start_slack(state, normalize=True):
    """Deadline minus earliest direct service start, in minutes or horizon units.

    The state clock is the time when the vehicle is ready to depart. Windows
    constrain service START, so candidate service duration is not subtracted.
    Negative values retain the degree of unavoidable lateness on a direct visit.
    Caller masks requests that are unavailable, visited or capacity-infeasible.
    """
    ids = state.ids[:, 0]
    current = state.prev_a[:, 0]
    distances = state.distance_matrix[ids, current]

    def row_scalar(value):
        tensor = torch.as_tensor(value, device=distances.device, dtype=distances.dtype)
        if tensor.numel() == 1:
            return tensor.reshape(1, 1)
        return tensor.reshape(-1, 1)[ids]

    arrival = state.cur_times + distances / row_scalar(state.speed)
    earliest_start = torch.maximum(arrival, state.window_starts[ids])
    slack = state.window_ends[ids] - earliest_start
    if normalize:
        slack = slack / (row_scalar(state.time_horizon) + row_scalar(state.latest_end))
    return slack


def action_lookahead_features(state, capacity_aware=False):
    """Direct leg, onward distance and delay costs for each candidate.

    The delay feature sums the additional lateness of visits to other
    revealed, unserved customers after completing the candidate. Capacity-aware
    estimates include a depot detour for customers that require a reload,
    both before and after the candidate. A depot visit resets the load and has
    zero service duration. Customer durations use the configured mean.
    """
    ids = state.ids[:, 0]
    distances = state.distance_matrix[ids, state.prev_a[:, 0]]
    travel = travel_time(state, distances)
    arrival = state.cur_times + travel
    starts = state.window_starts[ids]
    deadlines = state.window_ends[ids]
    waiting = (starts - arrival).clamp_min(0)
    earliest = arrival + waiting
    late_now = (earliest - deadlines).clamp_min(0)

    nodes = torch.arange(distances.shape[1], device=distances.device)
    customers = nodes != 0
    speed = _row_value(state.speed, state, distances)
    prior_mean = state.service_times_mean if state.service_times_mean is not None else DEFAULT_SERVICE_TIME_MEAN
    service_mean = _row_value(prior_mean, state, distances)
    finish = earliest + service_mean * customers
    pairwise_distances = state.distance_matrix[ids]
    followup_times = torch.round(pairwise_distances / speed.unsqueeze(2), decimals=4)
    followup_distances = pairwise_distances
    followup_arrivals = finish.unsqueeze(2) + followup_times
    reference_lateness = late_now
    if capacity_aware:
        demand = state.demands[ids]
        capacity = _row_value(state.vehicle_capacity, state, distances)
        load_after = torch.where(customers, state.used_cap + demand, 0)
        requires_reload = load_after.unsqueeze(2) + demand.unsqueeze(1) > capacity.unsqueeze(2)

        depot_departure = torch.maximum(followup_arrivals[:, :, 0], starts[:, :1])
        arrivals_via_depot = depot_departure.unsqueeze(2) + followup_times[:, :1, :]
        followup_arrivals = torch.where(requires_reload, arrivals_via_depot, followup_arrivals)
        distances_via_depot = pairwise_distances[:, :, :1] + pairwise_distances[:, :1, :]
        followup_distances = torch.where(requires_reload, distances_via_depot, pairwise_distances)

        # Compare against serving the same customer now, including any reload
        # that is already necessary with the current load.
        depot_departure_now = torch.maximum(arrival[:, :1], starts[:, :1])
        arrivals_via_depot_now = depot_departure_now + followup_times[:, 0, :]
        reference_arrivals = torch.where(state.used_cap + demand > capacity, arrivals_via_depot_now, arrival)
        reference_starts = torch.maximum(reference_arrivals, starts)
        reference_lateness = (reference_starts - deadlines).clamp_min(0)

    late_after_visit = (followup_arrivals - deadlines.unsqueeze(1)).clamp_min(0)
    known_unserved = ~(state.not_arrived_[:, 0] | state.visited_[:, 0]) & customers
    other_customers = (nodes[:, None] != nodes) & known_unserved.unsqueeze(1)
    onward_distance = followup_distances.masked_fill(~other_customers, float("inf")).amin(dim=2)
    onward_distance = torch.where(other_customers.any(dim=2), onward_distance, pairwise_distances[:, :, 0])
    added_lateness = (late_after_visit - reference_lateness.unsqueeze(1)).clamp_min(0)
    added_lateness = added_lateness.masked_fill(~other_customers, 0).sum(dim=2)

    scale = _row_value(state.time_horizon, state, distances) + _row_value(state.latest_end, state, distances)
    return torch.stack((distances, onward_distance, waiting / scale, late_now * customers / scale,
                        added_lateness / scale), dim=-1)

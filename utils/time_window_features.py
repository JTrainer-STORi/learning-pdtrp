"""Dynamic time-window features using the information in a decision state."""

import torch


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

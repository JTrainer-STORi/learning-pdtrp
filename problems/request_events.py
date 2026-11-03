"""Request revelation and idle waiting shared by all routing environments."""

import torch


def reveal_and_wait_if_idle(state):
    """Reveal calls through the clock; an empty queue waits for its next call."""
    times = state.cur_times.clone()
    pending = state.not_arrived_.clone()
    revealed = pending & (state.arrival_times[:, None, :] <= times[:, :, None])
    pending &= ~revealed
    arrivals = revealed.any(dim=-1).squeeze(1)
    for row in range(len(times)):
        if (state.visited_[row] | pending[row]).all() and not state.visited_[row].all():
            next_call = state.arrival_times[row][pending[row, 0]].min()
            times[row, 0] = torch.maximum(times[row, 0], next_call)
            newly_revealed = pending[row, 0] & (state.arrival_times[row] <= times[row, 0])
            pending[row, 0] &= ~newly_revealed
            arrivals[row] |= newly_revealed.any()
    return state._replace(cur_times=times, not_arrived_=pending, arrival_occured=arrivals)

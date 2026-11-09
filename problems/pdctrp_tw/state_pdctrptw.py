from typing import NamedTuple
import torch
from utils import scale_time
from problems.time_window_events import (
    reveal_and_wait_if_idle, slice_records, solution_tensors, update_time_window_state,
)

class StatePDCTRPTW(NamedTuple):

    # Fixed input
    coords: torch.Tensor # Depot followed by customers in their saved order
    arrival_times: torch.Tensor
    service_times: torch.Tensor
    window_starts: torch.Tensor
    window_ends: torch.Tensor
    demands: torch.Tensor
    speed: float
    time_horizon: float
    latest_end: float
    vehicle_capacity: float
    distance_matrix: torch.Tensor
    graph: torch.Tensor


    #ids for indexing environments
    ids: torch.Tensor

    #state
    prev_a: torch.Tensor
    cur_times: torch.Tensor
    cur_coords: torch.Tensor
    lengths: torch.Tensor
    used_cap: torch.Tensor
    visited_: torch.Tensor
    visit_times: list # Tracks the time at which each node was visited
    not_arrived_: torch.Tensor
    arrival_occured: torch.Tensor # Tracks if a new arrival has occurred in the last step
    routes: list
    last_action_executed: torch.Tensor
    i: torch.Tensor # current timestep

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)

        return self._replace(
            ids = self.ids[key],
            visit_times=slice_records(self.visit_times, key),
            routes=slice_records(self.routes, key),
            last_action_executed=self.last_action_executed[key],
            prev_a = self.prev_a[key],
            cur_times = self.cur_times[key],
            cur_coords = self.cur_coords[key],
            used_cap = self.used_cap[key],
            lengths = self.lengths[key],
            visited_ = self.visited_[key],
            not_arrived_ = self.not_arrived_[key],
        )
    
    @staticmethod
    def initialize(input):

        loc = input['all_nodes']
        service_times = input['service_times']
        arrival_times = input['arrival_times']
        window_starts = input['window_starts']
        window_ends = input['window_ends']
        demands = input['demand']
        speed = input['speed']
        time_horizon = input['time_horizon']
        latest_end = input['latest_end']
        distance_matrix = input['distance_matrix']
        vehicle_capacity = input['vehicle_capacity']
        graph = input['graph']

        batch_size, n_loc, _ = loc.size()
        visited = torch.zeros((batch_size, 1, n_loc), dtype=torch.bool, device=loc.device)
        visited[:, :, 0] = True  # depot starts visited
        
        state = StatePDCTRPTW(
            coords=loc,
            arrival_times=arrival_times,
            service_times=service_times,
            window_starts=window_starts,
            window_ends=window_ends,
            demands=demands,
            speed=speed,
            time_horizon=time_horizon,
            latest_end=latest_end,
            vehicle_capacity=vehicle_capacity,
            distance_matrix=distance_matrix,
            graph=graph,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a= torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            cur_times=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            cur_coords=loc[:, 0].unsqueeze(1), # start at the depot for every env
            used_cap=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            visited_= visited,
            visit_times=[[0.0] for _ in range(batch_size)],
            routes=[[0] for _ in range(batch_size)],
            last_action_executed=torch.zeros(batch_size, dtype=torch.bool, device=loc.device),
            lengths=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            not_arrived_= (arrival_times > 0).unsqueeze(1),
            arrival_occured=torch.zeros(batch_size, dtype=torch.bool, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )

        return reveal_and_wait_if_idle(state)

    def update(self, selected):
        return update_time_window_state(self, selected)

    def all_finished(self):
        return self.visited_.all()
    
    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        visited_loc = self.visited_[:, :, 1:]
        not_arrived_loc = self.not_arrived_[:, :, 1:]
        loc_mask = torch.logical_or(visited_loc, not_arrived_loc)
        capacities = self.vehicle_capacity[self.ids]
        exceeds_cap = self.demands[self.ids, 1:] + self.used_cap[self.ids] > capacities

        mask_loc = torch.logical_or(loc_mask, exceeds_cap)

        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)

        return torch.cat((mask_depot[:, :, None], mask_loc), dim=-1)
    
    def get_graph(self):
        return self.graph

    def get_graph_mask(self):
        batch_size, n_loc, _ = self.coords.size()
        if self.i.item() == 0:
            return torch.zeros(batch_size, 1, n_loc, dtype=torch.bool, device=self.coords.device)
        adjacent = self.graph.gather(1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_loc))
        return ~adjacent.bool()
    
    def arrival_at_last_timestep(self):
        return self.arrival_occured
    
    def construct_solutions(self, actions):
        return solution_tensors(self)[0]

    def get_visit_times(self):
        return solution_tensors(self)[1]

    def get_timestep(self, normalize=True):
        if normalize:
            return scale_time(self.cur_times, self.time_horizon+self.latest_end)
        else:
            return self.cur_times
        
    def get_remaining_capacity(self):
        # return the remaining capacity for each environment in the batch
        return self.vehicle_capacity - self.used_cap

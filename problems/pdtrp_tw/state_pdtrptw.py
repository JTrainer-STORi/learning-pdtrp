from typing import NamedTuple
import torch
from utils import scale_time

class StatePDTRPTW(NamedTuple):
    # Fixed input
    coords: torch.Tensor # Depot + static_loc + dynamic_loc
    service_times: torch.Tensor
    arrival_times: torch.Tensor
    speed: float
    time_horizon: float
    latest_end: float
    distance_matrix: torch.Tensor
    window_starts: torch.Tensor
    window_ends: torch.Tensor

    ids: torch.Tensor

    # State
    prev_a: torch.Tensor
    cur_times: torch.Tensor
    cur_coords: torch.Tensor
    lengths: torch.Tensor
    visited_: torch.Tensor # Tracks which nodes have been visited
    visit_times: list # Tracks the time at which each node was visited
    not_arrived_: torch.Tensor # Tracks which nodes have not arrived yet
    arrival_occured: torch.Tensor # Tracks if a new arrival has occurred in the last step
    i: torch.Tensor # Current timestep

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)

        return self._replace(
            ids = self.ids[key],
            prev_a = self.prev_a[key],
            cur_times = self.cur_times[key],
            cur_coords = self.cur_coords[key],
            lengths = self.lengths[key],
            visited_ = self.visited_[key],
            not_arrived_ = self.not_arrived_[key],
        )
    
    @staticmethod
    def initialize(input):

        loc = input['all_nodes']
        service_times = input['service_times']
        arrival_times = input['arrival_times']
        speed = input['speed']
        time_horizon = input['time_horizon']
        latest_end = input['latest_end']
        distance_matrix = input['distance_matrix']
        window_starts = input['window_starts']
        window_ends = input['window_ends']

        batch_size, n_loc, _ = loc.size()
        visited = torch.zeros((batch_size, 1, n_loc), dtype=torch.bool, device=loc.device)
        visited[:, :, 0] = True

        ids = torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None]

        return StatePDTRPTW(
            coords=loc,
            service_times=service_times,
            arrival_times=arrival_times,
            speed=speed,
            time_horizon=time_horizon,
            latest_end=latest_end,
            distance_matrix=distance_matrix,
            window_starts=window_starts,
            window_ends=window_ends,
            ids=ids,
            prev_a= torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            cur_times=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            cur_coords=loc[:, 0].unsqueeze(1), # start at the depot for every env
            visited_= visited,
            visit_times=[[0.0] for _ in range(batch_size)],
            not_arrived_= (arrival_times > 0).unsqueeze(1),
            arrival_occured=torch.zeros(batch_size, dtype=torch.bool, device=loc.device),
            lengths=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )

    def update(self, selected):

        batch_size, n_loc, _ = self.coords.size()
        
        prev_a = selected[:, None]

        cur_coords = self.coords[self.ids, prev_a]
        distances_travelled = self.distance_matrix[self.ids, self.prev_a, prev_a]
        
        # advance the current time by the maximum of the travel time and the beginning of the time window plus service time 
        service_starts = torch.maximum(self.cur_times  + torch.round((distances_travelled / self.speed), decimals=4), self.window_starts[self.ids, prev_a])

        cur_times = service_starts + (self.service_times[self.ids, prev_a])

        # record the time at which service began at the current location
        for i in range(batch_size):
            self.visit_times[i].append(service_starts[i, 0].item())

        arrival_occured = torch.zeros(batch_size, dtype=torch.bool, device=self.coords.device)

        # find out which dynamic nodes have arrived in the meantime
        new_arrival_mask = self.not_arrived_.clone()
        for i in range(batch_size):
            new_arrivals = torch.nonzero(torch.logical_and(torch.logical_and(self.arrival_times[i] >= self.cur_times[i], self.arrival_times[i] <= cur_times[i]), self.arrival_times[i] > 0))
            new_arrival_mask[i, 0, new_arrivals] = False
            if new_arrivals.numel() > 0:
                arrival_occured[i] = True
            
        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        # if, for any of the environments in the batch, after checking for new arrivals, there are no available nodes to visit but still dynamic nodes that have not arrived yet, fast forward the current time to the minimum of the remaining arrival times and update the mask accordingly
        for i in range(batch_size):            
            if torch.logical_or(visited_[i], new_arrival_mask[i]).all():
                if not visited_[i].all():
                    # fast forward the current time to the minimum of the remaining arrival times ...
                    min_arrival_times = torch.min(self.arrival_times[i][new_arrival_mask[i][0]])
                    cur_times[i] = torch.maximum(cur_times[i], min_arrival_times)
                    # ... and update the mask accordingly
                    new_arrivals = torch.nonzero(torch.logical_and(torch.logical_and(self.arrival_times[i] >= self.cur_times[i], self.arrival_times[i] <= cur_times[i]), self.arrival_times[i] > 0))
                    new_arrival_mask[i, 0, new_arrivals] = False
                    if new_arrivals.numel() > 0:
                        arrival_occured[i] = True

        # if all nodes are visited, calculate the distance back to the depot and what time it will be visited
        if visited_.all():
            distances_back_to_depot = (self.coords[self.ids, 0] - cur_coords).norm(p=2, dim=-1)
            time_back_at_depot = cur_times + torch.round((distances_back_to_depot / self.speed), decimals=4)
            for i in range(batch_size):
                self.visit_times[i].append(time_back_at_depot[i, 0].item())

        return self._replace(prev_a=prev_a, cur_coords=cur_coords, cur_times=cur_times, lengths=self.lengths + distances_travelled, visited_=visited_, not_arrived_=new_arrival_mask, i=self.i + 1,
        arrival_occured=arrival_occured)

    def all_finished(self):
        return self.visited_.all()
    
    def get_current_node(self):
        return self.prev_a
    
    def get_mask(self):
        return torch.logical_or(self.visited_, self.not_arrived_)
    
    def get_graph_mask(self):
        batch_size, n_loc, _ = self.loc.size()
        if self.i.item() == 0:
            return torch.zeros(batch_size, 1, n_loc, dtype=torch.bool, device=self.loc.device)
        else:
            return self.graph.gather(1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_loc))
            
    def get_graph(self):
        return self.graph

    def arrival_at_last_timestep(self):
        return self.arrival_occured

    def construct_solutions(self, actions):
        return actions

    def get_timestep(self, normalize=True):
        if normalize:
            return scale_time(self.cur_times, self.time_horizon+self.latest_end)
        else:
            return self.cur_times
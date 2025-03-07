from typing import NamedTuple
import torch

class StatePDTRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor # Depot + static_loc + dynamic_loc
    service_times: torch.Tensor
    arrival_times: torch.Tensor

    ids: torch.Tensor

    # State
    prev_a: torch.Tensor
    cur_times: torch.Tensor
    cur_coords: torch.Tensor
    lengths: torch.Tensor
    visited_: torch.Tensor # Tracks which nodes have been visited
    visit_times: list # Tracks the time at which each node was visited
    not_arrived_: torch.Tensor # Tracks which nodes have not arrived yet
    i: torch.Tensor # Current timestep
    SPEED = 16

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

        loc = input['loc']
        service_times = input['service_times']
        arrival_times = input['arrival_times']

        batch_size, n_loc, _ = loc.size()
        visited = torch.zeros((batch_size, 1, n_loc), dtype=torch.bool, device=loc.device)
        visited[:, :, 0] = True
        return StatePDTRP(
            coords=loc,
            service_times=service_times,
            arrival_times=arrival_times,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a= torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            cur_times=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            cur_coords=loc[:, 0].unsqueeze(1), # start at the depot for every env
            visited_= visited,
            visit_times=[[0.0] for _ in range(batch_size)],
            not_arrived_= (arrival_times > 0).unsqueeze(1),
            lengths=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )
    
    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0] - self.cur_coords).norm(p=2, dim=-1)

    def update(self, selected):

        batch_size, n_loc, _ = self.coords.size()
        
        prev_a = selected[:, None]

        cur_coords = self.coords[self.ids, prev_a]
        distances_travelled = (cur_coords - self.cur_coords).norm(p=2, dim=-1)

        service_starts = self.cur_times + torch.round((distances_travelled / self.SPEED), decimals=4)
        
        # advance the current time by the travel time and service time 
        cur_times = service_starts + (self.service_times[self.ids, prev_a])

        # record the time at which service began at the current location
        for i in range(batch_size):
            self.visit_times[i].append(service_starts[i, 0].item())

        # find out which dynamic nodes have arrived in the meantime
        new_arrival_mask = self.not_arrived_.clone()
        for i in range(batch_size):
            new_arrivals = torch.nonzero(torch.logical_and(self.arrival_times[i] >= self.cur_times[i], self.arrival_times[i] <= cur_times[i]))
            new_arrival_mask[i, 0, new_arrivals] = False
            
        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        # if, for any of the environments in the batch, after checking for new arrivals, there are no available nodes to visit but still dynamic nodes that have not arrived yet, fast forward the current time to the minimum of the remaining arrival times and update the mask accordingly
        for i in range(batch_size):            
            if torch.logical_or(visited_[i], new_arrival_mask[i]).all():
                if not visited_[i].all():
                    # fast forward the current time to the minimum of the remaining arrival times ...
                    min_arrival_times = torch.min(self.arrival_times[i][new_arrival_mask[i][0]])
                    cur_times[i] = torch.maximum(cur_times[i], min_arrival_times)
                    # ... and update the mask accordingly
                    new_arrivals = torch.nonzero(torch.logical_and(self.arrival_times[i] >= self.cur_times[i], self.arrival_times[i] <= cur_times[i]))
                    new_arrival_mask[i, 0, new_arrivals] = False

        return self._replace(prev_a=prev_a, cur_coords=cur_coords, cur_times=cur_times, lengths=self.lengths + distances_travelled, visited_=visited_, not_arrived_=new_arrival_mask, i=self.i + 1)

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

    def construct_solutions(self, actions):
        return actions
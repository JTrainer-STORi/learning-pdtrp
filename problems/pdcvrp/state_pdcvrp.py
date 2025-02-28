from typing import NamedTuple
import torch

class StatePDCVRP(NamedTuple):

    # Fixed input
    coords: torch.Tensor # Depot + static_loc + dynamic_loc
    arrival_times: torch.Tensor
    service_times: torch.Tensor
    demands: torch.Tensor

    #ids for indexing environments
    ids: torch.Tensor

    #state
    prev_a: torch.Tensor
    cur_times: torch.Tensor
    cur_coords: torch.Tensor
    visit_times: list
    lengths: torch.Tensor
    used_cap: torch.Tensor
    visited_: torch.Tensor
    not_arrived_: torch.Tensor
    i: torch.Tensor # current timestep

    SPEED = 16 # HARDCODED
    VEHICLE_CAPACITY = 1.0

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)

        return self._replace(
            ids = self.ids[key],
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

        loc = input['loc']
        service_times = input['service_times']
        arrival_times = input['arrival_times']
        demands = input['demand']

        batch_size, n_loc, _ = loc.size()
        visited = torch.zeros((batch_size, 1, n_loc), dtype=torch.bool, device=loc.device)
        return StatePDCVRP(
            coords=loc,
            arrival_times=arrival_times,
            service_times=service_times,
            demands=demands,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a= torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            cur_times=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            cur_coords=loc[:, 0].unsqueeze(1), # start at the depot for every env
            used_cap=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            visited_= visited,
            vist_times=[[] for _ in range(batch_size)],
            lengths=torch.zeros((batch_size, 1), dtype=torch.float32, device=loc.device),
            not_arrived_= (arrival_times > 0).unsqueeze(1),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )
    
    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0] - self.cur_coords).norm(p=2, dim=-1)

    def update(self, selected):

        batch_size, _, _ = self.coords.size()

        prev_a = selected[:, None]

        cur_coords = self.coords[self.ids, prev_a]
        distances_travelled = (cur_coords - self.cur_coords).norm(p=2, dim=-1)

        service_starts = self.cur_times + torch.round((distances_travelled / self.SPEED), decimals=4)

        # advance the current time by the travel time and service time
        cur_times = service_starts + (self.service_times[self.ids, prev_a])

        for i in range(batch_size):
            self.visit_times[i].append(cur_times[i].item())

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

        # update the used capacity
        selected_demand = self.demands[self.ids, prev_a]

        # increase capacity if depot is not visited, otherwise reset to 0
        used_capacity = (self.used_cap + selected_demand) * (prev_a != 0).float()

        return self._replace(
            prev_a=prev_a,
            cur_times=cur_times,
            cur_coords=cur_coords,
            used_cap=used_capacity,
            lengths=self.lengths + distances_travelled,
            visited_=visited_,
            not_arrived_=new_arrival_mask,
            i=self.i + 1
        )
    
    def all_finished(self):
        return self.visited_.all()
    
    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        visited_loc = self.visited_[:, :, 1:]
        not_arrived_loc = self.not_arrived_[:, :, 1:]
        loc_mask = torch.logical_or(visited_loc, not_arrived_loc)
        exceeds_cap = (self.demands[self.ids, 1:] + self.used_cap[:, :, None] > self.VEHICLE_CAPACITY)

        mask_loc = torch.logical_or(loc_mask, exceeds_cap)

        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)

        return torch.cat((mask_depot[:, :, None], mask_loc), dim=-1)
    
    def get_graph(self):
        return self.graph
    
    def construct_solutions(self, actions):
        return actions
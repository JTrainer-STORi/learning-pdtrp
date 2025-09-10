from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

from problems.pdcvrp.state_pdcvrp import StatePDCVRP
from problems.utils import nearest_neighbor_graph, sample_lognorm,subsample_ortec, generate_scaled_demands
from problems.location_generation import time_and_subregion_generation


class PDCVRP(object):

    NAME = 'pdcvrp'

    @staticmethod
    def get_costs(input, pi):

        distance_matrix = input['distance_matrix']
        vehicle_capacity = input['vehicle_capacity']
        demand = input['demand'] 

        B, N = demand.size()
        graph_size = N - 1
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(B, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(demand[:, :1], -vehicle_capacity[0].item()),
                demand[:, 1:]
            ),
            1
        )
        dem = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(demand[:, 0])
        for i in range(pi.size(1)):
            used_cap += dem[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= vehicle_capacity[0].item() + 1e-5).all(), "Used more than capacity, {}".format(used_cap)

        # Gather distances along the tour
        idx_from = pi[:, :-1]
        idx_to = pi[:, 1:]

        batch_idx = torch.arange(B, device=pi.device).unsqueeze(1).expand_as(idx_from)

        # Collect distances between consecutive nodes
        dists = distance_matrix[batch_idx, idx_from, idx_to]  # (B, N-1)

        # Add distance from last to first to complete the tour
        last_to_first = distance_matrix[torch.arange(B, device=pi.device), pi[:, -1], pi[:, 0]]  # (B,)

        tour_lengths = dists.sum(dim=1) + last_to_first  # (B,)

        return tour_lengths, {}
    
    @staticmethod
    def get_times(input, pi):
        """ Returns PDTRP visit times for a given set of tours (pi)"""

        batch_size, tour_length = pi.size()

        visit_times = torch.zeros(batch_size, tour_length + 1, device=pi.device)  # Initialize visit times tensor +1 for return to depot
        total_travel_times = torch.zeros(batch_size, device=pi.device)  # Initialize total travel time tensor
        total_service_times = torch.zeros(batch_size, device=pi.device)  # Initialize total service time tensor
        total_waiting_times = torch.zeros(batch_size, device=pi.device)  # Initialize total waiting time tensor

        for b in range(batch_size):
            time = 0.0
            total_travel_time = 0.0
            total_service_time = 0.0
            total_waiting_time = 0.0
            for i in range(tour_length):
                if i == 0:
                    visit_times[b, i] = time
                else:
                    prev_node = pi[b, i - 1]
                    curr_node = pi[b, i]
                    travel_time = torch.round(input['distance_matrix'][b, prev_node, curr_node] / input['speed'], decimals=4)  # Convert distance to time
                    total_travel_time += travel_time
                    service_time = input['service_times'][b, curr_node]
                    total_service_time += service_time
                    visit_time = max(time + travel_time, input['arrival_times'][b, curr_node] + travel_time)
                    total_waiting_time += visit_time - (time + travel_time)  # Calculate waiting time if any
                    visit_times[b, i] = visit_time 
                    time = visit_time + service_time
            # calculate the return time back to the depot
            return_time = input['distance_matrix'][b, pi[b, -1], pi[b, 0]] / input['speed']
            visit_times[b, tour_length] = time + return_time  # Last visit time is the return to depot

            total_travel_times[b] = total_travel_time
            total_service_times[b] = total_service_time
            total_waiting_times[b] = total_waiting_time
        
        return visit_times, {"total_travel_times": total_travel_times, "total_service_times": total_service_times, 'total_waiting_times': total_waiting_times}
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDCVRP.initialize(*args, **kwargs)
    
class PDCVRPDataset(Dataset):
    def __init__(self, min_total=20, max_total=100, min_dod=0.2, max_dod=0.8, speed=4, time_horizon=8, service_times_mean=3, service_times_var=5, n_subregions=9, arrival_weights = None, arrival_skews=None, min_time_window=None, max_time_window=None, batch_size=128, num_samples=12800, neighbors=20, knn_strat='percentage', filename=None, offset=0, gamma=None, theta=None, latest_end=None, reaction_time=None, vehicle_capacity=1.0, min_trips_required_lb=3, min_trips_required_ub=9, use_ortec=None):
        super(PDCVRPDataset, self).__init__()
        self.filename = filename
        self.min_total = min_total
        self.max_total = max_total
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.batch_size = batch_size
        self.n_subregions = n_subregions
        self.time_horizon = time_horizon * 60 # convert to minutes
        self.service_times_mean = service_times_mean
        self.service_times_std = np.sqrt(service_times_var)
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.offset = offset
        self.arrival_weights = arrival_weights
        self.arrival_skews = arrival_skews
        self.speed = speed / 60.0 # convert to units/minute
        self.vehicle_capacity = vehicle_capacity
        self.min_trips_required_lb = min_trips_required_lb
        self.min_trips_required_ub = min_trips_required_ub
        self.use_ortec = use_ortec        

        self.all_nodes = []
        self.arrival_times = []
        self.demands = []
        self.visit_times = []
        self.service_times = []

        if filename is not None:

            self.tours = []

            if self.use_ortec is not None:
                self.distance_matrix = []

            print("Loading from {}...".format(filename))

            for line in tqdm(open(filename, "r").readlines()[offset:offset + num_samples], ascii=True):
                line = line.split(" ")
                n_nodes = int(line.index('arrival_times')//2)
                self.all_nodes.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * n_nodes, 2)]
                )
                self.arrival_times.append([float(x) for x in line[line.index('arrival_times') + 1:line.index('service_times')]])
                self.service_times.append([float(x) for x in line[line.index('service_times') + 1:line.index('demands')]])
                self.demands.append([float(x) for x in line[line.index('demands') + 1:line.index('tour')]])
                self.tours.append([int(x) - 1 for x in line[line.index('tour') + 1:line.index('visit_times') - 1]])
                if self.use_ortec is not None:
                    # If using ORTEC, the distance matrix is already computed
                    self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:line.index('distance_matrix')]])
                    self.distance_matrix.append(np.reshape(np.array([float(x) for x in line[line.index('distance_matrix') + 1:]]), (n_nodes, n_nodes)))
                else: 
                    self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:]])

        elif self.use_ortec is not None:
            print("Generating PDCVRP instances from ORTEC instance: {}".format(self.use_ortec))
            # For this manner of generating 'Real' PDTRP instances, the dod arguments won't apply and instead the dod will be determined randomly based on the subsampling
            self.distance_matrix = []
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int) - 1
                # Step 2: Generate the locations and arrival times of the customers by subsampling ortec instances
                all_nodes = []
                arrival_times = []
                service_times = []
                distance_matrix = []
                demands = []
                for _ in range(batch_size):
                    batch_nodes, batch_service_times, batch_arrival_times, batch_distance_matrix, _, _,batch_demands = subsample_ortec(instance_file=self.use_ortec, problem='pdcvrp', n_total=n_total, time_horizon=self.time_horizon, speed=self.speed, vehicle_capacity=self.vehicle_capacity, min_trips_required_lb=self.min_trips_required_lb, min_trips_required_ub=self.min_trips_required_ub)
                    all_nodes.append(batch_nodes)
                    arrival_times.append(batch_arrival_times)
                    service_times.append(batch_service_times)
                    distance_matrix.append(batch_distance_matrix)
                    demands.append(batch_demands)

                # step 3: sort the customers by arrival time
                for i in range(batch_size):
                    idx = np.argsort(arrival_times[i])
                    all_nodes[i] = all_nodes[i][idx]
                    arrival_times[i] = arrival_times[i][idx]
                    service_times[i] = service_times[i][idx]
                    distance_matrix[i] = distance_matrix[i][idx][:, idx]
                    demands[i] = demands[i][idx]

                self.all_nodes += all_nodes
                self.arrival_times += arrival_times
                self.service_times += service_times
                self.distance_matrix += distance_matrix
                self.demands += demands

        else:
            if arrival_weights is not None and arrival_skews is not None:
                print("Generating PDCVRP instances with {} subregion(s) with arrival weights {} and skews {}...".format(self.n_subregions, self.arrival_weights, self.arrival_skews))
            elif arrival_weights is not None:
                print("Generating PDCVRP instances with {} subregion(s) with arrival weights {} and uniform arrival skews...".format(self.n_subregions, self.arrival_weights))
            elif arrival_skews is not None:
                print("Generating PDCVRP instances with {} subregion(s) with dirichlet arrival weights and skews {}...".format(self.n_subregions, self.arrival_skews))
            else:
                print("Generating PDCVRP instances with {} subregion(s) with dirichlet arrival weights and uniform skews...".format(self.n_subregions))  
            print("Vehicle Capacity: {}, Min Trips Required Lower Bound: {}, Upper Bound: {}".format(self.vehicle_capacity, self.min_trips_required_lb, self.min_trips_required_ub))    

            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int) - 1 # -1 to account for the depot
                # Step 2: Generate the locations and arrival times of the customers
                all_nodes = []
                arrival_times = []
                for _ in range(batch_size):
                    batch_nodes, batch_arrival_times = time_and_subregion_generation(n_total=n_total, n_subregions=self.n_subregions, arrival_weights=self.arrival_weights, arrival_skews=self.arrival_skews,time_horizon=self.time_horizon)
                    all_nodes.append(batch_nodes)
                    arrival_times.append(batch_arrival_times)
                # step 4: Generate customer service times
                service_times = list(np.round(sample_lognorm(self.service_times_mean, self.service_times_std, size=(batch_size, n_total)), decimals=3))
                # step 5: sample the number of advanced request customers for each instance in the batch
                n_imm = np.random.randint(int(self.min_dod * n_total), int(self.max_dod * n_total) + 1, size=(batch_size))
                n_adv = n_total - n_imm

                # step 6: Sample indices of advanced request customers for each instance in the batch
                adv_indices = [np.random.choice(n_total, n_adv[i], replace=False) for i in range(batch_size)]
                # step 7: set the arrival times of the advanced request customers to be 0
                for i in range(batch_size):
                    arrival_times[i][adv_indices[i]] = 0
                # step 8: sort the customers by arrival time
                for i in range(batch_size):
                    idx = np.argsort(arrival_times[i])
                    all_nodes[i] = all_nodes[i][idx]
                    arrival_times[i] = arrival_times[i][idx]
                    service_times[i] = service_times[i][idx]
                
                # Step 9: Generate the demands for each customer

                demands_without_depot = generate_scaled_demands(batch_size, n_total, self.vehicle_capacity, self.min_trips_required_lb, self.min_trips_required_ub)
            
                
                # step 10: Prepend the depot to the customers
                all_nodes = [np.concatenate([np.array([[0.5, 0.5]]), x]) for x in all_nodes]
                arrival_times = [np.concatenate((np.zeros(1), x)) for x in arrival_times]
                service_times = [np.concatenate((np.zeros(1), x)) for x in service_times]
                demands = [np.concatenate((np.zeros(1), x)) for x in demands_without_depot]


                self.all_nodes += all_nodes
                self.arrival_times += arrival_times
                self.service_times += service_times
                self.demands += demands

        self.size = len(self.all_nodes)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)  

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        all_nodes  = self.all_nodes[idx]
        arrival_times = self.arrival_times[idx]
        demands = self.demands[idx]
        service_times = self.service_times[idx]
        if self.use_ortec is not None:
            distance_matrix = self.distance_matrix[idx]
            nn_graph, _ = nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat, distance_matrix=distance_matrix)
        else:
            nn_graph, distance_matrix = nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat)

        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'arrival_times': torch.FloatTensor(arrival_times),
            'service_times': torch.FloatTensor(service_times),
            'graph': ~torch.BoolTensor(nn_graph),
            'demand': torch.FloatTensor(demands),
            'speed': torch.FloatTensor([self.speed]),
            'time_horizon': torch.FloatTensor([self.time_horizon]),
            'distance_matrix': torch.FloatTensor(distance_matrix),
            'vehicle_capacity': torch.FloatTensor([self.vehicle_capacity]),
        }

        if self.filename is not None:
            item['tour_nodes'] = torch.LongTensor(self.tours[idx])
            item['visit_times'] = torch.FloatTensor(self.visit_times[idx])

        return item
from itertools import islice

from torch.utils.data import Dataset
import torch
from problems.size_conventions import resolve_node_bounds
from problems.time_horizon import horizon_minutes, validate_hours_per_customer
from tqdm import tqdm
import numpy as np

from problems.pdctrp_tw.state_pdctrptw import StatePDCTRPTW

from problems.utils import nearest_neighbor_graph, sample_lognorm, subsample_ortec, generate_scaled_demands
from problems.location_generation import time_and_subregion_generation
from problems.time_windows import generate_time_windows, resolve_time_window_bounds

class PDCTRPTW(object):
    """
    Class representing the Partially Dynamic Travelling Repairman Problem
    """

    NAME = "pdctrptw"

    @staticmethod
    def get_costs(input, pi, visit_times, gamma, theta):
        """Returns cost based on the tour and missed customers as described in Larsen, Madsen and Solomon (2004)"""

        distance_matrix = input['distance_matrix']
        vehicle_capacity = input['vehicle_capacity']
        demand = input['demand'] 
        window_ends = input['window_ends']

        B, N = demand.size()
        capacities = vehicle_capacity.reshape(B, -1)[:, 0]
        graph_size = N - 1

        # Check that tours are valid, i.e. contain 0 to n - 1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(B, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                -capacities[:, None],
                input['demand'][:, 1:]
            ),
            1
        )
        dem = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(demand[:, 0])
        for i in range(pi.size(1)):
            used_cap += dem[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= capacities + 1e-5).all(), "Used more than capacity {}".format(used_cap)

        # Gather distances along the tour
        idx_from = pi[:, :-1]
        idx_to = pi[:, 1:]

        batch_idx = torch.arange(B, dtype=torch.int64, device=pi.device)[:, None]

        # Collect distances between consecutive nodes
        dists = distance_matrix[batch_idx, idx_from, idx_to]  # (B, N-1)

        # Add distance from last to first to complete the tour
        last_to_first = distance_matrix[torch.arange(B, device=pi.device), pi[:, -1], pi[:, 0]]  # (B,)

        tour_lengths = dists.sum(dim=1) + last_to_first  # (B,)

        # add the final depot visit to the pis so they match the visit times

        pi_with_depot_return = torch.cat((pi, torch.zeros((B, 1), dtype=torch.int64, device=pi.device)), dim=1)  # shape: [B, T+1]
        
        mask = pi_with_depot_return != 0  # [B, T]
        batch_indices = torch.arange(B, device=pi.device).unsqueeze(1).expand_as(pi_with_depot_return)  # [B, T]

        # Get valid pi indices and batch indices (flattened where mask is True)
        pi_wo_depot = pi_with_depot_return[mask]  # [N]
        batch_idx_flat = batch_indices[mask]  # [N]

        # Gather visit_times and corresponding window_ends values
        visit_times_wo_depots = visit_times[mask]  # [N]
        window_ends_ordered = window_ends[batch_idx_flat, pi_wo_depot]  # [N]

        # Compute lateness
        time_difference = torch.clamp(visit_times_wo_depots - window_ends_ordered, min=0.0)  # [N]

        # Accumulate lateness per batch
        overall_lateness_penalty = torch.zeros(B, device=visit_times.device)
        overall_lateness_penalty = overall_lateness_penalty.index_add(0, batch_idx_flat, time_difference)

        gamma = gamma.reshape(B, -1)[:, 0]
        theta = theta.reshape(B, -1)[:, 0]
        return theta * tour_lengths + gamma * overall_lateness_penalty, {
            'distance_penalty': tour_lengths,
            'lateness_penalty': overall_lateness_penalty
        }
    
    @staticmethod
    def get_times(input, pi):
        """ Returns PDTRP visit times for a given set of tours (pi)"""

        batch_size, tour_length = pi.size()

        visit_times = torch.zeros(batch_size, tour_length + 1, device=pi.device)  # Initialize visit times tensor +1 for return to depot
        total_travel_times = torch.zeros(batch_size, device=pi.device)  # Initialize total travel time tensor
        total_service_times = torch.zeros(batch_size, device=pi.device)  # Initialize total service time tensor
        total_waiting_times = torch.zeros(batch_size, device=pi.device)  # Initialize total waiting time tensor
        window_starts = input['window_starts']

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
                    visit_time = max(time + travel_time, input['arrival_times'][b, curr_node] + travel_time, window_starts[b, curr_node])  # Ensure visit time respects arrival time and time window start
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
        return PDCTRPTWDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDCTRPTW.initialize(*args, **kwargs)

class PDCTRPTWDataset(Dataset):
    def __init__(self, min_total=None, max_total=None, min_dod=0.2, max_dod=0.8, speed=4, time_horizon=8, service_times_mean=3, service_times_var=5, n_subregions=9, arrival_weights = None, arrival_skews=None, min_time_window=None, max_time_window=None, batch_size=128, num_samples=12800, neighbors=20, knn_strat='percentage', filename=None, offset=0, gamma=1.0, theta=1.0, latest_end=2, reaction_time=60, vehicle_capacity=1.0, min_trips_required_lb=3, min_trips_required_ub=9, use_ortec=None, dimension_scale=1.0, time_window_scenario="wide", min_customers=None, max_customers=None, time_horizon_per_customer=None):
        min_total, max_total = resolve_node_bounds(
            min_total, max_total, min_customers, max_customers
        )
        if batch_size < 1 or num_samples % batch_size != 0:
            raise ValueError("num_samples must be divisible by a positive batch_size")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        self.filename = filename
        self.min_total = min_total
        self.max_total = max_total
        self.min_customers = min_total - 1
        self.max_customers = max_total - 1
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.batch_size = batch_size
        self.n_subregions = n_subregions
        self.dimension_scale = dimension_scale
        self.time_horizon = time_horizon * 60 # convert to minutes
        self.time_horizon_per_customer = validate_hours_per_customer(time_horizon_per_customer)
        self.gamma = gamma
        self.theta = theta
        self.service_times_mean = service_times_mean
        self.service_times_std = np.sqrt(service_times_var)
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.offset = offset
        self.arrival_weights = arrival_weights
        self.arrival_skews = arrival_skews
        self.speed = speed / 60.0  # convert to minutes
        self.use_ortec = use_ortec
        self.latest_end = latest_end * 60 # maximum time after the time horizon that a time window can end
        self.min_time_window, self.max_time_window = resolve_time_window_bounds(
            time_window_scenario, min_time_window, max_time_window
        )
        self.reaction_time = reaction_time        
        self.vehicle_capacity = vehicle_capacity
        self.min_trips_required_lb = min_trips_required_lb
        self.min_trips_required_ub = min_trips_required_ub

        self.all_nodes = []
        self.arrival_times = []
        self.service_times = []
        self.window_starts = []
        self.window_ends = []
        self.demands = []

        depot_location = np.full(2, self.dimension_scale / 2)

        if filename is not None:

            self.tours = []
            self.visit_times = []

            if self.use_ortec is not None:
                self.distance_matrix = []

            print("Loading from {}...".format(filename))

            with open(filename, "r") as dataset_file:
                lines = islice(dataset_file, offset, offset + num_samples)
                for line in tqdm(lines, ascii=True):
                    line = line.split(" ")
                    n_nodes = int(line.index('arrival_times')//2)
                    self.all_nodes.append(
                        [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * n_nodes, 2)]
                    )
                    self.arrival_times.append([float(x) for x in line[line.index('arrival_times') + 1:line.index('service_times')]])
                    self.service_times.append([float(x) for x in line[line.index('service_times') + 1:line.index('window_starts')]])
                    self.window_starts.append([float(x) for x in line[line.index('window_starts') + 1:line.index('window_ends')]])
                    self.window_ends.append([float(x) for x in line[line.index('window_ends') + 1:line.index('demands')]])
                    self.demands.append([float(x) for x in line[line.index('demands') + 1:line.index('tour')]])
                    self.tours.append([int(x) - 1 for x in line[line.index('tour') + 1:line.index('visit_times') - 1]])
                    if self.use_ortec is not None:
                        # If using ORTEC, the distance matrix is already computed
                        self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:line.index('distance_matrix')]])
                        self.distance_matrix.append(np.reshape(np.array([float(x) for x in line[line.index('distance_matrix') + 1:]]), (n_nodes, n_nodes)))
                    else:
                        self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:]])

            if not self.all_nodes:
                raise ValueError(
                    f"No samples found in dataset {filename} at offset {offset}"
                )

        elif self.use_ortec is not None:
            print("Generating PDCTRPTW instances from ORTEC instance: {}".format(self.use_ortec))
            # For this manner of generating 'Real' PDTRP instances, the dod arguments won't apply and instead the dod will be determined randomly based on the subsampling
            self.distance_matrix = []
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int)
                batch_horizon = horizon_minutes(
                    n_total - 1, self.time_horizon, self.time_horizon_per_customer
                )
                # Step 2: Generate the locations and arrival times of the customers by subsampling ortec instances
                all_nodes = []
                arrival_times = []
                service_times = []
                distance_matrix = []
                window_starts = []
                window_ends = []
                demands = []
                for _ in range(batch_size):
                    # sample the number of immediate customers for this instance
                    n_imm = np.random.randint(int(self.min_dod * (n_total - 1)), int(self.max_dod * (n_total - 1)) + 1, dtype=int)
                    batch_nodes, batch_service_times, batch_arrival_times, batch_distance_matrix, batch_start_times, batch_end_times, batch_demands = subsample_ortec(instance_file=self.use_ortec, problem='pdctrptw', n_total=n_total, time_horizon=batch_horizon, speed=self.speed, latest_end=self.latest_end,vehicle_capacity=self.vehicle_capacity, min_trips_required_lb=self.min_trips_required_lb, min_trips_required_ub=self.min_trips_required_ub, n_imm=n_imm, reaction_time=self.reaction_time)
                    all_nodes.append(batch_nodes)
                    arrival_times.append(batch_arrival_times)
                    service_times.append(batch_service_times)
                    distance_matrix.append(batch_distance_matrix)
                    demands.append(batch_demands)
                    window_starts.append(batch_start_times)
                    window_ends.append(batch_end_times)

                # Keep sampled customer indices and their attached attributes.
                self.all_nodes += all_nodes
                self.arrival_times += arrival_times
                self.service_times += service_times
                self.distance_matrix += distance_matrix
                self.window_starts += window_starts
                self.window_ends += window_ends
                self.demands += demands

        else:
            # For now, spatial coordinates of customers will be generated using the subregion method rather than using a seed dataset        
            if arrival_weights is not None and arrival_skews is not None:
                print("Generating PDCTRPTW instances with {} subregion(s) with arrival weights {} and skews {}...".format(self.n_subregions, self.arrival_weights, self.arrival_skews))
            elif arrival_weights is not None:
                print("Generating PDCTRPTW instances with {} subregion(s) with arrival weights {} and uniform arrival skews...".format(self.n_subregions, self.arrival_weights))
            elif arrival_skews is not None:
                print("Generating PDCTRPTW instances with {} subregion(s) with dirichlet arrival weights and skews {}...".format(self.n_subregions, self.arrival_skews))
            else:
                print("Generating PDCTRPTW instances with {} subregion(s) with dirichlet arrival weights and uniform skews...".format(self.n_subregions))
            print("Time window min: {}, max: {}, latest end: {}, opening rule: call + depot travel, Vehicle Capacity: {}, Min Trips Required Lower Bound: {}, Upper Bound: {}".format(self.min_time_window, self.max_time_window, self.latest_end, self.vehicle_capacity, self.min_trips_required_lb, self.min_trips_required_ub))
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int) - 1 # -1 to account for the depot
                batch_horizon = horizon_minutes(
                    n_total, self.time_horizon, self.time_horizon_per_customer
                )
                # Step 2: Generate the locations and arrival times of the customers
                all_nodes = []
                arrival_times = []
                for _ in range(batch_size):
                    batch_nodes, batch_arrival_times = time_and_subregion_generation(n_total=n_total, n_subregions=self.n_subregions, arrival_weights=self.arrival_weights, arrival_skews=self.arrival_skews,time_horizon=batch_horizon, dimension_scale=self.dimension_scale)
                    all_nodes.append(batch_nodes)
                    arrival_times.append(batch_arrival_times)
                # step 4: Generate customer service times
                service_times = list(np.round(sample_lognorm(self.service_times_mean, self.service_times_std, size=(batch_size, n_total)), decimals=3))
                # step 5: sample the number of advanced request customers for each instance in the batch
                n_imm = np.random.randint(int(self.min_dod * n_total), int(self.max_dod * n_total) + 1, size=(batch_size))
                n_adv = n_total - n_imm
                # Mark advance requests before generating their distinct windows.
                adv_indices = [np.random.choice(n_total, n_adv[i], replace=False) for i in range(batch_size)]
                batch_start_times = []
                batch_end_times = []
                for i in range(batch_size):
                    arrival_times[i][adv_indices[i]] = 0
                    depot_travel = np.linalg.norm(all_nodes[i] - depot_location, axis=1) / self.speed
                    start_times, end_times = generate_time_windows(
                        arrival_times[i], depot_travel,
                        self.min_time_window, self.max_time_window,
                        batch_horizon + self.latest_end,
                    )
                    batch_start_times.append(start_times)
                    batch_end_times.append(end_times)
                # Retain sampled indices for coordinates, calls and windows.
                # Step 11: Generate the demands for each customer
                demands_without_depot = generate_scaled_demands(batch_size, n_total, self.vehicle_capacity, self.min_trips_required_lb, self.min_trips_required_ub)

                # step 12: Prepend the depot to the customers
                all_nodes = [np.concatenate([depot_location[None, :], x]) for x in all_nodes]
                arrival_times = [np.concatenate((np.zeros(1), x)) for x in arrival_times]
                service_times = [np.concatenate((np.zeros(1), x)) for x in service_times]
                batch_start_times = [np.concatenate((np.zeros(1), x)) for x in batch_start_times]
                batch_end_times = [np.concatenate((np.zeros(1), x)) for x in batch_end_times]
                demands = [np.concatenate((np.zeros(1), x)) for x in demands_without_depot]

                self.all_nodes += all_nodes
                self.arrival_times += arrival_times
                self.service_times += service_times
                self.window_starts += batch_start_times
                self.window_ends += batch_end_times
                self.demands += demands

        self.size = len(self.all_nodes)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):

        all_nodes = self.all_nodes[idx]
        arrival_times = self.arrival_times[idx]
        service_times = self.service_times[idx]
        window_starts = self.window_starts[idx]
        window_ends = self.window_ends[idx]
        demands = self.demands[idx]
        if self.use_ortec:
            distance_matrix = self.distance_matrix[idx]
            nn_graph, _ = nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat, distance_matrix=distance_matrix)
        else:
            nn_graph, distance_matrix = nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat)

        # Add groundtruth labels in case of SL         
        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'arrival_times': torch.FloatTensor(arrival_times),
            'service_times': torch.FloatTensor(service_times),
            'service_times_mean': torch.FloatTensor([self.service_times_mean]),
            'service_times_std': torch.FloatTensor([self.service_times_std]),
            'min_time_window': torch.FloatTensor([self.min_time_window]),
            'max_time_window': torch.FloatTensor([self.max_time_window]),
            'window_starts': torch.FloatTensor(window_starts),
            'window_ends': torch.FloatTensor(window_ends),
            'graph': ~torch.BoolTensor(nn_graph),
            'demand': torch.FloatTensor(demands),
            'speed': torch.FloatTensor([self.speed]),
            'vehicle_capacity': torch.FloatTensor([self.vehicle_capacity]),
            'distance_matrix': torch.FloatTensor(distance_matrix),
            'time_horizon': torch.FloatTensor([
                horizon_minutes(len(all_nodes) - 1, self.time_horizon, self.time_horizon_per_customer)
            ]),
            'latest_end': torch.FloatTensor([self.latest_end]),
            'gamma': torch.FloatTensor([self.gamma]),
            'theta': torch.FloatTensor([self.theta]),

        }
        if self.filename is not None:
           item['tour_nodes'] = torch.LongTensor(self.tours[idx])
           item['visit_times'] = torch.FloatTensor(self.visit_times[idx])

        return item

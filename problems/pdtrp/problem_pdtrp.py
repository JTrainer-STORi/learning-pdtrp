from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

from problems.pdtrp.state_pdtrp import StatePDTRP

from problems.utils import nearest_neighbor_graph, sample_lognorm, subsample_ortec 
from problems.location_generation import time_and_subregion_generation

class PDTRP(object):
    """
    Class representing the Partially Dynamic Travelling Repairman Problem
    """

    NAME = "pdtrp"

    @staticmethod
    def get_costs(input, pi):
        """
        Returns PDTRP tour length for a given set of tours (pi) and a precomputed distance matrix.
        
        Assumes:
            - input['distance_matrix']: shape (B, N, N)
            - pi: LongTensor of shape (B, N), each row is a permutation of node indices
        Returns:
            - tour length for each sample in batch (B,)
            - None (for compatibility with the original return type)
        """
        distance_matrix = input['distance_matrix']  # (B, N, N)
        B, N = pi.size()

        # Validate that each tour is a valid permutation
        assert (
            torch.arange(N, device=pi.device).view(1, -1).expand_as(pi) == pi.sort(dim=1)[0]
        ).all(), f"Invalid tour:\n{pi}"

        # Gather distances along the tour
        idx_from = pi[:, :-1]  # shape (B, N-1)
        idx_to = pi[:, 1:]     # shape (B, N-1)

        batch_idx = torch.arange(B, device=pi.device).unsqueeze(1).expand_as(idx_from)

        # Collect distances between consecutive nodes
        dists = distance_matrix[batch_idx, idx_from, idx_to]  # (B, N-1)

        # Add distance from last to first to complete the tour
        last_to_first = distance_matrix[torch.arange(B, device=pi.device), pi[:, -1], pi[:, 0]]  # (B,)

        tour_lengths = dists.sum(dim=1) + last_to_first  # (B,)

        return tour_lengths, {}

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDTRPDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDTRP.initialize(*args, **kwargs)
    
class PDTRPDataset(Dataset):
    def __init__(self, min_total=20, max_total=100, min_dod=0.2, max_dod=0.8, speed=4, time_horizon=8, service_times_mean=3, service_times_var=5, n_subregions=9, arrival_weights=None, arrival_skews=None, min_time_window=None, max_time_window=None, batch_size=128, num_samples=12800, neighbors=20, knn_strat='percentage', filename=None, offset=0, gamma=None, theta=None, latest_end=None, reaction_time=None, vehicle_capacity=None, min_trips_required=None, use_ortec=None):
        """
        Use this function for either loading a dataset from a file or generating a new dataset.

        Args:
            min_total (int): Minimum number of customers in the problem.
            max_total (int): Maximum number of customers in the problem.
            min_dod (float): Minimum fraction of immediate request customers.
            max_dod (float): Maximum fraction of immediate request customers.
            speed (float): Speed of the vehicle in the problem (in units/h).
            time_horizon (int): Time horizon for the problem (in hours).
            service_times_mean (float): Mean of the service times (in minutes).
            service_times_var (float): Variance of the service times (in minutes).
            n_subregions (int): Number of subregions to generate the customers in. Passing 1 will generate customers uniformly in the unit square.
            arrival_weights (list): Weights for the subregions, if None, subregion weights are sampled from a dirichlet distribution.
            batch_size (int): Number of samples to generate in one batch.
            num_samples (int): Total number of samples to generate.
            neighbors (int): Number of nearest neighbors to consider for the graph.
            knn_strat (str): Strategy for nearest neighbor graph construction, either 'percentage' or 'fixed'.

        """

        self.filename = filename
        self.min_total = min_total
        self.max_total = max_total
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.batch_size = batch_size
        self.n_subregions = n_subregions
        self.time_horizon = time_horizon * 60 # Convert time horizon to minutes
        self.service_times_mean = service_times_mean
        self.service_times_std = np.sqrt(service_times_var)
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.offset = offset
        self.arrival_weights = arrival_weights
        self.arrival_skews = arrival_skews
        self.speed = speed / 60.0  # Convert speed to units/minute
        self.use_ortec = use_ortec
        
        self.all_nodes = []
        self.arrival_times = []
        self.service_times = []
        self.visit_times = []
        self.tours = []

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
                self.service_times.append([float(x) for x in line[line.index('service_times') + 1:line.index('tour')]])
                self.tours.append([int(x) - 1 for x in line[line.index('tour') + 1:line.index('visit_times') - 1]])

                if self.use_ortec is not None:
                    # If using ORTEC, the distance matrix is already computed
                    self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:line.index('distance_matrix')]])
                    self.distance_matrix.append(np.reshape(np.array([float(x) for x in line[line.index('distance_matrix') + 1:]]), (n_nodes, n_nodes)))
                else: 
                    self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:]])

        elif self.use_ortec is not None:
            print("Generating PDTRP instances from ORTEC instance: {}".format(self.use_ortec))
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
                for _ in range(batch_size):
                    batch_nodes, batch_service_times, batch_arrival_times, batch_distance_matrix, _, _, _ = subsample_ortec(instance_file=self.use_ortec, problem='pdtrp', n_total=n_total, time_horizon=self.time_horizon, speed=self.speed)
                    all_nodes.append(batch_nodes)
                    arrival_times.append(batch_arrival_times)
                    service_times.append(batch_service_times)
                    distance_matrix.append(batch_distance_matrix)

                # step 3: sort the customers by arrival time
                for i in range(batch_size):
                    idx = np.argsort(arrival_times[i])
                    all_nodes[i] = all_nodes[i][idx]
                    arrival_times[i] = arrival_times[i][idx]
                    service_times[i] = service_times[i][idx]
                    distance_matrix[i] = distance_matrix[i][idx][:, idx]

                self.all_nodes += all_nodes
                self.arrival_times += arrival_times
                self.service_times += service_times
                self.distance_matrix += distance_matrix

        else:
            if arrival_weights is not None and arrival_skews is not None:
                print("Generating PDTRP instances with {} subregion(s) with arrival weights {} and skews {}...".format(self.n_subregions, self.arrival_weights, self.arrival_skews))
            elif arrival_weights is not None:
                print("Generating PDTRP instances with {} subregion(s) with arrival weights {} and uniform arrival skews...".format(self.n_subregions, self.arrival_weights))
            elif arrival_skews is not None:
                print("Generating PDTRP instances with {} subregion(s) with dirichlet arrival weights and skews {}...".format(self.n_subregions, self.arrival_skews))
            else:
                print("Generating PDTRP instances with {} subregion(s) with dirichlet arrival weights and uniform skews...".format(self.n_subregions))

            for _ in tqdm(range(num_samples//batch_size), ascii=True):   

                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int) - 1 # -1 to account for the depot
                # Step 2: Generate the locations and arrival times of the customers
                all_nodes = []
                arrival_times = []
                for _ in range(batch_size):
                    batch_nodes, batch_arrival_times = time_and_subregion_generation(n_total=n_total, n_subregions=self.n_subregions, arrival_weights=self.arrival_weights, arrival_skews=self.arrival_skews, time_horizon=self.time_horizon)
                    all_nodes.append(batch_nodes)
                    arrival_times.append(batch_arrival_times)
                # step 4: Generate customer service times
                service_times = list(np.round(sample_lognorm(self.service_times_mean, self.service_times_std, size=(batch_size, n_total)), decimals=3))
                # step 5: sample the number of immediate request customers for each instance in the batch
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
                # step 9: Prepend the depot to the customers
                all_nodes = [np.concatenate([np.array([[0.5, 0.5]]), x]) for x in all_nodes]
                arrival_times = [np.concatenate((np.zeros(1), x)) for x in arrival_times]
                service_times = [np.concatenate((np.zeros(1), x)) for x in service_times]

                self.all_nodes += all_nodes
                self.arrival_times += arrival_times
                self.service_times += service_times

        self.size = len(self.all_nodes)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):

        all_nodes = self.all_nodes[idx]
        arrival_times = self.arrival_times[idx]
        service_times = self.service_times[idx]
        if self.use_ortec is not None:
            # If using ortec, the distance matrix is already computed
            distance_matrix = self.distance_matrix[idx]
            nn_graph, _ = nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat, distance_matrix=distance_matrix)
        else:
            nn_graph, distance_matrix = nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat)

        # Add groundtruth labels in case of SL         
        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'arrival_times': torch.FloatTensor(arrival_times),
            'service_times': torch.FloatTensor(service_times),
            'graph': ~torch.BoolTensor(nn_graph),
            'speed': torch.FloatTensor([self.speed]),
            'time_horizon': torch.FloatTensor([self.time_horizon]),
            'distance_matrix': torch.FloatTensor(distance_matrix),
        }
        if self.filename is not None:
            item['tour_nodes'] = torch.LongTensor(self.tours[idx])
            item['visit_times'] = torch.FloatTensor(self.visit_times[idx])

        return item
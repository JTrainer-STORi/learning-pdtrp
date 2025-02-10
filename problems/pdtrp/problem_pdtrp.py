from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

from problems.pdtrp.state_pdtrp import StatePDTRP

from problems.utils import nearest_neighbor_graph
from problems.location_generation import subregion_generation

class PDTRP(object):
    """
    Class representing the Partially Dynamic Travelling Repairman Problem
    """

    NAME = "pdtrp"

    def get_costs(input, pi):
        """Returns PDTRP tour length for a given set of nodes and a tour"""

        all_nodes = input['loc']

        # Check that tours are valid, i.e. contain all nodes and each node exactly once
        sorted_pi = pi.data.sort(1)[0]
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(all_nodes, pi)

        # gather dataset in order of tour
        d = all_nodes.gather(1, pi.unsqueeze(-1).expand_as(all_nodes))

        # length is distance (l2-norm of difference) from each node location from its prev and of last to first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDTRPDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDTRP.initialize(*args, **kwargs)
    
class PDTRPDataset(Dataset):
    def __init__(self, min_total=20, max_total=100, min_dod=0.2, max_dod=0.8, time_horizon=480, service_times_mu=0.8777, service_times_sigma=0.6647, arrival_weights=None, batch_size=128, num_samples=12800, neighbors=20, knn_strat='percentage', filename=None, offset=0):
        self.filename = filename
        self.min_total = min_total
        self.max_total = max_total
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.batch_size = batch_size
        #self.time_horizon = time_horizon
        #self.service_times_mu = service_times_mu
        #self.service_times_sigma = service_times_sigma
        self.time_horizon = 1
        self.service_times_mu = -5.09
        self.service_times_sigma = np.sqrt(1.33)
        self.vehicle_speed = 32 # corresponds to 1000 units moved within the time horizon
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.offset = offset
        self.arrival_weights = arrival_weights

        self.all_nodes = []
        self.arrival_times = []
        self.service_times = []
        self.tours = []

        if filename is not None:
            self.tours = []

            print("Loading from {}...".format(filename))

            for line in tqdm(open(filename, "r").readlines()[offset:offset + num_samples], ascii=True):
                line = line.split(" ")
                n_nodes = int(line.index('arrival_times')//2)
                self.all_nodes.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * n_nodes, 2)]
                )
                self.arrival_times.append([float(x) for x in line[line.index('arrival_times') + 1:line.index('service_times')]])
                self.service_times.append([float(x) for x in line[line.index('service_times') + 1:line.index('output')]])
                self.tours.append([int(x) - 1 for x in line[line.index('output') + 1:-1]])

        else:

            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int) - 1 # -1 to account for the depot
                # Step 2: Generate the locations of the customers
                all_nodes = [subregion_generation(n_total=n_total, arrival_weights=self.arrival_weights) for _ in range(batch_size)]
                # step 3: Generate customer arrival times
                arrival_times = [np.round(np.random.uniform(0.001, self.time_horizon, size=(n_total)), decimals=3) for _ in range(batch_size)]
                # step 4: Generate customer service times
                service_times = list(np.round(np.random.lognormal(self.service_times_mu, self.service_times_sigma, size=(batch_size, n_total)), decimals=3))
                # step 5: sample the number of advanced request customers for each instance in the batch
                n_adv = np.random.randint(int(self.min_dod * n_total), int(self.max_dod * n_total), size=(batch_size))
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

        # Add groundtruth labels in case of SL         
        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'arrival_times': torch.FloatTensor(arrival_times),
            'service_times': torch.FloatTensor(service_times),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat)),
        }
        if self.filename is not None:
            item['tour_nodes'] = torch.LongTensor(self.tours[idx])

        return item
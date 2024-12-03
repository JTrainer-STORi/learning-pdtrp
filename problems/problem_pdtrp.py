from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

from problems.state_pdtrp import StatePDTRP

from problem_pdtsp import nearest_neighbor_graph

class PDTRP(object):
    """
    Class representing the Partially Dynamic Travelling Repairman Problem
    """

    NAME = "pdtrp"

    def get_costs(all_nodes, pi):
        """Returns PDTRP tour length for a given set of nodes and a tour"""

        # Check that tours are valid, i.e. contain all nodes and each node exactly once
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(all_nodes, pi)

        # gather dataset in order of tour
        d = all_nodes.gather(1, pi.unsqueeze(-1).expand_as(all_nodes))

        # length is distance (l2-norm of difference) from eahc node location from its prev and of last to first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDTRPDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDTRP.initialize(*args, **kwargs)
    
class PDTRPDataset(Dataset):
    def __init__(self, min_static=20, max_static=50, min_dyna=20,
                 max_dyna=50, batch_size=128, time_horizon=8, num_samples=12800, neighbours=20, knn_strat='percentage', filename=None, offset=0):
        self.filename = filename
        self.min_static = min_static
        self.max_static = max_static
        self.min_exp_dyna = min_dyna
        self.max_exp_dyna = max_dyna
        self.batch_size = batch_size
        self.time_horizon = time_horizon
        self.neighbours = neighbours
        self.knn_strat = knn_strat
        self.offset = offset

        self.static_v = []
        self.dyna_v = []
        self.all_nodes_v = []
        self.arrival_times_v = []
        self.service_times_v = []

        for _ in tqdm(range(num_samples//batch_size), ascii=True):
            # Step 1: Generate the number of static customers
            n_static_v = np.random.randint(self.min_static, self.max_static)
            # Step 2: Generate the locations of static customers
            loc_static_v = list(np.random.uniform(0, 1, size=(batch_size, n_static_v, 2)))
            # Step 3: Generate the number of dynamic customers for each instance in the batch from a poisson distribution
            exp_dyna_v = np.random.randint(self.min_exp_dyna, self.max_exp_dyna)
            n_dyna_v = np.random.poisson(lam=exp_dyna_v)
            # Step 4: Generate the locations of dynamic customers
            loc_dyna_v = list(np.random.uniform(0, 1, size=(batch_size, n_dyna_v, 2)))
            # Step 5: Generate the arrival times for dynamic customers
            arrival_times_v = list(np.random.uniform(0, self.time_horizon, size=(batch_size, n_dyna_v)))
            # Step 6: Generate the service times for both sets of customers
            service_times_v = list(np.random.lognormal(0.8777, 0.6647, size=(batch_size, n_static_v + n_dyna_v)))
            all_nodes = [np.concatenate((x, y)) for x, y in zip(loc_static_v, loc_dyna_v)]

            self.static_v += loc_static_v
            self.dyna_v += loc_dyna_v
            self.all_nodes_v += all_nodes
            self.arrival_times_v += arrival_times_v
            self.service_times_v += service_times_v

        self.size = len(self.times)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):

        nodes = self.static_v[idx]
        dyna_v = self.dyna_v[idx]
        arrival_times = self.arrival_times_v[idx]
        service_times = self.service_times_v[idx]
        all_nodes = self.all_nodes[idx]

        # Add groundtruth labels in case of SL         
        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'nodes': torch.FloatTensor(nodes),
            'new_nodes': torch.FloatTensor(dyna_v),
            'arrival_times': torch.LongTensor(np.array(arrival_times)),
            'service_times': torch.FloatTensor(np.array(service_times)),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph(nodes, self.neighbors, self.knn_strat)),
        }

        return item
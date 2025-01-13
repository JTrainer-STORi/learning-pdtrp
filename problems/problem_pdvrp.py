from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np


from problems.problem_pdtsp import nearest_neighbor_graph

from problems.state_pdcvrp import StatePDCVRP


class PDCVRP(object):

    NAME = 'pdcvrp'

    VEHICLE_CAPACITY = 1.0

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        graph_size -= 1
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -PDCVRP.VEHICLE_CAPACITY),
                dataset['demand'][:, 1:]
            ),
            1
        )
        dem = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += dem[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= PDCVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = dataset['loc']
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['loc'][:, 0, :]).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['loc'][:, 0, :]).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDCVRP.initialize(*args, **kwargs)
    
class VRPDataset(Dataset):
    def __init__(self, min_total=20, max_total=100, min_dod=0.2, max_dod=0.8, time_horizon=480, service_times_mu=0.8777, service_times_sigma=0.6647, batch_size=128, num_samples=12800, neighbors=20, knn_strat='percentage', filename=None, offset=0, seed_dataset=None):
        super(VRPDataset, self).__init__()
        self.filename = filename
        self.min_total = min_total
        self.max_total = max_total
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.num_samples = num_samples
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.time_horizon = time_horizon

        self.all_nodes = []
        self.arrival_times = []
        self.demands = []
        self.service_times = []

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
                self.service_times.append([float(x) for x in line[line.index('service_times') + 1:line.index('demand')]])
                self.demands.append([float(x) for x in line[line.index('demand') + 1:line.index('output')]])
                self.tours.append([int(x) - 1 for x in line[line.index('output') + 1:-1]][:-1])

        else:

            seed_coords = np.load(seed_dataset)

            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int)
                # Step 2: Generate the expected number of static customers for each instance in the batch
                lower_bound = np.floor((1-self.max_dod)*n_total)
                upper_bound = np.ceil((1-self.min_dod)*n_total)
                exp_n_static = np.random.randint(lower_bound, upper_bound)
                # Step 3: Sample the number of dynamic customers for each instance in the batch from a poisson distribution
                p_lambda = (n_total - exp_n_static) / time_horizon
                n_dyna = np.random.poisson(p_lambda * time_horizon, batch_size)
                n_dyna = np.clip(n_dyna, a_min=lower_bound, a_max=upper_bound)
                n_dyna = n_dyna.astype(int)
                # Step 4: Determine the number of static customers for each instance in the batch
                n_static = n_total - n_dyna
                # Step 5: Select n_total locations from the seed dataset
                all_nodes = [seed_coords[np.random.choice(seed_coords.shape[0], n_total, replace=False)] for _ in range(batch_size)]
                all_nodes = [np.concatenate([np.array([[0.5, 0.5]]), x]) for x in all_nodes]
                # Step 8: Generate the arrival times for dynamic customers
                dyna_arrival_times = [np.sort(np.random.uniform(0.1, self.time_horizon, size=(n_dyna[i]))) for i in range(batch_size)]
                arrival_times = [np.concatenate((np.zeros(n_static[i] + 1), dyna_arrival_times[i])) for i in range(batch_size)] # +1 for the depot
                #arrival_times = list(np.sort(np.random.uniform(0.1, self.time_horizon, size=(batch_size, int(n_dyna)))))
                # Step 9: Generate the demands for each customer
                capacity = np.floor(14*np.log(n_total + 3) - 15) # this function given similar capacities to the ones in ALTSRP implementation of the CVRP
                demands_without_depot = list(np.random.randint(1, 10, size=(batch_size, n_total)) / capacity)
                demands = [np.concatenate((np.zeros(1), x)) for x in demands_without_depot]
                # Step 10: Generate the service times for both sets of customers
                service_times = list(np.random.lognormal(service_times_mu, service_times_sigma, size=(batch_size, n_total))) 
                service_times = [np.concatenate((np.zeros(1), x)) for x in service_times] # 

                self.all_nodes += all_nodes
                self.arrival_times += arrival_times
                self.demands += demands
                self.service_times += service_times

        self.size = len(self.all_nodes)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)

        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        all_nodes  = self.all_nodes[idx]
        arrival_times = self.arrival_times[idx]
        demands = self.demands[idx]
        service_times = self.service_times[idx]

        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'arrival_times': torch.FloatTensor(arrival_times),
            'service_times': torch.FloatTensor(service_times),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat)),
            'demand': torch.FloatTensor(demands),
        }

        if self.filename is not None:
            item['tour_nodes'] = torch.LongTensor(self.tours[idx])

        return item
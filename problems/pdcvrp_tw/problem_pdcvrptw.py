from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

from problems.pdcvrp_tw.state_pdcvrptw import StatePDCVRPTW

from problems.utils import nearest_neighbor_graph
from problems.location_generation import subregion_generation

class PDCVRPTW(object):
    """
    Class representing the Partially Dynamic Travelling Repairman Problem
    """

    NAME = "pdcvrptw"

    VEHICLE_CAPACITY = 1.0

    def get_costs(input, pi, gamma):
        """Returns cost based on the tour and missed customers as described in Larsen, Madsen and Solomon (2004)"""

        theta = 3.0
        batch_size, graph_size = input['demand'].size()
        graph_size -= 1

        all_nodes = input['loc']
        window_ends = input['window_ends']
        visit_times = input['visit_times'] # The times that each customer was visited - should probably be passed as a seperate argument as it is not an input

        # Check that tours are valid, i.e. contain 0 to n - 1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(input['demand'][:, :1], -PDCVRPTW.VEHICLE_CAPACITY),
                input['demand'][:, 1:]
            ),
            1
        )
        dem = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(input['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += dem[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= PDCVRPTW.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # gather dataset in order of tour
        loc_with_depot = input['loc']
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        distance = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - input['loc'][:, 0, :]).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - input['loc'][:, 0, :]).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        )

        # need to reorder time windows in the order of the tour
        overall_difference = torch.zeros(batch_size, device=visit_times.device)

        for i in range(batch_size):
            mask = pi[i] != 0
            pi_wo_depot = pi[i][mask]
            visit_times_wo_depots = visit_times[i].squeeze()[mask]
            window_ends_ordered = window_ends[i][pi_wo_depot]
            time_difference = torch.maximum(visit_times_wo_depots - window_ends_ordered, torch.tensor([0], device=visit_times_wo_depots.device))
            overall_difference[i] = torch.sum(time_difference)

        return theta * distance + gamma * overall_difference, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDCVRPTWDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDCVRPTW.initialize(*args, **kwargs)
    
class PDCVRPTWDataset(Dataset):
    def __init__(self, min_total=20, max_total=100, min_dod=0.2, max_dod=0.8, time_horizon=1, service_times_mu=0.8777, service_times_sigma=0.6647, batch_size=128, num_samples=12800, neighbors=20, knn_strat='percentage', filename=None, offset=0, arrival_weights=None, tw_min=0.0625, tw_max=0.1875, real_data=None):
        self.filename = filename
        self.min_total = min_total
        self.max_total = max_total
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.batch_size = batch_size
        # self.time_horizon = time_horizon
        # self.service_times_mu = service_times_mu
        # self.service_times_sigma = service_times_sigma
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.offset = offset
        self.arrival_weights = arrival_weights
        self.L = 0.125
        #self.tw_min = tw_min
        #self.tw_max = tw_max
        #self.speed = 2000
        self.reaction_time = 0.125
        self.time_horizon = 1
        self.service_times_mu = -5.296
        self.service_times_sigma = np.sqrt(0.442)
        self.vehicle_speed = 16 # corresponds to 32 units moved within the time horizon
        self.tw_min = 0.0625
        self.tw_max = 0.1875

        self.all_nodes = []
        self.arrival_times = []
        self.service_times = []
        self.window_starts = []
        self.window_ends = []
        #self.capacities = []
        self.demand = []

        depot_location = np.array([0.5, 0.5])

        if filename is not None:


            print("Loading from {}...".format(filename))

            for line in tqdm(open(filename, "r").readlines()[offset:offset + num_samples], ascii=True):
                line = line.split(" ")
                n_nodes = int(line.index('arrival_times')//2)
                self.all_nodes.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * n_nodes, 2)]
                )
                self.arrival_times.append([float(x) for x in line[line.index('arrival_times') + 1:line.index('service_times')]])
                self.service_times.append([float(x) for x in line[line.index('service_times') + 1:line.index('window_starts')]])
                self.window_starts.append([float(x) for x in line[line.index('window_starts') + 1:line.index('window_ends')]])
                self.window_ends.append([float(x) for x in line[line.index('window_ends') + 1:line.index('demands')]])
                self.demand.append([float(x) for x in line[line.index('demands') + 1:line.index('tour')]])
                self.tours.append([int(x) - 1 for x in line[line.index('tour') + 1:line.index('visit_times') -1]])
                self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:]])

        else:
            # For now, spatial coordinates of customers will be generated using the subregion method rather than using a seed dataset        

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
                # step 6 Generate customer time windows
                batch_start_times = []
                batch_end_times = []
                for i in range(batch_size):
                    earliest_times = np.linalg.norm(all_nodes[i] - depot_location, axis=1) / self.vehicle_speed
                    start_times = np.round(np.maximum(earliest_times, arrival_times[i] + self.reaction_time), decimals=3)
                    batch_start_times.append(start_times)
                    batch_end_times.append(np.round(np.array([[np.random.uniform(start_times[j] + self.tw_min, np.min([start_times[j] + self.tw_max, self.time_horizon + self.L + self.reaction_time]))] for j in range(n_total)]).squeeze(), decimals=3))
                # step 7: Sample indices of advanced request customers for each instance in the batch
                adv_indices = [np.random.choice(n_total, n_adv[i], replace=False) for i in range(batch_size)]
                # step 8: set the arrival times of the advanced request customers to be 0 and set the time windows of the advance request customers
                for i in range(batch_size):
                    arrival_times[i][adv_indices[i]] = 0
                    earliest_times = np.linalg.norm(all_nodes[i][adv_indices[i]] - depot_location, axis=1) / self.vehicle_speed
                    adv_start_times = np.round(np.maximum(earliest_times, np.random.uniform(0, self.time_horizon + self.L - self.tw_min, n_adv[i])), decimals=3)
                    adv_end_times = np.round(np.array([np.random.uniform(adv_start_times[j] + self.tw_min, np.min([adv_start_times[j] + self.tw_max, self.time_horizon + self.L])) for j in range(n_adv[i])]), decimals=3)
                    batch_start_times[i][adv_indices[i]] = adv_start_times
                    batch_end_times[i][adv_indices[i]] = adv_end_times.squeeze()
                # step 10: sort the customers by arrival time
                for i in range(batch_size):
                    idx = np.argsort(arrival_times[i])
                    all_nodes[i] = all_nodes[i][idx]
                    arrival_times[i] = arrival_times[i][idx]
                    service_times[i] = service_times[i][idx]
                    batch_start_times[i] = batch_start_times[i][idx]
                    batch_end_times[i] = batch_end_times[i][idx]
                # Step 11: Generate the demands for each customer
                demands_without_depot = list(np.random.uniform(0.1,size=(batch_size, n_total)))
                capacities = [0.4*np.sum(demands_without_depot[i]) for i in range(batch_size)]
                # divide demands by the capacities
                demands_without_depot = [np.round(demands_without_depot[i]/capacities[i], decimals=3) for i in range(batch_size)]

                # step 12: Prepend the depot to the customers
                all_nodes = [np.concatenate([np.array([[0.5, 0.5]]), x]) for x in all_nodes]
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
                self.demand += demands
                #self.capacities += [capacities]

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
        #vehicle_capacities = self.capacities[idx]

        # Add groundtruth labels in case of SL         
        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'arrival_times': torch.FloatTensor(arrival_times),
            'service_times': torch.FloatTensor(service_times),
            'window_starts': torch.FloatTensor(window_starts),
            'window_ends': torch.FloatTensor(window_ends),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat)),
            'demand': torch.FloatTensor(self.demand[idx]),
            #'vehicle_capacities': torch.FloatTensor(vehicle_capacities),
        }
        if self.filename is not None:
           item['tour_nodes'] = torch.LongTensor(self.tours[idx])
           item['visit_times'] = torch.FloatTensor(self.visit_times[idx])

        return item
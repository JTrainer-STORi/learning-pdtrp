from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

from problems.state_pdtrptw import StatePDTRPTW

from problems.problem_pdtsp import nearest_neighbor_graph

class PDTRPTW(object):
    """
    Class representing the Partially Dynamic Travelling Repairman Problem
    """

    NAME = "pdtrptw"

    def get_costs(input, pi):
        """Returns cost based on the tour and missed customers as described in Larsen, Madsen and Solomon (2004)"""

        theta = 3.0
        gamma = 1.0

        all_nodes = input['loc']
        window_ends = input['window_ends']
        visit_times = input['visit_times'] # The times that each customer was visited - should probably be passed as a seperate argument as it is not an input

        # Check that tours are valid, i.e. contain all nodes and each node exactly once
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(all_nodes, pi)

        # gather dataset in order of tour
        d = all_nodes.gather(1, pi.unsqueeze(-1).expand_as(all_nodes))

        # length is distance (l2-norm of difference) from each node location from its prev and of last to first
        distance =  (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
 
        time_differences = torch.maximum(visit_times[:,0, :] - window_ends, torch.tensor([0]))

        overall_difference = torch.sum(time_differences, dim=1)

        return theta * distance + gamma * overall_difference, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDTRPTWDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDTRPTW.initialize(*args, **kwargs)
    
class PDTRPTWDataset(Dataset):
    def __init__(self, min_total=20, max_total=100, min_dod=0.2, max_dod=0.8, time_horizon=1, service_times_mu=0.8777, service_times_sigma=0.6647, batch_size=128, num_samples=12800, neighbors=20, knn_strat='percentage', filename=None, offset=0, arrival_weights=[1,2,1,2,4,2,1,2,1], tw_min=30/480, tw_max=90/480, seed_dataset=None):
        self.filename = filename
        self.min_total = min_total
        self.max_total = max_total
        self.min_dod = min_dod
        self.max_dod = max_dod
        self.batch_size = batch_size
        self.time_horizon = time_horizon
        self.service_times_mu = service_times_mu
        self.service_times_sigma = service_times_sigma
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.offset = offset
        self.arrival_weights = arrival_weights
        self.n_subreg = len(arrival_weights)
        self.L = 0.125
        self.tw_min = tw_min
        self.tw_max = tw_max
        self.speed = 2000
        self.reaction_time = 0.125

        self.all_nodes = []
        self.arrival_times = []
        self.service_times = []
        self.window_starts = []
        self.window_ends = []

        if filename is not None:

            self.tours = []
            self.visit_times = []

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
                self.window_ends.append([float(x) for x in line[line.index('window_ends') + 1:line.index('tour')]])
                self.tours.append([int(x) - 1 for x in line[line.index('tour') + 1:line.index('visit_times')]][:-1])
                self.visit_times.append([float(x) for x in line[line.index('visit_times') + 1:]])

        else:
            # For now, spatial coordinates of customers will be generated using the subregion method rather than using a seed dataset        

            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                
                # Step 1: Generate the total number of customers
                n_total = np.random.randint(self.min_total, self.max_total + 1, dtype=int) - 1 # -1 to account for the depot

                for _ in range(batch_size):

                    n_imm_subreg = np.zeros(self.n_subreg)
                    
                    while not np.sum(n_imm_subreg) == n_total:

                        y_ix = -1 # start at -1 so that the first increment is 0

                        n_imm_subreg = np.zeros(self.n_subreg)

                        depot_location = np.array([0.5, 0.5])

                        customer_locations = []
                        arrival_times = []
                        service_times = []
                        start_times = []
                        end_times = []

                        for j in range(self.n_subreg):
                            exp_n_imm = self.arrival_weights[j] * n_total / np.sum(self.arrival_weights)
                            arrival_rate = exp_n_imm / self.time_horizon

                            # Sample the Poisson distribution
                            n_imm = np.random.poisson(arrival_rate * self.time_horizon, size=1)
                            n_imm_subreg[j] = n_imm

                            # Generate arrival times for these customers
                            arrival_times.append(np.random.uniform(0.1, self.time_horizon, size=n_imm))

                            # Generate customer locations according to their subregion

                            if j % np.sqrt(self.n_subreg) == 0:
                                y_ix += 1
                                if y_ix % 2 ==0:
                                    x_ix = -1
                                else:
                                    x_ix = int(np.sqrt(self.n_subreg))
                            
                            if y_ix % 2 == 0:
                                x_ix += 1
                            else:
                                x_ix -= 1

                            x_limit_1 = x_ix / np.sqrt(self.n_subreg)
                            x_limit_2 = (x_ix+1) / np.sqrt(self.n_subreg)
                            x_coords = np.random.uniform(x_limit_1, x_limit_2, n_imm)
                            
                            y_limit_1 = y_ix / np.sqrt(self.n_subreg)
                            y_limit_2 = (y_ix+1) / np.sqrt(self.n_subreg)
                            y_coords = np.random.uniform(y_limit_1, y_limit_2, n_imm)

                            customer_locations.append(np.column_stack((x_coords, y_coords)))

                            # Generate customer service times
                            service_times.append(np.random.lognormal(0.8777, 0.6647, n_imm) / 480)

                            # Generate customer time windows
                            earliest_times = np.linalg.norm(customer_locations[j] - depot_location, axis=1) / self.speed
                            start_times.append(np.maximum(earliest_times, arrival_times[j] + self.reaction_time))
                            end_times.append(np.array([np.random.uniform(start_times[j][i] + self.tw_min, np.min([start_times[j][i] + self.tw_max, self.time_horizon + self.L + self.reaction_time])) for i in range(n_imm.item())]))
                    
                    ac_unsorted = np.concatenate(customer_locations)
                    at_unsorted = np.concatenate(arrival_times)
                    st_unsorted = np.concatenate(service_times)
                    ws_unsorted = np.concatenate(start_times)
                    we_unsorted = np.concatenate(end_times)

                    # prepend the depot to each array
                    ac_us = np.concatenate([np.array([[0.5, 0.5]]), ac_unsorted])
                    at_us = np.concatenate([np.zeros(1), at_unsorted])
                    st_us = np.concatenate([np.zeros(1), st_unsorted])
                    ws_us = np.concatenate([np.zeros(1), ws_unsorted])
                    we_us = np.concatenate([np.zeros(1), we_unsorted])

                    # now the advance request customers

                    n_adv = np.random.randint(np.ceil(n_total*self.min_dod), np.floor(n_total*self.max_dod))

                    # choose indices of advance request customers at random
                    adv_ix = np.random.choice(range(1, n_total), n_adv, replace=False)

                    # set arrival times of advance request customers to be 0
                    at_us[adv_ix] = 0

                    # Regenerate time windows for advance request customers
                    earliest_times = np.linalg.norm(ac_us[adv_ix] - depot_location, axis=1) / self.speed
                    adv_start_times = np.maximum(earliest_times, np.random.uniform(0, self.time_horizon + self.L - self.tw_min, n_adv))
                    adv_end_times = np.array([np.random.uniform(adv_start_times[i] + self.tw_min, np.min([adv_start_times[i] + self.tw_max, self.time_horizon + self.L])) for i in range(n_adv)])

                    # update the start and end time arrays
                    ws_us[adv_ix] = adv_start_times
                    we_us[adv_ix] = adv_end_times

                    # sort the arrays by arrival time
                    sort_ix = np.argsort(at_us)
                    ac_sorted = [ac_us[sort_ix]]
                    at_sorted = [at_us[sort_ix]]
                    st_sorted = [st_us[sort_ix]]
                    ws_sorted = [ws_us[sort_ix]]
                    we_sorted = [we_us[sort_ix]]

                    self.all_nodes += ac_sorted
                    self.arrival_times += at_sorted
                    self.service_times += st_sorted
                    self.window_starts += ws_sorted
                    self.window_ends += we_sorted

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

        # Add groundtruth labels in case of SL         
        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'arrival_times': torch.FloatTensor(arrival_times),
            'service_times': torch.FloatTensor(service_times),
            'window_starts': torch.FloatTensor(window_starts),
            'window_ends': torch.FloatTensor(window_ends),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat)),
        }
        if self.filename is not None:
           item['tour_nodes'] = torch.LongTensor(self.tours[idx])
           item['visit_times'] = torch.FloatTensor(self.visit_times[idx])

        return item
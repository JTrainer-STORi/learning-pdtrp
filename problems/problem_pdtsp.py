import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from problems.state_pdtsp import StatePDTSP, nearest_neighbor_graph

def truncated_bivariate_normal(alimits, blimits, mean, cov, n_samples):
    samples = np.zeros((0, 2))   # 2 columns now
    while samples.shape[0] < n_samples: 
        s = np.random.multivariate_normal(mean, cov, size=(n_samples,))
        accepted = s[(np.min(s - [alimits[0], blimits[0]], axis=1) >= 0) & (np.max(s - [alimits[1], blimits[1]], axis=1) <= 0)]
        samples = np.concatenate((samples, accepted), axis=0)
    samples = samples[:n_samples, :]
    return samples

class PDTSP(object):
    """
    Class representing the PDTSP problem.
    """

    NAME = "pdtspm"

    def get_costs(all_v, pi):
        """Returns DTSP tour length for given graph nodes and tour permutations
            This should remain fine provided that it is only used to find the cost of completed DTSP tours

        Args:
            all v: co-ordinates of all vertices in tour (torch.Tensor) dtype = torch.float32
            pi: vertex permutations representing tours (torch.Tensor) dtype = torch.int8
            
        Returns:
            TSP tour length, None
        """

        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(all_v, pi)

        # Gather dataset in order of tour
        d = all_v.gather(1, pi.unsqueeze(-1).expand_as(all_v))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDTSPDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDTSP.initialize(*args, **kwargs)
    
class PDTSPDataset(Dataset):

    def __init__(self, min_static=20, max_static=50, min_dyna=20, max_dyna=50, batch_size=128, num_samples=128000,  spatial_distribution=None, td1=None, td2 = None, mean1 = None, mean2 = None, cov1 = None, cov2 = None, proportion = None, allow_multiple_arrivals = False, eval=False, neighbors=20, knn_strat=None, filename=None, offset=0):
        self.filename = filename
        self.min_static = min_static
        self.max_static = max_static
        self.min_dyna = min_dyna
        self.max_dyna = max_dyna
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.offset = offset
        self.eval = eval
        self.neighbors = neighbors
        self.knn_strat = knn_strat

        self.times = []

        # if we have provided a filename, then we will be creating a dataset from that file 

        ### change this when it's more clear what the file format is ###

        self.static_v = []
        self.dyna_v = []
        self.all_nodes = []

        if filename is not None:
            self.tour_nodes = []

            print('\nLoading from {}...'.format(filename))

            for line in tqdm(open(filename, "r").readlines()[offset:offset + num_samples], ascii=True):
                line = line.split(" ")
                n_static_v = int(line.index('new_nodes')//2)
                n_dyna_v = int(line.index('times')//2) - n_static_v
                self.static_v.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * n_static_v, 2)]
                )
                self.dyna_v.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(2 * n_static_v + 1, 2*(n_dyna_v + n_static_v) + 1,2)]
                )
                self.times.append(
                    [int(time) for time in line[line.index('times') + 1:line.index('output')]]
                )
                if self.eval:
                    self.tour_nodes.append(
                        [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                    )

                self.all_nodes = [np.concatenate((x, y)) for x, y in zip(self.static_v, self.dyna_v)]
        else:
            self.masks = []

            print('\nGenerating {} training samples of DTSP with {}-{} static nodes and {}-{} dynamic nodes'.format(
            num_samples, min_static, max_static, min_dyna, max_dyna))
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                static_v, dyna_v, times, mask = self.generate_training_set(spatial_distribution, td1, td2, mean1, mean2, cov1, cov2, proportion, allow_multiple_arrivals)
                all_nodes = [np.concatenate((x, y)) for x, y in zip(static_v, dyna_v)]
                self.all_nodes += all_nodes
                self.static_v += static_v
                self.dyna_v += dyna_v
                self.masks += mask
                self.times += times

        self.size = len(self.times)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)
        
    def generate_training_set(self, spatial_distribution, td1, td2 = None, mean1 = None, mean2 = None, cov1 = None, cov2 = None, proportion = None, allow_multiple_arrivals = False):
        """
        Method for generating a batch of training samples for the PDTSP problem

        Args:
            spatial_distribution: string identifying which spatial distribution to use for dynamic nodes
            td1: string identifying the time distribution for the first distribution of dynamic nodes
            td2: string identifying the time distribution for the second distribution of dynamic nodes in the bimodal case
            mean1: if the spatial distribution is "unimodal" or "bimodal" this is the mean of the first normal distribution
            mean2: if the spatial distribution is "bimodal" this is the mean of the second normal distribution
            cov1: if the spatial distribution is "unimodal" or "bimodal" this is the covariance matrix of the first normal distribution
            cov2: if the spatial distribution is "bimodal" this is the covariance matrix of the second normal distribution
            proportions: if the spatial distribution is "bimodal" this is the proportion of points in the first normal distribution
            allow_multiple_arrivals: boolean indicating whether multiple dynamic nodes can arrive at the same time
        """

        # sample the numbr  of static and dynamic nodes for envs in this batch
        n_static_v = np.random.randint(low=self.min_static, high=self.max_static+1)
        n_dyna_v = np.random.randint(low=self.min_dyna, high=self.max_dyna+1)

        n_total = n_static_v + n_dyna_v

        # sample the co-ordinates of the static nodes uniformly
        static_v = list(np.random.random([self.batch_size, n_static_v, 2]).astype(np.float32))

        b_times = [] # list of lists of times for each batch
        b_dyna_v = [] # list of lists of dynamic vertex coordinates for each batch

        # At the moment, we have three options for temporal distributions: "am", "pm", "equal" corresponding to a higher probability of dynamic nodes being in the morning, evening or no preference in the equal case

        if td1 == "am":
            p1 = np.zeros_like(range(2, n_static_v + n_dyna_v))
            p1[:len(p1)//2] = 2
            p1[len(p1)//2:] = 1
            p1 = p1 / np.sum(p1)
        elif td1 == "pm":
            p1 = np.zeros_like(range(2, n_static_v + n_dyna_v))
            p1[:len(p1)//2] = 1
            p1[len(p1)//2:] = 2
            p1 = p1 / np.sum(p1)
        elif td1 == "equal":
            p1 = np.ones_like(range(2, n_static_v + n_dyna_v))
            p1 = p1 / np.sum(p1)

        if td2:
            if td2 == "am":
                p2 = np.zeros_like(range(2, n_static_v + n_dyna_v))
                p2[:len(p2)//2] = 2
                p2[len(p2)//2:] = 1
                p2 = p2 / np.sum(p2)
            elif td2 == "pm":
                p2 = np.zeros_like(range(2, n_static_v + n_dyna_v))
                p2[:len(p2)//2] = 1
                p2[len(p2)//2:] = 2
                p2 = p2 / np.sum(p2)
            elif td2 == "equal":
                p2 = np.ones_like(range(2, n_static_v + n_dyna_v))
                p2 = p2 / np.sum(p2)

        for _ in range(self.batch_size):
            if spatial_distribution == "uniform":
                dyna_v = np.random.random([n_dyna_v, 2]).astype(np.float32)
                times = np.random.choice(range(2, n_static_v + n_dyna_v), n_dyna_v, replace=allow_multiple_arrivals, p=p1)
                b_times.append(times)
                b_dyna_v.append(dyna_v)
            elif spatial_distribution == "unimodal":
                dyna_v = truncated_bivariate_normal([0, 1], [0, 1], mean1, [cov1[:2], cov1[2:]], n_dyna_v)
                times = np.random.choice(range(2, n_static_v + n_dyna_v), n_dyna_v, replace=allow_multiple_arrivals, p=p1)
                b_times.append(times)
                b_dyna_v.append(dyna_v)
            elif spatial_distribution == "bimodal":
                n_modes = np.random.multinomial(n_dyna_v, [proportion, 1-proportion], size=1)[0] # the number of locations sampled at the each mode
                t_times = []
                t_dyna_v = []
                a_times = list(range(2, n_static_v + n_dyna_v))
                probabilities = [p1, p2]
                means = [mean1, mean2]
                covariances = [[cov1[:2], cov1[2:]], [cov2[:2], cov2[2:]]]
                for mode in range(len(n_modes)):
                    probs = probabilities[mode][[i-2 for i in available_times]]/np.sum(probabilities[mode][[i-2 for i in available_times]])
                    times = np.random.choice(available_times, n_modes[mode], replace=allow_multiple_arrivals, p=probs)
                    t_times.extend(times)
                    if not allow_multiple_arrivals:
                        available_times = [x for x in available_times if x not in times]
                    t_dyna_v.extend(truncated_bivariate_normal([0, 1], [0, 1], means[mode], covariances[mode], n_modes[mode]))

                sorted_indices = np.argsort(t_times)
                b_times.append(np.array(t_times)[sorted_indices])
                b_dyna_v.append(np.array(t_dyna_v)[sorted_indices])


        mask = [np.zeros((n_total), dtype=bool) for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            mask[i][-n_dyna_v:] = True

        return static_v, b_dyna_v, b_times, mask

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):

        nodes = self.static_v[idx]
        dyna_v = self.dyna_v[idx]
        time = self.times[idx]
        all_nodes = self.all_nodes[idx]

        # Add groundtruth labels in case of SL         
        item = {
            'all_nodes': torch.FloatTensor(all_nodes),
            'nodes': torch.FloatTensor(nodes),
            'new_nodes': torch.FloatTensor(dyna_v),
            'times': torch.LongTensor(np.array(time)),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph(nodes, self.neighbors, self.knn_strat)),
        }
        if self.eval:
            # Add groundtruth labels in case of SL
            tour_nodes = self.tour_nodes[idx]
            item['tour_nodes'] = torch.LongTensor(tour_nodes)
        else:
            mask = self.masks[idx]
            item['all_nodes'] = torch.FloatTensor(all_nodes)
            item['complete_graph'] = ~torch.BoolTensor(nearest_neighbor_graph(all_nodes, self.neighbors, self.knn_strat))
            item['mask'] = torch.BoolTensor(mask)
        
        return item

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset

from problems.state_pdtsp_rd import statePDTSPReal, nearest_neighbor_graph_realDTSP, return_distance_matrix, return_distances

def generate_real_instance(nodes, batch_size, min_static=20, max_static=50, min_dyna=20, max_dyna=50, allow_multiple_arrivals=False):
    # these two values are generated once for the batch
    n_static = np.random.randint(low=min_static, high=max_static + 1)
    n_dynamic = np.random.randint(low=min_dyna, high=max_dyna + 1)

    n_total = n_static + n_dynamic

    all_nodes = np.array([np.random.choice(range(len(nodes)), n_total, replace=False) for i in range(batch_size)])

    dynamic_nodes = np.array([np.random.choice(x, n_dynamic, replace=False) for x in all_nodes])

    static_nodes = np.array([np.setdiff1d(x, y) for x,y in zip(all_nodes, dynamic_nodes)])
    node_coords = nodes[static_nodes]
    new_node_coords = nodes[dynamic_nodes]

    times = [np.sort(np.random.choice(range(2, n_total), n_dynamic, replace=allow_multiple_arrivals)) for _ in range(batch_size)]

    return list(node_coords), list(new_node_coords), times

class PDTSPReal(object):
    """Class for the Partially Dynamic Travelling Salesman Problem when we have real data"""

    NAME = 'pdtsp-rd'

    def __init__(self):
        pass

    def get_route_info(self, route_name, device):
        stop_coords = []
        distance_matrix = []
        route_name = 'route03'

        print('\nLoading from {}...'.format(route_name))
        for line in tqdm(open('data/routes/' + route_name + '_nodes.txt', "r").readlines(), ascii=True):
            line = line.split(" ")
            stop_coords.append(
                [float(line[0]), float(line[1])]
            )

        for line in tqdm(open('data/routes/' + route_name + '_distance_matrix.txt', "r").readlines(), ascii=True):
            line = line.split(" ")
            distance_matrix.append(
                [float(x) for x in line]
            )

        self.route_name = route_name
        self.stop_coords = torch.tensor(stop_coords, dtype=torch.float, device=device)
        self.distance_matrix = torch.tensor(distance_matrix, dtype=torch.float, device=device)

    def get_costs(self, dataset, pi):
        """Returns DTSP tour length for given graph nodes and tour permutations
            This should remain fine provided that it is only used to find the cost of completed DTSP tours

        Args:
            dataset: graph nodes (torch.Tensor)
            pi: node permutations representing tours (torch.Tensor)
            
        Returns:
            TSP tour length, None
        """

        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(dataset, pi)

        # Gather dataset in order of tour

        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        
        d2 = torch.roll(d, 1, 1)

        distances = return_distances(d, d2, self.stop_coords, self.distance_matrix)

        costs = torch.sum(distances, axis=1)
        
        return costs, None

    def make_dataset(self, *args, **kwargs):
        return PDTSPRealDataset(stop_coords=self.stop_coords, distance_matrix=self.distance_matrix, *args, **kwargs)

    def make_state(self, *args, **kwargs):
        return statePDTSPReal.initialize(stop_coords=self.stop_coords, distance_matrix=self.distance_matrix, *args, **kwargs)

class PDTSPRealDataset(Dataset):

    def __init__(self, stop_coords, distance_matrix, filename=None, min_static=20, max_static=50, min_dyna=20, max_dyna=50, batch_size=128, num_samples=128000, offset=0, spatial_distribution=None, td1=None, td2 = None, mean1 = None, mean2 = None, cov1 = None, cov2 = None, proportion = None, allow_multiple_arrivals = False, neighbors=20, knn_strat=None, eval=False):
            
        super(PDTSPRealDataset, self).__init__()

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.min_static = min_static
        self.max_static = max_static
        self.min_dyna = min_dyna
        self.max_dyna = max_dyna
        self.offset = offset
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.eval = eval
        self.filename = filename
        self.nodes_coords = stop_coords
        self.distance_matrix = distance_matrix

        self.static_nodes_coords = []
        self.dynamic_nodes_coords = []
        self.times = []

        cpu_distance_matrix = self.distance_matrix.cpu()
        
        if filename is not None:
            self.tour_nodes = []

            print('\nLoading from {}...'.format(filename))
            for line in tqdm(open(filename, "r").readlines()[offset:offset+num_samples], ascii=True):
                line = line.split(" ")
                num_old_nodes = int(line.index('new_nodes')//2)
                num_new_nodes = int(line.index('times')//2) - num_old_nodes
                self.static_nodes_coords.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_old_nodes, 2)]
                )
                self.dynamic_nodes_coords.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(2 * num_old_nodes + 1, 2*(num_new_nodes + num_old_nodes) + 1,2)]
                )
                self.times.append(
                    [int(time) for time in line[line.index('times') + 1:line.index('output')]]
                )
                if self.eval:
                    # Convert tour nodes to required format
                    # Don't add final connection for tour/cycle
                    tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                    self.tour_nodes.append(tour_nodes)

        else:
            print('\nGenerating {} samples of real DTSP with {}-{} static nodes and {}-{} dynamic nodes'.format(num_samples, min_static, max_static, min_dyna, max_dyna))
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                static_nodes_coords, dynamic_nodes_coords, times = generate_real_instance(np.array(cpu_distance_matrix), batch_size, self.min_static, self.max_static, self.min_dyna, self.max_dyna, allow_multiple_arrivals)
                self.static_nodes_coords += static_nodes_coords
                self.dynamic_nodes_coords += dynamic_nodes_coords
                self.times += times

        self.size = len(self.static_nodes_coords)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)
                
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        nodes = self.static_nodes_coords[idx]
        new_nodes = self.dynamic_nodes_coords[idx]
        time = self.times[idx]

        cpu_distance_matrix = self.distance_matrix.cpu()

        item = {
            'nodes': torch.FloatTensor(nodes),
            'new_nodes': torch.FloatTensor(new_nodes),
            'times': torch.LongTensor(np.array(time)),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph_realDTSP(return_distance_matrix(np.array(nodes, dtype=np.float32)[None,...], self.nodes_coords, np.array(cpu_distance_matrix))[0], self.neighbors, self.knn_strat)),
        }
        if self.eval:
            tour_nodes = self.tour_nodes[idx]
            item['tour_nodes'] = torch.LongTensor(tour_nodes)
        return item

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset


class PDTSPRealData(object):
    """Class for the Partially Dynamic Travelling Salesman Problem when we have real data"""

    NAME = 'rdtsp'

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

        # Normalize distance matrix to stop costs from being too massive
        self.distance_matrix -= self.distance_matrix.min(1, keepdim=True)[0]
        self.distance_matrix /= self.distance_matrix.max(1, keepdim=True)[0]

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
        return realDTSPDataset(self.route_name, *args, **kwargs)

    def make_state(self, *args, **kwargs):
        return staterealDTSP.initialize(stop_coords=self.stop_coords, distance_matrix=self.distance_matrix, *args, **kwargs)

class realDTSPDataset(Dataset):

    def __init__(self, route_name, filename=None, min_size=40, max_size=100, min_new=None, max_new=None, batch_size=128, num_samples=128000, offset=0, distribution=None, neighbors=20, knn_strat=None, eval=False):
            
        super(realDTSPDataset, self).__init__()

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.offset = offset
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.eval = eval
        self.filename = filename

        self.nodes_coords = []
        self.static_nodes_coords = []
        self.dynamic_nodes_coords = []
        self.times = []
        self.distance_matrix = []


        print('\nLoading from {}...'.format(route_name))
        for line in tqdm(open('data/routes/' + route_name + '_nodes.txt', "r").readlines(), ascii=True):
            line = line.split(" ")
            self.nodes_coords.append(
                [float(line[0]), float(line[1])]
            )

        for line in tqdm(open('data/routes/' + route_name + '_distance_matrix.txt', "r").readlines(), ascii=True):
            line = line.split(" ")
            self.distance_matrix.append(
                [float(x) for x in line]
            )
        
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
                if self.supervised:
                    # Convert tour nodes to required format
                    # Don't add final connection for tour/cycle
                    tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                    self.tour_nodes.append(tour_nodes)

        else:
            print('\nGenerating {} samples of realDTSP from the {} dataset'.format(
                    num_samples, route_name))
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                static_nodes_coords, dynamic_nodes_coords, times = generate_instance(np.array(self.nodes_coords), batch_size, min_size, max_size)
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

        item = {
            'nodes': torch.FloatTensor(nodes),
            'new_nodes': torch.FloatTensor(new_nodes),
            'times': torch.LongTensor(np.array(time)),
            'graph': ~torch.BoolTensor(nearest_neighbor_graph_realDTSP(return_distance_matrix(np.array(nodes)[None,...], self.nodes_coords, np.array(self.distance_matrix))[0], self.neighbors, self.knn_strat)),
        }
        if self.eval:
            tour_nodes = self.tour_nodes[idx]
            item['tour_nodes'] = torch.LongTensor(tour_nodes)
        return item

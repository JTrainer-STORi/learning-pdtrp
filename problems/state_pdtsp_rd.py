import torch
import numpy as np
from typing import NamedTuple

def return_distances(nodes_a, nodes_b, stop_coords, distance_matrix):

    indices_a = torch.where((nodes_a[:, :, None] == stop_coords).all(-1))[-1].view(nodes_a.shape[0], -1)
    indices_b = torch.where((nodes_b[:, :, None] == stop_coords).all(-1))[-1].view(nodes_b.shape[0], -1)

    return distance_matrix[indices_a, indices_b]

def return_distance_matrix(nodes, stop_coords, distance_matrix):
    """
    Takes a batch of nodes and creates a distance matrix for them which is a submatrix of the distance matrix for every node in the problem

    inputs: 
    nodes: array of nodes to produce the distance matrix for
    stop_coords: The array of all possible stop locations
    distance_matrix: Array containing all possible distances

    outputs:
    distance matrix for nodes, in the order of nodes. i.e. row 1 of distance matrix corresponds to distances from the first node in nodes to all other nodes in nodes.
    """

    indices = np.array([np.where((stop_coords == x[:, None]).all(-1))[1] for x in nodes])

    return np.array([distance_matrix[x, :][:, x] for x in indices])

def nearest_neighbor_graph_realDTSP(distance_matrix, neighbors, knn_strat):

    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = distance_matrix.shape[0]
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(distance_matrix, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W

class statePDTSPReal(NamedTuple):
    """class for keeping track of the PDTSP state when using real instances"""

    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor
    stop_coords: torch.Tensor
    distance_matrices: torch.Tensor

    ids: torch.Tensor # keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor # keeps track of nodes that have been visited
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    graph: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            loc = self.loc[key],
            graph = self.graph[key],
        )

    @staticmethod
    def initialize(loc, graph, stop_coords, distance_matrix, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        cpu_locs = loc.cpu()
        cpu_stops = stop_coords.cpu()
        cpu_dist = distance_matrix.cpu()
        return statePDTSPReal(
            loc=loc,
            stop_coords=stop_coords,
            dist=distance_matrix,
            distance_matrices=torch.tensor(return_distance_matrix(np.array(cpu_locs), np.array(cpu_stops), np.array(cpu_dist)), dtype=torch.float, device=loc.device),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
            ),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            graph=graph,
        )
    
    def update(self, selected):

        prev_a = self.prev_a

        for j, action in enumerate(selected):
            if action is not None:
                # Update the state
                prev_a[j] = selected[j]  # Add dimension for step

        cur_coord = self.loc[self.ids, prev_a]

        # Update should only be called with just 1 parallel step,
        # in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_, cur_coord=cur_coord, i=self.i + 1)
    
    def arrivals(self, new_node_locs):
        # For now, this will just add the new node to the list of nodes and update the mask. Later the graph will need to be updated.
        loc = torch.cat((self.loc, new_node_locs), 1)
        cpu_locs = loc.cpu()
        batch_size, _, _ = loc.size()
        visited_ = torch.cat((self.visited_, torch.zeros((batch_size,1,1), device=self.loc.device, dtype=torch.bool)), 2)
        cpu_stops = self.stop_coords.cpu()
        cpu_dist = self.dist.cpu()
        distance_matrices = return_distance_matrix(np.array(cpu_locs), np.array(cpu_stops), np.array(cpu_dist))
        graph = ~torch.BoolTensor(np.array([nearest_neighbor_graph_realDTSP(distance_matrices[i], neighbors=0.2, knn_strat='percentage') for i in range(len(loc))]))
        return self._replace(visited_=visited_, loc=loc, graph=graph, distance_matrices=torch.Tensor(distance_matrices, dtype=torch.float, device=loc.device))
    
    def all_finished(self):
        return self.visited_.all()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_
    
    def get_graph_mask(self):
        batch_size, n_loc, _ = self.loc.size()
        if self.i.item() == 0:
            return torch.zeros(batch_size, 1, n_loc, dtype=torch.bool, device=self.loc.device)
        else:
            return self.graph.gather(1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_loc))
            
    def get_graph(self):
        return self.graph
    
    def construct_solutions(self, actions):
        return actions
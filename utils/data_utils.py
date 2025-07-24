import os
import pickle
import numpy as np

from torch.utils.data import Sampler
from utils import move_to


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)
    
    
class BatchedRandomSampler(Sampler):
    """Samples elements randomly, while maintaining sequential order within a batch size

    Arguments:
        data_source (Dataset): dataset to sample from
        batch_size (int): batch size 
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        
        assert len(data_source) % batch_size == 0, "Number of samples must be divisible by batch size"

    def __iter__(self):
        batch_nums = np.arange(len(self.data_source) // self.batch_size)
        np.random.shuffle(batch_nums)
        idx = []
        for b_idx in batch_nums:
            idx += [self.batch_size*b_idx + s_idx for s_idx in range(self.batch_size)]
        return iter(idx)

    def __len__(self):
        return len(self.data_source)

def dataset_to_input(dataset, problem, device):
    """Convert a dataset to a model input format"""
    if dataset['all_nodes'].dim() == 2:
        if problem == 'pdtrp':
            input = {
                'loc': move_to(dataset['all_nodes'], device).unsqueeze(0),
                'service_times': move_to(dataset['service_times'], device).unsqueeze(0),
                'arrival_times': move_to(dataset['arrival_times'], device).unsqueeze(0),
                'graph': move_to(dataset['graph'], device).unsqueeze(0),
                'distance_matrix': move_to(dataset['distance_matrix'], device).unsqueeze(0),
                'speed': move_to(dataset['speed'], device),
                'time_horizon': move_to(dataset['time_horizon'], device)
            }
        elif problem == 'pdcvrp':
            input = {
                'loc': move_to(dataset['all_nodes'], device).unsqueeze(0),
                'service_times': move_to(dataset['service_times'], device).unsqueeze(0),
                'arrival_times': move_to(dataset['arrival_times'], device).unsqueeze(0),
                'demand': move_to(dataset['demand'], device).unsqueeze(0),
                'graph': move_to(dataset['graph'], device).unsqueeze(0),
                'distance_matrix': move_to(dataset['distance_matrix'], device).unsqueeze(0),
                'speed': move_to(dataset['speed'], device),
                'time_horizon': move_to(dataset['time_horizon'], device),
                'vehicle_capacity': move_to(dataset['vehicle_capacity'], device)
            }
        elif problem == 'pdtrptw':
            input = {
                'loc': move_to(dataset['all_nodes'], device).unsqueeze(0),
                'service_times': move_to(dataset['service_times'], device).unsqueeze(0),
                'arrival_times': move_to(dataset['arrival_times'], device).unsqueeze(0),
                'graph': move_to(dataset['graph'], device).unsqueeze(0),
                'window_starts': move_to(dataset['window_starts'], device).unsqueeze(0),
                'window_ends': move_to(dataset['window_ends'], device).unsqueeze(0),
                'distance_matrix': move_to(dataset['distance_matrix'], device).unsqueeze(0),
                'speed': move_to(dataset['speed'], device),
                'time_horizon': move_to(dataset['time_horizon'], device), 
                'gamma': move_to(dataset['gamma'], device),
                'theta': move_to(dataset['theta'], device),
            }
        elif problem == 'pdcvrptw':
            input = {
                'loc': move_to(dataset['all_nodes'], device).unsqueeze(0),
                'service_times': move_to(dataset['service_times'], device).unsqueeze(0),
                'arrival_times': move_to(dataset['arrival_times'], device).unsqueeze(0),
                'demand': move_to(dataset['demand'], device).unsqueeze(0),
                'graph': move_to(dataset['graph'], device).unsqueeze(0),
                'window_starts': move_to(dataset['window_starts'], device).unsqueeze(0),
                'window_ends': move_to(dataset['window_ends'], device).unsqueeze(0),
                'distance_matrix': move_to(dataset['distance_matrix'], device).unsqueeze(0),
                'speed': move_to(dataset['speed'], device),
                'time_horizon': move_to(dataset['time_horizon'], device),
                'vehicle_capacity': move_to(dataset['vehicle_capacity'], device),
                'gamma': move_to(dataset['gamma'], device),
                'theta': move_to(dataset['theta'], device),
            }
    else:
        if problem == 'pdtrp':
            input = {
                'loc': move_to(dataset['all_nodes'], device),
                'service_times': move_to(dataset['service_times'], device),
                'arrival_times': move_to(dataset['arrival_times'], device),
                'graph': move_to(dataset['graph'], device),
                'speed': move_to(dataset['speed'], device),
                'distance_matrix': move_to(dataset['distance_matrix'], device),
                'time_horizon': move_to(dataset['time_horizon'], device)
            }
        elif problem == 'pdcvrp':
            input = {
                'loc': move_to(dataset['all_nodes'], device),
                'service_times': move_to(dataset['service_times'], device),
                'arrival_times': move_to(dataset['arrival_times'], device),
                'demand': move_to(dataset['demand'], device),
                'graph': move_to(dataset['graph'], device),
                'distance_matrix': move_to(dataset['distance_matrix'], device),
                'speed': move_to(dataset['speed'], device),
                'time_horizon': move_to(dataset['time_horizon'], device),
                'vehicle_capacity': move_to(dataset['vehicle_capacity'], device)
            }
        elif problem == 'pdtrptw':
            input = {
                'loc': move_to(dataset['all_nodes'], device),
                'service_times': move_to(dataset['service_times'], device),
                'arrival_times': move_to(dataset['arrival_times'], device),
                'graph': move_to(dataset['graph'], device),
                'window_starts': move_to(dataset['window_starts'], device),
                'window_ends': move_to(dataset['window_ends'], device),
                'distance_matrix': move_to(dataset['distance_matrix'], device),
                'speed': move_to(dataset['speed'], device),
                'time_horizon': move_to(dataset['time_horizon'], device), 
                'gamma': move_to(dataset['gamma'], device),
                'theta': move_to(dataset['theta'], device),
            }
        elif problem == 'pdcvrptw':
            input = {
                'loc': move_to(dataset['all_nodes'], device),
                'service_times': move_to(dataset['service_times'], device),
                'arrival_times': move_to(dataset['arrival_times'], device),
                'demand': move_to(dataset['demand'], device),
                'graph': move_to(dataset['graph'], device),
                'window_starts': move_to(dataset['window_starts'], device),
                'window_ends': move_to(dataset['window_ends'], device),
                'distance_matrix': move_to(dataset['distance_matrix'], device),
                'speed': move_to(dataset['speed'], device),
                'time_horizon': move_to(dataset['time_horizon'], device),
                'vehicle_capacity': move_to(dataset['vehicle_capacity'], device),
                'gamma': move_to(dataset['gamma'], device),
                'theta': move_to(dataset['theta'], device),
            }

    return input
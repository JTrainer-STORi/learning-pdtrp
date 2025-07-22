import numpy as np
from scipy.spatial.distance import pdist, squareform

def nearest_neighbor_graph(nodes, neighbors, knn_strat, distance_matrix = None):
    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = len(nodes)
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        if distance_matrix is not None:
            W_val = distance_matrix
        else:
            W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W, W_val

def sample_lognorm(mean, std_dev, size):
    shape = np.sqrt(np.log((std_dev**2 / mean**2) + 1))
    scale = mean / np.exp(shape**2 / 2)
    return np.random.lognormal(mean=np.log(scale), sigma=shape, size=size)

def normalize_coordinates(coords):
    coords = np.array(coords)
    max_x = np.max(coords[:, 0])
    max_y = np.max(coords[:, 1])
    return coords / np.array([max_x, max_y])

def transform_and_subsample_rl_instance(rl_instance, problem, n_locations, time_horizon, speed, latest_end=None):

    # choose n_locations - 1 random indices (always retain depot)
    indices = np.random.choice(range(1, len(rl_instance['coords'])), n_locations - 1, replace=False)
    # create a new instance with the chosen locations

    indices_plus_depot = np.concatenate(([0], indices))  # +1 because depot is at index 0

    speed = 4.0 / 60.0
    time_horizon = 8 * 60
    distance_matrix = rl_instance['duration_matrix'][np.ix_(indices_plus_depot, indices_plus_depot)] * speed / 60.0
    arrival_times = (rl_instance['time_windows'][indices, 0] - np.min(rl_instance['time_windows'][indices, 0])) # shift to start at 0
    arrival_times = arrival_times * time_horizon / np.max(arrival_times)  # normalize to [0, time_horizon]
    arrival_times = np.concatenate(([0], arrival_times))  # depot has arrival time 0
    normalized_locations = normalize_coordinates(rl_instance['coords'][indices_plus_depot])
    service_times = rl_instance['service_times'][indices_plus_depot] / 60.0  # convert to minutes
    if problem == 'pdtrp_tw' or 'pdcvrp_tw':
        time_windows = rl_instance['time_windows'][indices] - np.min(rl_instance['time_windows']) / 60.0  # convert to minutes and shift to start at 0
        time_windows = time_windows * (time_horizon + latest_end) / np.max(time_windows)  # normalize to [0, time_horizon + latest_end]
        return np.round(np.array(normalized_locations), decimals=3), np.round(np.array(service_times), decimals=3), np.round(np.array(arrival_times), decimals=3), np.round(np.array(distance_matrix), decimals=3), np.round(np.array(time_windows), decimals=3)

    return np.round(np.array(normalized_locations), decimals=3), np.round(np.array(service_times), decimals=3), np.round(np.array(arrival_times), decimals=3), np.round(np.array(distance_matrix), decimals=3)


def read_vrplib(filename, rounded=True):
    loc = []
    demand = []
    mode = ''
    capacity = None
    edge_weight_type = None
    edge_weight_format = None
    duration_matrix = []
    service_t = []
    timewi = []
    with open(filename, 'r') as f:
        
        for line in f:
            line = line.strip(' \t\n')
            if line == "":
                continue
            elif line.startswith('CAPACITY'):
                capacity = int(line.split(" : ")[1])
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(" : ")[1]
            elif line.startswith('EDGE_WEIGHT_FORMAT'):
                edge_weight_format = line.split(" : ")[1]
            elif line == 'NODE_COORD_SECTION':
                mode = 'coord'
            elif line == 'DEMAND_SECTION':
                mode = 'demand'
            elif line == 'DEPOT_SECTION':
                mode = 'depot'
            elif line == "EDGE_WEIGHT_SECTION":
                mode = 'edge_weights'
                assert edge_weight_type == "EXPLICIT"
                assert edge_weight_format == "FULL_MATRIX"
            elif line == "TIME_WINDOW_SECTION":
                mode = "time_windows"
            elif line == "SERVICE_TIME_SECTION":
                mode = "service_t"
            elif line == "EOF":
                break
            elif mode == 'coord':
                node, x, y = line.split()  # Split by whitespace or \t, skip duplicate whitespace
                node = int(node)
                x, y = (int(x), int(y)) if rounded else (float(x), float(y))
                
                if node == 1:
                    depot = (x, y)
                else:
                    assert node == len(loc) + 2 # 1 is depot, 2 is 0th location
                    loc.append((x, y))
            elif mode == 'demand':
                node, d = [int(v) for v in line.split()]
                if node == 1:
                    assert d == 0
                demand.append(d)
            elif mode == 'edge_weights':
                duration_matrix.append(list(map(int if rounded else float, line.split())))
            elif mode == 'service_t':
                node, t = line.split()
                node = int(node)
                t = int(t) if rounded else float(t)
                if node == 1:
                    assert t == 0
                assert node == len(service_t) + 1
                service_t.append(t)
            elif mode == 'time_windows':
                node, l, u = line.split()
                node = int(node)
                l, u = (int(l), int(u)) if rounded else (float(l), float(u))
                assert node == len(timewi) + 1
                timewi.append([l, u])

    return {
        'is_depot': np.array([1] + [0] * len(loc), dtype=bool),
        'coords': np.array([depot] + loc),
        'demands': np.array(demand),
        'capacity': capacity,
        'time_windows': np.array(timewi),
        'service_times': np.array(service_t),
        'duration_matrix': np.array(duration_matrix) if len(duration_matrix) > 0 else None
    }
    
def subsample_ortec(instance_file, n_total, time_horizon, speed):
    raw_dict = read_vrplib(instance_file)
    return transform_and_subsample_rl_instance(raw_dict, n_total, time_horizon, speed)

def generate_scaled_demands(batch_size, n_total, vehicle_capacity, min_trips_required, min_demand=0.1):

    # Step 1: Generate raw uniform demands in shape [batch, customers]
    raw_demands = np.random.uniform(
        low=min_demand,
        high=vehicle_capacity,
        size=(batch_size, n_total)
    )

    # Step 2: Random target total demand ∈ [(min_trips - 1) * cap, min_trips * cap]
    lower = (min_trips_required - 1) * vehicle_capacity
    upper = min_trips_required * vehicle_capacity
    target_totals = np.random.uniform(lower, upper, size=(batch_size, 1))  # shape [batch, 1]

    # Step 3: Scale each instance's demand vector to match its target total
    current_totals = raw_demands.sum(axis=1, keepdims=True)  # shape [batch, 1]
    scale_factors = target_totals / current_totals
    scaled_demands = raw_demands * scale_factors

    # Step 4: Clip any demand > vehicle_capacity (optional — can remove to allow spillover)
    scaled_demands = np.clip(scaled_demands, min_demand, vehicle_capacity)

    return scaled_demands
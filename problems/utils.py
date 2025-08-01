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
    if std_dev == 0 and mean == 0:
        return np.zeros(size)
    else:
        shape = np.sqrt(np.log((std_dev**2 / mean**2) + 1))
        scale = mean / np.exp(shape**2 / 2)
        return np.random.lognormal(mean=np.log(scale), sigma=shape, size=size)

def normalize_coordinates(coords):
    coords = np.array(coords)
    max_x = np.max(coords[:, 0])
    max_y = np.max(coords[:, 1])
    return coords / np.array([max_x, max_y])

def transform_and_subsample_rl_instance(rl_instance, problem, n_locations, time_horizon, speed, latest_end=None, vehicle_capacity=None, min_trips_required_lb=None, min_trips_required_ub=None):

    # choose n_locations - 1 random indices (always retain depot)
    indices = np.random.choice(range(1, len(rl_instance['coords'])), n_locations - 1, replace=False)
    # create a new instance with the chosen locations

    indices_plus_depot = np.concatenate(([0], indices))  # +1 because depot is at index 0

    speed = 4.0 / 60.0
    time_horizon = 8 * 60
    unscaled_distance_matrix = rl_instance['duration_matrix']* speed / 60.0
    scaled_distance_matrix = unscaled_distance_matrix / np.max(unscaled_distance_matrix)
    distance_matrix = scaled_distance_matrix[np.ix_(indices_plus_depot, indices_plus_depot)] 
    arrival_times = (rl_instance['time_windows'][indices, 0] - np.min(rl_instance['time_windows'][indices, 0])) # shift to start at 0
    arrival_times = arrival_times * time_horizon / np.max(arrival_times)  # normalize to [0, time_horizon]
    arrival_times = np.concatenate(([0], arrival_times))  # depot has arrival time 0
    normalized_locations = normalize_coordinates(rl_instance['coords'][indices_plus_depot])
    service_times = rl_instance['service_times'][indices_plus_depot] / 60.0  # convert to minutes
    if problem == 'pdtrptw' or problem == 'pdcvrptw':
        time_windows = rl_instance['time_windows'][indices] - np.min(rl_instance['time_windows']) / 60.0  # convert to minutes and shift to start at 0
        time_windows = time_windows * (time_horizon + latest_end) / np.max(time_windows)  # normalize to [0, time_horizon + latest_end]
        time_windows = np.concatenate(([[0, 0]], time_windows), axis=0)  # depot has time window [0, 0]
        window_starts = time_windows[:, 0]
        window_ends = time_windows[:, 1]
    if problem == 'pdcvrp' or problem == 'pdcvrptw':
        raw_demands = rl_instance['demands'][indices]
        demands = generate_scaled_demands(
            batch_size=1,
            n_total=len(raw_demands),
            vehicle_capacity=vehicle_capacity,
            min_trips_required_lb=min_trips_required_lb,
            min_trips_required_ub=min_trips_required_ub,
            raw_demands=raw_demands[None, :]
        )[0]
        # depot has no demand
        demands = np.concatenate(([0], demands))  # depot has demand 0
    
    

    return np.round(np.array(normalized_locations), decimals=3), np.round(np.array(service_times), decimals=3), np.round(np.array(arrival_times), decimals=3), np.round(np.array(distance_matrix), decimals=3), np.round(np.array(window_starts), decimals=3) if problem == 'pdtrptw' or problem == 'pdcvrptw' else None, np.round(np.array(window_ends), decimals=3) if problem == 'pdtrptw' or problem == 'pdcvrptw' else None, np.round(np.array(demands), decimals=3) if problem == 'pdcvrp' or problem == 'pdcvrptw' else None


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
    
def subsample_ortec(instance_file, problem, n_total, time_horizon, speed, latest_end=None, vehicle_capacity=None, min_trips_required_lb=None, min_trips_required_ub=None):
    raw_dict = read_vrplib(instance_file)
    return transform_and_subsample_rl_instance(raw_dict, problem, n_total, time_horizon, speed, latest_end, vehicle_capacity, min_trips_required_lb, min_trips_required_ub)

def generate_scaled_demands(batch_size, n_total, vehicle_capacity, min_trips_required_lb, min_trips_required_ub, raw_demands=None):

    # Step 1: Generate raw uniform demands in shape [batch, customers]
    if raw_demands is None:
        raw_demands = np.random.uniform(
            low=0.1,
            high=vehicle_capacity,
            size=(batch_size, n_total)
        )

    min_trips_required = np.random.randint(min_trips_required_lb, min_trips_required_ub + 1, size=(batch_size, 1))  # shape [batch, 1]

    # Step 2: Random target total demand ∈ [(min_trips - 1) * cap, min_trips * cap]
    lower = (min_trips_required - 1) * vehicle_capacity
    upper = min_trips_required * vehicle_capacity
    target_totals = np.random.uniform(lower, upper, size=(batch_size, 1))  # shape [batch, 1]

    # Step 3: Scale each instance's demand vector to match its target total
    current_totals = raw_demands.sum(axis=1, keepdims=True)  # shape [batch, 1]
    scale_factors = target_totals / current_totals
    scaled_demands = raw_demands * scale_factors

    # Step 4: Clip any demand > vehicle_capacity (optional — can remove to allow spillover), have a minimum demand of 0.001 so that no node will have zero demand
    scaled_demands = np.round(np.clip(scaled_demands, a_min=0.001, a_max=vehicle_capacity), decimals=3)

    return scaled_demands

def fix_arrivals(input):
            # For problems with time windows like this one, we have an extra step in the initialization for rare cases in which the time window starts of the advanced request customers minus the time taken to travel to them is greater than arrival times of some of the immediate request customers.
        # In this case, we do some preprocessing of the instance to make it more realistic. The vehicle is sat in the depot and if some arrivals came during this time then it would consider them as well when it came time to chose an action

        # find the advanced request customers, i.e. those with arrival time 0
        advanced_request_customers = torch.nonzero(arrival_times == 0, as_tuple=False)[1:] # ignore the depot at index 0

        adv_customers_tw_starts = window_starts[advanced_request_customers]
        distances_to_depot = distance_matrix[:, 0, advanced_request_customers] # distance from depot to advanced request customers
        depot_leaving_times = adv_customers_tw_starts - distances_to_depot / speed
        earliest_depot_leaving_time = torch.min(depot_leaving_times, dim=1).values

        # if there is an arrival time of an immediate request customer that is less than the earliest depot leaving time, we need to adjust the instance
        immediate_request_customers = torch.nonzero(arrival_times > 0, as_tuple=False)[1:] # ignore the depot at index 0

        imm_customers_arrival_times = arrival_times[ immediate_request_customers]

        if (imm_customers_arrival_times < earliest_depot_leaving_time).any():
            pass

        return None
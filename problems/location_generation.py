# This script contains functions for generating customer locations in the partially dynamic routing problems.

import numpy as np

def subregion_generation(n_total: int = 40,
                         n_subregions: int = 9,
                         arrival_weights: list = None,
                         time_horizon: float = 480.0
                         ):
    """
    This function generates customer locations according to the subregion method of Larsen, Madsen and Solomon (2004). With a few changes made to make it more computationally efficient.

    inputs:
    n_total: int, the total number of customers to generate
    batch_size: int, the number of customers to generate in each batch
    n_subregions: int, the number of subregions to divide the service area into
    arrival_weights: list, the weights of the subregions. If None, the weights are sampled from a dirichlet distribution

    outputs:
    An array of customer locations

    """

    n_imm = np.zeros(n_subregions)
    
    if arrival_weights is None:
        arrival_weights = np.random.dirichlet(np.ones(n_subregions), size=1)[0]
    else:
        arrival_weights = np.array(arrival_weights)/np.sum(arrival_weights)

    while not np.sum(n_imm) == n_total:
        for j in range(n_subregions):
            exp_n_imm_j = arrival_weights[j] * n_total
            arrival_rate = exp_n_imm_j
            n_imm[j] = np.random.poisson(arrival_rate)
    
    customer_locations = []

    y_ix = -1 # start at -1 so that the first increment is 0
    for j, n_imm_j in enumerate(n_imm):
        # Generate customer locations according to their subregion

        if j % np.sqrt(n_subregions) == 0:
            y_ix += 1
            if y_ix % 2 ==0:
                x_ix = -1
            else:
                x_ix = int(np.sqrt(n_subregions))
        
        if y_ix % 2 == 0:
            x_ix += 1
        else:
            x_ix -= 1

        x_limit_1 = x_ix / np.sqrt(n_subregions)
        x_limit_2 = (x_ix+1) / np.sqrt(n_subregions)
        x_coords = np.round(np.random.uniform(x_limit_1, x_limit_2, int(n_imm_j)), decimals=3)
        
        y_limit_1 = y_ix / np.sqrt(n_subregions)
        y_limit_2 = (y_ix+1) / np.sqrt(n_subregions)
        y_coords = np.round(np.random.uniform(y_limit_1, y_limit_2, int(n_imm_j)), decimals=3)

        customer_locations.append(np.column_stack((x_coords, y_coords)))

    return np.concatenate(customer_locations)

# start with a simple example with 4 subregions
def time_and_subregion_generation(n_total: int = 40,
                         n_subregions: int = 4,
                         arrival_weights: list = None,
                         arrival_skews: list = None,
                         time_horizon: float = 480.0,
                         dimension_scale: float = 1.0
                         ):
    """
    This function is designed to generate customer locations and arrival_times in a manner such that it is possible to have spatially and temportally dependent arrivals. The subregion weight dictates which subregion is more likely to have customers, while the subregion skew dictates how the arrival times are distributed within that subregion.

    inputs:
    n_total: int, the total number of customers to generate
    n_subregions: int, the number of subregions to divide the service area into
    time_horizon: float, the time horizon of the problem in minutes
    subregion_weights: list, the weights of the subregions. If None, the weights are sampled from a dirichlet distribution
    subregion_skews: list, the skews of the subregions. If None, the skews are sampled from a uniform distribution

    outputs:
    A numpy array of customer locations and another array of customer arrival times.
    """

    if arrival_skews is None:
        # if no skews are provided then sample arrival times uniformly for every subregion
        arrival_skews = ['uniform' for _ in range(n_subregions)]
        
    n_imm = np.zeros(n_subregions)
    
    if arrival_weights is None:
        arrival_weights = np.random.dirichlet(np.ones(n_subregions), size=1)[0]
    else:
        arrival_weights = np.array(arrival_weights)/np.sum(arrival_weights)

    while not np.sum(n_imm) == n_total:
        for j in range(n_subregions):
            exp_n_imm_j = arrival_weights[j] * n_total
            arrival_rate = exp_n_imm_j
            n_imm[j] = np.random.poisson(arrival_rate)
    
    customer_locations = []
    arrival_times = []

    y_ix = -1 # start at -1 so that the first increment is 0
    for j, n_imm_j in enumerate(n_imm):
        # Generate customer locations according to their subregion

        if j % np.sqrt(n_subregions) == 0:
            y_ix += 1
            if y_ix % 2 ==0:
                x_ix = -1
            else:
                x_ix = int(np.sqrt(n_subregions))
        
        if y_ix % 2 == 0:
            x_ix += 1
        else:
            x_ix -= 1

        x_limit_1 = x_ix / np.sqrt(n_subregions) * dimension_scale
        x_limit_2 = (x_ix+1) / np.sqrt(n_subregions) * dimension_scale
        x_coords = np.round(np.random.uniform(x_limit_1, x_limit_2, int(n_imm_j)), decimals=3)

        y_limit_1 = y_ix / np.sqrt(n_subregions) * dimension_scale
        y_limit_2 = (y_ix+1) / np.sqrt(n_subregions) * dimension_scale
        y_coords = np.round(np.random.uniform(y_limit_1, y_limit_2, int(n_imm_j)), decimals=3)

        # Generate arrival times according to the skew of the subregion

        if arrival_skews[j] == 'uniform':
            alpha=1
            beta=1
        elif arrival_skews[j] == 'early':
            alpha=2
            beta=5
        elif arrival_skews[j] == 'late':
            alpha=5
            beta=2

        arrival_times_sr = np.round(1 + (time_horizon - 1) * np.random.beta(alpha, beta, int(n_imm_j)), decimals=3)

        arrival_times.append(arrival_times_sr)

        customer_locations.append(np.column_stack((x_coords, y_coords)))

    return np.concatenate(customer_locations), np.concatenate(arrival_times)
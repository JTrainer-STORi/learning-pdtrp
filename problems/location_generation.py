# This script contains functions for generating customer locations in the partially dynamic routing problems.

import numpy as np

def subregion_generation(n_total: int = 40,
                         n_subregions: int = 9,
                         arrival_weights: list = None,
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

    while not np.sum(n_imm) == n_total:
        for j in range(n_subregions):
            exp_n_imm_j = arrival_weights[j] * n_total
            arrival_rate = exp_n_imm_j / 1
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
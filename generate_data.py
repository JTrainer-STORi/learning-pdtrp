import numpy as np
import torch
from tqdm import tqdm

from problems.state_pdtrp import StatePDTRP

# set seeds for reproducibility
np.random.seed(12345)
torch.manual_seed(12345)

N_SAMPLES=1280 
N_TOTAL=100
TIME_HORIZON=8*60
E_DOD=0.5

filename = 'pdtrp100edod.5RR.txt'

all_nodes_list = []
arrival_times_list = []
service_times_list = []
tours_list = []

# Step 1: Generate the expected number of static customers for each instance
exp_n_static = E_DOD * N_TOTAL

for i in tqdm(range(N_SAMPLES), ascii=True):
    # Step 1: Sample the number of dynamic customers for each instance from a poisson distribution
    p_lambda = (N_TOTAL - exp_n_static) / TIME_HORIZON
    n_dyna = np.random.poisson(p_lambda * TIME_HORIZON)

    n_static = N_TOTAL - n_dyna

    # Step 2: Generate the locations of static customers
    loc_static = np.random.uniform(0, 1, size=(n_static, 2))
    # prepend the depot at (0.5, 0.5) to the static customers
    depot = np.array([0.5,0.5])
    loc_static = np.concatenate([depot[None, :], loc_static], axis=0).squeeze()
    # Step 2: Generate the locations of dynamic customers
    loc_dyna = np.random.uniform(0, 1, size=(n_dyna, 2))
    # Step 5: Generate the arrival times for dynamic customers (in minutes)
    arrival_times = np.sort(np.random.uniform(0.1, TIME_HORIZON, size=(n_dyna))).squeeze()
    arrival_times = np.concatenate((np.zeros((n_static + 1)), arrival_times), axis=0)
    # Step 6: Generate the service times for both sets of customers (in minutes)
    service_times = np.random.lognormal(0.8777, 0.6647, size=(N_TOTAL)).squeeze()
    service_times = np.concatenate((np.zeros(( 1)), service_times), axis=0)
    all_nodes = np.concatenate((loc_static, loc_dyna), axis=0)

    tour = [0]
    # Step 7: Solve the problem with a nearest neighbor heuristic
    state = StatePDTRP.initialize(torch.tensor(all_nodes).unsqueeze(0), torch.tensor(service_times).unsqueeze(0), torch.tensor(arrival_times).unsqueeze(0))
    selected = torch.zeros((1, 1), dtype=torch.int64)
    while not state.all_finished():
        visited_mask = state.get_visited()
        arrival_mask = state.get_not_arrived()
        mask = torch.logical_or(visited_mask, arrival_mask)
        distances_to_current = (torch.tensor(all_nodes).unsqueeze(0) - torch.tensor(all_nodes).unsqueeze(0)[0, selected]).norm(p=2, dim=-1)

        distances_to_current[mask[:,0,:]] = float('inf')

        selected = torch.argmin(distances_to_current, dim=1)

        state = state.update(selected)
        tour.append(selected.item())


    all_nodes_list += [all_nodes]
    arrival_times_list += [arrival_times]
    service_times_list += [service_times]
    tours_list += [tour]


with open(filename, "w") as f:
    for i in range(N_SAMPLES):
        f.write( " ".join( str(x)+str(" ")+str(y) for x,y in all_nodes_list[i]))
        f.write( str(" ") + str("arrival_times" + str(" ")))
        f.write( str(" ").join(str(x) for x in arrival_times_list[i]))
        f.write( str(" ") + str("service_times" + str(" ")))
        f.write( str(" ").join(str(x) for x in service_times_list[i]))
        f.write( str(" ") + str('output') + str(" ") )
        f.write( str(" ").join(str(node_idx+1) for node_idx in tours_list[i]) )
        f.write( str(" ") + str(tours_list[i][0]+1) + str(" ") )
        f.write( "\n" )
from problems.pdtrp.problem_pdtrp import PDTRP
from problems.pdtrp_tw.problem_pdtrptw import PDTRPTW
from problems.pdcvrp.problem_pdcvrp import PDCVRP
from problems.pdcvrp_tw.problem_pdcvrptw import PDCVRPTW

from torch.utils.data import DataLoader
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import argparse
import numpy as np
import torch
import sys
from functools import partial
from tqdm import tqdm
import os
import time

def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes

def get_times(solution, routing, manager, service_times=None):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle arrival times and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    times = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route_time = 0
        time = [route_time]
        while not routing.IsEnd(index):
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(prev_index, index, 0)
            if service_times is not None:
                time.append(route_time - service_times[manager.IndexToNode(index)])
            else:
                time.append(route_time)
        times.append(time)
    return times

def resolve_route(input_to_or_tools, partial_route = None, problem_type='PDTRP', solution_time_limit=3, final_run=False, gamma=None):
    """This function solves the problem PDTRP/PDTRPTW/PDCVRP/PDCVRPTW instance using an information dictionary about nodes currently present, the solution will have its first node visits according to the partial_route.
    
    inputs: 
    partial route (list): list of nodes visited so far that must be copied in this solution
    input_to_or_tools (dict): Dictionary containing information about nodes currently in the problem
    problem_type (str): Type of the problem to be solved, PDTRP, PDTRPTW, PDCVRP, PDCVRPTW

    output:
    solution: A solution of the problem
    """
    data = {}
    data['time_matrix'] = input_to_or_tools['time_matrix']
    data['num_vehicles'] = 1
    data['depot'] = 0
    data['service_times'] = input_to_or_tools['service_times']
    data['arrival_times'] = input_to_or_tools['arrival_times']
    if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
        data['time_windows'] = input_to_or_tools['time_windows']
    if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
        data['demands'] = input_to_or_tools['demands']
        data['vehicle_capacity'] = input_to_or_tools['vehicle_capacity']
        data['n_extra_depots'] = input_to_or_tools['n_extra_depots']
    if partial_route is not None:
        data['partial_routes'] = [partial_route]

    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # Create the time callback.

    def travel_time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['time_matrix'][from_node][to_node])
        
    def true_time_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(data['time_matrix'][i][j] + data['service_times'][j])
        

    transit_callback_index = routing.RegisterTransitCallback(travel_time_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def create_demand_evaluator(data):
        """Creates callback to get demands at each location."""
        _demands = data['demands']

        def demand_evaluator(manager, from_node):
            """Returns the demand of the current node"""
            return _demands[manager.IndexToNode(from_node)]

        return demand_evaluator

    def add_capacity_constraints(routing, manager, data, demand_evaluator_index):
        """Adds capacity constraint"""
        vehicle_capacity = data['vehicle_capacity']
        capacity = 'Capacity'
        routing.AddDimension(
            demand_evaluator_index,
            vehicle_capacity,
            vehicle_capacity,
            True,  # start cumul to zero
            capacity)

        # Add Slack for reseting to zero unload depot nodes.
        # e.g. vehicle with load 10/15 arrives at node 1 (depot unload)
        # so we have CumulVar = 10(current load) + -15(unload) + 5(slack) = 0.
        capacity_dimension = routing.GetDimensionOrDie(capacity)
        # Allow to drop reloading nodes with zero cost.
        for node in range(1, data['n_extra_depots'] + 1):
            node_index = manager.NodeToIndex(node)
            routing.AddDisjunction([node_index], 0)

        # Allow to drop regular node with a cost.
        for node in range(data['n_extra_depots'] + 1, len(data['demands'])):
            node_index = manager.NodeToIndex(node)
            capacity_dimension.SlackVar(node_index).SetValue(0)
            routing.AddDisjunction([node_index], 10_000_000)

    if problem_type == 'PDCVRPTW' or problem_type == 'PDCVRP':
        # Add Capacity constraints
        demand_evaluator_index = routing.RegisterUnaryTransitCallback(
        partial(create_demand_evaluator(data), manager))
        add_capacity_constraints(routing, manager, data, demand_evaluator_index)

    
    true_time_callback_index = routing.RegisterTransitCallback(true_time_callback)

    if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
        # Add Time Window constraints

        time = 'Time'
        routing.AddDimension(
            true_time_callback_index,
            sys.maxsize,  # allow waiting time
            sys.maxsize,  # maximum time per vehicle
            True,  # start cumul to zero
            time)
        
        time_dimension = routing.GetDimensionOrDie(time)
        
        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx == data["depot"]:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetMin(time_window[0])
            time_dimension.SetCumulVarSoftUpperBound(index, time_window[1], gamma)
            routing.AddToAssignment(time_dimension.SlackVar(index))
        # Add time window constraints for each vehicle start node.
        depot_idx = data["depot"]
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
            )
            routing.AddToAssignment(time_dimension.SlackVar(index))
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))
    else:
        # add time windows based on arrival times
        time = 'Time'
        routing.AddDimension(
            true_time_callback_index,
            sys.maxsize,  # allow waiting time
            sys.maxsize,  # maximum time per vehicle
            True,  # start cumul to zero
            time)
        
        time_dimension = routing.GetDimensionOrDie(time)

        for location_idx, arrival_time in enumerate(data["arrival_times"]):
            if location_idx == data["depot"]:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(arrival_time, sys.maxsize)
            slack_var = time_dimension.SlackVar(index)
            routing.AddVariableMinimizedByFinalizer(slack_var) 

    WAITING_PENALTY = 1
    time_dimension.SetSlackCostCoefficientForAllVehicles(WAITING_PENALTY) # This will ensure that the vehicle only waits when it absolutely has to
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)  # pylint: disable=no-member
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(solution_time_limit)

    solution = routing.SolveWithParameters(search_parameters)

    route = get_routes(solution, routing, manager)[0]

    if final_run:
        """Prints assignment on console."""
        print(f"Objective: {solution.ObjectiveValue()}")
        # Display dropped nodes.
        dropped_nodes = "Dropped nodes:"
        for node in range(routing.Size()):
            if routing.IsStart(node) or routing.IsEnd(node):
                continue
            if solution.Value(routing.NextVar(node)) == node:
                dropped_nodes += f" {manager.IndexToNode(node)}"
        print(dropped_nodes)

    return route

parser = argparse.ArgumentParser(description='Solve a problem using OR-Tools')

parser.add_argument('--problem', type=str, help="Type of the problem to be solved, 'PDTRP', 'PDTRPTW', 'PCVRP' or 'PDCVRPTW", required=True)

parser.add_argument('--num_samples', type=int, default=128, help="Number of samples to be solved")

parser.add_argument('--min_total', type=int, default=40, 
                        help="The minimum number of total nodes")
parser.add_argument('--max_total', type=int, default=100, 
                    help="The maximum number of static nodes")
parser.add_argument('--min_dod', type=float, default=0.2,
                    help='The minimum ratio of dynamic nodes')
parser.add_argument('--max_dod', type=float, default=0.8,
                    help='The maximum ratio of dynamic nodes')
parser.add_argument('--speed', type=float, default=4.0,
                    help='The speed of the vehicle in units/h')
parser.add_argument('--time_horizon', type=int, default=8,
                    help='Time horizon for dynamic node arrivals in hours, default is 8 hours (480 minutes)')
parser.add_argument('--arrival_weights', nargs='+', type=int, default=None,
                    help='The weights for the subregions, if None, subregion weights are sampled from a dirichlet distribution. Pass as a space-separated list, e.g. "1 2 3"')
parser.add_argument('--arrival_skews', nargs='+', type=str, default=None,
                    help='The skews for the arrival times distribution in each subregion, if None, uniform distribution is used. Pass as a space-separated list, e.g. "uniform early late"')
parser.add_argument('--use_ortec', type=str, default=None,
                    help='filename of ortec instance to subsample from when generating customer locations. If None, no subsampling is done.')
parser.add_argument('--n_subregions', type=int, default=9,
                        help='Number of subregions to generate the customers in. Passing 1 will generate customers uniformly in the unit square.')
parser.add_argument('--filename', type=str, default=None,
                    help='The filename to save the data to')
parser.add_argument('--batch_size', type=int, default=16,
                    help='The batch size for the problem')
parser.add_argument('--seed', type=int, default=41394125,
                    help='The seed for the problem')
parser.add_argument('--scenario', type=str, default=None,
                    help='Scenario for arrival weights and time windows')
parser.add_argument('--read_data_filepath', type=str, default=None,
                    help='Filepath to read data from in real data scenario')
parser.add_argument('--gamma', type=int, default=1,
                    help='How much to penalize lateness in time windows')
parser.add_argument('--theta', type=float, default=1.0, 
                    help='weighting for distance in costs for problems with time windows')
parser.add_argument('--latest_end', type=int, default=2,
                        help='number of hours after the time horizon that a customer\'s time window can end, default is 2 hours (120 minutes)')
parser.add_argument('--reaction_time', type=int, default=60,
                        help="Reaction time in minutes, the reaction time is the minimum amount of time that must pass between a customer arriving and their time window starting. ")
parser.add_argument('--vehicle_capacity', type=float, default=1.0,
                    help='Vehicle capacity, default is 1.0 (1 unit of demand)')
parser.add_argument('--min_trips_required', type=int, default=3,
                    help='Minimum number of trips required to service all customers, default is 3')
parser.add_argument('--min_time_window', type=int, default=60,
                        help='Minimum time window length in minutes')
parser.add_argument('--max_time_window', type=int, default=100,
                        help='Maximum time window length in minutes')
parser.add_argument('--stmean', type=float, default=3,
                    help='mean of service times')
parser.add_argument('--stvar', type=float, default=5,
                    help='variance of service times')
parser.add_argument('--solution_time_limit', type=int, default=3,
                    help='Time limit for solving each instance in seconds')

args = parser.parse_args()

if args.arrival_weights is not None:
    assert len(args.arrival_weights) == args.n_subregions, "The number of arrival weights must match the number of subregions"
if args.arrival_skews is not None:
    assert len(args.arrival_skews) == args.n_subregions, "The number of arrival skews must match the number of subregions"

print("Parameters for the problem:")
print(f"Problem type: {args.problem}")
print(f"Number of samples: {args.num_samples}")
print(f"Batch size: {args.batch_size}")
print(f"Filename: {args.filename}")
print(f"Min total nodes: {args.min_total}")
print(f"Max total nodes: {args.max_total}")
print(f"Min dod: {args.min_dod}")
print(f"Max dod: {args.max_dod}")
print(f"solution time limit: {args.solution_time_limit} seconds")

# Set the parameters for the problem here

problem_type = args.problem
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
filename = args.filename
SPEED = args.speed / 60.0

# set seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if problem_type == 'PDTRP':
    problem = PDTRP()
    dataset = problem.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, speed=args.speed, time_horizon=args.time_horizon, n_subregions=args.n_subregions, arrival_weights=args.arrival_weights, arrival_skews=args.arrival_skews, use_ortec=args.use_ortec, service_times_mean=args.stmean, service_times_var=args.stvar)
elif problem_type == 'PDCVRP':
    problem = PDCVRP()
    dataset = problem.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, speed=args.speed, time_horizon=args.time_horizon, n_subregions=args.n_subregions, arrival_weights=args.arrival_weights, arrival_skews=args.arrival_skews, use_ortec=args.use_ortec, vehicle_capacity=args.vehicle_capacity, min_trips_required=args.min_trips_required, service_times_mean=args.stmean, service_times_var=args.stvar)
elif problem_type == 'PDTRPTW':
    problem = PDTRPTW()
    dataset = problem.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, speed=args.speed, min_time_window=args.min_time_window, max_time_window=args.max_time_window, time_horizon=args.time_horizon, n_subregions=args.n_subregions, arrival_weights=args.arrival_weights, arrival_skews=args.arrival_skews, use_ortec=args.use_ortec, gamma=args.gamma, theta=args.theta, latest_end=args.latest_end, reaction_time=args.reaction_time, service_times_mean=args.stmean, service_times_var=args.stvar)
elif problem_type == 'PDCVRPTW':
    problem = PDCVRPTW()
    dataset = problem.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, speed=args.speed, min_time_window=args.min_time_window, max_time_window=args.max_time_window, time_horizon=args.time_horizon, n_subregions=args.n_subregions, arrival_weights=args.arrival_weights, arrival_skews=args.arrival_skews, use_ortec=args.use_ortec, gamma=args.gamma, theta=args.theta, latest_end=args.latest_end, reaction_time=args.reaction_time, vehicle_capacity=args.vehicle_capacity, min_trips_required=args.min_trips_required, service_times_mean=args.stmean, service_times_var=args.stvar)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_nodes_list = []
arrival_times_list = []
service_times_list = []
tours_list = []
visit_times_list = []
if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
    window_starts_list = []
    window_ends_list = []
if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
    demands_list = []
if args.use_ortec is not None:
    distance_matrix_list = []

run_times = []

for input in tqdm(dataloader, ascii=True):

    batch_routes = []
    batch_times = []

    for j in range(BATCH_SIZE):

        all_nodes = input['all_nodes'][j]
        n_original_nodes = len(all_nodes)
        arrival_times = input['arrival_times'][j]
        service_times = input['service_times'][j]
        distance_matrix = input['distance_matrix'][j]
        
        if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
            window_starts = input['window_starts'][j]
            window_ends = input['window_ends'][j]
            time_windows = torch.stack([window_starts, window_ends], dim=1)
            window_starts_list += [np.asarray(input['window_starts'][j])]
            window_ends_list += [np.asarray(input['window_ends'][j])]
        if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
            demands = input['demand'][j]
            vehicle_capacity = 1.0
            demands_list += [np.asarray(demands)]
            # if we are solving a capacitated problem, we need to add dummy depots to allow for restocking the vehicle
            #n_extra_depots = torch.ceil(torch.sum(demands)/ vehicle_capacity).to(torch.int32).item() - 1
            n_extra_depots = args.min_trips_required - 1 # we need at least min_trips_required depots to ensure that the vehicle can service all customers (original brings total number of depots to 1 + n_extra_depots)
            # add n_extra depots to data
            all_nodes = torch.cat([all_nodes[0].repeat(n_extra_depots, 1), all_nodes], dim=0)
            demands = torch.cat([torch.zeros(1), -torch.tensor(vehicle_capacity).repeat(n_extra_depots), demands[1:]], dim=0)

            arrival_times = torch.cat([torch.tensor(0).repeat(n_extra_depots), arrival_times], dim=0)
            service_times = torch.cat([torch.tensor(0).repeat(n_extra_depots), service_times], dim=0)
            # need to modify distance matrix to account for the new depots
            temp_distance_matrix = np.zeros((len(all_nodes), len(all_nodes)))
            for i in range(n_extra_depots):
                temp_distance_matrix[i, n_extra_depots:] = np.asarray(distance_matrix[0, :])
                temp_distance_matrix[n_extra_depots:, i] = np.asarray(distance_matrix[:, 0])
            temp_distance_matrix[n_extra_depots:, n_extra_depots:] = np.asarray(distance_matrix)
            distance_matrix = torch.tensor(temp_distance_matrix)

            if problem_type == 'PDCVRPTW':
                window_starts = torch.cat([window_starts[0].repeat(n_extra_depots), window_starts], dim=0)
                window_ends = torch.cat([torch.zeros(1), torch.tensor(100).repeat(n_extra_depots), window_ends[1:]], dim=0)

                time_windows = torch.stack([window_starts, window_ends], dim=1)

        all_nodes_list += [np.asarray(input['all_nodes'][j])]
        arrival_times_list += [np.asarray(input['arrival_times'][j])]
        service_times_list += [np.asarray(input['service_times'][j])]
        if args.use_ortec is not None:
            distance_matrix_list += [np.asarray(input['distance_matrix'][j])]

        start_time = time.perf_counter()

        input_to_or_tools = {}
        input_to_get_times = {}

        input_to_get_times['speed'] = SPEED
        input_to_get_times['arrival_times'] = arrival_times.unsqueeze(0)

        input_to_or_tools['time_matrix'] = torch.round((distance_matrix/SPEED)*1e4).to(torch.int32).tolist()
        input_to_or_tools['service_times'] = (service_times*1e4).to(torch.int32).tolist()
        input_to_or_tools['arrival_times'] = (arrival_times*1e4).to(torch.int32).tolist()
        input_to_get_times['distance_matrix'] = distance_matrix.unsqueeze(0)
        input_to_get_times['service_times'] = service_times.unsqueeze(0)
        if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
            input_to_or_tools['time_windows'] = (time_windows*1e4).to(torch.int32).tolist()
            input_to_get_times['window_starts'] = window_starts.unsqueeze(0)
        if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
            input_to_or_tools['demands'] = (demands*1e3).to(torch.int32).tolist()
            input_to_or_tools['vehicle_capacity'] = int(vehicle_capacity*1e3)
            new_time_matrix = np.array(input_to_or_tools['time_matrix'])
            new_time_matrix[0:n_extra_depots+1, 0:n_extra_depots+1] = 1_000_000 
            input_to_or_tools['time_matrix'] = new_time_matrix.tolist()
            input_to_or_tools['n_extra_depots'] = n_extra_depots 

        route = resolve_route(input_to_or_tools, None, problem_type, final_run=True, gamma=args.gamma, solution_time_limit=args.solution_time_limit)

        route_wo_final_depot_return = route[:-1]  # remove the last depot return

        visit_times, _ = problem.get_times(input_to_get_times, torch.tensor(route_wo_final_depot_return).unsqueeze(0))

        visit_times = visit_times.squeeze(0)

        # for CVRP problems, we need to remove dummy depots from the output and also pad solutions to account for the fact that different solutions may have different lengths

        if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
            route = np.maximum(np.zeros_like(route), np.array(route) - n_extra_depots).tolist()

        # new method of assessing the route's validity is to sort the route, strip all zeros from the beginning and check that it is equal to arange from 1 to the number of nodes
        sorted_route = sorted(route)
        stripped_sorted_route = [x for x in sorted_route if x != 0]

        assert stripped_sorted_route == list(range(1, n_original_nodes)), f"Stripped sorted route {stripped_sorted_route} does not match expected range from 1 to {n_original_nodes}" 

        batch_routes += [route]

        batch_times += [visit_times.tolist()]

        finish_time = time.perf_counter()
        run_times.append(finish_time - start_time)

    max_len = max(len(route) for route in batch_routes)
    padded_batch_routes = [route + [0]*(max_len - len(route)) for route in batch_routes]
    padded_batch_times = [times + [times[-1]]*(max_len - len(times)) for times in batch_times]

    tours_list += padded_batch_routes
    visit_times_list += padded_batch_times

# for each run, we need to save the information to file so that it can be used to validate the model

os.makedirs("runtime_stats/", exist_ok=True)

with open("runtime_stats/" + filename + "_run_times.txt", "w") as f:
    f.write("Run times for each sample:\n")
    for i, run_time in enumerate(run_times):
        f.write(f"Sample {i+1}: {run_time:.4f} seconds\n")
    f.write(f"Average run time: {np.mean(run_times):.4f} seconds\n")
    f.write(f"Std dev of run times: {np.std(run_times):.4f} seconds\n")

os.makedirs("new_data/" + args.problem.lower(), exist_ok=True)

with open("new_data/" + args.problem.lower() + "/" + filename, "w") as f:
    for i in range(NUM_SAMPLES):
        f.write( " ".join( str(x)+str(" ")+str(y) for x,y in all_nodes_list[i]))
        f.write( str(" ") + str("arrival_times" + str(" ")))
        f.write( str(" ").join(str(x) for x in arrival_times_list[i]))
        f.write( str(" ") + str("service_times" + str(" ")))
        f.write( str(" ").join(str(x) for x in service_times_list[i]))
        if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
            f.write( str(" ") + str("window_starts" + str(" ")))
            f.write( str(" ").join(str(x) for x in window_starts_list[i]))
            f.write( str(" ") + str("window_ends" + str(" ")))
            f.write( str(" ").join(str(x) for x in window_ends_list[i]))
        if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
            f.write( str(" ") + str("demands" + str(" ")))
            f.write( str(" ").join(str(x) for x in demands_list[i]))
        f.write( str(" ") + str("tour") + str(" ") )
        f.write( str(" ").join(str(node_idx+1) for node_idx in tours_list[i]) )
        f.write( str(" ") + str("visit_times") + str(" ") )
        f.write( str(" ").join(str(x) for x in visit_times_list[i]))
        if args.use_ortec is not None:
            f.write( str(" ") + str("distance_matrix" + str(" ")))
            f.write( str(" ").join(str(x) for x in distance_matrix_list[i].flatten()))

        f.write( "\n" )
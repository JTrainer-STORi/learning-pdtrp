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

def resolve_route(current_input, partial_route = None, problem_type='PDTRP', solution_time_limit=3, visit_times=False):
    """This function solves the problem PDTRP/PDTRPTW/PDCVRP/PDCVRPTW instance using an information dictionary about nodes currently present, the solution will have its first node visits according to the partial_route.
    
    inputs: 
    partial route (list): list of nodes visited so far that must be copied in this solution
    current_input (dict): Dictionary containing information about nodes currently in the problem
    problem_type (str): Type of the problem to be solved, PDTRP, PDTRPTW, PDCVRP, PDCVRPTW

    output:
    solution: A solution of the problem
    """
    data = {}
    data['time_matrix'] = current_input['time_matrix']
    data['num_vehicles'] = 1
    data['depot'] = 0
    data['service_times'] = current_input['service_times']
    if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
        data['time_windows'] = current_input['time_windows']
    if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
        data['demands'] = current_input['demands']
        data['vehicle_capacity'] = current_input['vehicle_capacity']
        data['n_extra_depots'] = current_input['n_extra_depots']
    if partial_route is not None:
        data['partial_routes'] = [partial_route]

    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # Create the time callback.

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node] + data['service_times'][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

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
            routing.AddDisjunction([node_index], 1_000_000)

    if problem_type == 'PDCVRPTW' or problem_type == 'PDCVRP':
        # Add Capacity constraints
        demand_evaluator_index = routing.RegisterUnaryTransitCallback(
        partial(create_demand_evaluator(data), manager))
        add_capacity_constraints(routing, manager, data, demand_evaluator_index)

    if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
        # Add Time Window constraints

        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
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
            time_dimension.SetCumulVarSoftUpperBound(index, time_window[1], 10)
        # Add time window constraints for each vehicle start node.
        depot_idx = data["depot"]
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
            )
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  # pylint: disable=no-member
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(solution_time_limit)

    if partial_route is not None:
        routing.CloseModel()
        partial_route_list = data['partial_routes']
        routing.ApplyLocksToAllVehicles(partial_route_list, False)

    solution = routing.SolveWithParameters(search_parameters)

    route = get_routes(solution, routing, manager)[0]
    if visit_times:
        times = get_times(solution, routing, manager, data['service_times'])[0]
    else:
        times = get_times(solution, routing, manager)[0]

    return route, times

parser = argparse.ArgumentParser(description='Solve a problem using OR-Tools')

parser.add_argument('--problem', type=str, help="Type of the problem to be solved, 'PDTRP', 'PDTRP-TW', 'PCVRP' or 'PDCVRP-TW", required=True)

parser.add_argument('--num_samples', type=int, default=128, help="Number of samples to be solved")

parser.add_argument('--min_total', type=int, default=40, 
                        help="The minimum number of total nodes")
parser.add_argument('--max_total', type=int, default=100, 
                    help="The maximum number of static nodes")
parser.add_argument('--min_dod', type=float, default=0.2,
                    help='The minimum ratio of dynamic nodes')
parser.add_argument('--max_dod', type=float, default=0.8,
                    help='The maximum ratio of dynamic nodes')
parser.add_argument('--filename', type=str, default=None,
                    help='The filename to save the data from')
parser.add_argument('--batch_size', type=int, default=16,
                    help='The batch size for the problem')
parser.add_argument('--seed', type=int, default=41394125,
                    help='The seed for the problem')
parser.add_argument('--scenario', type=str, default=None,
                    help='Scenario for arrival weights and time windows')

# import debugpy
# debugpy.listen(5678)  # 5678 is port
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

args = parser.parse_args()

# Set the parameters for the problem here

problem_type = args.problem
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
filename = args.filename

# set seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

arrival_weights = None

if args.scenario is not None:
    if args.scenario == 'N1':
        arrival_weights = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    elif args.scenario == 'N2':
        arrival_weights = [5, 1, 1, 10, 6, 3, 1, 15, 12]

if problem_type == 'PDTRP':
    dataset = PDTRP.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, arrival_weights=arrival_weights)
elif problem_type == 'PDCVRP':
    dataset = PDCVRP.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, arrival_weights=arrival_weights)
elif problem_type == 'PDTRPTW':
    dataset = PDTRPTW.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, arrival_weights=arrival_weights)
elif problem_type == 'PDCVRPTW':
    dataset = PDCVRPTW.make_dataset(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, min_total=args.min_total, max_total=args.max_total, min_dod=args.min_dod, max_dod=args.max_dod, arrival_weights=arrival_weights)

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

for input in tqdm(dataloader, ascii=True):

    batch_routes = []
    batch_times = []

    for j in range(BATCH_SIZE):

        all_nodes = input['all_nodes'][j]
        arrival_times = input['arrival_times'][j]*1e4
        service_times = input['service_times'][j]
        
        if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
            window_starts = input['window_starts'][j]
            window_ends = input['window_ends'][j]
            time_windows = torch.stack([window_starts, window_ends], dim=1)
            window_starts_list += [np.array(input['window_starts'][j])]
            window_ends_list += [np.array(input['window_ends'][j])]
        if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
            demands = input['demand'][j]
            vehicle_capacity = 1.0
            demands_list += [np.array(demands)]
            # if we are solving a capacitated problem, we need to add dummy depots to allow for restocking the vehicle
            n_extra_depots = torch.ceil(torch.sum(demands)/ vehicle_capacity).to(torch.int32).item() - 1
            # add n_extra depots to data
            all_nodes = torch.cat([all_nodes[0].repeat(n_extra_depots, 1), all_nodes], dim=0)
            demands = torch.cat([torch.zeros(1), -torch.tensor(vehicle_capacity).repeat(n_extra_depots), demands[1:]], dim=0)
            arrival_times = torch.cat([torch.tensor(0).repeat(n_extra_depots), arrival_times], dim=0)
            service_times = torch.cat([torch.tensor(0).repeat(n_extra_depots), service_times], dim=0)
            if problem_type == 'PDCVRPTW':
                window_starts = torch.cat([window_starts[0].repeat(n_extra_depots), window_starts], dim=0)
                window_ends = torch.cat([torch.zeros(1), torch.tensor(100).repeat(n_extra_depots), window_ends[1:]], dim=0)

                time_windows = torch.stack([window_starts, window_ends], dim=1)

        all_nodes_list += [np.array(input['all_nodes'][j])]
        arrival_times_list += [np.array(input['arrival_times'][j])]
        service_times_list += [np.array(input['service_times'][j])]

        mask = arrival_times == 0

        partial_route = None

        total_customers = len(all_nodes[mask])

        total_time_advanced_by = 0
        if problem_type == 'PDTRP' or problem_type == 'PDCVRP':
            time_advances = {}

        done = False
        
        while not done:

            current_input = {}

            current_input['time_matrix'] = torch.round((torch.cdist(all_nodes[mask], all_nodes[mask], p=2)/dataset.vehicle_speed)*1e4).to(torch.int32).tolist()
            current_input['service_times'] = (service_times[mask]*1e4).to(torch.int32).tolist()
            if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
                current_input['time_windows'] = (time_windows[mask]*1e4).to(torch.int32).tolist()
            if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
                current_input['demands'] = (demands[mask]*1e3).to(torch.int32).tolist()
                current_input['vehicle_capacity'] = int((vehicle_capacity*1e3))
                new_time_matrix = np.array(current_input['time_matrix'])
                new_time_matrix[0:n_extra_depots+1, 0:n_extra_depots+1] = 1_000_000 
                current_input['time_matrix'] = new_time_matrix.tolist()
                current_input['n_extra_depots'] = n_extra_depots  

            route, times = resolve_route(current_input, partial_route, problem_type)

            n_arrivals = 0

            for i, time in enumerate(times[1:]):
                n_arrivals = torch.sum(arrival_times[total_customers:] < (time + total_time_advanced_by)).item()
                if n_arrivals > 0:
                    last_idx = min(i+2, len(times[1:]))
                    partial_route = route[1:last_idx]
                    break

            if n_arrivals == 0: 
                time_advanced_by = arrival_times[total_customers] - (times[-1] + total_time_advanced_by - current_input['service_times'][route[-2]])
                total_time_advanced_by += time_advanced_by
                n_arrivals = torch.sum(arrival_times[total_customers:] <= (times[-1] + total_time_advanced_by)).item()
                partial_route = route[1:-1]
                time_advances[total_customers] = time_advanced_by.item()

            mask[:total_customers+n_arrivals] = True

            total_customers += n_arrivals

            # if the sum of new arrivals is greater than the remaining vehicle capacity and there are no more dummy depots to visit, we need to add a new dummy depot
            if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
                route_demands = [current_input['demands'][i] for i in partial_route]
                # current_capacity is the sum of the demands since the last depot visit, indicated by a negative demand
                depot_visits = [i for i, x in enumerate(route_demands) if x < 0]
                if len(depot_visits) > 0:
                    last_depot_visit = np.max(depot_visits)
                    current_capacity = current_input['vehicle_capacity'] - sum(route_demands[last_depot_visit+1:])
                    unvisited_customers = [i for i in range(1, len(all_nodes)) if i not in partial_route]
                    remaining_demand = sum([(demands*1e3).to(torch.int32).tolist()[i] for i in unvisited_customers])
                    if current_capacity < remaining_demand:
                        # in this case we need to add a new depot
                        n_extra_depots += 1
                        all_nodes = torch.cat([all_nodes[0].repeat(1,1), all_nodes], dim=0)
                        demands = torch.cat([-torch.tensor(vehicle_capacity).repeat(1), demands], dim=0)
                        arrival_times = torch.cat([torch.tensor(0).repeat(1), arrival_times], dim=0)
                        service_times = torch.cat([torch.tensor(0).repeat(1), service_times], dim=0)
                        mask = torch.cat([torch.tensor(True).repeat(1), mask], dim=0)
                        if problem_type == 'PDCVRPTW':
                            window_starts = torch.cat([window_starts[0].repeat(1), window_starts], dim=0)
                            window_ends = torch.cat([torch.zeros(1), torch.tensor(100).repeat(1), window_ends[1:]], dim=0)

                            time_windows = torch.stack([window_starts, window_ends], dim=1)
                        # add 1 to every node index in the partial route to account for the new depot being added at index 0
                        partial_route = [x + 1 for x in partial_route]

            done = mask.all()

        current_input = {}

        current_input['time_matrix'] = torch.round((torch.cdist(all_nodes[mask], all_nodes[mask], p=2)/dataset.vehicle_speed)*1e4).to(torch.int32).tolist()
        current_input['service_times'] = (service_times[mask]*1e4).to(torch.int32).tolist()
        if problem_type == 'PDTRPTW' or problem_type == 'PDCVRPTW':
            current_input['time_windows'] = (time_windows[mask]*1e4).to(torch.int32).tolist()
        if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
            current_input['demands'] = (demands[mask]*1e3).to(torch.int32).tolist()
            current_input['vehicle_capacity'] = int(vehicle_capacity*1e3)
            new_time_matrix = np.array(current_input['time_matrix'])
            new_time_matrix[0:n_extra_depots+1, 0:n_extra_depots+1] = 1_000_000 
            current_input['time_matrix'] = new_time_matrix.tolist()
            current_input['n_extra_depots'] = n_extra_depots 

        route, times = resolve_route(current_input, partial_route, problem_type, visit_times=True)

        # this won't work for the problems with time windows as these will naturally advance some of the times
        if problem_type == 'PDTRP' or problem_type == 'PDCVRP':
            if bool(time_advances):
                for key, value in time_advances.items():
                    times = [x + value if i >= key else x for i, x in enumerate(times)]

        # for CVRP problems, we need to remove dummy depots from the output and also pad solutions to account for the fact that different solutions may have different lengths

        if problem_type == 'PDCVRP' or problem_type == 'PDCVRPTW':
            route = np.maximum(np.zeros_like(route), np.array(route) - n_extra_depots).tolist()


        batch_routes += [route]
        # if problem_type == 'PDCVRPTW':
        #     # remove extra visits to dummy depots from visit times
        #     new_times = np.insert(np.array(times)[np.array(route) != 0], 0, times[0])
        #     batch_times += [[np.round(x * 1e-4, decimals=4) for x in new_times]]
        # else:
        batch_times += [[np.round(x * 1e-4, decimals=4) for x in times[:-1]]]

    max_len = max(len(route) for route in batch_routes)
    padded_batch_routes = [route + [0]*(max_len - len(route)) for route in batch_routes]
    padded_batch_times = [times + [-1]*(max_len - len(times)) for times in batch_times]

    tours_list += padded_batch_routes
    visit_times_list += padded_batch_times

# for each run, we need to save the information to file so that it can be used to validate the model

with open("data/" + args.problem.lower() + "/" + filename, "w") as f:
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

        f.write( "\n" )
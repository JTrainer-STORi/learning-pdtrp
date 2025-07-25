from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib import cm
from matplotlib.patches import Rectangle
import numpy as np

import torch

plt.rcParams['figure.constrained_layout.use'] = True

def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
               reinforce_loss, bl_loss, tb_logger, wandb_run, opts, info):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('\nepoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('actor_loss: {}, nll: {}'.format(reinforce_loss.item(), -log_likelihood.mean().item()))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        for key, value in info.items():
            tb_logger.log_value(key, value.mean().item(), step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)     
    # Log values to wandb
    if not opts.no_wandb:
        wandb_run.log({
            'avg_cost': avg_cost,
            'actor_loss': reinforce_loss.item(),
            'nll': -log_likelihood.mean().item(),
            'grad_norm': grad_norms[0],
            'grad_norm_clipped': grad_norms_clipped[0],
        }, step=step)

        for key, value in info.items():
            wandb_run.log({key: value.mean().item()}, step=step)

        if opts.baseline == 'critic':
            wandb_run.log({
                'critic/critic_loss': bl_loss.item(),
                'critic/critic_grad_norm': grad_norms[1],
                'critic/critic_grad_norm_clipped': grad_norms_clipped[1],
            }, step=step)
            
def update_anim(problem, data, fig, ax, pi=None, visit_times=None):
    """
    Function for updating the animation in PDTRP.
    This function should be implemented to update the animation based on the current state of the PDTRP problem.
    """
    if pi is None:
        pi = data['tour_nodes']
    if visit_times is None:
        visit_times = data['visit_times']

    pi = pi.squeeze().cpu()
    visit_times = visit_times.squeeze().cpu()
    arrival_times = data['arrival_times'].squeeze().cpu()
    coords = data['loc'].squeeze().cpu()
    visited_mask = torch.zeros_like(arrival_times, dtype=torch.bool)


    # add a return to the depot at the end of the tour
    pi = torch.cat((pi, torch.tensor([0], device=pi.device)), dim=0)

    if problem == 'pdtrp':

        def update(i):
            current_time = visit_times[i]
            current_tour = pi[:i+1]
            #keep track of which nodes have been visited
            visited_mask[current_tour] = True

            num_customers = len(arrival_times) - 1

            available_mask = arrival_times[1:] <= current_time
            unavailable_mask = ~available_mask
            available_customers = coords[1:][available_mask]
            unavailable_customers = coords[1:][unavailable_mask]

            ax.clear()

            # Plot depot
            depot = ax.scatter(*coords[0], c='green', marker='s', label='Depot')

            # Plot available and unavailable customers
            artists = [depot]
            if len(available_customers) > 0:
                avail_visited_mask = visited_mask[1:][available_mask]
                visited_customers = available_customers[avail_visited_mask]
                available_customers = available_customers[~avail_visited_mask]
                if len(visited_customers) > 0:
                    artists.append(ax.scatter(visited_customers[:, 0], visited_customers[:, 1], c='green', label='Visited'))
                if len(available_customers) > 0:
                    artists.append(ax.scatter(available_customers[:, 0], available_customers[:, 1], c='blue', label='Available'))
            if len(unavailable_customers) > 0:
                unavail = ax.scatter(unavailable_customers[:, 0], unavailable_customers[:, 1], c='gray', label='Not Yet Arrived')
                artists.append(unavail)

            # Customer labels
            for j in range(1, num_customers + 1):
                x, y = coords[j]
                txt = ax.text(x, y + 0.02, str(j), fontsize=8, ha='center')
                artists.append(txt)

            # Tour lines
            lines = []
            length = 0.0
            for j in range(1, len(current_tour)):
                a = coords[current_tour[j - 1]]
                b = coords[current_tour[j]]
                lines.append([a, b])
                length += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

            if lines:
                lc = mc.LineCollection(lines, colors='red', linewidths=2)
                ax.add_collection(lc)
                artists.append(lc)

            ax.set_title(f"Step {i} | Time Elapsed {current_time:.1f} Minutes | Distance Traveled {length:.2f} Units")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

            # Adjust the figure to make space for the legend
            fig.subplots_adjust(right=0.75)

            return artists  # This is key!
        
    elif problem=='pdtrptw':

        window_starts = data['window_starts'].squeeze().cpu()
        window_ends = data['window_ends'].squeeze().cpu()

        def update(i):
            current_time = visit_times[i]
            current_tour = pi[:i+1]

            visited_mask[current_tour] = True
            not_visited_mask = ~visited_mask
            num_customers = len(arrival_times) - 1

            available_mask = arrival_times[1:] <= current_time
            unavailable_mask = ~available_mask
            available_customers = coords[1:][available_mask]
            unavailable_customers = coords[1:][unavailable_mask]

            visit_times_by_index = torch.full_like(window_starts[1:], float('nan'))

            visit_times_by_index[pi[1:-1] - 1] = visit_times[1:-1]

            on_time_visits = (window_starts[1:] <= visit_times_by_index) & (visit_times_by_index <= window_ends[1:])

            lateness = torch.maximum(torch.zeros_like(visit_times_by_index), visit_times_by_index - window_ends[1:])

            late_visits = visit_times_by_index > window_ends[1:]

            ax.clear()

            # Plot depot
            depot = ax.scatter(*coords[0], c='green', marker='s', label='Depot')

            # Plot available and unavailable customers
            artists = [depot]
            if len(available_customers) > 0:
                # deal with nodes wich have been visited first
                visited_on_time = coords[1:][available_mask & visited_mask[1:] & on_time_visits]
                visited_late = coords[1:][available_mask & visited_mask[1:] & late_visits]

                if len(visited_on_time) > 0:
                    artists.append(ax.scatter(visited_on_time[:, 0], visited_on_time[:, 1], c='green', label='Visited Within Time Window'))
                if len(visited_late) > 0:
                    artists.append(ax.scatter(visited_late[:, 0], visited_late[:, 1], c='purple', label='Visited After Time Window'))

                # Next deal with unvisited nodes

                before_mask = current_time < window_starts[1:] # this is fine as it is 
                during_mask = (window_starts[1:]<= current_time) & (window_ends[1:] >= current_time)
                after_mask = window_ends[1:] < current_time

                before = coords[1:][before_mask]
                during_not_visited = coords[1:][not_visited_mask[1:] & during_mask & available_mask]
                after_not_visited = coords[1:][not_visited_mask[1:] & after_mask & available_mask]

                if len(before) > 0:
                    artists.append(ax.scatter(before[:, 0], before[:, 1], c='orange', label='Before Time Window'))
                if len(during_not_visited) > 0:
                    artists.append(ax.scatter(during_not_visited[:, 0], during_not_visited[:, 1], c='blue', label='Within Time Window'))
                if len(after_not_visited) > 0:
                    artists.append(ax.scatter(after_not_visited[:, 0], after_not_visited[:, 1], c='red', label='After Time Window'))

            if len(unavailable_customers) > 0:
                unavail = ax.scatter(unavailable_customers[:, 0], unavailable_customers[:, 1], c='gray', label='Not Yet Arrived')
                artists.append(unavail)

            # Customer labels - if customer was visited late, show the amount of time they were late
            for j in range(1, num_customers + 1):
                x, y = coords[j]
                node_label = ax.text(x, y + 0.02, str(j), fontsize=8, ha='center')
                artists.append(node_label)
                if len(visited_late) > 0:
                    if j in pi[1:-1] and late_visits[j - 1] and visited_mask[j]:
                        late_label = ax.text(x, y - 0.04, f"{lateness[j-1]:.1f}", fontsize=8, ha='center', color='purple')
                        artists.append(late_label)
                

            # Tour lines
            lines = []
            length = 0.0
            for j in range(1, len(current_tour)):
                a = coords[current_tour[j - 1]]
                b = coords[current_tour[j]]
                lines.append([a, b])
                length += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

            if lines:
                lc = mc.LineCollection(lines, colors='red', linewidths=2)
                ax.add_collection(lc)
                artists.append(lc)

            lateness_including_depot = torch.cat((torch.tensor([0.0], device=lateness.device), lateness))

            accumulated_lateness = lateness_including_depot[current_tour].sum()

            ax.set_title(f"Step {i} | Time Elapsed {current_time:.1f} Minutes | Distance Traveled {length:.2f} Units \n Accumulated Lateness {accumulated_lateness:.2f} Minutes")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

            # Adjust the figure to make space for the legend
            fig.subplots_adjust(right=0.75)

            return artists  # This is key!
        
    elif problem=='pdcvrp':

        vehicle_capacity = data['vehicle_capacity'].squeeze().cpu()
        demands = data['demand'].squeeze().cpu()
        cmap = cm.get_cmap('tab10')

        def update(i):

            current_time = visit_times[i]
            current_tour = pi[:i+1]
            visited_mask[current_tour] = True

            num_customers = len(arrival_times) - 1
            available_mask = arrival_times[1:] <= current_time
            unavailable_mask = ~available_mask
            available_customers = coords[1:][available_mask]
            unavailable_customers = coords[1:][unavailable_mask]

            ax.clear()

            # Plot depot
            depot = ax.scatter(*coords[0], c='green', marker='s', label='Depot')
            artists = [depot]

            # Plot available/unavailable customers
            if len(available_customers) > 0:
                avail_visited_mask = visited_mask[1:][available_mask]
                visited_customers = available_customers[avail_visited_mask]
                available_customers = available_customers[~avail_visited_mask]
                if len(visited_customers) > 0:
                    artists.append(ax.scatter(visited_customers[:, 0], visited_customers[:, 1], c='green', label='Visited'))
                if len(available_customers) > 0:
                    artists.append(ax.scatter(available_customers[:, 0], available_customers[:, 1], c='blue', label='Available'))
            if len(unavailable_customers) > 0:
                unavail = ax.scatter(unavailable_customers[:, 0], unavailable_customers[:, 1], c='gray', label='Not Yet Arrived')
                artists.append(unavail)

            # Customer labels
            for j in range(1, num_customers + 1):
                x, y = coords[j]
                txt = ax.text(x, y + 0.02, str(j), fontsize=8, ha='center')
                artists.append(txt)

            # Split tour into subroutes using depot (0) as delimiter
            routes = []
            current_route = [current_tour[0]]
            for node in current_tour[1:]:
                current_route.append(node)
                if node == 0:
                    routes.append(torch.tensor(current_route, device=coords.device))
                    current_route = [0]
            if len(current_route) > 1:  # leftover
                routes.append(torch.tensor(current_route, device=coords.device))

            total_length = 0.0

            dem_rects = []
            used_rects = []
            cap_rects = []

            route_handles = [] # for legend

            current_capacity = vehicle_capacity.item()

            # reminder this is CVRP
            for idx, route in enumerate(routes):
                route_demands = demands[route]
                cum_demand = 0
                lines = []
                current_capacity = vehicle_capacity.item()
                for j in range(1, len(route)):
                    a = coords[route[j - 1]]
                    b = coords[route[j]]
                    lines.append([a, b])
                    cap_rects.append(Rectangle(b, 0.01, 0.1))
                    used_rects.append(Rectangle(b, 0.01, 0.1 * cum_demand / vehicle_capacity))
                    dem_rects.append(Rectangle((b[0], b[1] + 0.1 * cum_demand / vehicle_capacity), 0.01, 0.1 * route_demands[j - 1].item() / vehicle_capacity))
                    cum_demand += route_demands[j].item()
                    total_length += np.linalg.norm(a - b)
                    current_capacity -= route_demands[j].item()
                if lines:
                    lc = mc.LineCollection(lines, colors=[cmap(idx)], linewidths=2)
                    ax.add_collection(lc)
                    artists.append(lc)

                    # add a custom legend handle for the route
                    route_handle = plt.Line2D([0], [0], color=cmap(idx), lw=2, label=f'Route {idx + 1}')
                    route_handles.append(route_handle)
                if dem_rects:
                    pc_cap = mc.PatchCollection(cap_rects, facecolor='whitesmoke', edgecolor='lightgray', alpha=1.0)
                    ax.add_collection(pc_cap)
                    artists.append(pc_cap)
                    pc_used = mc.PatchCollection(used_rects, facecolor='lightgray', edgecolor='lightgray', alpha=1.0)
                    ax.add_collection(pc_used)
                    artists.append(pc_used)
                    pc_dem = mc.PatchCollection(dem_rects, facecolor='black', edgecolor='black', alpha=1.0)
                    ax.add_collection(pc_dem)
                    artists.append(pc_dem)
                    

            ax.set_title(f"Step {i} | Time Elapsed {current_time:.1f} Minutes | Distance Traveled {total_length:.2f} Units \n Current Capacity {current_capacity:.2f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            # Position the legend to the right

            leg1 = ax.legend(title='points', loc='upper left', bbox_to_anchor=(1.02, 1.0))

            ax.add_artist(leg1)

            leg2 = ax.legend(handles=route_handles, title='routes', loc='upper left', bbox_to_anchor=(1.02, 0.5))

            ax.add_artist(leg2)

            # Adjust the figure to make space for the legends
            fig.subplots_adjust(right=0.7)

            return artists
        
    elif problem=='pdcvrptw':

        vehicle_capacity = data['vehicle_capacity'].squeeze().cpu()
        demands = data['demand'].squeeze().cpu()
        window_starts = data['window_starts'].squeeze().cpu()
        window_ends = data['window_ends'].squeeze().cpu()
        cmap = cm.get_cmap('tab10')

        def update(i):
            current_time = visit_times[i]
            current_tour = pi[:i+1]
            visited_mask[current_tour] = True
            not_visited_mask = ~visited_mask

            num_customers = len(arrival_times) - 1
            available_mask = arrival_times[1:] <= current_time
            unavailable_mask = ~available_mask
            available_customers = coords[1:][available_mask]
            unavailable_customers = coords[1:][unavailable_mask]

            visit_times_by_index = torch.full_like(window_starts[1:], float('nan'))

            depot_mask = pi != 0
            pi_wo_depots = pi[depot_mask]
            visit_times_wo_depots = visit_times[depot_mask]

            visit_times_by_index[pi_wo_depots - 1] = visit_times_wo_depots

            on_time_visits = (window_starts[1:] <= visit_times_by_index) & (visit_times_by_index <= window_ends[1:])

            lateness = torch.maximum(torch.zeros_like(visit_times_by_index), visit_times_by_index - window_ends[1:])

            late_visits = visit_times_by_index > window_ends[1:]

            ax.clear()

            # Plot depot
            depot = ax.scatter(*coords[0], c='green', marker='s', label='Depot')
            artists = [depot]

            if len(available_customers) > 0:
                # deal with nodes wich have been visited first
                visited_on_time = coords[1:][available_mask & visited_mask[1:] & on_time_visits]
                visited_late = coords[1:][available_mask & visited_mask[1:] & late_visits]

                if len(visited_on_time) > 0:
                    artists.append(ax.scatter(visited_on_time[:, 0], visited_on_time[:, 1], c='green', label='Visited Within Time Window'))
                if len(visited_late) > 0:
                    artists.append(ax.scatter(visited_late[:, 0], visited_late[:, 1], c='purple', label='Visited After Time Window'))

                # Next deal with unvisited nodes

                before_mask = current_time < window_starts[1:] # this is fine as it is 
                during_mask = (window_starts[1:]<= current_time) & (window_ends[1:] >= current_time)
                after_mask = window_ends[1:] < current_time

                before = coords[1:][before_mask]
                during_not_visited = coords[1:][not_visited_mask[1:] & during_mask & available_mask]
                after_not_visited = coords[1:][not_visited_mask[1:] & after_mask & available_mask]

                if len(before) > 0:
                    artists.append(ax.scatter(before[:, 0], before[:, 1], c='orange', label='Before Time Window'))
                if len(during_not_visited) > 0:
                    artists.append(ax.scatter(during_not_visited[:, 0], during_not_visited[:, 1], c='blue', label='Within Time Window'))
                if len(after_not_visited) > 0:
                    artists.append(ax.scatter(after_not_visited[:, 0], after_not_visited[:, 1], c='red', label='After Time Window'))

            if len(unavailable_customers) > 0:
                unavail = ax.scatter(unavailable_customers[:, 0], unavailable_customers[:, 1], c='gray', label='Not Yet Arrived')
                artists.append(unavail)

            # Customer labels - if customer was visited late, show the amount of time they were late
            for j in range(1, num_customers + 1):
                x, y = coords[j]
                node_label = ax.text(x, y + 0.02, str(j), fontsize=8, ha='center')
                artists.append(node_label)
                if len(visited_late) > 0:
                    if j in pi[1:-1] and late_visits[j - 1] and visited_mask[j]:
                        late_label = ax.text(x, y - 0.04, f"{lateness[j-1]:.1f}", fontsize=8, ha='center', color='purple')
                        artists.append(late_label)

            # Split tour into subroutes using depot (0) as delimiter
            routes = []
            current_route = [current_tour[0]]
            for node in current_tour[1:]:
                current_route.append(node)
                if node == 0:
                    routes.append(torch.tensor(current_route, device=coords.device))
                    current_route = [0]
            if len(current_route) > 1:  # leftover
                routes.append(torch.tensor(current_route, device=coords.device))

            # Color map for routes
            total_length = 0.0

            dem_rects = []
            used_rects = []
            cap_rects = []

            current_capacity = vehicle_capacity.item()

            #reminder this is CVRPTW
            for idx, route in enumerate(routes):
                route_demands = demands[route]
                cum_demand = 0
                lines = []
                current_capacity = vehicle_capacity.item()
                for j in range(1, len(route)):
                    a = coords[route[j - 1]]
                    b = coords[route[j]]
                    lines.append([a, b])
                    cap_rects.append(Rectangle(b, 0.01, 0.1))
                    used_rects.append(Rectangle(b, 0.01, 0.1 * cum_demand / vehicle_capacity))
                    dem_rects.append(Rectangle((b[0], b[1] + 0.1 * cum_demand / vehicle_capacity), 0.01, 0.1 * route_demands[j].item() / vehicle_capacity))
                    cum_demand += route_demands[j].item()
                    total_length += np.linalg.norm(a - b)
                    current_capacity -= route_demands[j].item()
                if lines:
                    lc = mc.LineCollection(lines, colors=[cmap(idx)], linewidths=2)
                    ax.add_collection(lc)
                    artists.append(lc)
                if dem_rects:
                    pc_cap = mc.PatchCollection(cap_rects, facecolor='whitesmoke', edgecolor='lightgray', alpha=1.0)
                    ax.add_collection(pc_cap)
                    artists.append(pc_cap)
                    pc_used = mc.PatchCollection(used_rects, facecolor='lightgray', edgecolor='lightgray', alpha=1.0)
                    ax.add_collection(pc_used)
                    artists.append(pc_used)
                    pc_dem = mc.PatchCollection(dem_rects, facecolor='black', edgecolor='black', alpha=1.0)
                    ax.add_collection(pc_dem)
                    artists.append(pc_dem)
                    
            lateness_including_depot = torch.cat((torch.tensor([0.0], device=lateness.device), lateness))

            accumulated_lateness = lateness_including_depot[current_tour].sum()

            ax.set_title(f"Step {i} | Time Elapsed {current_time:.1f} Minutes | Distance Traveled {total_length:.2f} Units \n Current Capacity {current_capacity:.2f} | Accumulated Lateness {accumulated_lateness:.2f} Minutes")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            # Position the legend to the right
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

            # Adjust the figure to make space for the legend
            fig.subplots_adjust(right=0.75)

            return artists
        
    return update





from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib import cm
import numpy as np

import torch

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
            
def update_anim(problem, data, ax, pi=None, visit_times=None):
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
            ax.legend()

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
            ax.legend()

            return artists  # This is key!
        
    elif problem=='pdcvrp':

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
                    routes.append(current_route)
                    current_route = [0]
            if len(current_route) > 1:  # leftover
                routes.append(current_route)

            # Color map for routes
            cmap = cm.get_cmap('tab10', len(routes))
            total_length = 0.0

            for idx, route in enumerate(routes):
                lines = []
                for j in range(1, len(route)):
                    a = coords[route[j - 1]]
                    b = coords[route[j]]
                    lines.append([a, b])
                    total_length += np.linalg.norm(a - b)
                if lines:
                    lc = mc.LineCollection(lines, colors=[cmap(idx)], linewidths=2)
                    ax.add_collection(lc)
                    artists.append(lc)

            ax.set_title(f"Step {i} | Time Elapsed {current_time:.1f} Minutes | Distance Traveled {total_length:.2f} Units")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()

            return artists
        

    return update





from matplotlib import pyplot as plt
from matplotlib import collections as mc

import torch

def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
               reinforce_loss, bl_loss, tb_logger, wandb_run, opts):
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

    arrival_times = data['arrival_times']
    coords = data['all_nodes']

    # add a return to the depot at the end of the tour
    pi = torch.cat((pi, torch.tensor([0], device=pi.device)), dim=0)

    if problem == 'pdtrp':

        def update(i):
            current_time = visit_times.squeeze()[i]
            current_tour = pi[:i+1]
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
                avail = ax.scatter(available_customers[:, 0], available_customers[:, 1], c='blue', label='Available')
                artists.append(avail)
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

    return update





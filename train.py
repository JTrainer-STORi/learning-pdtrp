import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
import tempfile
import wandb

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel

from utils.log_utils import log_values, update_anim
from utils.data_utils import BatchedRandomSampler, dataset_to_input
from utils import move_to, count_trailing_zeros


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def should_run_periodic(epoch, every, epoch_start, n_epochs):
    """Return whether periodic work is due, always including the final epoch."""
    is_final_epoch = epoch == epoch_start + n_epochs - 1
    return (epoch + 1) % every == 0 or is_final_epoch


def validate(model, dataset, problem, opts):
    # Validate
    print(f'\nValidating on {dataset.size} samples from {dataset.filename}...')      
    cost, info = rollout(model, dataset, opts)
    gt_cost, info = rollout_groundtruth(problem, dataset, opts)
    opt_gap = (((cost/gt_cost) - 1) * 100)
    
    print('Validation groundtruth cost: {:.3f} +- {:.3f}'.format(
        gt_cost.mean(), torch.std(gt_cost)))
    print('Validation average cost: {:.3f} +- {:.3f}'.format(
        cost.mean(), torch.std(cost)))
    print('Validation optimality gap: {:.3f}% +- {:.3f}'.format(
        opt_gap.mean(), torch.std(opt_gap)))
    
    model_best = opt_gap.argmax()
    model_worst = opt_gap.argmin()

    return cost.mean(), opt_gap.mean(), gt_cost.mean(), model_best, model_worst


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    if opts.val_batch_size == 1:
        mode = opts.eval_mode
    else:
        mode='masked'
    def eval_model_bat(bat):
        input = dataset_to_input(bat, opts.problem, opts.device)
        with torch.no_grad():
            cost, _, info = model(input, mode=mode)
        return cost, info
    
    costs = []
    infos = []
    
    for bat in tqdm(
            DataLoader(dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers), 
            disable=opts.no_progress_bar, ascii=True
        ):
        cost, info = eval_model_bat(bat)
        costs.append(cost)
        if info:
            infos.append(info)

    return torch.cat(costs, 0), infos

def rollout_baseline(model, dataset, opts, mode):
        # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    def eval_model_bat(bat):
        input = dataset_to_input(bat, opts.problem, opts.device)
        with torch.no_grad():
            cost, _, info = model(input, mode=mode)
        return cost, info
    
    costs = []
    infos = []
    
    for bat in tqdm(
            DataLoader(dataset, batch_size=opts.rollout_batch_size, shuffle=False, num_workers=opts.num_workers), 
            disable=opts.no_progress_bar, ascii=True
        ):
        cost, info = eval_model_bat(bat)
        costs.append(cost)
        if info:
            infos.append(info)

    return torch.cat(costs, 0), infos

def rollout_groundtruth(problem, dataset, opts):
    if opts.problem in ('pdctrp', 'pdcvrp'):
        costs = []
        infos = []
        for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers):
    
            cost, info = problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device), 'distance_matrix': move_to(bat['distance_matrix'], opts.device), 'demand': move_to(bat['demand'], opts.device), 'vehicle_capacity': move_to(bat['vehicle_capacity'], opts.device)}, move_to(bat['tour_nodes'], opts.device))
            costs.append(cost)
            if info:
                infos.append(info)
        return torch.cat(costs, 0), infos
    if opts.problem == 'pdtrp':
        costs = []
        infos = []
        for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers):
            
            cost, info = problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device), 'distance_matrix': move_to(bat['distance_matrix'], opts.device)}, move_to(bat['tour_nodes'], opts.device))
            costs.append(cost)
            if info:
                infos.append(info)
        return torch.cat(costs, 0), infos
    elif opts.problem == 'pdtrptw':
        costs = []
        infos = []
        for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers):
            
            cost, info = problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device), 'window_ends': move_to(bat['window_ends'], opts.device), 'distance_matrix': move_to(bat['distance_matrix'], opts.device)}, move_to(bat['tour_nodes'], opts.device), move_to(bat['visit_times'], opts.device), move_to(bat['gamma'], opts.device), move_to(bat['theta'], opts.device))
            costs.append(cost)
            if info:
                infos.append(info)
        return torch.cat(costs, 0), infos
    elif opts.problem in ('pdctrptw', 'pdcvrptw'):
        costs = []
        infos = []
        for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers):
            
            cost, info = problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device), 'window_ends': move_to(bat['window_ends'], opts.device), 'distance_matrix': move_to(bat['distance_matrix'], opts.device), 'demand': move_to(bat['demand'], opts.device), 'vehicle_capacity': move_to(bat['vehicle_capacity'], opts.device)}, move_to(bat['tour_nodes'], opts.device), move_to(bat['visit_times'], opts.device), move_to(bat['gamma'], opts.device), move_to(bat['theta'], opts.device))
            costs.append(cost)
            if info:
                infos.append(info)
        return torch.cat(costs, 0), infos


def clip_grad_norms(param_groups, max_norm=math.inf):
    """Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_datasets, problem, tb_logger, wandb_run, opts):
    print("\nStart train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
    if not opts.no_wandb:
        wandb_run.log({'learnrate_pg0': optimizer.param_groups[0]['lr']}, step=step)

    # Generate new training data for each epoch
    train_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            min_total=opts.min_total, max_total=opts.max_total, min_dod=opts.min_dod, max_dod=opts.max_dod, speed=opts.speed, batch_size=opts.batch_size, n_subregions=opts.n_subregions, num_samples=opts.epoch_size, time_horizon=opts.time_horizon, service_times_mean = opts.stmean, service_times_var = opts.stvar, neighbors=opts.neighbors, knn_strat=opts.knn_strat, arrival_weights = opts.arrival_weights, arrival_skews=opts.arrival_skews, min_time_window=opts.min_time_window, max_time_window=opts.max_time_window, gamma = opts.gamma, theta=opts.theta, latest_end=opts.latest_end, reaction_time=opts.reaction_time, vehicle_capacity=opts.vehicle_capacity, min_trips_required_lb=opts.min_trips_required_lb, min_trips_required_ub=opts.min_trips_required_ub, use_ortec = opts.use_ortec, dimension_scale=opts.dimension_scale
        ))
    train_dataloader = DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    # Put model in train mode!
    model.train()
    optimizer.zero_grad()
    set_decode_type(model, "sampling") # This differs from the sampling decoding in the actual paper.

    if opts.profiler:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profile'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)):
                prof.step()
                train_batch(
                    model,
                    optimizer,
                    baseline,
                    epoch,
                    batch_id,
                    step,
                    batch,
                    tb_logger,
                    wandb_run,
                    opts
                )

                step += 1
    else:
        for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)):
                train_batch(
                    model,
                    optimizer,
                    baseline,
                    epoch,
                    batch_id,
                    step,
                    batch,
                    tb_logger,
                    wandb_run,
                    opts
                )

                step += 1
    
    lr_scheduler.step()

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    should_validate = should_run_periodic(
        epoch, opts.val_every, opts.epoch_start, opts.n_epochs
    )
    if not should_validate:
        print("Skipping validation at epoch {} (validation interval: {})".format(epoch, opts.val_every))

    for val_idx, val_dataset in enumerate(val_datasets if should_validate else []):
        avg_reward, avg_opt_gap, avg_gt_cost, model_worst, model_best = validate(model, val_dataset, problem, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val{}/avg_reward'.format(val_idx+1), avg_reward, step)
            tb_logger.log_value('val{}/opt_gap'.format(val_idx+1), avg_opt_gap, step)
            tb_logger.log_value('val{}/avg_gt_cost'.format(val_idx+1), avg_gt_cost, step)
        if not opts.no_wandb:
            # do a single rollout with the model to get a tour
            if not opts.no_videos:
                for i in [model_best, model_worst]:
                    print(f"Logging videos for validation dataset {val_idx + 1}, sample {i} (model {'best' if i == model_best else 'worst'})")
                    for j in range(2):
                        dataset_instance = val_dataset.__getitem__(i)
                        input_instance = dataset_to_input(dataset_instance, opts.problem, opts.device)
                        fig, ax = plt.subplots()
                        if j == 0: # use the model
                            set_decode_type(model, "greedy")
                            model.eval()
                            with torch.no_grad():
                                _, _, _, pi, visit_times = model(input_instance, return_pi=True, return_times=True, mode='recursive')
                                anim = FuncAnimation(fig, update_anim(opts.problem, input_instance, fig, ax, pi=pi, visit_times=visit_times, dimension_scale=opts.dimension_scale), frames=len(visit_times.squeeze()), blit=False)
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    video_path = os.path.join(tmpdir, 'validation.mp4')
                                    anim.save(video_path, writer='ffmpeg', fps=2)

                                    # Log the video to wandb
                                    if i == model_best:
                                        wandb.log({"videos_val{}/model_best/model".format(val_idx + 1): wandb.Video(video_path, format="mp4")}, step=step)
                                    else:
                                        wandb.log({"videos_val{}/model_worst/model".format(val_idx + 1): wandb.Video(video_path, format="mp4")}, step=step)
                        else: # use the baseline 
                            # remove any padding from the baseline
                            padding = count_trailing_zeros(dataset_instance['tour_nodes'])
                            tour = dataset_instance['tour_nodes'][:-padding] if padding > 0 else dataset_instance['tour_nodes']
                            visit_times = dataset_instance['visit_times'][:-(padding - 1)] if padding > 1 else dataset_instance['visit_times']
                            anim = FuncAnimation(fig, update_anim(opts.problem, input_instance, fig, ax, pi=move_to(tour, opts.device), visit_times=move_to(visit_times, opts.device), dimension_scale=opts.dimension_scale), frames=len(visit_times), blit=False)

                            with tempfile.TemporaryDirectory() as tmpdir:
                                video_path = os.path.join(tmpdir, 'validation.mp4')
                                anim.save(video_path, writer='ffmpeg', fps=2)

                                # Log the video to wandb
                                if i == model_best:
                                    wandb.log({"videos_val{}/model_best/baseline".format(val_idx + 1): wandb.Video(video_path, format="mp4")}, step=step)
                                else:
                                    wandb.log({"videos_val{}/model_worst/baseline".format(val_idx + 1): wandb.Video(video_path, format="mp4")}, step=step)
                        plt.close('all')
            wandb_run.log({
                'val{}/avg_reward'.format(val_idx+1): avg_reward,
                'val{}/opt_gap'.format(val_idx+1): avg_opt_gap,
                'val{}/avg_gt_cost'.format(val_idx+1): avg_gt_cost
            }, step=step)
                
    baseline.epoch_callback(model, epoch)


def train_batch(model, optimizer, baseline, epoch, 
                batch_id, step, batch, tb_logger, wandb_run, opts):
    
    use_pomo = opts.pomo_batch_size > 1 

    if use_pomo:
        bat = batch
    else:
        # if not using pomo then the baseline can be calculated before passing the input to the model
        bat, bl_val = baseline.unwrap_batch(batch)
        # Optionally move Tensors to GPU
        bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    

    input = dataset_to_input(bat, opts.problem, opts.device, opts.pomo_batch_size)

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, info = model(input, pomo_batch_size=opts.pomo_batch_size, print_query_times=opts.print_query_times)

    # Evaluate baseline, get baseline loss if any (only for critic)
    if use_pomo:
        cost_grouped = cost.view(-1, opts.pomo_batch_size)
        baseline_per_batch = cost_grouped.mean(dim=1, keepdim=True)
        bl_val = baseline_per_batch.expand(-1, opts.pomo_batch_size).reshape(-1)
        bl_loss = 0
    else:
        bl_val, bl_loss = baseline.eval(input, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    
    # Normalize loss for gradient accumulation
    loss = loss / opts.accumulation_steps

    # Perform backward pass
    loss.backward()
    
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    
    # Perform optimization step after accumulating gradients
    if step % opts.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, tb_logger, wandb_run, opts, info)

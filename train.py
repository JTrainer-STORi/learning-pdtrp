import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel

from utils.log_utils import log_values
from utils.data_utils import BatchedRandomSampler
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def validate(model, dataset, problem, opts):
    # Validate
    print(f'\nValidating on {dataset.size} samples from {dataset.filename}...')
    cost = rollout(model, dataset, opts)
    gt_cost = rollout_groundtruth(problem, dataset, opts)
    opt_gap = (((cost/gt_cost) - 1) * 100)
    
    print('Validation groundtruth cost: {:.3f} +- {:.3f}'.format(
        gt_cost.mean(), torch.std(gt_cost)))
    print('Validation average cost: {:.3f} +- {:.3f}'.format(
        cost.mean(), torch.std(cost)))
    print('Validation optimality gap: {:.3f}% +- {:.3f}'.format(
        opt_gap.mean(), torch.std(opt_gap)))

    return cost.mean(), opt_gap.mean(), gt_cost.mean()


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    def eval_model_bat(bat):
        if opts.problem == 'pdtrp':
            input = {
                'loc': move_to(bat['all_nodes'], opts.device),
                'service_times': move_to(bat['service_times'], opts.device),
                'arrival_times': move_to(bat['arrival_times'], opts.device),
                'graph': move_to(bat['graph'], opts.device)
            }
        elif opts.problem == 'pdcvrp':
            input = {
                'loc': move_to(bat['all_nodes'], opts.device),
                'service_times': move_to(bat['service_times'], opts.device),
                'arrival_times': move_to(bat['arrival_times'], opts.device),
                'demand': move_to(bat['demand'], opts.device),
                'graph': move_to(bat['graph'], opts.device)
            }
        elif opts.problem == 'pdtrptw':
            input = {
                'loc': move_to(bat['all_nodes'], opts.device),
                'service_times': move_to(bat['service_times'], opts.device),
                'arrival_times': move_to(bat['arrival_times'], opts.device),
                'graph': move_to(bat['graph'], opts.device),
                'window_starts': move_to(bat['window_starts'], opts.device),
                'window_ends': move_to(bat['window_ends'], opts.device),
                'graph': move_to(bat['graph'], opts.device)
            }
        elif opts.problem == 'pdcvrptw':
            input = {
                'loc': move_to(bat['all_nodes'], opts.device),
                'service_times': move_to(bat['service_times'], opts.device),
                'arrival_times': move_to(bat['arrival_times'], opts.device),
                'graph': move_to(bat['graph'], opts.device),
                'window_starts': move_to(bat['window_starts'], opts.device),
                'window_ends': move_to(bat['window_ends'], opts.device),
                'demand': move_to(bat['demand'], opts.device),
                'graph': move_to(bat['graph'], opts.device)
            }
        with torch.no_grad():
            cost, _ = model(input)
        return cost
    
    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(
            DataLoader(dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers), 
            disable=opts.no_progress_bar, ascii=True
        )
    ], 0)


def rollout_groundtruth(problem, dataset, opts):
    if opts.problem == 'pdcvrp':
        return torch.cat([
            problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device), 'demand': move_to(bat['demand'], opts.device)}, move_to(bat['tour_nodes'], opts.device))[0]
            for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers)
        ], 0)
    elif opts.problem == 'pdtrp':
        return torch.cat([
            problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device)}, move_to(bat['tour_nodes'], opts.device))[0]
            for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers)
        ], 0)
    elif opts.problem == 'pdtrptw':
        return torch.cat([
            problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device), 'window_ends': move_to(bat['window_ends'], opts.device), 'visit_times': move_to(bat['visit_times'].unsqueeze(1), opts.device)}, move_to(bat['tour_nodes'], opts.device))[0]
            for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers)
        ], 0)
    elif opts.problem == 'pdcvrptw':
        return torch.cat([
            problem.get_costs({'loc': move_to(bat['all_nodes'], opts.device), 'demand': move_to(bat['demand'], opts.device), 'window_ends': move_to(bat['window_ends'], opts.device), 'visit_times': move_to(bat['visit_times'].unsqueeze(1), opts.device)}, move_to(bat['tour_nodes'], opts.device))[0]
            for bat in DataLoader(
                dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=opts.num_workers)
        ], 0)


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


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_datasets, problem, tb_logger, opts):
    print("\nStart train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    train_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            min_total=opts.min_total, max_total=opts.max_total, min_dod=opts.min_dod, max_dod=opts.max_dod, batch_size=opts.batch_size, 
            num_samples=opts.epoch_size, time_horizon=opts.time_horizon, service_times_mu = opts.stmu, service_times_sigma = opts.stsig, neighbors=opts.neighbors, knn_strat=opts.knn_strat
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
                    opts
                )

                step += 1
    
    lr_scheduler.step(epoch)

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

    for val_idx, val_dataset in enumerate(val_datasets):
        avg_reward, avg_opt_gap, avg_gt_cost = validate(model, val_dataset, problem, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val{}/avg_reward'.format(val_idx+1), avg_reward, step)
            tb_logger.log_value('val{}/opt_gap'.format(val_idx+1), avg_opt_gap, step)
            tb_logger.log_value('val{}/avg_gt_cost'.format(val_idx+1), avg_gt_cost, step)

    baseline.epoch_callback(model, epoch)


def train_batch(model, optimizer, baseline, epoch, 
                batch_id, step, batch, tb_logger, opts):
    # Unwrap baseline
    bat, bl_val = baseline.unwrap_batch(batch)

    if opts.problem == 'pdtrp':
        input = {
            'loc': move_to(bat['all_nodes'], opts.device),
            'service_times': move_to(bat['service_times'], opts.device),
            'arrival_times': move_to(bat['arrival_times'], opts.device),
            'graph': move_to(bat['graph'], opts.device)
        }
    elif opts.problem == 'pdcvrp':
        input = {
            'loc': move_to(bat['all_nodes'], opts.device),
            'service_times': move_to(bat['service_times'], opts.device),
            'arrival_times': move_to(bat['arrival_times'], opts.device),
            'demand': move_to(bat['demand'], opts.device),
            'graph': move_to(bat['graph'], opts.device)
        }
    elif opts.problem == 'pdtrptw':
        input = {
            'loc': move_to(bat['all_nodes'], opts.device),
            'service_times': move_to(bat['service_times'], opts.device),
            'arrival_times': move_to(bat['arrival_times'], opts.device),
            'graph': move_to(bat['graph'], opts.device),
            'window_starts': move_to(bat['window_starts'], opts.device),
            'window_ends': move_to(bat['window_ends'], opts.device),
            'graph': move_to(bat['graph'], opts.device)
        }
    elif opts.problem == 'pdcvrptw':
        input = {
            'loc': move_to(bat['all_nodes'], opts.device),
            'service_times': move_to(bat['service_times'], opts.device),
            'arrival_times': move_to(bat['arrival_times'], opts.device),
            'graph': move_to(bat['graph'], opts.device),
            'window_starts': move_to(bat['window_starts'], opts.device),
            'window_ends': move_to(bat['window_ends'], opts.device),
            'demand': move_to(bat['demand'], opts.device),
            'graph': move_to(bat['graph'], opts.device)
        }
    
    # Optionally move Tensors to GPU
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(input)

    # Evaluate baseline, get baseline loss if any (only for critic)
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
        log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
                   reinforce_loss, bl_loss, tb_logger, opts)

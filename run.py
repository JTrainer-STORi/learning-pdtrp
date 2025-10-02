#!/usr/bin/env python

import os
import json
import pprint as pp
import numpy as np

import torch
import torch.optim as optim
import wandb

from options import get_options
from train import train_epoch, get_inner_model

from tensorboard_logger import Logger as TbLogger
from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.tgnn_encoder import IncrementalUpdateEncoder
from nets.critic_network import CriticNetwork

from reinforce_baselines import ExponentialBaseline, CriticBaseline, RolloutBaseline, NoBaseline, WarmupBaseline

from utils import load_problem, torch_load_cpu

def run_rl(opts) -> None:
    """
    Top Level Function for running RL experiments
    """

    # torch.autograd.set_detect_anomaly(True) # un-comment this line for help detecting specific Pytorch autograd errors

    # pretty print the run args
    pp.pprint(vars(opts))

    # set the random seeds
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}-{}-{}-{}".format(opts.problem, opts.min_total, opts.max_total, opts.min_dod, opts.max_dod), opts.run_name))
    if not opts.no_wandb:
        wandb_run = wandb.init(
            project="dynamic VRP",
            name=opts.run_name,
            config=vars(opts),
            dir=opts.log_dir,
            resume="allow" if opts.resume else None
        )
    else:
        wandb_run = None

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load the problem
    problem = load_problem(opts.problem)

    # load data from load path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('\nLoading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    if opts.use_incremental_encoder:
        encoder_class = IncrementalUpdateEncoder
    else:
        encoder_class = GNNEncoder

    model = DynamicAttentionModel(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        encoder_class=encoder_class, # change this at some point if we only go with this encoder 
        n_encode_layers=opts.n_encode_layers, 
        aggregation=opts.aggregation,
        aggregation_graph=opts.aggregation_graph,
        normalization=opts.normalization,
        learn_norm=opts.learn_norm,
        track_norm=opts.track_norm,
        gated=opts.gated,
        n_heads=opts.n_heads,
        tanh_clipping=opts.tanh_clipping,
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        checkpoint_encoder=opts.checkpoint_encoder,
        edge_features=opts.edge_features,
        use_time_feature=opts.use_time_feature,
        functional_time_encoding=opts.functional_time_encoding,
        knn_strat=opts.knn_strat,
        neighbors=opts.neighbors,
        scale_times=opts.scale_times,
        use_arrival_lstm=opts.use_arrival_lstm,
        use_arrival_times=opts.use_arrival_times,
        recursively_remove_visited_nodes=opts.recursively_remove_visited_nodes,
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Compute number of network parameters
    print(model)
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    if not opts.no_wandb:
        if opts.watch_gradients:
            wandb.watch(model, log='all', log_graph=True)

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetwork(
                    embedding_dim=opts.embedding_dim,
                    encoder_class=GNNEncoder,
                    n_encode_layers=opts.n_encode_layers,
                    aggregation=opts.aggregation,
                    normalization=opts.normalization,
                    learn_norm=opts.learn_norm,
                    track_norm=opts.track_norm,
                    gated=opts.gated,
                    n_heads=opts.n_heads
                )
            ).to(opts.device)
        )
        
        print(baseline.critic)
        nb_param = 0
        for param in baseline.get_learnable_parameters():
            nb_param += np.prod(list(param.data.size()))
        print('Number of parameters (BL): ', nb_param)
        
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    
    elif opts.baseline == 'pomo':
        # pomo baseline is handled in the model itself
        baseline = NoBaseline()
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Load/generate datasets
    val_datasets = []
    for val_filename in opts.val_datasets:
        val_datasets.append(
            problem.make_dataset(
                filename=val_filename, batch_size=opts.val_batch_size, num_samples=opts.val_size, 
                neighbors=opts.neighbors, knn_strat=opts.knn_strat, speed=opts.speed, time_horizon=opts.time_horizon, gamma=opts.gamma, theta=opts.theta, latest_end=opts.latest_end, reaction_time=opts.reaction_time, vehicle_capacity=opts.vehicle_capacity, min_trips_required_lb=opts.min_trips_required_lb,
                min_trips_required_ub=opts.min_trips_required_ub,use_ortec=opts.use_ortec, dimension_scale=opts.dimension_scale
            ))

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # Start training loop
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_datasets,
            problem,
            tb_logger,
            wandb_run,
            opts
        )

    # finish wandb run
    if not opts.no_wandb:
        wandb.finish()

if __name__ == "__main__":

    # import debugpy
    # debugpy.listen(5678)  # 5678 is port
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()

    run_rl(get_options())

import os
import time
import argparse
import json
import math
import torch
from problems.size_conventions import CUSTOMER_SIZE_CONVENTION, resolve_node_bounds
from problems.time_windows import add_time_window_arguments, validate_time_window_arguments
from problems.time_horizon import validate_hours_per_customer

# Note on the store_true and store_false arguments in argparse:
# The store_true action is used to create a boolean flag that is set to True if the option is provided, and False otherwise.
# Similarly, the store_false action creates a boolean flag that is set to False if the option is provided, and True otherwise.

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters for training learning-driven solvers for dynamic routing problems",)

    # Data
    parser.add_argument('--problem', default='pdtrp', 
                        help="The problem to solve, options are: 'pdtrp', 'pdtrptw', 'pdctrp' or 'pdctrptw'")
    parser.add_argument('--min-customers', '--min_customers', type=int,
                        help="Minimum N, excluding the depot (default: 40)")
    parser.add_argument('--max-customers', '--max_customers', type=int,
                        help="Maximum N, excluding the depot (default: 100)")
    parser.add_argument('--min_total', type=int,
                        help="Legacy minimum total-node count INCLUDING the depot; use --min-customers for N")
    parser.add_argument('--max_total', type=int,
                        help="Legacy maximum total-node count INCLUDING the depot; use --max-customers for N")
    parser.add_argument('--min_dod', type=float, default=0.2,
                        help='The minimum ratio of dynamic nodes')
    parser.add_argument('--max_dod', type=float, default=0.8,
                        help='The maximum ratio of dynamic nodes')
    parser.add_argument('--n_subregions', type=int, default=1,
                        help='Number of subregions to generate the customers in. Passing 1 will generate customers uniformly in the unit square.')
    parser.add_argument('--dimension_scale', type=float, default=1.0,
                        help='Scale of generated coordinates, default is the unit square')
    parser.add_argument('--speed', type=float, default=4.0, 
                        help='The speed of the vehicle in units/h, the dimension of the unit square is 1x1 units^2')
    parser.add_argument('--arrival_weights', nargs='+', type=int, default=None,
                        help='The weights for the subregions, if None, subregion weights are sampled from a dirichlet distribution. Pass as a space-separated list, e.g. "1 2 3"')
    parser.add_argument('--arrival_skews', nargs='+', type=str, default=None,
                        help='The skews for the arrival times distribution in each subregion. An "early" skew means that arrival times for that subregion are generated using a beta(2,5) distribution, a "late" skew means beta(5,2), and a "uniform" skew means uniform distribution. If None, all subregions are sampled uniformly. Pass as a space-separated list, e.g. "early late uniform"')
    parser.add_argument('--neighbors', type=float, default=20, 
                        help="The k-nearest neighbors for graph sparsification, if the knn_strat is 'percentage', this is the percentage of total nodes to connect to each node otherwise is the fixed number of neighbors to connect to each node")
    parser.add_argument('--knn_strat', type=str, choices=('fixed', 'percentage'), default='fixed',
                        help="Strategy for k-nearest neighbors (default: fixed)")
    parser.add_argument('--n_epochs', type=int, default=100, 
                        help='The number of epochs to train')
    parser.add_argument('--epoch_size', type=int, default=12800,
                        help='Number of instances per epoch during training')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Number of instances per batch during training')
    parser.add_argument('--accumulation_steps', type=int, default=1, 
                        help='Gradient accumulation step during training (effective batch_size = batch_size * accumulation_steps)')
    parser.add_argument('--val_datasets', type=str, nargs='+', default=None, 
                        help='Paths to dataset files to use for validation')
    parser.add_argument('--validation-seed', '--validation_seed', type=int, default=None,
                        help='Seed for generated held-out validation data, independent of training seed (default: 12345)')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Number of instances per batch during validation')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_every', type=int, default=1,
                        help='Run validation every n epochs (and always on the final epoch)')
    parser.add_argument('--rollout_batch_size', type=int, default=128,
                        help='Number of instances per batch during rollout baseline evaluation')
    parser.add_argument('--rollout_size', type=int, default=1280,
                        help='Number of instances used for updating rollout baseline')
    parser.add_argument('--baseline_update_every', type=int, default=1,
                        help='Challenge and potentially update the rollout baseline every n epochs (and always on the final epoch)')
    horizon_options = parser.add_mutually_exclusive_group()
    horizon_options.add_argument('--time-horizon', '--time_horizon', type=float, default=None,
                                 help='Fixed call horizon in hours (default: 8 when scaling is unset)')
    horizon_options.add_argument('--time-horizon-per-customer', '--time_horizon_per_customer', type=float,
                                 help='Set T=N times this many hours, excluding the depot; use 0.2 for T=N/5')
    parser.add_argument('--stmean', type=float, default=3.0,
                        help='mean of service times, in minutes')
    parser.add_argument('--stvar', type=float, default=5.0,
                        help='variance of service times, in minutes')
    parser.add_argument('--gamma', type=float, default=1.0, 
                        help='weighting for missed time windows in costs for problems with time windows')
    parser.add_argument('--theta', type=float, default=1.0, 
                        help='weighting for distance in costs for problems with time windows')
    parser.add_argument('--latest_end', type=int, default=2,
                        help='maximum amount of time after the time horizon passes that a customer\'s time window can end, default is 2 hours')
    parser.add_argument('--reaction_time', type=int, default=60,
                        help="Legacy ORTEC call-to-opening delay in minutes; synthetic TW generation uses depot travel instead")
    add_time_window_arguments(parser)
    parser.add_argument('--vehicle_capacity', type=float, default=1.0,
                        help='Vehicle capacity, default is 1.0 (1 unit of demand)')
    parser.add_argument('--min_trips_required_lb', type=int, default=3,
                        help='lower bound on the minimum number of trips required to service all customers, default is 3')
    parser.add_argument('--min_trips_required_ub', type=int, default=9,
                        help='upper bound on the minimum number of trips required to service all customers, default is 9')
    parser.add_argument('--use_ortec', type=str, default=None,
                        help='filename of ortec instance to subsample from when generating customer ugations. If None, no subsampling is done.')

    # Model/GNN Encoder
    parser.add_argument('--model', default='attention', 
                        help="Model: 'attention'/'nar'")
    parser.add_argument('--encoder', default='gnn', 
                        help="Graph encoder: 'gat'/'gnn'/'mlp'")
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, 
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--aggregation', default='max', 
                        help="Neighborhood aggregation function: 'sum'/'mean'/'max'")
    parser.add_argument('--aggregation_graph', default='mean', 
                        help="Graph embedding aggregation function: 'sum'/'mean'/'max'")
    parser.add_argument('--normalization', default='layer', 
                        help="Normalization type: 'batch'/'layer'/None")
    parser.add_argument('--learn_norm', action='store_true', 
                        help="Enable learnable affine transformation during normalization")
    parser.add_argument('--track_norm', action='store_true',
                        help="Enable tracking batch statistics during normalization")
    parser.add_argument('--gated', action='store_true', 
                        help="Enable edge gating during neighborhood aggregation")
    parser.add_argument('--n_heads', type=int, default=8, 
                        help="Number of attention heads")
    parser.add_argument('--tanh_clipping', type=float, default=10., 
                        help='Clip the parameters to within +- this value using tanh. Set to 0 to not do clipping.')
    parser.add_argument('--edge_features', type=str, default='distance',
                        help='Edge features to use: distance or adjacency')
    parser.add_argument('--use_time_feature', action='store_true',
                        help='Use time as an additional feature during decoding')
    parser.add_argument('--functional_time_encoding', action='store_true',
                        help='Whether to pass any time features in the model through thte functional time encoding from TGAT paper')
    parser.add_argument('--scale_times', action='store_true',
    help='scale time features to an approximate [0,1] range by dividing by the time horizon')
    parser.add_argument('--use_arrival_lstm', action='store_true',
                        help='Use an LSTM hidden state as part of the decoder context')
    parser.add_argument('--use_arrival_times', action='store_true',
                        help='Use arrival time as an additional node feature in the encoder')
    parser.add_argument('--use_incremental_encoder', action='store_true', help='Use incremental encoder for the model')
    parser.add_argument('--recursively_remove_visited_nodes', action='store_true', help='Recursively remove visited nodes from the graph')
    parser.add_argument('--eval_mode', type=str, default='recursive', choices=['recursive', 'MSA'])

    # Training
    parser.add_argument('--pomo_batch_size', type=int, default=1,
                        help='Batch size for POMO training, i.e. number of start points per instance')
    parser.add_argument('--lr_model', type=float, default=1e-4, 
                        help="Set the learning rate for the actor network, i.e. the main model")
    parser.add_argument('--lr_critic', type=float, default=1e-4, 
                        help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, 
                        help='Learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                        help='Maximum L2 norm for gradient clipping (0 to disable clipping)')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'pomo' or 'exponential'.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed to use')
    parser.add_argument('--profiler', type=bool, default=False,
                        help="whether to run pytorch profiler on this run or not")

    # Misc
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for DataLoaders')
    parser.add_argument('--log_step', type=int, default=100, 
                        help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', 
                        help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', 
                        help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', 
                        help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', 
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', 
                        help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', 
                        help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', 
                        help='Disable progress bar')
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--no_videos', action='store_true',
                        help='Disable video logging on wandb')
    parser.add_argument('--profile', type=bool, default=False,
                        help='run cuda profiler')
    parser.add_argument('--watch_gradients', action='store_true',
                        help='Watch gradients in Weights & Biases')
    parser.add_argument('--print_query_times', action='store_true',
                        help='Print query times for the model to stdout')
    opts = parser.parse_args(args)
    validate_time_window_arguments(parser, opts)

    saved = {}
    if opts.resume:
        saved_args = os.path.join(os.path.dirname(opts.resume), 'args.json')
        if os.path.isfile(saved_args):
            with open(saved_args) as handle:
                saved = json.load(handle)
    if opts.time_horizon is None and opts.time_horizon_per_customer is None:
        opts.time_horizon = saved.get('time_horizon', 8.0)
        opts.time_horizon_per_customer = saved.get('time_horizon_per_customer')
    if opts.time_horizon is None:
        opts.time_horizon = 8.0  # Legacy fallback; per-customer scaling overrides it.
    try:
        validate_hours_per_customer(opts.time_horizon_per_customer)
        if not math.isfinite(opts.time_horizon) or opts.time_horizon <= 0:
            raise ValueError('time_horizon must be finite and positive')
    except ValueError as error:
        parser.error(str(error))
    if opts.validation_seed is None:
        opts.validation_seed = saved.get('validation_seed', 12345)
    if not 0 <= opts.validation_seed < 2**32:
        parser.error('--validation-seed must be between 0 and 2**32 - 1')

    # Resuming without size overrides inherits the saved bounds, including the
    # old total-node convention used by checkpoints without size metadata.
    if opts.resume and all(value is None for value in (
        opts.min_total, opts.max_total, opts.min_customers, opts.max_customers
    )):
        if saved:
            if 'min_customers' in saved and 'max_customers' in saved:
                opts.min_customers, opts.max_customers = saved['min_customers'], saved['max_customers']
            else:
                opts.min_total, opts.max_total = saved.get('min_total'), saved.get('max_total')
    try:
        opts.min_total, opts.max_total = resolve_node_bounds(
            opts.min_total, opts.max_total, opts.min_customers, opts.max_customers,
            default_min_customers=40,
        )
    except ValueError as error:
        parser.error(str(error))
    opts.min_customers, opts.max_customers = opts.min_total - 1, opts.max_total - 1
    opts.size_convention = CUSTOMER_SIZE_CONVENTION

    for size, batch in ((opts.epoch_size, opts.batch_size),
                        (opts.rollout_size, opts.rollout_batch_size),
                        (opts.val_size, opts.val_batch_size)):
        if size < 1 or batch < 1 or size % batch:
            parser.error('Training, rollout and validation sizes must be divisible by their positive batch sizes')
    if opts.baseline == 'rollout' and opts.min_total != opts.max_total and opts.batch_size % opts.rollout_batch_size:
        parser.error('--rollout_batch_size must divide --batch_size for variable-size training; otherwise baseline batches mix graph sizes')

    if opts.accumulation_steps < 1:
        parser.error("--accumulation_steps must be at least 1")
    if opts.val_every < 1:
        parser.error("--val_every must be at least 1")
    if opts.baseline_update_every < 1:
        parser.error("--baseline_update_every must be at least 1")

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}-{}-{}-{}".format(opts.problem, opts.min_customers, opts.max_customers, opts.min_dod, opts.max_dod),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    
    return opts

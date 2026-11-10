"""Generation settings shared by training, validation and rollout baselines."""

import numpy as np


def training_dataset_options(opts):
    return {
        "min_total": opts.min_total, "max_total": opts.max_total,
        "min_dod": opts.min_dod, "max_dod": opts.max_dod,
        "speed": opts.speed, "time_horizon": opts.time_horizon,
        "time_horizon_per_customer": getattr(opts, "time_horizon_per_customer", None),
        "n_subregions": opts.n_subregions,
        "arrival_weights": opts.arrival_weights, "arrival_skews": opts.arrival_skews,
        "neighbors": opts.neighbors, "knn_strat": opts.knn_strat,
        "service_times_mean": opts.stmean, "service_times_var": opts.stvar,
        "min_time_window": opts.min_time_window, "max_time_window": opts.max_time_window,
        "gamma": opts.gamma, "theta": opts.theta, "latest_end": opts.latest_end,
        "reaction_time": opts.reaction_time, "vehicle_capacity": opts.vehicle_capacity,
        "min_trips_required_lb": opts.min_trips_required_lb,
        "min_trips_required_ub": opts.min_trips_required_ub,
        "use_ortec": opts.use_ortec, "dimension_scale": opts.dimension_scale,
    }


def make_validation_datasets(problem, opts):
    """Use common held-out data without consuming the training RNG stream."""
    random_state = np.random.get_state()
    datasets = []
    try:
        np.random.seed(opts.validation_seed)
        for source in opts.val_datasets or (0.2, 0.5, 0.8):
            generated = not isinstance(source, str)
            options = training_dataset_options(opts)
            options.update(
                filename=None if generated else source,
                min_total=opts.max_total, max_total=opts.max_total,
                min_dod=source if generated else opts.min_dod,
                max_dod=source if generated else opts.max_dod,
                batch_size=opts.val_batch_size, num_samples=opts.val_size,
            )
            datasets.append(problem.make_dataset(**options))
    finally:
        np.random.set_state(random_state)
    return datasets


def training_epoch_seed(opts, epoch):
    """Keep epoch samples independent of model-dependent baseline updates."""
    seed = getattr(opts, "training_data_seed", None)
    if seed is None:
        return None
    return int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])


def make_training_dataset(problem, opts, epoch):
    options = dict(batch_size=opts.batch_size, num_samples=opts.epoch_size,
                   **training_dataset_options(opts))
    seed = training_epoch_seed(opts, epoch)
    if seed is None:
        return problem.make_dataset(**options)
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        return problem.make_dataset(**options)
    finally:
        np.random.set_state(random_state)

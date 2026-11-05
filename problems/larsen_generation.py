"""Larsen chapter 5 Poisson candidates with exact dynamism quotas."""

import numpy as np

from problems.utils import nearest_neighbor_graph, sample_lognorm


SCENARIOS = {"A": 20, "B": 30, "C": 40}
DYNAMISM_LEVELS = tuple(range(0, 101, 5))


def dynamism_level(advance, dynamic):
    """Return an exact multiple of 5 percent, or None; exclude empty instances."""
    total = advance + dynamic
    if total == 0 or (20 * dynamic) % total:
        return None
    return 100 * dynamic // total


def sample_instance(advance, dynamic, rng):
    """Generate normalized PDTRP data; customer counts exclude the depot.

    The unit square represents the stated 10-km square, so speed 4 units/hour
    corresponds to 40 km/hour. Dynamic times are continuous uniform on (0,T],
    with no one-minute offset or rounding to zero.
    """
    total = advance + dynamic
    coords = rng.uniform(0.0, 1.0, size=(total, 2))
    service = sample_lognorm(3.0, np.sqrt(5.0), total, rng=rng)
    calls = np.concatenate((np.zeros(advance), 480.0 * (1.0 - rng.random(dynamic))))
    order = np.argsort(calls, kind="stable")
    coords = np.concatenate(([[0.5, 0.5]], coords[order]))
    graph, distances = nearest_neighbor_graph(coords, -1, "fixed")
    return {
        "all_nodes": coords,
        "arrival_times": np.concatenate(([0.0], calls[order])),
        "service_times": np.concatenate(([0.0], service[order])),
        "graph": ~graph.astype(bool),
        "distance_matrix": distances,
        "speed": np.array([4.0 / 60.0]),
        "time_horizon": np.array([480.0]),
        "latest_end": np.array([120.0]),
        "service_times_mean": np.array([3.0]),
        "service_times_std": np.array([np.sqrt(5.0)]),
    }


def generate_balanced_instances(expected_customers, instances_per_dod=50, seed=20,
                               max_candidates=1_000_000):
    """Draw candidates until each exact dynamism group reaches its quota.

    Given advance count a, dynamic count is Poisson(expected_customers - a).
    Uniform sorted call times conditional on that count implement a homogeneous
    Poisson process. Quota rejection conditions the resulting dataset.
    Independent per-candidate feature seeds keep counts independent of feature
    generation and preserve each group's prefix when quotas are increased.
    """
    if expected_customers < 1 or int(expected_customers) != expected_customers:
        raise ValueError("expected_customers must be a positive integer")
    if instances_per_dod < 1 or max_candidates < 1:
        raise ValueError("quota and candidate limit must be positive")
    counts_rng = np.random.default_rng(seed)
    groups = {level: [] for level in DYNAMISM_LEVELS}
    rejected = {"empty": 0, "off_grid": 0, "quota_full": 0}
    for attempt in range(1, max_candidates + 1):
        advance = int(counts_rng.integers(0, expected_customers + 1))
        dynamic = int(counts_rng.poisson(expected_customers - advance))
        level = dynamism_level(advance, dynamic)
        if advance + dynamic == 0:
            rejected["empty"] += 1
        elif level is None:
            rejected["off_grid"] += 1
        elif len(groups[level]) == instances_per_dod:
            rejected["quota_full"] += 1
        else:
            feature_rng = np.random.default_rng(np.random.SeedSequence([seed, attempt, 1]))
            groups[level].append({
                "candidate_index": attempt, "advance_customers": advance,
                "dynamic_customers": dynamic, "customers": advance + dynamic,
                "dod_percent": level,
                "arrival_rate_per_hour": (expected_customers - advance) / 8.0,
                "sample": sample_instance(advance, dynamic, feature_rng),
            })
            if all(len(group) == instances_per_dod for group in groups.values()):
                return groups, {"candidates": attempt, "rejected": rejected}
    missing = {level: instances_per_dod - len(group) for level, group in groups.items()
               if len(group) < instances_per_dod}
    raise RuntimeError(f"Candidate limit reached before quotas were filled: {missing}")

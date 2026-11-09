"""ORTEC coordinate pools with the synthetic routing problem's other features."""

import hashlib
from pathlib import Path

import numpy as np

from problems.time_windows import generate_time_windows
from problems.utils import nearest_neighbor_graph, read_vrplib


def load_ortec_locations(filename, dimension_scale=1.0):
    """Fit the complete source map into a square without changing aspect ratio."""
    if not np.isfinite(dimension_scale) or dimension_scale <= 0:
        raise ValueError("dimension-scale must be finite and positive")
    path = Path(filename)
    coordinates = np.asarray(read_vrplib(path, rounded=False)["coords"], dtype=float)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2
            or not np.isfinite(coordinates).all()):
        raise ValueError("ORTEC locations require a depot and finite customer coordinates")
    origin = coordinates.min(axis=0)
    extent = coordinates.max(axis=0) - origin
    if extent.max() <= 0:
        raise ValueError("ORTEC source coordinates must span a nonzero area or line")
    scale = float(dimension_scale / extent.max())
    padding = (dimension_scale - extent * scale) / 2
    normalized = (coordinates - origin) * scale + padding
    _, distance = nearest_neighbor_graph(normalized, -1, "fixed")
    return {
        "filename": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "coords": normalized, "distance": distance, "locations_only": True,
        "normalization": {"method": "source_bounds_preserve_aspect_ratio",
                          "origin": origin.tolist(), "scale": scale,
                          "padding": padding.tolist(), "dimension_scale": dimension_scale},
    }


def make_ortec_location_dataset(problem, options, locations, seed):
    """Reuse synthetic features, then apply source geography before scoring.

    The fully dynamic base keeps service/demand draws independent of the target
    DoD, including for TW variants. Independent per-instance streams select
    source IDs, nested dynamic sets and windows, preserving prefixes and pairing.
    """
    dod = options["min_dod"]
    if dod != options["max_dod"] or not 0 <= dod <= 1:
        raise ValueError("Coordinate-only benchmarks require one fixed DoD in [0, 1]")
    base_options = {**options, "min_dod": 1.0, "max_dod": 1.0}
    previous_state = np.random.get_state()
    try:
        np.random.seed(seed)
        dataset = problem.make_dataset(**base_options)
    finally:
        np.random.set_state(previous_state)
    source_ids = []
    for index, original in enumerate(dataset.all_nodes):
        customers = len(original) - 1
        if customers >= len(locations["coords"]):
            raise ValueError(
                f"ORTEC source has only {len(locations['coords']) - 1} customers; requested {customers}"
            )
        location_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
        priority_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
        window_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 2]))
        selected = np.concatenate(([0], location_rng.permutation(len(locations["coords"]) - 1)[:customers] + 1))
        coordinates = locations["coords"][selected].copy()
        calls = np.asarray(dataset.arrival_times[index]).copy()
        dynamic = np.zeros(customers, dtype=bool)
        dynamic[priority_rng.permutation(customers)[:int(dod * customers)]] = True
        calls[1:][~dynamic] = 0.0
        dataset.all_nodes[index] = coordinates
        dataset.arrival_times[index] = calls
        if hasattr(dataset, "window_starts"):
            travel = np.linalg.norm(coordinates[1:] - coordinates[0], axis=1) / dataset.speed
            starts, ends = generate_time_windows(
                calls[1:], travel, dataset.min_time_window, dataset.max_time_window,
                dataset.time_horizon + dataset.latest_end, rng=window_rng,
            )
            dataset.window_starts[index] = np.concatenate(([0.0], starts))
            dataset.window_ends[index] = np.concatenate(([0.0], ends))
        source_ids.append(selected.tolist())
    dataset.min_dod = dataset.max_dod = dod
    return dataset, source_ids

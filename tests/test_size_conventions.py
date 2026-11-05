import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

import generate_data
from options import get_options
from problems import PDCTRP, PDCTRPTW, PDTRP, PDTRPTW
from problems.size_conventions import resolve_node_bounds
from utils.functions import load_args


PROBLEMS = (PDTRP, PDCTRP, PDTRPTW, PDCTRPTW)


def dataset_options():
    return dict(batch_size=1, num_samples=1, n_subregions=1, neighbors=-1, knn_strat="fixed")


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("dod,expected_dynamic", [(0.2, 10), (0.5, 25), (0.8, 40), (1.0, 50)])
def test_n50_means_fifty_customers_and_one_depot(problem, dod, expected_dynamic):
    dataset = problem.make_dataset(
        min_customers=50, max_customers=50, min_dod=dod, max_dod=dod, **dataset_options()
    )
    sample = dataset[0]
    assert dataset.min_customers == dataset.max_customers == 50
    assert dataset.min_total == dataset.max_total == 51
    assert sample["all_nodes"].shape == (51, 2)
    assert sample["distance_matrix"].shape == (51, 51)
    assert sample["arrival_times"][0] == 0
    assert sample["arrival_times"].gt(0).sum() == expected_dynamic


@pytest.mark.parametrize("problem", PROBLEMS)
def test_explicit_legacy_bounds_reproduce_original_instance_sizes(problem):
    np.random.seed(97)
    old = problem.make_dataset(min_total=50, max_total=50, **dataset_options())[0]
    np.random.seed(97)
    new = problem.make_dataset(min_customers=49, max_customers=49, **dataset_options())[0]
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert len(old["all_nodes"]) == 50


@pytest.mark.parametrize("problem", PROBLEMS)
def test_dataset_customer_bounds_cannot_be_mixed_with_legacy_node_bounds(problem):
    with pytest.raises(ValueError, match="not both"):
        problem.make_dataset(min_customers=50, max_total=50, **dataset_options())


@pytest.mark.parametrize("bounds", [dict(min_customers=0), dict(min_customers=5, max_customers=4),
                                   dict(min_total=1), dict(min_customers=4.5)])
def test_invalid_size_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="Size bounds"):
        resolve_node_bounds(**bounds)


def test_training_options_use_customer_counts_and_customer_directory_names():
    opts = get_options(["--min-customers", "50", "--max-customers", "100"])
    assert (opts.min_customers, opts.max_customers) == (50, 100)
    assert (opts.min_total, opts.max_total) == (51, 101)
    assert opts.size_convention == "customers_excluding_depot"
    assert "/pdtrp_50-100-" in opts.save_dir
    defaults = get_options([])
    assert (defaults.min_customers, defaults.max_customers) == (40, 100)
    assert (defaults.min_total, defaults.max_total) == (41, 101)


def test_training_legacy_flags_still_explicitly_count_nodes():
    opts = get_options(["--min_total", "50", "--max_total", "100"])
    assert (opts.min_customers, opts.max_customers) == (49, 99)
    assert (opts.min_total, opts.max_total) == (50, 100)
    with pytest.raises(SystemExit):
        get_options(["--min-customers", "50", "--max_total", "100"])


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_metadata_and_resume_preserve_saved_sizes(tmp_path, legacy):
    args = {"min_total": 20, "max_total": 50}
    if not legacy:
        args.update(min_customers=20, max_customers=50, min_total=21, max_total=51,
                    size_convention="customers_excluding_depot")
    path = tmp_path / "args.json"
    path.write_text(json.dumps(args))
    loaded = load_args(path)
    assert loaded["min_customers"] == (19 if legacy else 20)
    assert loaded["max_customers"] == (49 if legacy else 50)
    assert loaded["size_convention"] == (
        "total_nodes_including_depot" if legacy else "customers_excluding_depot"
    )
    resumed = get_options(["--resume", str(tmp_path / "epoch-0.pt")])
    assert (resumed.min_total, resumed.max_total) == (args["min_total"], args["max_total"])
    assert path.read_text() == json.dumps(args)


def test_benchmark_default_and_explicit_legacy_node_sizes(tmp_path):
    default = generate_data.parse_args(["--output", str(tmp_path / "new.txt")])
    assert (default.num_customers, default.num_nodes) == (50, 51)
    legacy = generate_data.parse_args(["--num-nodes", "50", "--output", str(tmp_path / "old.txt")])
    assert (legacy.num_customers, legacy.num_nodes) == (49, 50)
    result = generate_data.generate(generate_data.parse_args([
        "--num-nodes", "50", "--num-samples", "1", "--output", str(tmp_path / "old.txt")
    ]))
    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["num_customers"] == 49
    assert metadata["num_nodes_including_depot"] == 50


def test_training_wrapper_passes_customer_count_flags(tmp_path):
    stub = tmp_path / "python"
    captured = tmp_path / "arguments.txt"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$PDTSP_CAPTURE"\n')
    stub.chmod(0o755)
    import os
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                       PDTSP_CAPTURE=str(captured))
    subprocess.run(["bash", "scripts/train-policy.sh", "pdtrp", "50", "0"],
                   check=True, env=environment)
    arguments = captured.read_text().splitlines()
    assert arguments[arguments.index("--max-customers") + 1] == "50"
    assert arguments[arguments.index("--min-customers") + 1] == "20"
    assert "--max_total" not in arguments

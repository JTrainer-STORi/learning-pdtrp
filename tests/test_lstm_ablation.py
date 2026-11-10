import csv
import json

import pytest

from compare_lstm_ablation import compare_runs


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def pair(tmp_path):
    roots = []
    for lstm, values in ((False, [10., 20.]), (True, [9., 18.])):
        root = tmp_path / ("on" if lstm else "off")
        root.mkdir()
        roots.append(root)
        (root / "training-args.json").write_text(json.dumps(dict(
            problem="pdctrptw", use_arrival_lstm=lstm, seed=0, max_customers=50,
            training_data_seed=0, n_epochs=2, run_name=root.name,
        )))
        (root / "manifest.json").write_text(json.dumps(dict(
            problem="pdctrptw", checkpoint=f"/{root.name}/epoch-1.pt", dataset_parameters={"time_horizon": 10},
            evaluation_seed=12346, offset=0, num_samples_requested=2,
            datasets=[{"dataset": "/shared/check.txt", "samples": 2}],
        )))
        write_csv(root / "training-history.csv", [dict(
            epoch=i, epoch_data_seed=123 + i, epoch_seconds=2, peak_gpu_allocated_bytes=100,
        ) for i in range(2)])
        write_csv(root / "check.samples.csv", [dict(
            sample=i, objective=value, runtime_seconds=.1,
        ) for i, value in enumerate(values)])
    return roots


def test_comparison_preserves_pairs_and_computes_the_ratio_of_means(pair, tmp_path):
    summary = compare_runs(*pair, tmp_path / "comparison")
    case, = summary["datasets"]
    assert case["mean_without"] == 15
    assert case["mean_with"] == 13.5
    assert case["change_percent"] == pytest.approx(-10)
    assert case["lstm_wins"] == 2
    with (tmp_path / "comparison/paired-results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["change"]) for row in rows] == [-1, -2]


@pytest.mark.parametrize("field,value", [("seed", 1), ("n_epochs", 3), ("training_data_seed", 7)])
def test_comparison_rejects_different_training_conditions(pair, tmp_path, field, value):
    path = pair[1] / "training-args.json"
    args = json.loads(path.read_text())
    args[field] = value
    path.write_text(json.dumps(args))
    with pytest.raises(ValueError, match="Training settings differ"):
        compare_runs(*pair, tmp_path / "comparison")


def test_comparison_rejects_unpaired_examples_and_different_data_streams(pair, tmp_path):
    write_csv(pair[1] / "check.samples.csv", [dict(sample=5, objective=9, runtime_seconds=.1)])
    with pytest.raises(ValueError, match="Sample indices differ"):
        compare_runs(*pair, tmp_path / "comparison")
    write_csv(pair[1] / "training-history.csv", [dict(
        epoch=i, epoch_data_seed=999, epoch_seconds=2, peak_gpu_allocated_bytes=100,
    ) for i in range(2)])
    with pytest.raises(ValueError, match="per-epoch data seeds must match"):
        compare_runs(*pair, tmp_path / "comparison")


def test_comparison_rejects_a_checkpoint_from_an_earlier_epoch(pair, tmp_path):
    path = pair[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint"] = "/on/epoch-0.pt"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="final checkpoint"):
        compare_runs(*pair, tmp_path / "comparison")

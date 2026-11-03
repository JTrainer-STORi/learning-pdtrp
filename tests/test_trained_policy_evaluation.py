"""Check sample selection through real dataset loaders and policy output files."""

import csv
import json

import pytest
import torch

import evaluate_trained_policy as policy


def dataset_row(problem, marker):
    row = f"0 0 {marker} 0 0 1 arrival_times 0 0 0 service_times 0 0 0 "
    if problem in ("pdtrptw", "pdctrptw"):
        row += "window_starts 0 0 0 window_ends 0 1000 1000 "
    if problem in ("pdctrp", "pdctrptw"):
        row += "demands 0 0.4 0.4 "
    return row + "tour 1 2 3 1 visit_times 0 15 36.2132 51.2132\n"


class RecordingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def forward(self, inputs, mode):
        assert mode == "recursive"
        marker = inputs["all_nodes"][:, 1, 0]
        self.seen.extend(marker.tolist())
        return marker, None, None


def run_evaluation(tmp_path, monkeypatch, problem, datasets, count, offset=0):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    model = RecordingPolicy()
    monkeypatch.setattr(policy, "load_model", lambda path: (model, {"seed": 0}))
    output = tmp_path / "results"
    policy.main([
        "--problem", problem, "--checkpoint", str(checkpoint),
        "--datasets", *map(str, datasets), "--output-dir", str(output),
        "--device", "cpu", "--num-samples", str(count), "--offset", str(offset),
    ])
    return model, output


@pytest.mark.parametrize("problem", ["pdtrp", "pdctrp", "pdtrptw", "pdctrptw"])
@pytest.mark.parametrize("offset", [0, 1])
def test_sample_selection_in_each_dataset(tmp_path, monkeypatch, problem, offset):
    datasets = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for marker, dataset in enumerate(datasets, start=1):
        # Reading beyond the requested count would fail in the actual loader.
        dataset.write_text(
            "skipped malformed row\n" * offset
            + dataset_row(problem, marker)
            + "unselected malformed row\n"
        )

    model, output = run_evaluation(tmp_path, monkeypatch, problem, datasets, count=1, offset=offset)

    assert model.seen == [1.0, 2.0]
    assert model.decode_type == "greedy"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["num_samples_requested"] == 1
    assert manifest["offset"] == offset
    assert len(manifest["datasets"]) == 2
    for marker, dataset in enumerate(datasets, start=1):
        summary = json.loads((output / f"{dataset.stem}.summary.json").read_text())
        assert summary["samples"] == summary["num_samples_requested"] == 1
        assert summary["offset"] == offset
        with (output / f"{dataset.stem}.samples.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert int(rows[0]["sample"]) == offset
        assert float(rows[0]["objective"]) == marker


@pytest.mark.parametrize("offset", [0, 1])
def test_sample_limit_larger_than_file_reports_actual_count(tmp_path, monkeypatch, offset):
    dataset = tmp_path / "short.txt"
    dataset.write_text("skipped malformed row\n" * offset + dataset_row("pdtrp", 1))

    model, output = run_evaluation(tmp_path, monkeypatch, "pdtrp", [dataset], count=3, offset=offset)

    assert model.seen == [1.0]
    summary = json.loads((output / "short.summary.json").read_text())
    assert summary["samples"] == 1
    assert summary["num_samples_requested"] == 3
    assert summary["offset"] == offset


def test_multiple_samples_keep_original_indices(tmp_path, monkeypatch):
    dataset = tmp_path / "data.txt"
    dataset.write_text("".join(dataset_row("pdtrp", marker) for marker in (1, 2, 3, 4)))

    model, output = run_evaluation(tmp_path, monkeypatch, "pdtrp", [dataset], count=2, offset=1)

    assert model.seen == [2.0, 3.0]
    with (output / "data.samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["sample"]) for row in rows] == [1, 2]


def test_offset_past_end_rejects_empty_selection(tmp_path, monkeypatch):
    dataset = tmp_path / "data.txt"
    dataset.write_text(dataset_row("pdtrp", 1))

    with pytest.raises(ValueError, match="No samples found"):
        run_evaluation(tmp_path, monkeypatch, "pdtrp", [dataset], count=1, offset=1)
    assert not (tmp_path / "results" / "manifest.json").exists()


@pytest.mark.parametrize("dataset_count", [1, 2])
def test_cli_prints_result_dictionary_matching_return_value_and_manifest(
    tmp_path, monkeypatch, capsys, dataset_count
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    datasets = [tmp_path / f"data-{index}.txt" for index in range(dataset_count)]
    for dataset in datasets:
        dataset.write_text(dataset_row("pdtrp", 1))

    def load_model(path):
        print(f"Loading model from {path}")
        return RecordingPolicy(), {"seed": 0}

    monkeypatch.setattr(policy, "load_model", load_model)
    output = tmp_path / "results"
    returned = policy.main([
        "--problem", "pdtrp", "--checkpoint", str(checkpoint),
        "--datasets", *map(str, datasets), "--output-dir", str(output),
        "--device", "cpu", "--num-samples", "1",
    ])

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result == returned == json.loads((output / "manifest.json").read_text())
    assert result["problem"] == "pdtrp"
    assert result["offset"] == 0
    assert result["num_samples_requested"] == 1
    assert len(result["datasets"]) == dataset_count
    assert "Loading model from" in captured.err
    assert "Evaluating" in captured.err
    for summary in result["datasets"]:
        assert summary["samples"] == 1
        assert summary["mean_objective"] == 1.0
        assert summary["mean_runtime_seconds"] >= 0


@pytest.mark.parametrize("count", ["0", "-1", "invalid"])
def test_cli_rejects_invalid_sample_count(count):
    with pytest.raises(SystemExit):
        policy.parse_args([
            "--problem", "pdtrp", "--checkpoint", "model.pt", "--datasets", "data.txt",
            "--output-dir", "results", "--num-samples", count,
        ])


@pytest.mark.parametrize("offset", ["-1", "-5", "invalid"])
def test_cli_rejects_invalid_offset(offset):
    with pytest.raises(SystemExit):
        policy.parse_args([
            "--problem", "pdtrp", "--checkpoint", "model.pt", "--datasets", "data.txt",
            "--output-dir", "results", "--offset", offset,
        ])


def test_cli_preserves_previous_sample_limit_by_default():
    args = policy.parse_args([
        "--problem", "pdtrp", "--checkpoint", "model.pt", "--datasets", "data.txt",
        "--output-dir", "results",
    ])
    assert args.num_samples == 12800
    assert args.offset == 0

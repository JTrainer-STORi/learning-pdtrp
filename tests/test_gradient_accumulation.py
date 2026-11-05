from types import SimpleNamespace

import pytest
import torch

import train
from options import get_options
from reinforce_baselines import NoBaseline


class LinearPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.2]))

    def forward(self, features, **kwargs):
        return torch.ones(len(features)), features @ self.weight, {}


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("accumulation_steps", [1, 2, 3])
@pytest.mark.parametrize("num_batches", [1, 4, 5])
@pytest.mark.parametrize("max_grad_norm", [0, 1.0])
def test_accumulation_matches_combined_batch_updates(
    monkeypatch, optimizer_class, accumulation_steps, num_batches, max_grad_norm
):
    model, reference = LinearPolicy(), LinearPolicy()
    optimizer = optimizer_class(model.parameters(), lr=0.1)
    reference_optimizer = optimizer_class(reference.parameters(), lr=0.1)
    # Opposing gradients reveal premature clipping as well as wrong boundaries.
    batches = torch.tensor([
        [[8.0, 4.0], [4.0, 2.0]],
        [[-6.0, -2.0], [-2.0, 0.0]],
        [[3.0, -5.0], [1.0, -3.0]],
        [[-5.0, 7.0], [-3.0, 5.0]],
        [[2.0, 6.0], [4.0, 8.0]],
    ])[:num_batches]
    opts = SimpleNamespace(
        pomo_batch_size=1, problem="pdtrp", device="cpu", print_query_times=False,
        accumulation_steps=accumulation_steps, max_grad_norm=max_grad_norm, log_step=1,
    )
    monkeypatch.setattr(train, "dataset_to_input", lambda batch, *args: batch)
    monkeypatch.setattr(train, "log_values", lambda *args: None)
    clip_batches = []
    original_clip = train.clip_grad_norms

    def record_clip(groups, max_norm):
        clip_batches.append(batch_id)
        return original_clip(groups, max_norm)

    monkeypatch.setattr(train, "clip_grad_norms", record_clip)
    for batch_id, batch in enumerate(batches):
        # The logging step deliberately differs from the epoch-local batch ID.
        train.train_batch(
            model, optimizer, NoBaseline(), 2, batch_id, 101 + batch_id, batch,
            None, None, opts, num_batches=num_batches,
        )
        if (batch_id + 1) % accumulation_steps == 0 or batch_id + 1 == num_batches:
            start = (batch_id // accumulation_steps) * accumulation_steps
            features = batches[start:batch_id + 1].flatten(0, 1)
            costs, likelihood, _ = reference(features)
            (costs * likelihood).mean().backward()
            original_clip(reference_optimizer.param_groups, max_grad_norm)
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            torch.testing.assert_close(model.weight, reference.weight)
            assert all(p.grad is None or p.grad.eq(0).all() for p in model.parameters())

    expected = [
        i for i in range(num_batches)
        if (i + 1) % accumulation_steps == 0 or i + 1 == num_batches
    ]
    assert clip_batches == expected


@pytest.mark.parametrize("value", ["0", "-1"])
def test_accumulation_steps_must_be_positive(value):
    with pytest.raises(SystemExit):
        get_options(["--accumulation_steps", value])

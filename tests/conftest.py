import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_randomness():
    """Keep generated problem instances repeatable across test runs."""
    np.random.seed(1234)
    torch.manual_seed(1234)


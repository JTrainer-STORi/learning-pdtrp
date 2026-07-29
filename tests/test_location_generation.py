import numpy as np
import pytest

from problems.location_generation import (
    subregion_generation,
    time_and_subregion_generation,
)


def test_uniform_generation_has_requested_shape_and_bounds():
    locations = subregion_generation(n_total=100, n_subregions=1)

    assert locations.shape == (100, 2)
    assert np.all((locations >= 0) & (locations <= 1))


@pytest.mark.parametrize(
    ("skew", "expected_side"),
    [("early", "early"), ("uniform", "middle"), ("late", "late")],
)
def test_time_and_location_generation_respects_bounds(skew, expected_side):
    locations, arrival_times = time_and_subregion_generation(
        n_total=200,
        n_subregions=1,
        arrival_weights=[1],
        arrival_skews=[skew],
        time_horizon=480,
        dimension_scale=2,
    )

    assert locations.shape == (200, 2)
    assert arrival_times.shape == (200,)
    assert np.all((locations >= 0) & (locations <= 2))
    assert np.all((arrival_times >= 1) & (arrival_times <= 480))

    normalized_mean = (arrival_times.mean() - 1) / 479
    if expected_side == "early":
        assert normalized_mean < 0.4
    elif expected_side == "late":
        assert normalized_mean > 0.6
    else:
        assert 0.4 < normalized_mean < 0.6


def test_weighted_subregions_place_all_customers_in_selected_region():
    locations = subregion_generation(
        n_total=40,
        n_subregions=4,
        arrival_weights=[1, 0, 0, 0],
    )

    assert np.all((locations[:, 0] >= 0) & (locations[:, 0] <= 0.5))
    assert np.all((locations[:, 1] >= 0) & (locations[:, 1] <= 0.5))


@pytest.mark.parametrize("n_subregions", [0, 2, 3, 5])
def test_subregions_must_form_square_grid(n_subregions):
    with pytest.raises(ValueError, match="perfect square"):
        subregion_generation(n_total=10, n_subregions=n_subregions)


def test_generation_rejects_invalid_weights_and_skews():
    with pytest.raises(ValueError, match="arrival_weights"):
        subregion_generation(n_total=10, n_subregions=4, arrival_weights=[1, 1])
    with pytest.raises(ValueError, match="Unsupported arrival skew"):
        time_and_subregion_generation(
            n_total=10,
            n_subregions=1,
            arrival_skews=["sometimes"],
        )

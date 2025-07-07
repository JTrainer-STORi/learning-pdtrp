"""Testing Suite for the various problems and their variants."""

import matplotlib.pyplot as plt

# Testing utils common to all problems

from problems.location_generation import subregion_generation

# Test that the subregion generation function is uniform 

def test_uniform_generation():
    """
    Test that the subregion generation function generates points uniformly in the unit square.
    """
    n_total = 100
    n_subregions = 1
    arrival_weights = None

    locations = subregion_generation(n_total, n_subregions, arrival_weights)

    assert locations.shape == (n_total, 2), "Generated locations shape mismatch"

    # Plot the generated locations for visual inspection
    plt.scatter(locations[:, 0], locations[:, 1], alpha=0.5)
    plt.show()


from problems.pdtrp.problem_pdtrp import PDTRP, PDTRPDataset

# Tests for the PDTRP problem

if __name__ == "__main__":    
    # Run the uniform generation test
    test_uniform_generation()
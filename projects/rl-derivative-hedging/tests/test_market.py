import numpy as np
from rl_hedging.market import simulate_gbm


def test_shape_and_initial_spot():
    paths = simulate_gbm(100, 0, 0.2, 1.0, 12, 50, seed=4)
    assert paths.shape == (50, 13)
    assert np.all(paths[:, 0] == 100)


def test_reproducible_seed():
    a = simulate_gbm(100, 0, 0.2, 1.0, 12, 5, seed=9)
    b = simulate_gbm(100, 0, 0.2, 1.0, 12, 5, seed=9)
    assert np.allclose(a, b)

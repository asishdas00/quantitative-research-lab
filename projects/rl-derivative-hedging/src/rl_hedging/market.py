"""Geometric Brownian motion scenario generation."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_gbm(spot: float, drift: float, volatility: float, maturity: float,
                 steps: int, paths: int, seed: int | None = None) -> NDArray[np.float64]:
    """Return price paths with shape (paths, steps + 1), including initial spot."""
    if spot <= 0 or maturity <= 0 or steps < 1 or paths < 1:
        raise ValueError("spot, maturity, steps and paths must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    z = rng.standard_normal((paths, steps))
    increments = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
    log_returns = np.concatenate(
        [np.zeros((paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return (spot * np.exp(log_returns)).astype(np.float64)

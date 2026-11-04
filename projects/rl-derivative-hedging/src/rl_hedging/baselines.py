"""Classical hedging policies."""
import numpy as np
from .pricing import call_delta


def delta_target(spot: float, strike: float, tau: float, rate: float,
                 volatility: float, dividend: float = 0.0) -> float:
    return call_delta(spot, strike, tau, rate, volatility, dividend)


def no_hedge_target(*args, **kwargs) -> float:
    return 0.0

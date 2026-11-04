"""Black-Scholes pricing helpers for European calls."""
from __future__ import annotations

import math
from scipy.stats import norm


def _validate(spot: float, strike: float, maturity: float, volatility: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity cannot be negative")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")


def call_price(spot: float, strike: float, maturity: float, rate: float,
               volatility: float, dividend: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European call."""
    _validate(spot, strike, maturity, volatility)
    if maturity == 0:
        return max(spot - strike, 0.0)
    if volatility == 0:
        return max(spot * math.exp(-dividend * maturity)
                   - strike * math.exp(-rate * maturity), 0.0)
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike)
          + (rate - dividend + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    return (spot * math.exp(-dividend * maturity) * norm.cdf(d1)
            - strike * math.exp(-rate * maturity) * norm.cdf(d2))


def call_delta(spot: float, strike: float, maturity: float, rate: float,
               volatility: float, dividend: float = 0.0) -> float:
    """Delta of a European call with respect to the underlying spot."""
    _validate(spot, strike, maturity, volatility)
    if maturity == 0:
        return 1.0 if spot > strike else (0.0 if spot < strike else 0.5)
    if volatility == 0:
        forward = spot * math.exp((rate - dividend) * maturity)
        return math.exp(-dividend * maturity) if forward > strike else 0.0
    d1 = (math.log(spot / strike)
          + (rate - dividend + 0.5 * volatility**2) * maturity) / (volatility * math.sqrt(maturity))
    return math.exp(-dividend * maturity) * norm.cdf(d1)

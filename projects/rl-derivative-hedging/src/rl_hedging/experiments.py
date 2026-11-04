"""Common-path simulation and policy evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import delta_target, no_hedge_target
from .metrics import summarize_pnl


def evaluate_rule(volatility: float, paths: np.ndarray, policy: str,
                  transaction_cost: float, rate: float = 0.0,
                  strike: float = 100.0, maturity: float = 1.0) -> dict:
    """Evaluate a deterministic hedge rule on supplied common price paths."""
    pnl, costs = [], []
    steps = paths.shape[1] - 1
    dt = maturity / steps
    from .pricing import call_price

    for path in paths:
        premium = call_price(path[0], strike, maturity, rate, volatility)
        cash, inventory, total_cost = premium, 0.0, 0.0
        for t in range(steps):
            tau = maturity - t * dt
            if policy == "delta":
                target = delta_target(path[t], strike, tau, rate, volatility)
            elif policy == "none":
                target = no_hedge_target()
            else:
                raise ValueError(f"unknown policy: {policy}")
            trade = target - inventory
            cost = transaction_cost * path[t] * abs(trade)
            cash -= trade * path[t] + cost
            total_cost += cost
            inventory = target
        terminal = cash + inventory * path[-1] - max(path[-1] - strike, 0.0)
        pnl.append(terminal)
        costs.append(total_cost)
    return summarize_pnl(np.asarray(pnl), np.asarray(costs))

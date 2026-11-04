"""Evaluate trained SB3 policies and classical baselines on held-out GBM paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .experiments import evaluate_rule
from .market import simulate_gbm
from .metrics import summarize_pnl
from .pricing import call_price


def evaluate_model(model, paths: np.ndarray, volatility: float, transaction_cost: float,
                   rate: float = 0.0, strike: float = 100.0,
                   maturity: float = 1.0, max_inventory: float = 2.0,
                   deterministic: bool = True) -> dict[str, float]:
    """Evaluate an SB3 model on fixed price paths; action sees current state only."""
    pnl_values, cost_values = [], []
    n_steps = paths.shape[1] - 1
    dt = maturity / n_steps
    for path in paths:
        cash = call_price(path[0], strike, maturity, rate, volatility)
        inventory = 0.0
        total_cost = 0.0
        last_trade = 0.0
        for t in range(n_steps):
            tau = maturity - t * dt
            obs = np.array([path[t] / path[0], strike / path[0], tau / maturity,
                            volatility, inventory / max_inventory,
                            last_trade / max_inventory], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=deterministic)
            # SB3's DQN emits an integer; PPO/SAC emit a one-element float array.
            if np.isscalar(action) or np.asarray(action).ndim == 0:
                grid = np.linspace(-max_inventory, max_inventory, 9)
                target = float(grid[int(action)])
            else:
                target = float(np.clip(np.asarray(action).reshape(-1)[0], -max_inventory, max_inventory))
            trade = target - inventory
            cost = transaction_cost * path[t] * abs(trade)
            cash -= trade * path[t] + cost
            total_cost += cost
            inventory = target
            last_trade = trade
        pnl_values.append(cash + inventory * path[-1] - max(path[-1] - strike, 0.0))
        cost_values.append(total_cost)
    return summarize_pnl(np.asarray(pnl_values), np.asarray(cost_values))


def run_regime_benchmark(episodes: int = 2000, steps: int = 30, seed: int = 100,
                         transaction_cost: float = 0.001) -> pd.DataFrame:
    """Generate common held-out paths and evaluate no hedge/delta hedge regimes."""
    rows = []
    for vol in (0.10, 0.20, 0.40):
        paths = simulate_gbm(100.0, 0.0, vol, 1.0, steps, episodes, seed + int(vol * 1000))
        for policy in ("none", "delta"):
            rows.append({"volatility": vol, "policy": policy,
                         **evaluate_rule(vol, paths, policy, transaction_cost)})
    return pd.DataFrame(rows)

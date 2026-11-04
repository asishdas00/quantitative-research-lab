"""Risk metrics for terminal P&L (higher is better)."""
from __future__ import annotations

import numpy as np


def summarize_pnl(pnl: np.ndarray, costs: np.ndarray, alpha: float = 0.95) -> dict[str, float]:
    pnl = np.asarray(pnl, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if pnl.size == 0:
        raise ValueError("pnl cannot be empty")
    # Loss is -P&L; VaR/CVaR are reported as positive loss thresholds.
    losses = -pnl
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return {
        "episodes": float(pnl.size),
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0,
        "pnl_5pct": float(np.quantile(pnl, 0.05)),
        "loss_var_95": var,
        "loss_cvar_95": float(np.mean(tail)) if tail.size else var,
        "mean_transaction_cost": float(np.mean(costs)),
        "mean_absolute_loss": float(np.mean(np.abs(losses))),
    }

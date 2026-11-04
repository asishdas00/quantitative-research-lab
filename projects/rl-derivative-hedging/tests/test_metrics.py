import numpy as np
from rl_hedging.metrics import summarize_pnl


def test_metrics_are_finite_for_simple_sample():
    result = summarize_pnl(np.array([1.0, 0.0, -1.0]), np.array([0.1, 0.2, 0.3]))
    assert result["episodes"] == 3.0
    assert np.isclose(result["mean_pnl"], 0.0)
    assert result["mean_transaction_cost"] > 0

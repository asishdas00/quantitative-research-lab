#!/usr/bin/env python
"""Run no-hedge and Black-Scholes delta benchmarks across volatility regimes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rl_hedging.market import simulate_gbm
from rl_hedging.experiments import evaluate_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cost", type=float, default=0.001)
    args = parser.parse_args()
    rows = []
    for sigma in (0.10, 0.20, 0.40):
        paths = simulate_gbm(100.0, 0.0, sigma, 1.0, args.steps,
                             args.episodes, seed=args.seed)
        for policy in ("none", "delta"):
            metrics = evaluate_rule(sigma, paths, policy, args.cost)
            rows.append({"volatility": sigma, "policy": policy, **metrics})
    out = Path("results")
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "baseline_metrics.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

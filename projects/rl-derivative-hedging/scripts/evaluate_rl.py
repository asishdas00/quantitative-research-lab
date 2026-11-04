#!/usr/bin/env python
"""Evaluate a saved RL model on held-out paths and compare with baselines."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from rl_hedging.evaluation import evaluate_model
from rl_hedging.experiments import evaluate_rule
from rl_hedging.market import simulate_gbm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["PPO", "SAC", "DQN"], required=True)
    parser.add_argument("--model", required=True, help="Path to saved Stable-Baselines3 model (omit .zip)")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--cost", type=float, default=0.001)
    args = parser.parse_args()

    from stable_baselines3 import PPO, SAC, DQN
    cls = {"PPO": PPO, "SAC": SAC, "DQN": DQN}[args.algo]
    model = cls.load(args.model)
    rows = []
    for vol in (0.10, 0.20, 0.40):
        paths = simulate_gbm(100.0, 0.0, vol, 1.0, args.steps,
                             args.episodes, args.seed + int(vol * 1000))
        for policy in ("none", "delta"):
            rows.append({"volatility": vol, "policy": policy,
                         **evaluate_rule(vol, paths, policy, args.cost)})
        rows.append({"volatility": vol, "policy": args.algo,
                     **evaluate_model(model, paths, vol, args.cost)})
    out = Path("results")
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    target = out / f"{args.algo.lower()}_heldout_comparison.csv"
    frame.to_csv(target, index=False)
    print(frame.to_string(index=False))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()

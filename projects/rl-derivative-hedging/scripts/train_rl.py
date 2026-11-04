#!/usr/bin/env python
"""Train a Stable-Baselines3 policy in the hedging environment."""
from __future__ import annotations

import argparse
from pathlib import Path

from rl_hedging.env import HedgingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["PPO", "SAC", "DQN"], default="PPO")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--volatility", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, SAC, DQN
    except ImportError as exc:
        raise SystemExit('Install optional dependencies with: pip install -e ".[rl]"') from exc

    continuous = args.algo != "DQN"
    env = HedgingEnv(volatility=args.volatility, continuous=continuous, seed=args.seed)
    algorithm = {"PPO": PPO, "SAC": SAC, "DQN": DQN}[args.algo]
    model = algorithm("MlpPolicy", env, verbose=1, seed=args.seed,
                      learning_rate=3e-4, gamma=1.0)
    model.learn(total_timesteps=args.timesteps)
    output = Path("models")
    output.mkdir(exist_ok=True)
    model.save(output / f"{args.algo.lower()}_sigma_{args.volatility:.2f}")
    print(f"Saved model to {output.resolve()}")


if __name__ == "__main__":
    main()

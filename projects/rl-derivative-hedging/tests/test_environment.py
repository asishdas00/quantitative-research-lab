import numpy as np
from rl_hedging.env import HedgingEnv


def test_environment_contract_and_termination():
    env = HedgingEnv(steps=3, seed=2)
    obs, info = env.reset(seed=2)
    assert env.observation_space.contains(obs)
    total_reward = 0.0
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        total_reward += reward
    assert terminated and not truncated
    assert np.isfinite(env.terminal_pnl())
    assert abs(total_reward - (env.terminal_pnl() - env.initial_premium)) < 1e-6


def test_transaction_cost_is_charged_on_trade():
    env = HedgingEnv(steps=2, transaction_cost=0.01, seed=1)
    env.reset(seed=1)
    _, _, _, _, first = env.step(np.array([1.0], dtype=np.float32))
    assert first["transaction_cost"] > 0
    _, _, _, _, second = env.step(np.array([1.0], dtype=np.float32))
    assert second["transaction_cost"] == 0.0

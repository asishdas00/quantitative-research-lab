"""Gymnasium environment for hedging one short European call."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .pricing import call_delta, call_price


class HedgingEnv(gym.Env):
    """Choose a target stock inventory; cash and transaction costs are accounted for.

    Positive inventory means long shares. The option liability is one short call.
    Reward is incremental change in marked portfolio value net of trading costs;
    the final step also deducts the call payoff.
    """

    metadata = {"render_modes": []}

    def __init__(self, spot: float = 100.0, strike: float = 100.0,
                 maturity: float = 1.0, volatility: float = 0.20,
                 rate: float = 0.0, dividend: float = 0.0, drift: float | None = None,
                 steps: int = 30, transaction_cost: float = 0.001,
                 max_inventory: float = 2.0, continuous: bool = True,
                 seed: int | None = None):
        super().__init__()
        if spot <= 0 or strike <= 0 or maturity <= 0 or steps < 1:
            raise ValueError("invalid instrument or horizon parameters")
        if volatility < 0 or transaction_cost < 0 or max_inventory <= 0:
            raise ValueError("volatility/cost must be nonnegative; inventory limit positive")
        self.spot0, self.strike, self.maturity = float(spot), float(strike), float(maturity)
        self.volatility, self.rate, self.dividend = float(volatility), float(rate), float(dividend)
        self.drift = float(rate if drift is None else drift)
        self.steps, self.dt = int(steps), float(maturity) / int(steps)
        if continuous and max_inventory < 1.0:
            raise ValueError("continuous inventory limit must allow a 1-share delta hedge")
        self._discrete_grid = np.linspace(-float(max_inventory), float(max_inventory), 9)
        self.cost_rate, self.max_inventory = float(transaction_cost), float(max_inventory)
        self.continuous = continuous
        self._seed = seed

        self.action_space = (spaces.Box(low=np.array([-max_inventory], dtype=np.float32),
                                        high=np.array([max_inventory], dtype=np.float32),
                                        dtype=np.float32) if continuous
                             else spaces.Discrete(9))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -max_inventory, -1.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, np.inf, max_inventory, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def _obs(self) -> np.ndarray:
        tau = max(self.maturity - self.t * self.dt, 0.0)
        return np.array([
            self.spot / self.spot0,
            self.strike / self.spot0,
            tau / self.maturity,
            self.volatility,
            self.inventory,
            self.last_trade / self.max_inventory,
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.spot = self.spot0
        self.inventory = 0.0
        self.last_trade = 0.0
        self.cash = call_price(self.spot0, self.strike, self.maturity,
                               self.rate, self.volatility, self.dividend)
        self.initial_premium = self.cash
        self.total_cost = 0.0
        self._terminated = False
        return self._obs(), {"premium": self.initial_premium}

    def _target_inventory(self, action) -> float:
        if self.continuous:
            target = float(np.asarray(action, dtype=float).reshape(-1)[0])
        else:
            action_index = int(action)
            if not self.action_space.contains(action_index):
                raise ValueError("discrete action must be an integer in [0, 8]")
            target = float(self._discrete_grid[action_index])
        return float(np.clip(target, -self.max_inventory, self.max_inventory))

    def step(self, action):
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset()")
        old_mark = self.cash + self.inventory * self.spot
        target = self._target_inventory(action)
        trade = target - self.inventory
        transaction_cost = self.cost_rate * self.spot * abs(trade)
        self.cash -= trade * self.spot + transaction_cost
        self.inventory = target
        self.total_cost += transaction_cost
        self.last_trade = trade

        z = self._rng.standard_normal()
        self.spot *= np.exp((self.drift - 0.5 * self.volatility**2) * self.dt
                            + self.volatility * np.sqrt(self.dt) * z)
        self.t += 1

        new_mark = self.cash + self.inventory * self.spot
        terminated = self.t >= self.steps
        payoff = max(self.spot - self.strike, 0.0) if terminated else 0.0
        # Transaction costs have already reduced cash, so they are included exactly once.
        reward = (new_mark - old_mark) - payoff
        self._terminated = terminated
        obs = self._obs()
        info = {"spot": self.spot, "inventory": self.inventory, "trade": trade,
                "transaction_cost": transaction_cost, "total_cost": self.total_cost,
                "payoff": payoff}
        return obs, float(reward), terminated, False, info

    def delta_action(self) -> np.ndarray | int:
        tau = max(self.maturity - self.t * self.dt, 0.0)
        target = call_delta(self.spot, self.strike, tau, self.rate,
                            self.volatility, self.dividend)
        return np.array([target], dtype=np.float32) if self.continuous else int(
            np.argmin(np.abs(self._discrete_grid - target))
        )

    def terminal_pnl(self) -> float:
        if not self._terminated:
            raise RuntimeError("terminal P&L is available only after the episode")
        return float(self.cash + self.inventory * self.spot
                     - max(self.spot - self.strike, 0.0))

    def render(self):
        return None

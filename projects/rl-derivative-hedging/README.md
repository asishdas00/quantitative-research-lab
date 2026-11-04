# Reinforcement Learning for Derivative Hedging

An independently implemented research project that studies dynamic hedging of a short European call under transaction costs. It compares a textbook Black–Scholes delta hedge with reinforcement-learning policies trained on simulated market paths.

> Research simulator only. It is not investment advice or a production risk system. Results are meaningful only with documented seeds, costs, training/test separation, and repeated experiments.

## Research questions

1. Can a learned hedge reduce terminal hedging risk after accounting for transaction costs?
2. How does performance change across constant-volatility regimes?
3. How sensitive are policies to proportional trading costs and risk aversion?
4. Does a learned policy generalize to unseen Monte Carlo paths?

## Scope and model

The first experiment uses one short European call, one underlying asset, discrete hedge times, zero interest rate, and geometric Brownian motion (GBM). The seller receives the option premium and holds a short option liability. The hedge inventory is the number of shares held.

The simulator uses exact GBM transitions:
`S_{t+dt}=S_t\exp((\mu-\frac12\sigma^2)dt+\sigma\sqrt{dt}Z_t)`.

At each hedge time the agent chooses a target stock inventory. Rebalancing costs are charged as
`cost_t = c S_t |h_t-h_{t-1}|`.
Cash is updated by paying for the shares bought (or receiving proceeds from shares sold). At maturity, cash plus stock value minus the call payoff is the hedger's terminal P&L.

The benchmark target inventory is the Black–Scholes call delta. For the initial short call, the seller buys positive delta shares.

## Reward and objective

The environment returns incremental mark-to-market P&L net of trading costs. The terminal step additionally subtracts the call payoff. This makes the policy reward consistent with the actual cash/inventory ledger.

Evaluation reports mean P&L, standard deviation, lower-tail VaR/CVaR of P&L, mean absolute hedging loss, and transaction costs. The training reward is not itself a claim of lower risk. For comparable policies we evaluate on identical held-out paths.

## Volatility regimes

The default illustrative regimes are 10%, 20%, and 40% annualized volatility. These are controlled simulation parameters, not empirical estimates. Train and test seeds are separate.

## Repository layout

- `src/rl_hedging/pricing.py`: Black–Scholes call price and delta.
- `src/rl_hedging/market.py`: GBM path generation.
- `src/rl_hedging/env.py`: Gymnasium hedging environment and accounting.
- `src/rl_hedging/baselines.py`: no-hedge and delta-hedge policies.
- `src/rl_hedging/metrics.py`: terminal risk and cost metrics.
- `src/rl_hedging/experiments.py`: common-path benchmark runner.
- `src/rl_hedging/evaluation.py`: held-out RL policy evaluator.
- `scripts/run_baselines.py`: run reproducible classical benchmarks.
- `scripts/train_rl.py`: train PPO, SAC or DQN.
- `scripts/evaluate_rl.py`: compare a saved RL model with both classical benchmarks on held-out paths.
- `tests/`: pricing, simulation and ledger checks.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[rl,test]"
```

## Run the baseline experiment

```bash
python scripts/run_baselines.py --episodes 5000 --steps 30 --seed 7
```

It writes CSV results to `results/baseline_metrics.csv`.

## Train an RL policy

```bash
python scripts/train_rl.py --algo PPO --timesteps 100000 --volatility 0.20
python scripts/train_rl.py --algo SAC --timesteps 100000 --volatility 0.20
python scripts/train_rl.py --algo DQN --timesteps 100000 --volatility 0.20
```

The DQN environment uses a finite nine-point target-inventory grid. PPO and SAC use a continuous target inventory in [-2, 2] shares. Evaluate a saved model on held-out paths with, for example: `python scripts/evaluate_rl.py --algo PPO --model models/ppo_sigma_0.20`. The evaluator reports no-hedge and delta-hedge results on the same test paths. Training returns are not reported as test performance.

## Run tests

Run from this project directory after installation.

```bash
pytest -q
```

## Tests

```bash
pytest
```

## Important implementation notes

- The option is short one call; the hedge is in shares of the underlying.
- The default interest rate and dividend yield are zero.
- Volatility is constant within an episode in the first version. Regime robustness is tested by evaluating separate volatilities.
- The risk-neutral drift is set to the risk-free rate for pricing consistency by default. A separate physical drift can be specified for experiments.
- The environment exposes only current and past information. It does not expose future stock prices.
- The terminal option payoff is charged exactly once.
- Transaction costs are charged on changes in inventory, not on the full inventory every step.
- A finite-horizon RL reward with per-step mark-to-market P&L is not identical to the monetary-utility optimization in the original Deep Hedging paper. The comparison is documented rather than conflated.

## Reference material

This project is an independent implementation informed by:
- Bühler, Gonon, Teichmann & Wood (2019), *Deep Hedging*, https://arxiv.org/abs/1802.03042
- Hans Buehler's reference implementation: https://github.com/hansbuehler/deephedging

The reference repository uses TensorFlow/Keras, custom Monte Carlo worlds, neural policies and monetary utility objectives. This project instead starts with Gymnasium and Stable-Baselines3 algorithms. Do not describe the reference implementation as authored by this project maintainer.

## Next research extensions

1. Stochastic volatility with observable/latent volatility comparison.
2. Volatility jumps and regime switching.
3. Multiple options and hedge instruments.
4. CVaR/entropic utility objectives and reward ablations.
5. Slippage, bid/ask spreads, inventory limits and discrete trading constraints.
6. Hyperparameter search and confidence intervals across independent training seeds.
7. Compare learned policies against delta-hedging variants with identical information and cost assumptions.

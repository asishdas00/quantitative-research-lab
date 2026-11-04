# Experiment design

## Primary grid

- Realized volatility: 10%, 20%, 40%.
- Transaction cost: 0, 5, 10, 20 basis points per dollar of traded notional.
- Hedge steps: 10, 30, 60.
- Baselines: no hedge and Black–Scholes delta hedge.
- Policies: PPO, SAC, DQN (discrete inventory grid).

## Data protocol

Use independent seeds for training, validation and final test. Report mean and standard deviation across multiple policy-training seeds. Use common test paths for all policies to reduce comparison noise. Do not select hyperparameters using the final test set.

## Report

For each setting report mean terminal P&L, P&L standard deviation, 5th-percentile P&L, 95% loss VaR and CVaR, and mean transaction costs. Include confidence intervals from path-level bootstrap or repeated independent test batches.

## Limitations

GBM with constant volatility is a controlled first step. It omits jumps, stochastic volatility, volatility smiles, discrete dividends, stochastic rates, market impact, bid-ask dynamics and hedging constraints beyond a simple inventory bound. A policy trained on exposed constant volatility is not being tested for volatility estimation.

# Methodology

## Portfolio accounting

The desk starts short one call and receives its Black–Scholes premium. Let q_t denote stock inventory and B_t cash. For a trade Δq_t=q_t−q_{t−1}, proportional transaction cost is c S_t |Δq_t|. Cash changes by

B_t = B_{t-1} − S_t Δq_t − c S_t |Δq_t|.

At expiry, terminal P&L is B_T + q_{T-1} S_T − (S_T−K)^+. The simulator charges the transaction cost once at the time of trade and subtracts the payoff once at maturity.

## Reward

The environment reward is the change in marked cash-plus-stock value, less the terminal liability at maturity. Summing rewards therefore equals terminal P&L minus the initial premium. This is a pathwise accounting identity and is tested.

## Information set

The observation includes normalized spot, normalized strike, normalized time to expiry, the episode's volatility parameter, current stock inventory and normalized last trade. The agent never receives future path values. For a more realistic partially observed regime experiment, volatility should be hidden and estimated from trailing returns; the current version intentionally exposes constant episode volatility.

## Benchmark

At each hedge time, delta hedging targets the Black–Scholes call delta. This is a frictionless theoretical target executed with the same discrete hedge schedule and transaction cost model as the learned policies. The simulator evaluates baselines on identical paths.

# Mathematical framework

## Price process

Under the risk-neutral GBM model with continuous dividend yield q,

\[
dS_t=(r-q)S_tdt+\sigma S_tdW_t^Q.
\]

The simulator uses the exact time-discrete transition, so it does not introduce Euler discretization error for GBM.

## Short-call liability and hedge account

The trader sells one European call at premium C(0). The trader starts with cash B_0=C(0) and zero stock inventory h_0=0. At rebalancing time t, the target position is h_t and the trade is Δh_t=h_t-h_{t-1}. With proportional cost rate c,

\[
B_t=B_{t-1}-S_t\Delta h_t-cS_t|\Delta h_t|.
\]

At expiry,

\[
\mathrm{PnL}_T=B_T+h_{T-1}S_T-(S_T-K)^+.
\]

This sign convention is from the option seller's perspective. Positive stock inventory hedges the positive delta of the short-call liability.

## Black–Scholes benchmark

For a European call with time-to-maturity τ and continuous dividend yield q,

\[
d_1=\frac{\log(S/K)+(r-q+\frac12\sigma^2)\tau}{\sigma\sqrt{\tau}},
\qquad
\Delta=e^{-q\tau}\Phi(d_1).
\]

The delta strategy targets h_t=Δ_t before each price move. The experiment executes the hedge at the same discrete observation schedule and charges the same transaction-cost function used by RL policies.

## Learning formulation

The Markov observation in this first experiment is

\[
s_t=(S_t/S_0,K/S_0,\tau_t/T,\sigma,h_{t-1},\Delta h_{t-1}/h_{max}).
\]

The action is a target stock inventory. PPO and SAC use continuous Box actions. DQN uses a finite nine-point inventory grid. The episode reward is the change in cash-plus-stock mark-to-market, with the option payoff subtracted at maturity. Transaction costs reduce cash at the trade time, so they must not be deducted again from reward.

The discounted return is set with γ=1 because the objective is terminal hedging P&L in this finite-horizon experiment.

## Risk measures

For P&L X, report standard deviation and lower-tail metrics. Loss L=-X. At confidence α,

\[
\mathrm{VaR}_\alpha(L)=Q_\alpha(L),\qquad
\mathrm{CVaR}_\alpha(L)=E[L\mid L\geq \mathrm{VaR}_\alpha(L)].
\]

The sample CVaR implementation averages empirical losses at or above the sample VaR threshold. This convention is stated because finite-sample CVaR estimators differ.

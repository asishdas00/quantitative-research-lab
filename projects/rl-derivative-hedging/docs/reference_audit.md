# Reference implementation audit

Reference: https://github.com/hansbuehler/deephedging (main branch).

The reference source provides a TensorFlow/Keras Deep Hedging implementation:
- `world.py` supplies simulated market worlds and path features.
- `gym.py` implements a Monte Carlo hedging training model.
- `agents.py` defines feed-forward and recurrent policy networks.
- `objectives.py` provides monetary utility / risk objectives, including utility-based formulations.
- `trainer.py` wraps model training and caching.
- `plot_training.py` and `plot_bs_hedge.py` provide diagnostics and hedge comparisons.
- `Network.md` documents recurrent states and feature construction.

The reference README describes the code as an illustrative sample, not optimized for speed or production. The reference is not a DQN/PPO/SAC Gymnasium project. The present repository's first implementation is independently structured around a Gymnasium step interface and Stable-Baselines3 agents.

This project does not copy the reference implementation's source modules. It uses the reference as a methodological source and cites it.

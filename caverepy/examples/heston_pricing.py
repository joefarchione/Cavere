"""Heston stochastic volatility model pricing."""

import numpy as np

from cavere import CavereClient, call_payoff, heston

spec = heston(
    spot=100.0,
    rate=0.05,
    v0=0.04,  # initial variance (vol = 0.20)
    kappa=2.0,  # mean reversion speed
    theta=0.04,  # long-run variance
    xi=0.3,  # vol of vol
    rho=-0.7,  # stock-vol correlation
    steps=252,
    payoff=call_payoff(strike=100.0),
)

with CavereClient("localhost:5000") as client:
    values = client.fold(spec, num_scenarios=100_000)
    price = np.mean(values)
    stderr = np.std(values) / np.sqrt(len(values))
    print(f"Heston Call Price: {price:.4f} +/- {stderr:.4f}")

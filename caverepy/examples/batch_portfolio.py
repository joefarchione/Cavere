"""Batch simulation — price GBM calls across multiple strikes at once."""

import numpy as np

from cavere import CavereClient, call_payoff, gbm

# GBM model; batch_values will replace the payoff strike
spec = gbm(spot=100.0, rate=0.05, vol=0.20, steps=252, payoff=call_payoff(strike=100.0))

strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]

with CavereClient("localhost:5000") as client:
    # batch_fold_means returns one mean per strike
    means = client.batch_fold_means(spec, num_scenarios=50_000, batch_values=strikes)
    print("Strike | Price")
    print("-------+-------")
    for strike, price in zip(strikes, means):
        print(f"{strike:6.0f} | {price:.4f}")

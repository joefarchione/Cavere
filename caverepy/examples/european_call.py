"""European call option pricing via GBM template."""

import numpy as np

from cavere import CavereClient, call_payoff, gbm

# GBM with call payoff, discounted
spec = gbm(spot=100.0, rate=0.05, vol=0.20, steps=252, payoff=call_payoff(strike=100.0))

with CavereClient("localhost:5000") as client:
    values = client.fold(spec, num_scenarios=100_000)
    price = np.mean(values)
    stderr = np.std(values) / np.sqrt(len(values))
    print(f"European Call Price: {price:.4f} +/- {stderr:.4f}")

    # Also get the generated C# source
    source = client.source(spec)
    print(f"\nGenerated C# source ({len(source)} chars):")
    print(source[:500])

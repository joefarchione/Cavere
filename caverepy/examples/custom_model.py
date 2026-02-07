"""Custom model built from Python expression trees."""

import numpy as np

from cavere import CavereClient
from cavere.expr import Const, ModelBuilder, Normal, exp, max_, sqrt

# Build a GBM model from scratch using the Expr DSL
b = ModelBuilder()
dt = Const(1 / 252)
z = Normal(0)
rate = Const(0.05)
vol = Const(0.20)
spot = Const(100.0)
strike = Const(100.0)

# Stock process: S(t+1) = S(t) * exp((r - 0.5*v^2)*dt + v*sqrt(dt)*z)
stock = b.add_accum(
    init=spot,
    body_fn=lambda s: s * exp((rate - Const(0.5) * vol * vol) * dt + vol * sqrt(dt) * z),
)

# Discount factor: df(t+1) = df(t) * exp(-r*dt)
df = b.add_accum(
    init=Const(1.0),
    body_fn=lambda d: d * exp(-rate * dt),
)

# Discounted call payoff
result = max_(stock - strike, 0.0) * df
spec = b.build(result, normal_count=1, uniform_count=0, steps=252)

with CavereClient("localhost:5000") as client:
    values = client.fold(spec, num_scenarios=100_000)
    price = np.mean(values)
    stderr = np.std(values) / np.sqrt(len(values))
    print(f"Custom GBM Call Price: {price:.4f} +/- {stderr:.4f}")

    source = client.source(spec)
    print(f"\nGenerated C# ({len(source)} chars)")

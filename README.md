# Cavere

GPU-accelerated Monte Carlo simulation in F#. Write models as composable expression trees, compile them to GPU kernels through Roslyn and ILGPU, and run millions of paths at device speed.

```fsharp
let callModel = model {
    let! dt = scheduleDt (Schedule.constant (1.0f / 252.0f) 252)
    let! z = normal
    let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
    let! df    = decay 0.05f.C dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}

use sim = Simulation.create CPU 100_000 252
let prices = Simulation.fold sim callModel
printfn "Call price: %.4f" (Array.average prices)
```

## Why Cavere

- **Composable DSL** — stochastic processes snap together with `let!`. One `model { }` block wires up dependencies, allocates random variables, and registers accumulators automatically.
- **GPU-compiled** — the expression tree compiles to flat C# via Roslyn, then to GPU kernels through ILGPU. Switch between `CPU` and `GPU` with a single flag.
- **Automatic differentiation** — mark any parameter as `Dual` or `HyperDual` and get Greeks (delta, gamma, vega) computed alongside prices. Forward, reverse, and symbolic modes included.
- **Extensible generators** — GBM, Heston, Vasicek, CIR, local vol, multi-asset, and AAA ESG models ship out of the box. Writing a custom generator is just an F# function.
- **Python interop** — a gRPC server exposes all simulation modes to Python (or any gRPC client) with template and custom expression tree builders.

## Installation

```xml
<ProjectReference Include="path/to/src/Core/Core.fsproj" />
<ProjectReference Include="path/to/src/Generators/Generators.fsproj" />
```

```fsharp
open Cavere.Core
open Cavere.Generators
```

**Dependencies**: ILGPU 1.5.3, Microsoft.CodeAnalysis.CSharp 4.12.0

## Quick Start

```fsharp
open Cavere.Core
open Cavere.Generators

// 1. Define a schedule (252 daily steps = 1 year)
let sched = Schedule.constant (1.0f / 252.0f) 252

// 2. Build a model
let callModel = model {
    let! dt = scheduleDt sched
    let! z = normal                                  // standard normal
    let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt   // GBM: 5% rate, 20% vol, S0=100
    let! df = decay 0.05f.C dt                       // discount factor
    return Expr.max (stock - 100.0f) 0.0f.C * df     // discounted call payoff
}

// 3. Create simulator and run
use sim = Simulation.create CPU 100_000 sched.Steps
let prices = Simulation.fold sim callModel
printfn "European call price: %.4f" (Array.average prices)
```

## A Taste of What's Possible

**Correlated multi-asset basket:**

```fsharp
let basketModel = model {
    let! dt = scheduleDt sched
    let corr = array2D [| [| 1.0f; 0.6f |]; [| 0.6f; 1.0f |] |]
    let! zs = correlatedNormals corr
    let! s1 = gbm zs.[0] 0.05f.C 0.20f.C 100.0f.C dt
    let! s2 = gbm zs.[1] 0.05f.C 0.25f.C 50.0f.C dt
    let! df = decay 0.05f.C dt
    return Expr.max ((s1 + s2) / 2.0f - 75.0f) 0.0f.C * df
}
```

**Heston stochastic volatility:**

```fsharp
let hestonModel = model {
    let! dt = scheduleDt sched
    let! z = normal
    let! stock = heston z 0.05f.C 0.04f.C 1.5f.C 0.04f.C 0.3f.C -0.7f 100.0f.C dt
    let! df = decay 0.05f.C dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}
```

**Custom generator — just a function:**

```fsharp
let ornsteinUhlenbeck kappa theta sigma x0 dt : ModelCtx -> Expr = fun ctx ->
    let z = normal ctx
    evolve x0 (fun x -> x + kappa * (theta - x) * dt + sigma * Expr.sqrt dt * z) ctx
```

## Examples

| Example | Description |
|---------|-------------|
| [EuropeanCall.fs](examples/EuropeanCall.fs) | Vanilla call with GBM and flat rate |
| [HestonModel.fs](examples/HestonModel.fs) | Heston stochastic volatility |
| [LocalVol.fs](examples/LocalVol.fs) | Local volatility surface pricing |
| [CustomGenerator.fs](examples/CustomGenerator.fs) | OU mean-reversion and jump-diffusion |
| [ForwardRateCurve.fs](examples/ForwardRateCurve.fs) | Term-structure-aware pricing |
| [NestedSimulation.fs](examples/NestedSimulation.fs) | Conditional expectations via nested MC |
| [FixedIndexedAnnuity.fs](examples/FixedIndexedAnnuity.fs) | FIA crediting with batch simulation |

```bash
dotnet run --project examples/Examples.fsproj              # all examples
dotnet run --project examples/Examples.fsproj -- heston    # just Heston
```

## Documentation

Full reference documentation is available on the [GitHub Wiki](https://github.com/joefarchione/Cavere/wiki):

- [Core Concepts](https://github.com/joefarchione/Cavere/wiki/Core-Concepts) — model builder, evolve, validation, compilation pipeline
- [Expression Language](https://github.com/joefarchione/Cavere/wiki/Expression-Language) — operators, math functions, special expressions
- [Simulation API](https://github.com/joefarchione/Cavere/wiki/Simulation-API) — fold, scan, batch, nested simulation
- [Generators](https://github.com/joefarchione/Cavere/wiki/Generators) — equity, rates, multi-asset, AAA ESG
- [Automatic Differentiation](https://github.com/joefarchione/Cavere/wiki/Automatic-Differentiation) — Dual, HyperDual, adjoint modes
- [Custom Generators](https://github.com/joefarchione/Cavere/wiki/Custom-Generators) — writing your own stochastic processes
- [gRPC & Python](https://github.com/joefarchione/Cavere/wiki/gRPC-and-Python) — server setup, Python client, template and custom models
- [GPU Optimization](https://github.com/joefarchione/Cavere/wiki/GPU-Optimization) — pinned memory, multi-device, output formats

## Build & Test

```bash
dotnet build                                    # Build entire solution
dotnet test                                     # Run all tests
dotnet run --project src/App/App.fsproj         # Run console demo
```

## Development

After cloning, enable the pre-commit hooks:

```bash
git config core.hooksPath hooks
```

This runs automatically on each commit:

| Language | Tool | Check |
|----------|------|-------|
| F# | Fantomas | Formatting (`fantomas --check`) |
| Python | ruff | Formatting (`ruff format --check`) |
| Python | ruff | Linting (`ruff check`) |
| Python | ty | Type checking (`uv run ty check`) |

**Git workflow**: all changes go through `feature/<name>` branches, PR'd into `staging` (squash merge), then `staging` PR'd into `main` (squash merge). Direct pushes to `main` and `staging` are blocked.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

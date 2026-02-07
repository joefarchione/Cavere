# Examples

Runnable example models demonstrating various features. References Core and Generators.

## Running

```bash
dotnet run --project examples/Examples.fsproj              # all examples
dotnet run --project examples/Examples.fsproj -- heston    # specific example
```

## Structure

Each example is a module with a `run()` function, dispatched from `Program.fs`:

```fsharp
module Cavere.Examples.MyExample

let run () =
    let sched = Schedule.constant (1.0f / 252.0f) 252
    let m = model { ... }
    use sim = Simulation.create CPU 100_000 sched.Steps
    let results = Simulation.fold sim m
    printfn "  Mean: %.4f" (Array.average results)
```

## Examples

- **EuropeanCall** — Vanilla GBM call pricing
- **LocalVol** — Local volatility surface interpolation
- **HestonModel** — Stochastic volatility with correlation
- **CustomGenerator** — Writing your own process (Ornstein-Uhlenbeck, jump diffusion)
- **ForwardRateCurve** — Term-structure-aware pricing
- **CalendarSchedule** — Business day calendar with holidays
- **NestedSimulation** — Conditional expectations via nested MC
- **FixedIndexedAnnuity** — Batch simulation for FIA portfolio (uses `batchInput`, `BatchSimulation`)

## Conventions

- Print output indented with 2 spaces for consistent formatting
- Use `Simulation.create CPU` (not GPU) so examples run anywhere
- Each example should be self-contained — define model, run, print results

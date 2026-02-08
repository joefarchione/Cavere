# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build & Test

```bash
dotnet build                                          # Build entire solution
dotnet test                                           # Run all tests
dotnet run --project src/Cavere.App/Cavere.App.fsproj # Run console demo
dotnet run --project examples/Examples.fsproj         # Run all examples
dotnet clean                                          # Clean build artifacts
```

## Pre-commit Hooks

```bash
pip install pre-commit       # or: pipx install pre-commit
pre-commit install           # installs hooks into .git/hooks/
pre-commit run --all-files   # verify all hooks pass
```

Hooks run automatically on `git commit` and check:
- **Fantomas** — F# formatting (requires `dotnet tool install -g fantomas`)
- **ruff format** + **ruff lint** — Python formatting and linting (auto-downloaded by pre-commit)
- **ty** — Python type-checking (requires `uv` installed)

## Solution Structure

```
Cavere.sln
├── src/Core/                  (F#, net8.0)  — Expr DSL, compiler, engine
│   ├── Dsl.Expr.fs            — Expr DU, operator overloads (.C, .>, .<, etc.)
│   ├── Dsl.Schedule.fs        — Time grid construction
│   ├── Dsl.Model.fs           — Model record, ModelBuilder CE, DSL primitives
│   ├── Ast.Symbolic.fs        — Expression simplification, symbolic differentiation
│   ├── Ast.Analysis.fs        — Closed-form detection, path dependence analysis
│   ├── Compiler.Common.fs     — Surface layout, expression emission, topo sort
│   ├── Compiler.Codegen.fs    — C# templates, dynamic emitters
│   ├── Compiler.Regular.fs    — Fold, FoldWatch, Scan kernels
│   ├── Compiler.Batch.fs      — FoldBatch, FoldBatchWatch kernels
│   ├── Compiler.Diff.fs       — Forward-mode AD expression transformation
│   ├── Compiler.Adjoint.fs    — Reverse-mode AD kernel compilation
│   ├── Compiler.fs            — Unified compiler API facade
│   ├── Sim.Device.fs          — ILGPU Context/Accelerator management
│   ├── Sim.Transfer.fs        — Pinned memory pool, DMA transfers
│   ├── Sim.Engine.fs          — Surface packing, kernel launch, result extraction
│   ├── Sim.Watcher.fs         — Observer buffer mgmt, value extraction
│   ├── Sim.Kernel.fs          — Kernel compilation and caching
│   ├── Sim.Simulation.fs      — High-level orchestration
│   └── Sim.Output.fs          — CSV and Parquet export
├── src/Generators/        (F#, net8.0)  — Finance-specific model primitives
│   ├── Common.fs          — decay (discount factor accumulator)
│   ├── Equity.fs          — gbm, heston, multiAssetHeston
│   ├── Rates.fs           — vasicek, cir, cirpp, forward curves
│   ├── AAA.Common.fs      — SLV parameter types
│   ├── AAA.Rates.fs       — SLV interest rate model
│   ├── AAA.Equity.fs      — SLV equity with leverage effect
│   ├── AAA.Bonds.fs       — Duration-based bond returns
│   ├── AAA.Selection.fs   — Scenario selection algorithms
│   └── AAA.Generator.fs   — Combined AAA model builder
├── src/Actuarial/         (F#, net8.0)  — Insurance product definitions
│   ├── Decrements.Common.fs    — Survival prob, multi-decrement
│   ├── Decrements.Mortality.fs — Mortality tables, joint life
│   ├── Decrements.Surrender.fs — Lapse rates, persistency
│   ├── Decrements.Withdrawal.fs— Partial withdrawals, free amounts
│   ├── Product.Common.fs  — CDSC, MVA, RMD, fees, benefit bases
│   ├── Product.Riders.fs  — GMxB rider definitions
│   ├── Product.Life.fs    — Term, UL, whole life
│   ├── Product.Fixed.fs   — Fixed annuities, MYGAs
│   ├── Product.FixedIndexed.fs — FIA crediting strategies
│   ├── Product.RILA.fs    — Buffer/floor annuities
│   ├── Product.Variable.fs— VA sub-accounts, allocations
│   └── Policy.fs          — Policyholder, Policy, AccountStructure
├── src/App/               (F#, exe)  — Console demo
│   ├── Stats.fs
│   └── Program.fs
├── tests/                 (F#, xUnit)  — Test suite
│   ├── CompilerTests.fs   — CE allocation, topo sort, Roslyn
│   ├── PricingTests.fs    — BS analytical comparison
│   ├── RatesTests.fs      — Rate model validation
│   ├── NestedSimTests.fs  — Nested simulation patterns
│   └── ScheduleTests.fs   — Schedule construction
└── examples/              (F#, exe)  — Runnable examples
    ├── EuropeanCall.fs, LocalVol.fs, HestonModel.fs
    ├── CustomGenerator.fs, ForwardRateCurve.fs
    ├── CalendarSchedule.fs, NestedSimulation.fs
    └── FixedIndexedAnnuity.fs
```

**NuGet**: ILGPU 1.5.3, ILGPU.Algorithms 1.5.3, Microsoft.CodeAnalysis.CSharp 4.12.0

## Architecture

Expr DSL + code generation. Models are expression trees (`Expr` DU), compiled to flat C# via Roslyn, loaded into ILGPU as GPU kernels.

```
F# model { } CE  →  Expr AST  →  C# source  →  Roslyn  →  ILGPU  →  PTX
```

### Core Abstractions

- **`Expr`** — DU: Const, TimeIndex, Normal(id), Uniform(id), AccumRef(id), Lookup1D, BatchRef, Interp1D, Interp2D, arithmetic/comparison/unary ops
- **`Model`** — Record: Result expr, accumulators, surfaces, observers, counts
- **`ModelBuilder`** — Computation expression builder (`model { }` CE)
- **`evolve init body`** — Accumulator (like Seq.scan), returns AccumRef
- **`normal`** / **`uniform`** — Fresh random variable allocation
- **`batchInput values`** — Per-batch data vector as BatchRef
- **`observe name expr`** — Record values for path inspection
- **`correlatedNormals corrMatrix`** — Cholesky-decomposed correlated normals (Cholesky runs on CPU at build time, correlation coefficients baked into Expr graph)

## Code Style

- 4-space indentation, 120-character max line
- Single-precision floats only (`float32` in F#)
- Prefer functional style over mutable state
- Prefer F# interpolated strings (`$"..."`) and multi-line strings (`$"""..."""`)
- Prefer piping (`|>`) and functional combinators over imperative loops
- `.C` converts `float32` to `Const` expr: `0.05f.C`
- Comparison operators return `Expr`, not `bool`: `.>`, `.>=`, `.<`, `.<=`

## File & Namespace Conventions

- Break namespaces into files prefixed with the namespace name
  - `Compiler.Common.fs`, `Compiler.Codegen.fs` → modules in `Cavere.Core`
  - `AAA.Common.fs`, `AAA.Rates.fs` → modules in `Cavere.Generators.AAA`
  - `Decrements.Common.fs`, `Decrements.Mortality.fs` → modules in `Cavere.Actuarial.Decrements`
  - `Product.Common.fs`, `Product.Fixed.fs` → modules in `Cavere.Actuarial.Product`
- Non-prefixed files stay in the parent namespace (e.g., `Policy.fs` → `Cavere.Actuarial`)
- F# compilation order matters — files listed in `.fsproj` must respect dependencies

## Generator Pattern

Generators are functions with signature `... -> ModelCtx -> Expr` that compose inside `model { }` via `let!`:

```fsharp
let callModel = model {
    let dt = (1.0f / 252.0f).C
    let! z = normal
    let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
    let! df = decay 0.05f.C dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}
```

Writing a custom generator is just a function that calls `normal` and `evolve`:

```fsharp
let myProcess (kappa: Expr) (theta: Expr) (sigma: Expr) (x0: float32) (dt: Expr)
    : ModelCtx -> Expr = fun ctx ->
    let z = normal ctx
    evolve x0.C (fun x -> x + kappa * (theta - x) * dt + sigma * Expr.sqrt dt * z) ctx
```

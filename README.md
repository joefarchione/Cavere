# Cavere

GPU-accelerated Monte Carlo simulation engine for F#.

Models are built as composable expression trees using an F# computation expression, compiled to flat C# via Roslyn, and executed on the GPU through ILGPU. No nesting limits, no fixed buffers, no runtime interpretation — just generated kernels that run at device speed.

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

## Installation

Add the project reference to your F# project:

```xml
<ProjectReference Include="path/to/src/Core/Core.fsproj" />
<ProjectReference Include="path/to/src/Generators/Generators.fsproj" />
```

Open the namespaces:

```fsharp
open Cavere.Core
open Cavere.Generators
```

**Dependencies**: ILGPU 1.5.3, Microsoft.CodeAnalysis.CSharp 4.12.0

## Quick start

```fsharp
open Cavere.Core
open Cavere.Generators

// 1. Define a schedule (252 daily steps = 1 year)
let sched = Schedule.constant (1.0f / 252.0f) 252

// 2. Build a model
let callModel = model {
    let! dt = scheduleDt sched
    let! z = normal                                  // allocate a standard normal
    let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt   // GBM with 5% rate, 20% vol, S0=100
    let! df = decay 0.05f.C dt                       // discount factor
    return Expr.max (stock - 100.0f) 0.0f.C * df     // discounted call payoff
}

// 3. Create simulator and run
use sim = Simulation.create CPU 100_000 sched.Steps
let prices = Simulation.fold sim callModel
printfn "European call price: %.4f" (Array.average prices)
```

---

## Core concepts

### The `model { }` builder

The `model { }` computation expression is the primary way to define simulations. Each `let!` binds a stochastic process or surface, and the builder handles all ID allocation, accumulator registration, and dependency wiring automatically:

```fsharp
let callModel = model {
    let! dt = scheduleDt sched              // load time steps onto GPU
    let! z = normal                           // allocate normal random
    let! stock   = gbm z rate vol spot dt     // GBM stock process
    let! df      = decay rate dt              // discount factor
    do!  observe "stock" stock                // record for path inspection
    return Expr.max (stock - strike) 0.0f.C * df
}
```

Processes compose naturally. The output of one `let!` is an `Expr` that feeds directly into the next — a stochastic rate can drive a stock, which can drive a payoff, all in the same model:

```fsharp
let model = model {
    let! dt = scheduleDt sched
    let! z = normal
    let! rate  = vasicek 0.5f.C 0.05f.C 0.01f.C 0.03f dt  // stochastic rate
    let! stock = gbm z rate 0.20f.C 100.0f.C dt           // rate feeds into stock
    let! df    = decay rate dt                             // discount at stochastic rate
    return Expr.max (stock - 100.0f) 0.0f.C * df
}
```

Three evolving states, two sources of randomness, automatic dependency ordering — all compiled to a single GPU kernel.

### How it works

```
F# model { } CE  →  Expr AST  →  validate  →  C# source  →  Roslyn  →  ILGPU  →  GPU kernel
```

1. **Build**: The `model { }` computation expression constructs an expression tree (`Expr` discriminated union)
2. **Validate**: All surface IDs and accumulator references are checked for integrity before code generation
3. **Layout**: Surfaces are packed into a single float32 array with baked offsets
4. **Sort**: Accumulators are topologically sorted by their dependencies
5. **Generate**: A complete C# class is generated with specialized kernel methods
6. **Compile**: Roslyn compiles the C# to an in-memory assembly
7. **Load**: ILGPU loads the kernel onto the GPU/CPU accelerator
8. **Launch**: Each simulation thread runs an independent path with its own random stream

### Model validation

The compiler validates every model before code generation. This catches structural errors at compile time rather than producing a kernel that silently reads garbage memory:

- **Surface ID integrity**: Every `Lookup1D`, `SurfaceAt`, and `BatchRef` references a surface that was actually registered via `surface1d`, `scheduleDt`, `batchInput`, etc.
- **AccumRef integrity**: Every `AccumRef` references an accumulator that was actually created by `evolve`

Validation walks the entire expression tree — the result expression, all accumulator init/body expressions, and all observer expressions. If any dangling reference is found, compilation fails with a clear error message listing the missing IDs.

### `evolve` — composable state

The core primitive is `evolve`:

```fsharp
let! stock = evolve 100.0f.C (fun price ->
    price * Expr.exp (drift * dt + vol * Expr.sqrt dt * z))
```

`evolve` defines a value that updates itself each time step — like `Seq.scan` but compiled to a GPU register. It takes an initial value (`Expr`) and a body that receives the current value and returns the next. The returned `Expr` (an `AccumRef`) can be used anywhere in subsequent expressions.

What makes this powerful is that evolve calls compose. One evolving state can depend on another:

```fsharp
let! variance = evolve v0 (fun v ->
    v + kappa * (theta - v) * dt + xi * Expr.sqrt v * dw2)
let! stock = evolve spot (fun s ->
    s * Expr.exp ((rate - 0.5f * variance) * dt + Expr.sqrt variance * dw1))
```

The compiler topologically sorts all accumulators so dependencies are always evaluated first. You write natural math; the compiler figures out the execution order.

---

## Expression language

The `Expr` type represents mathematical expressions that compile to GPU code. All values are single-precision floats (`float32`).

### Constants and conversion

```fsharp
let x = Const 3.14f          // explicit constructor
let y = 3.14f.C              // extension property (preferred)
let z = 0.0f.C               // zero
```

### Arithmetic operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `a + b`, `a + 1.0f` |
| `-` | Subtraction | `a - b`, `a - 1.0f` |
| `*` | Multiplication | `a * b`, `a * 2.0f` |
| `/` | Division | `a / b`, `a / 2.0f` |
| `~-` | Negation | `-a` |

Mixed `Expr` and `float32` operands are supported — the float is automatically wrapped in `Const`.

### Comparison operators

| Operator | Description | Example |
|----------|-------------|---------|
| `.>` | Greater than | `a .> b`, `a .> 0.0f` |
| `.>=` | Greater or equal | `a .>= b` |
| `.<` | Less than | `a .< b`, `a .< 100.0f` |
| `.<=` | Less or equal | `a .<= b` |

Comparisons return `1.0f` for true, `0.0f` for false (GPU-friendly representation).

### Math functions

| Function | Description | Example |
|----------|-------------|---------|
| `Expr.exp` | Exponential | `Expr.exp x` |
| `Expr.log` | Natural logarithm | `Expr.log x` |
| `Expr.sqrt` | Square root | `Expr.sqrt x` |
| `Expr.abs` | Absolute value | `Expr.abs x` |
| `Expr.max` | Maximum | `Expr.max a b` |
| `Expr.min` | Minimum | `Expr.min a b` |
| `Expr.select` | Conditional | `Expr.select cond ifTrue ifFalse` |
| `Expr.clip` | Clamp to range | `Expr.clip lo hi x` |

### Special expressions

| Expression | Description |
|------------|-------------|
| `TimeIndex` | Current time step index (0 to steps-1) |
| `Normal id` | Standard normal random variable |
| `Uniform id` | Uniform [0,1] random variable |
| `AccumRef id` | Reference to evolving accumulator |
| `Lookup1D sid` | Direct array lookup by time index |
| `Interp1D(sid, t)` | 1D interpolated surface lookup |
| `Interp2D(sid, t, s)` | 2D interpolated surface lookup |
| `BatchRef sid` | Per-batch data lookup (batch simulation) |

---

## Simulation API

### Creating a simulator

```fsharp
use sim = Simulation.create deviceType numSims steps
```

| Parameter | Description |
|-----------|-------------|
| `deviceType` | `CPU` or `GPU` |
| `numSims` | Number of simulation paths |
| `steps` | Number of time steps |

The simulator owns GPU resources and must be disposed.

### Simple simulation functions

These require a non-batch model (no `batchInput` calls):

| Function | Signature | Description |
|----------|-----------|-------------|
| `fold` | `Simulation -> Model -> float32[]` | Terminal values only |
| `foldWatch` | `Simulation -> Model -> Frequency -> float32[] * WatchResult` | Terminal values + observed paths |
| `scan` | `Simulation -> Model -> float32[,]` | Full path history `[step, sim]` |

```fsharp
// Just terminal values
let prices = Simulation.fold sim callModel

// Terminal values + watched variables
let prices, watch = Simulation.foldWatch sim callModel Monthly
let stockPaths = Watcher.values "stock" watch

// Full path scan (memory intensive)
let paths = Simulation.scan sim callModel
```

### Batch simulation functions

These require a batch model (uses `batchInput`):

| Function | Signature | Description |
|----------|-----------|-------------|
| `foldBatch` | `Simulation -> Model -> int -> float32[]` | Raw results `[batch * numSims]` |
| `foldBatchWatch` | `Simulation -> Model -> int -> Frequency -> float32[] * WatchResult` | Results + observations |
| `foldBatchMeans` | `Simulation -> Model -> int -> float32[]` | Per-batch means `[batch]` |

```fsharp
let means = Simulation.foldBatchMeans sim batchModel numSims
```

All batch elements share the same random scenarios, making results directly comparable.

### Kernel reuse and source export

Compiled kernels are cached by model identity — calling `fold` twice with the same model object skips Roslyn recompilation. You can also pre-compile and inspect:

```fsharp
// Inspect generated C#
let cs = Simulation.source callModel
printfn "%s" cs

// Pre-compile, then run multiple times without recompilation
let kernel = Simulation.compile callModel
let results1 = Simulation.foldKernel sim kernel
let results2 = Simulation.foldKernel sim kernel
```

### Observation frequencies

| Frequency | Interval |
|-----------|----------|
| `Daily` | Every step |
| `Weekly` | steps / 52 |
| `Monthly` | steps / 12 |
| `Quarterly` | steps / 4 |
| `Annually` | steps |
| `Terminal` | Final step only |

Intervals are derived from the schedule's step count, so they work correctly for any schedule length.

### Watcher extraction

| Function | Returns | Description |
|----------|---------|-------------|
| `Watcher.values "name" watch` | `float32[obs, sim]` | Full 2D path history |
| `Watcher.terminals "name" watch` | `float32[]` | Final observation values |
| `Watcher.sliceObs "name" idx watch` | `float32[]` | Single observation across all sims |

---

## Built-in generators

### Equity processes (`Cavere.Generators.Equity`)

| Generator | Signature | Description |
|-----------|-----------|-------------|
| `gbm` | `z:Expr -> rate:Expr -> vol:Expr -> spot:Expr -> dt:Expr -> ModelCtx -> Expr` | Geometric Brownian motion |
| `gbmLocalVol` | `z:Expr -> surfId:int -> rate:Expr -> spot:Expr -> dt:Expr -> ModelCtx -> Expr` | GBM with local volatility surface |
| `heston` | `z:Expr -> rate -> v0 -> kappa -> theta -> xi -> rho -> spot -> dt -> ModelCtx -> Expr` | Heston stochastic volatility |

### Common utilities (`Cavere.Generators.Common`)

| Generator | Signature | Description |
|-----------|-----------|-------------|
| `decay` | `rate:Expr -> dt:Expr -> ModelCtx -> Expr` | Discount factor accumulator |

**Note**: `gbm`, `gbmLocalVol`, and `heston` all take a normal `z` as the first parameter. Use `let! z = normal` for single assets, or `let! zs = correlatedNormals corrMatrix` for correlated multi-asset models.

**GBM dynamics**: $dS = (r - \frac{1}{2}\sigma^2)S\,dt + \sigma S\,dW$

**Heston dynamics**:
- $dS = (r - \frac{1}{2}v)S\,dt + \sqrt{v}S\,dW_1$
- $dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2$
- $\text{Corr}(dW_1, dW_2) = \rho$

### Rate models (`Cavere.Generators.Rates`)

| Generator | Signature | Description |
|-----------|-----------|-------------|
| `vasicek` | `kappa theta sigma r0 dt -> ModelCtx -> Expr` | Vasicek mean-reverting rate |
| `cir` | `kappa theta sigma r0 dt -> ModelCtx -> Expr` | Cox-Ingersoll-Ross |
| `cirpp` | `kappa theta sigma x0 shiftSurfId dt -> ModelCtx -> Expr` | CIR++ with deterministic shift |

**Vasicek**: $dr = \kappa(\theta - r)\,dt + \sigma\,dW$

**CIR**: $dr = \kappa(\theta - r)\,dt + \sigma\sqrt{r}\,dW$

### Multi-asset processes

For portfolios with correlated assets, use `correlatedNormals` to generate correlated random shocks via Cholesky decomposition:

```fsharp
// Two correlated stocks with 60% correlation
let twoStockModel = model {
    let! dt = scheduleDt sched
    let correlation = array2D [| [| 1.0f; 0.6f |]
                                 [| 0.6f; 1.0f |] |]
    let! zs = correlatedNormals correlation  // zs: Expr[2]
    let! stock1 = gbm zs.[0] 0.05f.C 0.20f.C 100.0f.C dt
    let! stock2 = gbm zs.[1] 0.05f.C 0.25f.C 50.0f.C dt
    return stock1 + stock2  // portfolio value
}
```

For multi-asset Heston with full correlation structure, use `multiAssetHeston`:

```fsharp
open Cavere.Generators.Equity

let basketModel = model {
    let! dt = scheduleDt sched

    let assets = [|
        { Spot = 100.0f.C; V0 = 0.04f.C; Kappa = 1.5f.C
          Theta = 0.04f.C; Xi = 0.3f.C; Rho = -0.7f }
        { Spot = 50.0f.C; V0 = 0.06f.C; Kappa = 2.0f.C
          Theta = 0.05f.C; Xi = 0.4f.C; Rho = -0.6f }
    |]

    // Stock-stock correlation (0.5 between assets)
    let stockCorr = array2D [| [| 1.0f; 0.5f |]
                               [| 0.5f; 1.0f |] |]
    // Vol-vol correlation (0.3 between assets)
    let volCorr = array2D [| [| 1.0f; 0.3f |]
                             [| 0.3f; 1.0f |] |]

    let! stocks = multiAssetHeston 0.05f.C stockCorr volCorr assets dt
    // stocks: Expr[2] — terminal values for each asset

    let! df = decay 0.05f.C dt
    let basket = (stocks.[0] + stocks.[1]) / 2.0f  // equal-weighted basket
    return Expr.max (basket - 75.0f) 0.0f.C * df
}
```

The correlation structure for `multiAssetHeston`:
- `Corr(dW_stock_i, dW_stock_j)` = `stockCorrelation[i,j]`
- `Corr(dW_vol_i, dW_vol_j)` = `volCorrelation[i,j]`
- `Corr(dW_stock_i, dW_vol_i)` = `assets[i].Rho` (per-asset stock-vol correlation)
- `Corr(dW_stock_i, dW_vol_j)` = 0 for i≠j (no cross stock-vol between different assets)

### Rate curve utilities

| Function | Description |
|----------|-------------|
| `Rates.flat rate` | Constant rate (returns `Expr`) |
| `Rates.curve surfId` | Time-varying rate from surface |
| `Rates.linearForwards tenors zeroRates steps` | Bootstrap forward rates from zero curve |
| `Rates.logDiscountForwards tenors fwdRates steps` | Forward rates from log-discount interpolation |

---

## Schedules and calendars

### Uniform schedule

```fsharp
let sched = Schedule.constant (1.0f / 252.0f) 252  // daily for 1 year
```

### Business day schedule

```fsharp
let holidays = set [
    DateTime(2024, 12, 25)
    DateTime(2024, 1, 1)
]
let sched = Schedule.businessDays startDate endDate holidays
```

Business day schedules exclude weekends and holidays. The `dt` values vary based on actual calendar gaps (measured in calendar days / 365).

### Using schedules in models

```fsharp
let! dt = scheduleDt sched   // time step sizes (most common)
let! t  = scheduleT sched   // year fractions (when needed)
```

Use `scheduleDt` for time step sizes and `scheduleT` for cumulative year fractions. Each only allocates the surface it needs — no wasted GPU memory if you only use one.

---

## Surfaces

Market data is packed into GPU memory as interpolated surfaces.

### 1D surfaces (curves)

```fsharp
let! sid = surface1d fwdRates steps
let rate = Interp1D(sid, TimeIndex)  // interpolated lookup
// or
let rate = Lookup1D sid              // direct array access (no interpolation)
```

### 2D surfaces (vol surfaces, etc.)

```fsharp
let! sid = surface2d times spots vols steps
let vol = Interp2D(sid, TimeIndex, stock)  // bilinear interpolation
```

All surfaces are packed into a single flat array with baked offsets — zero runtime overhead.

---

## Watching variables

Add observers to record values during simulation:

```fsharp
let callModel = model {
    let! dt = scheduleDt sched
    let! z = normal
    let! stock = gbm z rate vol spot dt
    let! df    = decay rate dt
    do! observe "stock" stock       // record stock at each observation
    do! observe "df" df             // record discount factor
    return Expr.max (stock - strike) 0.0f.C * df
}

let finals, watch = Simulation.foldWatch sim callModel Monthly
let stockPaths = Watcher.values "stock" watch      // float32[12, numSims]
let terminalDfs = Watcher.terminals "df" watch     // float32[numSims]
```

Observers are metadata — you can add or remove watches without recompiling the kernel.

---

## Batch simulation

Run many parameter sets in a single kernel launch. Every batch element sees the **same random scenarios**, making results directly comparable.

### Using `batchInput`

```fsharp
let premiums = [| 100.0f; 150.0f; 200.0f |]  // 3 policies
let caps = [| 0.05f; 0.06f; 0.07f |]

let batchModel = model {
    let! premium = batchInput premiums
    let! cap = batchInput caps
    // ... model using premium and cap per batch element
}

use sim = Simulation.create CPU (3 * numSims) steps
let means = Simulation.foldBatchMeans sim batchModel numSims
// means: float32[3] — one mean per policy
```

### How it works

- `batchIdx = threadIdx / numSims` — which parameter set
- `scenarioIdx = threadIdx % numSims` — which random scenario
- Random seed uses `scenarioIdx`, so all batch elements share the same draws

### Use cases

- **Policy portfolios**: Value many insurance policies with different parameters
- **Parameter sweeps**: Sensitivity analysis across rate/vol assumptions
- **Nested simulation**: Outer paths become batch inputs for inner simulation

---

## Nested simulation

Run inner simulations conditioned on outer paths — for exposure profiles, XVA, and American option pricing:

```fsharp
// Outer simulation
let outerModel = model {
    let! dt = scheduleDt sched
    let! z = normal
    let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
    do! observe "stock" stock
    return stock
}

use outerSim = Simulation.create CPU numOuter sched.Steps
let _, watch = Simulation.foldWatch outerSim outerModel Quarterly
let outerStocks = Watcher.sliceObs "stock" 0 watch  // stock values at t=0.25

// Inner simulation conditioned on outer paths
let innerModel = model {
    let! dt = scheduleDt sched
    let! z = normal
    let! stock0 = batchInput outerStocks           // each outer path is a batch element
    let! stock = gbm z 0.05f.C 0.20f.C stock0 dt
    let! df = decay 0.05f.C dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}

use innerSim = Simulation.create CPU (numOuter * numInner) remainingSteps
let expectations = Simulation.foldBatchMeans innerSim innerModel numInner
// expectations[i] = E[payoff | S_outer[i]] — conditional expectation for each outer path
```

---

## Writing custom generators

Generators are ordinary F# functions. No interface, no base class — just call `normal`, `uniform`, and `evolve`:

```fsharp
// Ornstein-Uhlenbeck mean-reverting process
let ornsteinUhlenbeck kappa theta sigma x0 dt : ModelCtx -> Expr = fun ctx ->
    let z = normal ctx
    evolve x0 (fun x ->
        x + kappa * (theta - x) * dt + sigma * Expr.sqrt dt * z) ctx

// Jump-diffusion with Poisson jumps
let jumpDiffusion rate vol lambda muJ sigmaJ spot dt : ModelCtx -> Expr = fun ctx ->
    let z = normal ctx       // diffusion shock
    let zJ = normal ctx      // jump size shock
    let u = uniform ctx      // uniform [0,1] for jump timing
    let jumpProb = lambda * dt
    let jumpSize = muJ + sigmaJ * zJ
    evolve spot (fun price ->
        let hasJump = u .< jumpProb
        let jump = Expr.select hasJump jumpSize 0.0f.C
        price * Expr.exp (drift + diffusion + jump)) ctx
```

Your generator composes with all built-in processes:

```fsharp
let model = model {
    let! dt = scheduleDt sched
    let! z = normal
    let! rate = ornsteinUhlenbeck 0.5f.C 0.05f.C 0.01f.C 0.03f.C dt
    let! stock = gbm z rate 0.20f.C 100.0f.C dt  // stochastic rate drives stock
    return stock
}
```

See [examples/CustomGenerator.fs](examples/CustomGenerator.fs) for more patterns.

---

## Examples

| Example | Description |
|---------|-------------|
| [EuropeanCall.fs](examples/EuropeanCall.fs) | Vanilla call with GBM and flat rate |
| [LocalVol.fs](examples/LocalVol.fs) | Local volatility surface pricing |
| [HestonModel.fs](examples/HestonModel.fs) | Heston stochastic volatility |
| [CustomGenerator.fs](examples/CustomGenerator.fs) | Custom generators: OU mean-reversion and jump-diffusion |
| [ForwardRateCurve.fs](examples/ForwardRateCurve.fs) | Term-structure-aware pricing |
| [CalendarSchedule.fs](examples/CalendarSchedule.fs) | Business day calendar with holidays |
| [NestedSimulation.fs](examples/NestedSimulation.fs) | Conditional expectations via nested MC |
| [FixedIndexedAnnuity.fs](examples/FixedIndexedAnnuity.fs) | FIA call-spread crediting with batch simulation |

Run examples:

```bash
dotnet run --project examples/Examples.fsproj              # all examples
dotnet run --project examples/Examples.fsproj -- heston    # just Heston
dotnet run --project examples/Examples.fsproj -- fia       # just FIA
```

---

## Build and test

```bash
dotnet build                                    # Build entire solution
dotnet test                                     # Run all tests (152 tests)
dotnet run --project src/App/App.fsproj         # Run console demo
```

---

## gRPC Service & Python Client

Cavere exposes all simulation operations over gRPC, letting Python (or any gRPC client) submit models and receive results from the GPU engine.

### Server setup

```bash
# Start the gRPC server (listens on http://localhost:5000 with HTTP/2)
dotnet run --project src/Grpc/Grpc.fsproj
```

The server supports all simulation modes: Fold, FoldWatch, Scan, BatchFold, BatchFoldWatch, BatchFoldMeans, plus server-side streaming for large results and session management for accelerator reuse.

### Python client installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
cd caverepy
uv sync                        # Create venv and install dependencies
uv run generate                # Generate protobuf stubs from simulation.proto
```

Verify the setup:

```bash
uv run lint                    # Lint with ruff
uv run typecheck               # Type check with ty
```

### Template-based models

Python builders construct `ModelSpec` protobuf messages that the server maps to `model { }` CEs using existing generators:

```python
from cavere import CavereClient, gbm, heston, call_payoff
import numpy as np

with CavereClient("localhost:5000") as client:
    # GBM European call
    spec = gbm(spot=100, rate=0.05, vol=0.20, steps=252, payoff=call_payoff(strike=100))
    values = client.fold(spec, num_scenarios=100_000)
    print(f"Call price: {np.mean(values):.4f}")

    # Heston stochastic vol
    spec = heston(spot=100, rate=0.05, v0=0.04, kappa=2.0, theta=0.04,
                  xi=0.3, rho=-0.7, steps=252, payoff=call_payoff(strike=100))
    values = client.fold(spec, num_scenarios=100_000)
    print(f"Heston price: {np.mean(values):.4f}")
```

Available templates: `gbm`, `gbm_local_vol`, `heston`, `multi_asset_heston`, `vasicek`, `cir`, `cirpp`.

### Custom expression tree models

For full flexibility, build expression trees in Python that serialize to the F# `Expr` AST:

```python
from cavere.expr import Const, Normal, ModelBuilder, exp, sqrt, max_

b = ModelBuilder()
dt, z = Const(1/252), Normal(0)
rate, vol, spot = Const(0.05), Const(0.20), Const(100.0)

stock = b.add_accum(spot, lambda s: s * exp((rate - Const(0.5) * vol * vol) * dt + vol * sqrt(dt) * z))
df = b.add_accum(Const(1.0), lambda d: d * exp(-rate * dt))
result = max_(stock - Const(100.0), Const(0.0)) * df

spec = b.build(result, normal_count=1, uniform_count=0, steps=252)

with CavereClient("localhost:5000") as client:
    values = client.fold(spec, num_scenarios=100_000)
```

### Batch simulation

Run multiple parameter sets in a single kernel launch (shared random scenarios):

```python
spec = gbm(spot=100, rate=0.05, vol=0.20, steps=252, payoff=call_payoff(strike=100))
strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

with CavereClient("localhost:5000") as client:
    means = client.batch_fold_means(spec, num_scenarios=50_000, batch_values=strikes)
    for strike, price in zip(strikes, means):
        print(f"K={strike:.0f}: {price:.4f}")
```

### Python examples

| Example | Description |
|---------|-------------|
| [european_call.py](caverepy/examples/european_call.py) | GBM call pricing with template builder |
| [heston_pricing.py](caverepy/examples/heston_pricing.py) | Heston stochastic vol pricing |
| [batch_portfolio.py](caverepy/examples/batch_portfolio.py) | Batch simulation across strikes |
| [custom_model.py](caverepy/examples/custom_model.py) | Custom Expr tree built in Python |

---

## AAA Economic Scenario Generator

Implementation of the Academy Interest Rate Generator (AIRG) Stochastic Log Volatility methodology for regulatory reserve and capital calculations.

### Interest Rate Model (SLV)

Three correlated mean-reverting processes:
- **Long rate** (20-year Treasury): log-rate with stochastic volatility
- **Spread**: short rate excess over long rate
- **Log volatility**: mean-reverting vol-of-vol process

```fsharp
open Cavere.Generators.AAA.Generator
open Cavere.Generators.AAA.Common

let model = buildRatesOnlyModel defaultRateParams
```

### Equity Model (SLV)

Stochastic log volatility capturing:
- Volatility clustering
- Negative skewness (leverage effect)
- Fat tails

### Scenario Selection

Reduce large scenario sets to representative subsets while preserving statistical properties:

```fsharp
open Cavere.Generators.AAA.Selection

// Stratified selection by percentile buckets
let selected = stratifiedSelect terminalEquity defaultBucketEdges 20

// Tail-preserving selection for CTE calculations
let selected = tailPreservingSelect terminalEquity 50 50 400

// Evaluate selection quality
let quality = evaluateSelection terminalEquity selected
printQuality quality
```

---

## Project structure

```
Cavere.sln
├── src/Core/                  Core DSL, compiler, engine
│   ├── Expr.fs                Expression discriminated union (DiffVar for AD)
│   ├── Schedule.fs            Time step schedules
│   ├── Model.fs               Model record and builder CE
│   ├── Symbolic.fs            Expression simplification and symbolic differentiation
│   ├── Analysis.fs            Closed-form detection and analytical evaluation
│   ├── Compiler.Common.fs     Surface layout, expression emission, topo sort
│   ├── Compiler.Codegen.fs    C# templates, dynamic emitters
│   ├── Compiler.Regular.fs    Fold, FoldWatch, Scan kernels
│   ├── Compiler.Batch.fs      FoldBatch, FoldBatchWatch kernels
│   ├── Compiler.Diff.fs       Forward-mode AD via expression transformation
│   ├── Compiler.Adjoint.fs    Reverse-mode AD kernel (tape + backward pass)
│   ├── Compiler.fs            Unified compiler API facade
│   ├── Device.fs              ILGPU context management, multi-device support
│   ├── Transfer.fs            Pinned memory pool and async GPU transfers
│   ├── Engine.fs              Kernel launch, result extraction, pinned + multi-device
│   ├── Watcher.fs             Observation buffer management
│   ├── Kernel.fs              Kernel compilation and caching
│   ├── Simulation.fs          High-level orchestration API
│   └── Output.fs              CSV and Parquet export
├── src/Generators/            Finance-specific generators
│   ├── Common.fs              Discount factor (decay)
│   ├── Equity.fs              GBM, local vol, Heston
│   ├── Rates.fs               Vasicek, CIR, CIR++, curve utilities
│   ├── AAA.Common.fs          AAA ESG types and parameters
│   ├── AAA.Rates.fs           SLV interest rate model
│   ├── AAA.Equity.fs          SLV equity model
│   ├── AAA.Bonds.fs           Duration-based bond funds
│   ├── AAA.Selection.fs       Scenario selection utilities
│   └── AAA.Generator.fs       Combined AAA scenario generator
├── src/Actuarial/             Actuarial product definitions
│   ├── Decrements.*.fs        Mortality, surrender, withdrawal decrements
│   ├── Product.*.fs           Life, Fixed, FIA, RILA, VA product definitions
│   ├── Policy.fs              Policy and policyholder records
│   └── Model.*.fs             Actuarial projection models
├── src/Grpc.Proto/            C# protobuf code generation
│   └── Grpc.Proto.csproj      Grpc.Tools requires a .csproj
├── src/Grpc/                  F# gRPC server
│   ├── Protos/simulation.proto Service + message definitions
│   ├── ModelFactory.fs         Proto messages → Model via generators
│   ├── SimulationServiceImpl.fs gRPC service implementation
│   └── Program.fs              ASP.NET Core host
├── src/App/                   Console demo application
├── tests/                     xUnit test suite
├── examples/                  Runnable example project
└── caverepy/                  Python client package
    ├── pyproject.toml          uv project config, ruff + ty settings
    ├── src/cavere/client.py    High-level gRPC client wrapper
    ├── src/cavere/expr.py      Python Expr DSL + ModelBuilder
    ├── scripts/generate_protos.py  Stub generation from .proto
    └── examples/               Python usage examples
```

---

## Performance notes

- **CPU vs GPU**: Use `CPU` for development and testing. Switch to `GPU` for production workloads with >100k simulations.
- **Memory**: Each simulation thread uses registers for accumulators. Surfaces are shared across threads. Observer buffers scale with `numObs * numSims * numObservers`.
- **Compilation**: First run incurs Roslyn compilation overhead (~100-500ms). Subsequent runs reuse the compiled kernel.
- **Batch efficiency**: Batch simulation is more efficient than sequential kernel launches — one launch for N parameter sets vs N launches.

---

## Automatic differentiation

Cavere supports automatic differentiation for computing sensitivities (Greeks) alongside simulation values. Mark parameters as differentiable with `DiffVar`, then choose from three AD modes based on your needs.

### DiffVar — differentiable parameters

Replace `Const` with `DiffVar(index, value)` for any parameter you want derivatives with respect to:

```fsharp
let deltaModel = model {
    let dt = (1.0f / 252.0f).C
    let! z = normal
    // Mark spot (index 0) and vol (index 1) as differentiable
    let! stock = gbm z 0.05f.C (DiffVar(1, 0.20f)) (DiffVar(0, 100.0f)) dt
    let! df = decay 0.05f.C dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}
```

### Dual mode — first-order forward

Computes value + all first-order derivatives in a single forward pass. Best for 1-4 DiffVars.

```fsharp
let expanded, diffVars = CompilerDiff.transformDual deltaModel

use sim = Simulation.create CPU 100_000 252
let finals, watch = Simulation.foldWatch sim expanded Monthly
let deltas = Watcher.terminals "__deriv_0" watch  // dV/dSpot
let vegas = Watcher.terminals "__deriv_1" watch   // dV/dVol
printfn "Delta: %.4f  Vega: %.4f" (Array.average deltas) (Array.average vegas)
```

### HyperDual mode — second-order (gamma)

Computes value + first-order + second-order derivatives. Supports diagonal mode (d²V/dSi² only, for gamma) and full mode (all cross-terms d²V/dSi·dSj, for vanna/volga).

```fsharp
// Diagonal: only d²V/dSi² (gamma, volga)
let expanded, _ = CompilerDiff.transformHyperDual true deltaModel

use sim = Simulation.create CPU 100_000 252
let finals, watch = Simulation.foldWatch sim expanded Monthly
let deltas = Watcher.terminals "__deriv_0" watch    // dV/dSpot (delta)
let gammas = Watcher.terminals "__deriv2_0" watch   // d²V/dSpot² (gamma)

// Full: all cross-terms including d²V/dSpot·dVol (vanna)
let expandedFull, _ = CompilerDiff.transformHyperDual false deltaModel
// Observers: __deriv2_0_0 (gamma), __deriv2_0_1 (vanna), __deriv2_1_1 (volga)
```

### Adjoint mode — reverse-mode for many parameters

Computes value + all first-order derivatives via forward tape + backward adjoint propagation. Best when numDiffVars > 4 (e.g. calibration). Uses extra GPU memory for the tape (numAccums × steps × numScenarios floats).

```fsharp
// Model with many differentiable parameters
let calibrationModel = model {
    let dt = (1.0f / 252.0f).C
    let! z = normal
    let! stock = gbm z (DiffVar(0, 0.05f)) (DiffVar(1, 0.20f)) (DiffVar(2, 100.0f)) dt
    let! df = decay (DiffVar(3, 0.05f)) dt
    return Expr.max (stock - (DiffVar(4, 100.0f))) 0.0f.C * df
}

use sim = Simulation.create CPU 100_000 252
let values, adjoints = Simulation.foldAdjoint sim calibrationModel
// values: float32[numScenarios] — option prices
// adjoints: float32[numScenarios, numDiffVars] — all sensitivities per path
let meanDelta = Array.init sim.NumScenarios (fun i -> adjoints.[i, 2]) |> Array.average
```

### Choosing an AD method

`CompilerDiff.recommend` analyzes a model and suggests the best approach:

```fsharp
let mode, description = CompilerDiff.recommend myModel
printfn "%s" description
// "2 DiffVar(s): Dual (forward mode) recommended.
//   Adds 2 derivative accumulators (1 per DiffVar).
//   All derivatives computed in a single forward pass.
//   For 2nd-order (gamma): use HyperDual(diagonal=true)."
```

### DiffMode types

```fsharp
type DiffMode =
    | Dual                         // 1st-order forward (fastest for few DiffVars)
    | HyperDual of diagonal: bool  // 2nd-order forward (gamma, vanna, volga)
    | Jet of order: int            // Higher-order Taylor (reserved for future use)
    | Adjoint                      // Reverse mode (best for many DiffVars)
```

### Comparison

| Method | Order | Cost | Memory | Best for |
|--------|-------|------|--------|----------|
| Dual (forward) | 1st | 1 pass, N×accums extra | Registers only | 1-4 DiffVars |
| HyperDual | 2nd | 1 pass, N²×accums extra | Registers only | Gamma, convexity |
| Adjoint (reverse) | 1st | 2 passes (fwd+bwd) | Tape buffer | 5+ DiffVars |
| Finite differences | Any | 2N+1 passes | None | Validation, quick checks |
| Symbolic (`Symbolic.diff`) | Any | Pre-computation | None | Closed-form only |

---

## Symbolic analysis

The symbolic engine provides algebraic simplification, symbolic differentiation, and closed-form pattern detection — all operating on the `Expr` AST before compilation.

### Expression simplification

```fsharp
open Cavere.Core

let expr = Mul(Const 2.0f, Add(Const 0.0f, Mul(Const 1.0f, x)))
let simplified = Symbolic.fullySimplify expr
// Result: Mul(Const 2.0f, x)  — identity and zero rules applied until fixed point

let nodes_before = Symbolic.countNodes expr    // 5
let nodes_after = Symbolic.countNodes simplified // 3
```

`Symbolic.simplify` applies one pass of algebraic rules (identity elimination, constant folding, double negation, exp/log cancellation). `Symbolic.fullySimplify` iterates until the expression stops changing (max 100 iterations).

### Symbolic differentiation

```fsharp
// Differentiate expression w.r.t. DiffVar index 0
let derivative = Symbolic.diff expr 0
// Applies standard rules: sum, product, quotient, chain
// Result is automatically simplified
```

### Closed-form detection

`Analysis.analyzeModel` inspects a model's structure and attempts to match it against known analytical solutions:

```fsharp
let callModel = model {
    let dt = (1.0f / 252.0f).C
    let! z = normal
    let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
    let! df = decay 0.05f.C dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}

match Analysis.analyzeModel callModel with
| ClosedForm solution ->
    let price = Analysis.evaluate solution
    let greeks = Analysis.evaluateGreeks solution
    printfn "BS Price: %.4f, Delta: %.4f" price greeks.Delta
| RequiresMC reason ->
    printfn "Needs Monte Carlo: %s" reason
```

Detected patterns: `BlackScholesCall`, `BlackScholesPut`, `Forward`, `ZeroCouponBond`.

---

## GPU memory optimization

### Pinned memory transfers

Page-locked (pinned) memory enables DMA transfers between CPU and GPU, bypassing the OS page fault mechanism. The `PinnedPool` class manages a reusable pool of pinned buffers:

```fsharp
use sim = Simulation.create GPU 1_000_000 252
use pool = new PinnedPool(sim.Accelerator)

// Pinned fold — uses DMA for surface upload and result download
let results = Simulation.foldPinned sim pool myModel
```

For lower-level control:

```fsharp
// Direct transfer helpers
let deviceBuf = Transfer.copyToDevicePinned accel hostData
let hostData = Transfer.copyFromDevicePinned accel deviceBuf

// Async transfers via streams
use stream = accel.CreateStream()
Transfer.copyToDeviceAsync stream pinnedSrc deviceTarget
Transfer.copyFromDeviceAsync stream deviceSrc pinnedDst
stream.Synchronize()
```

### Multi-device simulation

Split work across multiple GPU accelerators:

```fsharp
let config = { DeviceType = GPU; DeviceCount = 2 }
use sim = Simulation.createMulti config 1_000_000 252
let results = Simulation.foldMulti sim myModel
// Scenarios split across GPUs with unique random seeds per device slice
```

Each device gets a slice of scenarios with `indexOffset` ensuring unique random streams. Results are concatenated automatically.

### Output formats

Export simulation results to CSV or Parquet:

```fsharp
// CSV export
Output.writeFold "results.csv" Csv values
Output.writeScan "paths.csv" Csv scanData

// Parquet export (columnar, compressed)
Output.writeFold "results.parquet" Parquet values
Output.writeScan "paths.parquet" Parquet scanData
```

---

## Feature composability

The features compose naturally because they all operate on the same `Expr` AST:

### AD + any generator

`DiffVar` composes with all generators — `gbm`, `heston`, `vasicek`, `cir`, and any custom generator. The AD transformation is applied after model construction, so generators don't need to know about differentiation:

```fsharp
// Heston vega via AD — just replace vol parameter with DiffVar
let vegaModel = model {
    let! dt = scheduleDt sched
    let! z = normal
    let! stock = heston z 0.05f.C (DiffVar(0, 0.04f)) 1.5f.C 0.04f.C 0.3f.C -0.7f 100.0f.C dt
    let! df = decay 0.05f.C dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}
let expanded, _ = CompilerDiff.transformDual vegaModel
```

### Symbolic simplify on derivatives

Derivative expressions from AD can be simplified before compilation, reducing kernel size:

```fsharp
let expanded, _ = CompilerDiff.transformDual myModel
// Derivative accumulators contain expressions like:
//   Mul(Const 0.0f, x) + Mul(Const 1.0f, y)
// fullySimplify collapses these to just: y
```

### Kernel caching across multi-device

Compiled kernels are cached by model identity (`ConditionalWeakTable`). When using multi-device simulation, all accelerators share the same compiled kernel — Roslyn compilation happens once.

### Batch + AD

AD composes with batch simulation. Each batch element computes its own derivatives:

```fsharp
let batchModel = model {
    let! spot = batchInput [| 90.0f; 100.0f; 110.0f |]
    let! z = normal
    let dt = (1.0f / 252.0f).C
    let! stock = gbm z (DiffVar(0, 0.05f)) 0.20f.C spot dt
    let! df = decay (DiffVar(0, 0.05f)) dt
    return Expr.max (stock - 100.0f) 0.0f.C * df
}
let expanded, _ = CompilerDiff.transformDual batchModel
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Commercial Support

Need help integrating Cavere into your production pipeline? 
I offer consulting services for:
- Custom Generator implementation (ESG, complex riders)
- GPU infrastructure setup and optimization
- Legacy model migration (AXIS/Prophet to Cavere)

[Contact me](mailto:your.email@example.com) for rates and availability.

# DSL Capabilities and Extensions

This document details the current capabilities of the Cavere DSL, identifies gaps for actuarial use cases, and proposes extensions.

---

## Current DSL Primitives

### Expression Types

| Primitive | Description | Use Case |
|-----------|-------------|----------|
| `Const` | Constant value | Parameters, strikes, rates |
| `Normal` | Standard normal random | Diffusion shocks |
| `Uniform` | Uniform [0,1] random | Event triggers, sampling |
| `TimeIndex` | Current time step | Time-dependent logic |
| `AccumRef` | Reference to accumulator | Evolving state |
| `Lookup1D` | Direct surface lookup | Rate curves |
| `Interp1D` | Interpolated 1D lookup | Smooth curves |
| `Interp2D` | Interpolated 2D lookup | Vol surfaces |
| `BatchRef` | Per-batch element data | Policy parameters |

### Operators and Functions

| Category | Operations |
|----------|------------|
| Arithmetic | `+`, `-`, `*`, `/`, unary `-` |
| Comparison | `.>`, `.>=`, `.<`, `.<=` (return 0/1) |
| Math | `exp`, `log`, `sqrt`, `abs`, `max`, `min` |
| Conditional | `select` (branchless ternary) |

### Model Building

| Primitive | Description |
|-----------|-------------|
| `normal` | Allocate standard normal random variable |
| `uniform` | Allocate uniform random variable |
| `evolve` | Create accumulator with init and step function |
| `surface1d` | Register 1D curve data |
| `surface2d` | Register 2D surface data |
| `batchInput` | Register per-policy/batch data |
| `observe` | Record variable for path inspection |
| `correlatedNormals` | Cholesky-correlated normals |

### Simulation Modes

| Mode | Output | Use Case |
|------|--------|----------|
| `fold` | Terminal values | Pricing, valuation |
| `foldWatch` | Terminal + observed paths | Debugging, analysis |
| `scan` | Full path history | Path-dependent analysis |
| `foldBatch` | Terminal values, shared scenarios | Portfolio valuation |
| `foldBatchMeans` | Per-batch averages | Nested stochastic |

---

## Use Case Coverage

### Valuation ✓

**Requirements:**
- Run stochastic scenarios
- Compute present value of cash flows
- Aggregate across policies
- Nested stochastic for reserves

**DSL Support:** Complete. Build product definitions as functions composing primitives.

```fsharp
let valuationModel = model {
    let! dt = scheduleDt schedule
    let! z = normal
    let! accountValue = evolve initialAV (fun av ->
        av * Expr.exp(fundReturn * dt) - fees * av * dt - withdrawal)
    let! df = decay rate dt
    let! claimsPV = evolve 0.0f.C (fun pv ->
        pv + Expr.max(guarantee - accountValue, 0.0f.C) * df * mortalityRate * dt)
    return claimsPV
}
```

### Pricing ✓

**Requirements:**
- Same as valuation
- Often simpler (no nested stochastic)
- Fast iteration for what-if analysis

**DSL Support:** Complete.

### Hedging ⚠️

**Requirements:**
- Greeks: Delta, Gamma, Vega, Rho, Theta
- Fast recomputation as market moves
- Hedge ratio calculation

**DSL Support:** Partial. Finite difference works but is slow. See [Automatic Differentiation](#automatic-differentiation-for-greeks) section.

---

## Identified Gaps

### 1. Automatic Differentiation (High Priority)

See dedicated section below.

### 2. Multiple Outputs Per Path (Medium Priority)

**Current limitation:** Single return value per model.

```fsharp
// Current - only one value
return discountedPayoff

// Needed - structured output
return {| PV = pv; Claims = claims; Fees = fees |}
```

**Proposed extension:**

```fsharp
type ModelOutput = {
    PV: Expr
    ClaimsPaid: Expr
    FeeIncome: Expr
    AccountValue: Expr
}

let model = model {
    // ... computations ...
    return {
        PV = discountedPayoff
        ClaimsPaid = totalClaims
        FeeIncome = totalFees
        AccountValue = finalAV
    }
}

// Generates kernel that writes multiple output arrays
let results = Simulation.foldMulti sim model
// results.PV: float32[]
// results.ClaimsPaid: float32[]
// etc.
```

### 3. Probabilistic Event Primitives (Low Priority)

**Current workaround:**

```fsharp
let! u = uniform
let died = u .< (mortalityRate * dt)
let factor = Expr.select died 0.0f.C 1.0f.C
```

**Proposed primitive:**

```fsharp
let! died = bernoulli (mortalityRate * dt)  // Returns Expr (0 or 1)
let! numEvents = poisson lambda              // For jump processes
```

### 4. Named Accumulator Collections (Medium Priority)

See [Accumulator Arrays with Named Indexing](#accumulator-arrays-with-named-indexing) section.

---

## Automatic Differentiation for Greeks

### The Problem

Hedging requires sensitivities (Greeks) of option values to market inputs:

| Greek | Sensitivity To | Hedge Instrument |
|-------|---------------|------------------|
| Delta (Δ) | Spot price | Underlying |
| Gamma (Γ) | Delta | Options |
| Vega (ν) | Volatility | Vol swaps, options |
| Rho (ρ) | Interest rate | Bonds, swaps |
| Theta (θ) | Time | N/A (P&L attribution) |

### Current Approach: Finite Difference

```fsharp
let computeGreeksFD (model: Model) (sim: Simulation) (bumpSize: float32) =
    let basePrice = Simulation.fold sim model |> Array.average

    // Delta: bump spot up and down
    let priceUp = Simulation.fold sim (model |> bumpSpot (+bumpSize)) |> Array.average
    let priceDn = Simulation.fold sim (model |> bumpSpot (-bumpSize)) |> Array.average
    let delta = (priceUp - priceDn) / (2.0f * bumpSize)

    // Gamma: second derivative
    let gamma = (priceUp - 2.0f * basePrice + priceDn) / (bumpSize * bumpSize)

    // Vega: bump vol
    let priceVolUp = Simulation.fold sim (model |> bumpVol (+0.01f)) |> Array.average
    let priceVolDn = Simulation.fold sim (model |> bumpVol (-0.01f)) |> Array.average
    let vega = (priceVolUp - priceVolDn) / 0.02f

    // ... repeat for each Greek
    { Delta = delta; Gamma = gamma; Vega = vega; ... }
```

**Problems:**
- Requires 2N+1 simulations for N Greeks (Delta, Gamma, Vega, Rho = 9 simulations)
- Numerical instability (bump size tradeoff)
- Doesn't scale for large portfolios with daily hedging

### Better Approach: Automatic Differentiation

AD computes exact derivatives in a single forward pass (forward mode) or backward pass (adjoint/reverse mode).

#### Forward Mode AD

Compute derivative alongside value using dual numbers:

```
Regular:  f(x) → y
Forward:  f(x, dx) → (y, dy)  where dy = df/dx * dx
```

For Monte Carlo, forward mode computes one Greek per pass:

```fsharp
// Single simulation gives value AND delta
let (value, delta) = Simulation.foldWithDelta sim model
```

#### Adjoint Mode AD (Reverse Mode)

Propagate sensitivities backward from output to all inputs:

```
Forward:  inputs → intermediate values → output
Adjoint:  d(output)/d(inputs) ← adjoint values ← seed (1.0)
```

For Monte Carlo, adjoint mode computes ALL Greeks in one backward pass:

```fsharp
// Single simulation gives value AND all Greeks
let (value, greeks) = Simulation.foldWithAD sim model
// greeks.Delta, greeks.Gamma, greeks.Vega, greeks.Rho all computed
```

### Implementing AD in the DSL

#### Option A: Dual Number Expr (Forward Mode)

Extend Expr to carry derivative alongside value:

```fsharp
// Current Expr
type Expr =
    | Const of float32
    | Add of Expr * Expr
    | Mul of Expr * Expr
    | Exp of Expr
    // ...

// Extended for forward-mode AD
type DualExpr = {
    Value: Expr      // The value
    Deriv: Expr      // Derivative w.r.t. some input
}

// Dual number arithmetic
let dualAdd (a: DualExpr) (b: DualExpr) = {
    Value = Add(a.Value, b.Value)
    Deriv = Add(a.Deriv, b.Deriv)  // d(a+b) = da + db
}

let dualMul (a: DualExpr) (b: DualExpr) = {
    Value = Mul(a.Value, b.Value)
    Deriv = Add(Mul(a.Deriv, b.Value), Mul(a.Value, b.Deriv))  // d(a*b) = da*b + a*db
}

let dualExp (a: DualExpr) = {
    Value = Exp(a.Value)
    Deriv = Mul(Exp(a.Value), a.Deriv)  // d(exp(a)) = exp(a) * da
}
```

**Generated code for Delta:**

```csharp
// Value computation
float stock = stock * exp(drift + vol * sqrt(dt) * z);

// Derivative computation (forward mode, seeded with dStock/dSpot = 1)
float d_stock = d_stock * exp(drift + vol * sqrt(dt) * z)
              + stock * exp(drift + vol * sqrt(dt) * z) * d_drift;
```

#### Option B: Tape-Based AD (Adjoint Mode)

Record operations in a tape, then replay backward:

```fsharp
type TapeEntry =
    | TapeAdd of resultId: int * leftId: int * rightId: int
    | TapeMul of resultId: int * leftId: int * rightId: int
    | TapeExp of resultId: int * argId: int
    // ...

type ADTape = {
    Entries: TapeEntry[]
    Values: float32[]      // Forward pass values
    Adjoints: float32[]    // Backward pass adjoints
}

let backward (tape: ADTape) =
    // Seed output adjoint with 1.0
    tape.Adjoints.[outputId] <- 1.0f

    // Replay tape backward
    for i in tape.Entries.Length - 1 .. -1 .. 0 do
        match tape.Entries.[i] with
        | TapeAdd(r, l, ri) ->
            tape.Adjoints.[l] <- tape.Adjoints.[l] + tape.Adjoints.[r]
            tape.Adjoints.[ri] <- tape.Adjoints.[ri] + tape.Adjoints.[r]
        | TapeMul(r, l, ri) ->
            tape.Adjoints.[l] <- tape.Adjoints.[l] + tape.Adjoints.[r] * tape.Values.[ri]
            tape.Adjoints.[ri] <- tape.Adjoints.[ri] + tape.Adjoints.[r] * tape.Values.[l]
        | TapeExp(r, a) ->
            tape.Adjoints.[a] <- tape.Adjoints.[a] + tape.Adjoints.[r] * tape.Values.[r]
```

#### Option C: Source Transformation (Compile-Time AD)

Transform the Expr AST to generate adjoint code:

```fsharp
// Original model
let model = model {
    let! z = normal
    let! stock = evolve spot (fun s -> s * Expr.exp(drift + vol * sqrt(dt) * z))
    return Expr.max (stock - strike) 0.0f.C * df
}

// AD-transformed model (generated)
let modelWithAdjoints = model {
    // Forward pass (same as original)
    let! z = normal
    let! stock = evolve spot (fun s -> s * Expr.exp(drift + vol * sqrt(dt) * z))
    let payoff = Expr.max (stock - strike) 0.0f.C * df

    // Backward pass (generated)
    let! adj_payoff = Const 1.0f  // Seed
    let! adj_stock = ... // Propagate through max, subtract, multiply
    let! adj_spot = ... // Accumulate through evolve

    return (payoff, adj_spot)  // Value and Delta
}
```

### Recommended Approach

**Phase 1: Forward Mode (Simpler)**
- Extend Expr to DualExpr
- Generate code that computes value + one derivative
- Run N times for N Greeks (still better than 2N+1 finite diff)

**Phase 2: Adjoint Mode (Optimal)**
- Source transformation at Expr AST level
- Generate adjoint accumulation code
- All Greeks in single simulation

### AD Performance Comparison

| Method | Simulations | Relative Cost |
|--------|-------------|---------------|
| Finite Difference | 2N+1 = 9 | 9x |
| Forward Mode AD | N = 4 | 4x |
| Adjoint Mode AD | 1 + backward | ~2-3x |

For daily hedging with 4 Greeks on 100k policies:
- Finite Diff: 9 × base time
- Adjoint AD: 2-3 × base time
- **Speedup: 3-4x**

### AD Through `evolve` (The Hard Part)

The challenge is differentiating through the time-stepping loop:

```fsharp
let! stock = evolve spot (fun s -> s * Expr.exp(drift + vol * sqrt(dt) * z))
```

**Forward mode:** Carry derivative alongside value through each step.

```csharp
// Generated code
float stock = spot;
float d_stock = 1.0f;  // d(stock)/d(spot) = 1 initially

for (int step = 0; step < numSteps; step++) {
    float z = normal(rng);
    float growth = exp(drift + vol * sqrt(dt) * z);

    // Forward mode: propagate derivative
    d_stock = d_stock * growth;  // Chain rule through multiplication
    stock = stock * growth;
}

// d_stock now contains d(finalStock)/d(spot) = Delta
```

**Adjoint mode:** Record tape forward, replay backward.

```csharp
// Forward pass - record values
float[] stockHistory = new float[numSteps + 1];
stockHistory[0] = spot;
for (int step = 0; step < numSteps; step++) {
    stockHistory[step + 1] = stockHistory[step] * growth[step];
}

// Backward pass - accumulate adjoints
float adj_spot = 0.0f;
float adj_stock = adj_payoff;  // Seeded from output
for (int step = numSteps - 1; step >= 0; step--) {
    adj_spot += adj_stock * (stockHistory[step + 1] / stockHistory[step]);
    // ... propagate to other inputs
}
```

---

## AD Through Evolve Loops: Deep Dive

The `evolve` primitive creates a recurrence relation:

```
s₀ = initial
s₁ = f(s₀, z₁)
s₂ = f(s₁, z₂)
...
sₙ = f(sₙ₋₁, zₙ)
```

Differentiating through this loop is the core challenge of AD for Monte Carlo.

### Forward Mode Through Evolve

**Idea:** Carry derivative alongside value at each step using the chain rule.

For GBM: `sₜ₊₁ = sₜ × exp(drift + σ√dt × zₜ)`

Let `ṡ = ∂s/∂s₀` (sensitivity to initial spot).

```
Initial:
  s₀ = spot
  ṡ₀ = 1.0  (∂s₀/∂spot = 1)

Step t → t+1:
  g = exp(drift + σ√dt × zₜ)
  sₜ₊₁ = sₜ × g
  ṡₜ₊₁ = ṡₜ × g  (chain rule: ∂sₜ₊₁/∂spot = ∂sₜ₊₁/∂sₜ × ∂sₜ/∂spot = g × ṡₜ)

Terminal:
  ṡₙ = ∂sₙ/∂spot = ∏ᵢ gᵢ = sₙ/s₀  (for GBM specifically)
```

**Generated code:**

```csharp
// Forward mode AD through evolve
float s = spot;
float ds_dspot = 1.0f;  // Seed: ∂s/∂spot = 1

for (int t = 0; t < numSteps; t++) {
    float z = normal(rng, t);
    float g = exp(drift + vol * sqrt(dt) * z);

    // Value update
    s = s * g;

    // Derivative update (chain rule)
    ds_dspot = ds_dspot * g;
}

// Payoff: max(s - K, 0)
float payoff = max(s - strike, 0.0f);

// Delta = ∂payoff/∂spot = ∂payoff/∂s × ∂s/∂spot
float dpayoff_ds = (s > strike) ? 1.0f : 0.0f;  // Derivative of max
float delta = dpayoff_ds * ds_dspot;
```

**Key insight:** Forward mode only needs O(1) extra storage per accumulator - just the derivative value.

### Adjoint Mode Through Evolve

**Idea:** Store forward trajectory, then propagate sensitivities backward.

**Forward pass:** Run simulation, store all intermediate values.

```csharp
float[] s = new float[numSteps + 1];
float[] g = new float[numSteps];

s[0] = spot;
for (int t = 0; t < numSteps; t++) {
    float z = normal(rng, t);
    g[t] = exp(drift + vol * sqrt(dt) * z);
    s[t + 1] = s[t] * g[t];
}

float payoff = max(s[numSteps] - strike, 0.0f);
```

**Backward pass:** Propagate adjoints from output to inputs.

```csharp
// Adjoint of payoff w.r.t. itself = 1 (seed)
float adj_payoff = 1.0f;

// Adjoint through max
float adj_s_n = (s[numSteps] > strike) ? adj_payoff : 0.0f;

// Propagate backward through time
float adj_s = adj_s_n;
float adj_spot = 0.0f;
float adj_vol = 0.0f;

for (int t = numSteps - 1; t >= 0; t--) {
    // s[t+1] = s[t] * g[t]
    // ∂L/∂s[t] += ∂L/∂s[t+1] × g[t]
    // ∂L/∂g[t] += ∂L/∂s[t+1] × s[t]

    float adj_g = adj_s * s[t];
    adj_s = adj_s * g[t];  // Propagate to previous step

    // g[t] = exp(drift + vol * sqrt(dt) * z[t])
    // ∂L/∂vol += ∂L/∂g[t] × g[t] × sqrt(dt) × z[t]
    adj_vol += adj_g * g[t] * sqrt(dt) * z[t];
}

// adj_s now contains ∂L/∂s[0] = ∂L/∂spot
adj_spot = adj_s;

// Results: delta = adj_spot, vega = adj_vol
```

**Memory cost:** O(numSteps) to store forward trajectory. For 360 steps, this is 360 floats per accumulator per thread - manageable on GPU.

### Checkpointing for Memory-Constrained AD

For very long simulations, store only checkpoints and recompute:

```
Full storage:     [s₀, s₁, s₂, s₃, s₄, s₅, s₆, s₇, ...]  O(n)
Checkpointing:    [s₀,     s₂,     s₄,     s₆,     ...]  O(√n)
                        ↑ recompute s₁ from s₀ when needed
```

**Binomial checkpointing** achieves O(log n) memory with O(n log n) recomputation.

---

## Higher-Order Derivatives (Gamma)

Gamma (Γ) is the second derivative: `Γ = ∂²V/∂S²`

### Approach 1: Finite Difference on Delta

```fsharp
let gamma spot bumpSize model =
    let deltaUp = computeDelta (spot + bumpSize) model
    let deltaDn = computeDelta (spot - bumpSize) model
    (deltaUp - deltaDn) / (2.0f * bumpSize)
```

**Cost:** 4 simulations (2 for deltaUp, 2 for deltaDn) + numerical instability.

### Approach 2: Hyper-Dual Numbers (Forward Mode)

Extend dual numbers to track second derivatives:

```
Dual:       (value, derivative)
Hyper-dual: (value, derivative, second_derivative)
```

**Hyper-dual arithmetic:**

```fsharp
type HyperDual = {
    V: float32   // f(x)
    D: float32   // f'(x)
    D2: float32  // f''(x)
}

// Addition: (a + b)'' = a'' + b''
let hyperAdd a b = {
    V = a.V + b.V
    D = a.D + b.D
    D2 = a.D2 + b.D2
}

// Multiplication: (a × b)'' = a'' × b + 2 × a' × b' + a × b''
let hyperMul a b = {
    V = a.V * b.V
    D = a.D * b.V + a.V * b.D
    D2 = a.D2 * b.V + 2.0f * a.D * b.D + a.V * b.D2
}

// Exponential: (exp(a))'' = exp(a) × (a'' + (a')²)
let hyperExp a = {
    V = exp(a.V)
    D = exp(a.V) * a.D
    D2 = exp(a.V) * (a.D2 + a.D * a.D)
}
```

**Generated code for Gamma:**

```csharp
// Hyper-dual through evolve
float s = spot;
float ds = 1.0f;    // First derivative seed
float d2s = 0.0f;   // Second derivative seed

for (int t = 0; t < numSteps; t++) {
    float z = normal(rng, t);
    float g = exp(drift + vol * sqrt(dt) * z);

    // Hyper-dual multiplication: s_new = s * g
    // (g is constant w.r.t. spot, so dg = 0, d2g = 0)
    float s_new = s * g;
    float ds_new = ds * g;
    float d2s_new = d2s * g;

    s = s_new;
    ds = ds_new;
    d2s = d2s_new;
}

// Payoff and its derivatives
float payoff, dpayoff, d2payoff;
if (s > strike) {
    payoff = s - strike;
    dpayoff = ds;
    d2payoff = d2s;
} else {
    payoff = 0.0f;
    dpayoff = 0.0f;
    d2payoff = 0.0f;
}

// Results
float price = payoff;
float delta = dpayoff;
float gamma = d2payoff;
```

### Approach 3: Forward-over-Adjoint

Differentiate the adjoint code itself using forward mode:

```
1. Forward pass:  compute values
2. Backward pass: compute adjoints (first derivatives)
3. Forward-on-backward: differentiate adjoint computation for second derivatives
```

This is efficient when you need Gamma for many inputs - compute ∂(∂V/∂Sᵢ)/∂Sⱼ for all i,j.

---

## Cross-Derivatives (Vanna, Volga)

Cross-derivatives measure sensitivity to multiple inputs:

| Greek | Definition | Meaning |
|-------|------------|---------|
| Vanna | ∂²V/∂S∂σ | Delta sensitivity to vol |
| Volga | ∂²V/∂σ² | Vega sensitivity to vol |
| Charm | ∂²V/∂S∂t | Delta decay |

### Approach 1: Finite Difference

```fsharp
let vanna spot vol bumpS bumpV model =
    let delta_volUp = computeDelta spot (vol + bumpV) model
    let delta_volDn = computeDelta spot (vol - bumpV) model
    (delta_volUp - delta_volDn) / (2.0f * bumpV)
```

**Cost:** 4+ simulations per cross-derivative.

### Approach 2: Multi-Variate Hyper-Dual

Track derivatives with respect to multiple inputs simultaneously:

```fsharp
type MultiDual = {
    V: float32                      // f(x, y)
    Dx: float32                     // ∂f/∂x
    Dy: float32                     // ∂f/∂y
    Dxx: float32                    // ∂²f/∂x²
    Dxy: float32                    // ∂²f/∂x∂y  (cross-derivative)
    Dyy: float32                    // ∂²f/∂y²
}

// Multiplication with cross-derivatives
let multiMul a b = {
    V = a.V * b.V
    Dx = a.Dx * b.V + a.V * b.Dx
    Dy = a.Dy * b.V + a.V * b.Dy
    Dxx = a.Dxx * b.V + 2.0f * a.Dx * b.Dx + a.V * b.Dxx
    Dxy = a.Dxy * b.V + a.Dx * b.Dy + a.Dy * b.Dx + a.V * b.Dxy  // Mixed partial
    Dyy = a.Dyy * b.V + 2.0f * a.Dy * b.Dy + a.V * b.Dyy
}
```

**For spot (x) and vol (y):**

```csharp
// Initialize multi-dual for spot and vol
float s = spot;
float ds_dspot = 1.0f;   // ∂s/∂spot
float ds_dvol = 0.0f;    // ∂s/∂vol
float d2s_dspot2 = 0.0f; // ∂²s/∂spot²
float d2s_dspot_dvol = 0.0f;  // ∂²s/∂spot∂vol (Vanna contribution)
float d2s_dvol2 = 0.0f;  // ∂²s/∂vol²

for (int t = 0; t < numSteps; t++) {
    float z = normal(rng, t);

    // g = exp(drift + vol * sqrt(dt) * z)
    float g = exp(drift + vol * sqrt(dt) * z);
    float dg_dvol = g * sqrt(dt) * z;       // ∂g/∂vol
    float d2g_dvol2 = g * dt * z * z;       // ∂²g/∂vol²

    // Update s = s * g with multi-dual arithmetic
    float s_new = s * g;

    // First derivatives
    float ds_dspot_new = ds_dspot * g;
    float ds_dvol_new = ds_dvol * g + s * dg_dvol;

    // Second derivatives (using product rule)
    float d2s_dspot2_new = d2s_dspot2 * g;
    float d2s_dspot_dvol_new = d2s_dspot_dvol * g + ds_dspot * dg_dvol;  // Cross!
    float d2s_dvol2_new = d2s_dvol2 * g + 2.0f * ds_dvol * dg_dvol + s * d2g_dvol2;

    // Update all
    s = s_new;
    ds_dspot = ds_dspot_new;
    ds_dvol = ds_dvol_new;
    d2s_dspot2 = d2s_dspot2_new;
    d2s_dspot_dvol = d2s_dspot_dvol_new;
    d2s_dvol2 = d2s_dvol2_new;
}
```

### Approach 3: Adjoint-over-Adjoint (Hessian)

For the full Hessian matrix (all second derivatives), apply adjoint mode twice:

```
Forward:        inputs → values
Adjoint 1:      ∂output/∂values → ∂output/∂inputs (gradient)
Adjoint 2:      ∂(gradient)/∂inputs → Hessian
```

**Generated structure:**

```fsharp
// Original model
let model = computePayoff spot vol rate

// First adjoint (gradient)
let gradModel = adjoint model  // Generates: delta, vega, rho

// Second adjoint (Hessian)
let hessModel = adjoint gradModel  // Generates: gamma, vanna, volga, etc.
```

---

## Greeks Summary: AD Approach Selection

| Greek | Order | Best AD Approach | Cost |
|-------|-------|------------------|------|
| Delta (∂V/∂S) | 1st | Adjoint | 1 backward pass |
| Vega (∂V/∂σ) | 1st | Adjoint | 1 backward pass |
| Rho (∂V/∂r) | 1st | Adjoint | 1 backward pass |
| **All 1st order** | 1st | **Single adjoint** | **~2x base cost** |
| Gamma (∂²V/∂S²) | 2nd | Forward-over-adjoint or hyper-dual | ~3x base cost |
| Vanna (∂²V/∂S∂σ) | 2nd cross | Multi-dual or adjoint² | ~3-4x base cost |
| Volga (∂²V/∂σ²) | 2nd | Hyper-dual | ~3x base cost |
| **All 2nd order** | 2nd | **Adjoint-over-adjoint** | **~4-5x base cost** |

**Comparison to finite difference:**

| Method | 1st Order (4 Greeks) | 2nd Order (6 Greeks) |
|--------|----------------------|----------------------|
| Finite Difference | 9 sims | 25+ sims |
| AD | ~2x cost (1 sim) | ~4-5x cost (1 sim) |
| **Speedup** | **~4-5x** | **~5-6x** |

---

## Example: Delta of European Call with Local Volatility

This section provides a complete example of computing Delta for a European call option under a Local Volatility model using forward-mode AD.

### Model Definition

**Local Volatility dynamics:**
```
dS = (r - ½σ(t,S)²) S dt + σ(t,S) S dW
```

Where `σ(t,S)` is the local volatility looked up from a 2D surface.

**DSL model (without AD):**

```fsharp
let lvCallModel spot strike rate volSurface steps = model {
    let dt = Const(1.0f / float32 steps)
    let! surfId = surface2d volSurface.Times volSurface.Spots volSurface.Vols steps
    let! z = normal

    let! stock = evolve (Const spot) (fun s ->
        let vol = Interp2D(surfId, TimeIndex, s)
        let drift = (Const rate - Const 0.5f * vol * vol) * dt
        let diffusion = vol * Expr.sqrt(dt) * z
        s * Expr.exp(drift + diffusion)
    )

    let! df = decay (Const rate) dt
    return Expr.max(stock - Const strike, Const 0.0f) * df
}
```

### Forward-Mode AD: Mathematical Derivation

Let `S₀ = spot` and we want `∂V/∂S₀` (Delta).

**Step-by-step derivative propagation:**

```
Step 0 (initialization):
  S₀ = spot
  Ṡ₀ = ∂S₀/∂spot = 1.0

Step t → t+1:
  σₜ = σ(t, Sₜ)              // Vol lookup
  σ̇ₜ = ∂σ/∂S × Ṡₜ           // Vol sensitivity to spot (chain rule)

  gₜ = exp(drift + σₜ√dt × zₜ)
  ġₜ = gₜ × (∂drift/∂σ × σ̇ₜ + √dt × zₜ × σ̇ₜ)
     = gₜ × σ̇ₜ × (-σₜ × dt + √dt × zₜ)

  Sₜ₊₁ = Sₜ × gₜ
  Ṡₜ₊₁ = Ṡₜ × gₜ + Sₜ × ġₜ   // Product rule

Terminal:
  payoff = max(Sₙ - K, 0) × df
  ∂payoff/∂spot = (Sₙ > K ? 1 : 0) × df × Ṡₙ
```

### DSL Model with AD Annotation

```fsharp
// Extended model that requests Delta computation
let lvCallModelWithDelta spot strike rate volSurface steps = model {
    let dt = Const(1.0f / float32 steps)
    let! surfId = surface2d volSurface.Times volSurface.Spots volSurface.Vols steps
    let! z = normal

    // Mark spot as differentiable input
    let! spotDual = dualInput spot  // Returns DualExpr { Value = spot, Deriv = 1.0 }

    let! stock = evolveDual spotDual (fun s ->
        // s is DualExpr { Value, Deriv }
        let vol = Interp2D(surfId, TimeIndex, s.Value)

        // Vol surface derivative w.r.t. spot (finite diff approximation or analytic)
        let dVol_dS = Interp2DGradient(surfId, TimeIndex, s.Value)  // ∂σ/∂S
        let volDual = { Value = vol; Deriv = dVol_dS * s.Deriv }

        let drift = (Const rate - Const 0.5f * volDual * volDual) * dt
        let diffusion = volDual * Expr.sqrt(dt) * z

        s * dualExp(drift + diffusion)
    )

    let! df = decay (Const rate) dt

    // Payoff with derivative
    let payoff = dualMax(stock - Const strike, Const 0.0f) * df

    return (payoff.Value, payoff.Deriv)  // (price, delta)
}
```

### Generated C# Code (Forward-Mode AD)

```csharp
[Kernel]
public static void LVCallWithDelta(
    Index1D index,
    ArrayView<float> results,        // Output: prices
    ArrayView<float> deltas,         // Output: deltas
    ArrayView<float> volSurface,     // [time, spot] -> vol
    ArrayView<float> volTimes,
    ArrayView<float> volSpots,
    float spot,
    float strike,
    float rate,
    int numSteps,
    ArrayView<XorShift128Plus> rng)
{
    float dt = 1.0f / numSteps;
    float sqrtDt = MathF.Sqrt(dt);

    // Initialize dual number for stock
    float S = spot;
    float dS = 1.0f;  // ∂S/∂spot = 1 (seed)

    // Discount factor (not differentiated w.r.t. spot)
    float df = 1.0f;

    for (int t = 0; t < numSteps; t++)
    {
        // Generate random shock
        float z = GenerateNormal(ref rng[index]);

        // Local vol lookup: σ(t, S)
        float vol = Interp2D(volSurface, volTimes, volSpots, t * dt, S);

        // Vol gradient: ∂σ/∂S (for chain rule)
        float dVol_dS = Interp2DGradientS(volSurface, volTimes, volSpots, t * dt, S);

        // Dual vol: { vol, dVol_dS * dS }
        float vol_deriv = dVol_dS * dS;

        // Drift = (r - 0.5 * vol^2) * dt
        float drift = (rate - 0.5f * vol * vol) * dt;

        // ∂drift/∂spot = -vol * dVol * dt = -vol * vol_deriv * dt
        float drift_deriv = -vol * vol_deriv * dt;

        // Diffusion = vol * sqrt(dt) * z
        float diffusion = vol * sqrtDt * z;

        // ∂diffusion/∂spot = dVol * sqrt(dt) * z = vol_deriv * sqrt(dt) * z
        float diffusion_deriv = vol_deriv * sqrtDt * z;

        // Growth factor: g = exp(drift + diffusion)
        float g = MathF.Exp(drift + diffusion);

        // ∂g/∂spot = g * (∂drift/∂spot + ∂diffusion/∂spot)
        float g_deriv = g * (drift_deriv + diffusion_deriv);

        // Update S: S_new = S * g
        float S_new = S * g;

        // ∂S_new/∂spot = ∂S/∂spot * g + S * ∂g/∂spot (product rule)
        float dS_new = dS * g + S * g_deriv;

        S = S_new;
        dS = dS_new;

        // Update discount factor
        df = df * MathF.Exp(-rate * dt);
    }

    // Payoff: max(S - K, 0)
    float payoff;
    float dPayoff;

    if (S > strike)
    {
        payoff = S - strike;
        dPayoff = dS;  // ∂max(S-K,0)/∂spot = ∂S/∂spot when S > K
    }
    else
    {
        payoff = 0.0f;
        dPayoff = 0.0f;  // ∂max(S-K,0)/∂spot = 0 when S ≤ K
    }

    // Apply discount factor
    float price = payoff * df;
    float delta = dPayoff * df;

    // Store results
    results[index] = price;
    deltas[index] = delta;
}
```

### Vol Surface Gradient Helper

The `Interp2DGradientS` function computes ∂σ/∂S using finite differences on the vol surface:

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static float Interp2DGradientS(
    ArrayView<float> surface,
    ArrayView<float> times,
    ArrayView<float> spots,
    float t, float s)
{
    // Central difference for ∂σ/∂S
    float h = 0.01f * s;  // 1% bump
    float volUp = Interp2D(surface, times, spots, t, s + h);
    float volDn = Interp2D(surface, times, spots, t, s - h);
    return (volUp - volDn) / (2.0f * h);
}
```

Alternatively, if the vol surface is parameterized analytically, compute the gradient analytically.

### Usage

```fsharp
// Build model
let model = lvCallModelWithDelta 100.0f 100.0f 0.05f volSurface 252

// Run simulation - get price AND delta in single pass
use sim = Simulation.create GPU 100_000 252
let prices, deltas = Simulation.foldWithDelta sim model

let price = Array.average prices
let delta = Array.average deltas

printfn "Price: %.4f" price
printfn "Delta: %.4f" delta
```

### Numerical Validation

Compare AD delta to finite-difference delta:

```fsharp
let validateDelta spot strike rate volSurface steps numSims =
    // AD Delta (single simulation)
    let model = lvCallModelWithDelta spot strike rate volSurface steps
    use sim = Simulation.create GPU numSims steps
    let _, deltas = Simulation.foldWithDelta sim model
    let adDelta = Array.average deltas

    // Finite-difference Delta (three simulations)
    let bumpSize = 0.01f * spot
    let modelBase = lvCallModel spot strike rate volSurface steps
    let modelUp = lvCallModel (spot + bumpSize) strike rate volSurface steps
    let modelDn = lvCallModel (spot - bumpSize) strike rate volSurface steps

    let priceUp = Simulation.fold sim modelUp |> Array.average
    let priceDn = Simulation.fold sim modelDn |> Array.average
    let fdDelta = (priceUp - priceDn) / (2.0f * bumpSize)

    printfn "AD Delta: %.6f" adDelta
    printfn "FD Delta: %.6f" fdDelta
    printfn "Difference: %.6f" (abs(adDelta - fdDelta))

// Expected output:
// AD Delta: 0.543217
// FD Delta: 0.543185
// Difference: 0.000032  (< 0.01% - validates correctness)
```

### Key Observations

1. **Single simulation gives both price and delta** - no need for separate bump-and-reprice runs.

2. **Chain rule through vol surface** - the LV model requires propagating derivatives through the `Interp2D` lookup, which needs the vol surface gradient `∂σ/∂S`.

3. **Product rule through evolve** - at each step, `Ṡₜ₊₁ = Ṡₜ × g + S × ġ` captures both the direct effect and the vol-feedback effect.

4. **Discontinuity at strike** - the `max(S-K, 0)` creates a discontinuity in the derivative. AD handles this correctly by checking `S > strike`.

5. **Same random numbers** - AD delta uses the exact same random path as the price, eliminating Monte Carlo noise in the hedge ratio (unlike finite difference which uses different random seeds).

---

## Implementation Roadmap for AD

### Phase 1: First-Order Greeks (Adjoint Mode)

**Scope:** Delta, Vega, Rho, Theta in single simulation.

**Implementation:**
1. Extend Expr AST with adjoint node types
2. Generate forward pass code (same as now)
3. Generate backward pass code (adjoint accumulation)
4. Store forward trajectory in thread-local array

**Effort:** 4-6 weeks

**Deliverable:**
```fsharp
let value, greeks = Simulation.foldWithGreeks sim model
// greeks.Delta, greeks.Vega, greeks.Rho
```

### Phase 2: Second-Order Greeks (Hyper-Dual)

**Scope:** Gamma, Vanna, Volga.

**Implementation:**
1. Extend Expr to generate hyper-dual arithmetic
2. Track second derivatives through evolve loops
3. Handle cross-derivatives with multi-dual

**Effort:** 4-6 weeks (after Phase 1)

**Deliverable:**
```fsharp
let value, greeks = Simulation.foldWithFullGreeks sim model
// greeks.Delta, greeks.Gamma, greeks.Vega, greeks.Vanna, greeks.Volga
```

### Phase 3: Checkpointing (Memory Optimization)

**Scope:** Reduce memory for long simulations.

**Implementation:**
1. Implement binomial checkpointing
2. Trade recomputation for memory

**Effort:** 2-3 weeks

**Deliverable:**
```fsharp
let config = { MaxMemoryPerThread = 1000 * sizeof<float32> }  // 4KB limit
let value, greeks = Simulation.foldWithGreeks config sim model
// Automatic checkpointing if needed
```

---

## Accumulator Arrays with Named Indexing

### The Problem

Multi-fund products require tracking many account values:

```fsharp
// Tedious - each fund is a separate accumulator
let! fund1 = evolve init1 (fun av -> av * return1 - fee1)
let! fund2 = evolve init2 (fun av -> av * return2 - fee2)
let! fund3 = evolve init3 (fun av -> av * return3 - fee3)
// ... repeat for 20+ funds
```

### Proposed Solution: Named Accumulator Collections

**DSL Layer (User-Facing):** Use string keys for readability and safety.

```fsharp
let! funds = evolveMap [
    "LargeCap", initialAllocation.["LargeCap"]
    "SmallCap", initialAllocation.["SmallCap"]
    "Bond", initialAllocation.["Bond"]
    "MoneyMarket", initialAllocation.["MoneyMarket"]
] (fun funds ->
    funds
    |> Map.map (fun name av ->
        let ret = fundReturns.[name]
        let fee = fundFees.[name]
        av * Expr.exp(ret * dt) - fee * av * dt)
)

// Access by name in DSL
let totalAV = funds.["LargeCap"] + funds.["SmallCap"] + funds.["Bond"]
```

**Compilation:** Names resolved to integer indices at compile time.

```fsharp
// Name-to-index mapping (computed at model build time)
let fundIndices = Map [
    "LargeCap", 0
    "SmallCap", 1
    "Bond", 2
    "MoneyMarket", 3
]

// Generated C# uses integer indexing
```

**Generated C#/ILGPU Code:**

```csharp
// Accumulator array (integer indexed)
float acc_funds_0 = initialLargeCap;   // LargeCap
float acc_funds_1 = initialSmallCap;   // SmallCap
float acc_funds_2 = initialBond;       // Bond
float acc_funds_3 = initialMoneyMarket; // MoneyMarket

for (int step = 0; step < numSteps; step++) {
    // Update each fund (unrolled, integer indexed)
    acc_funds_0 = acc_funds_0 * exp(ret_0 * dt) - fee_0 * acc_funds_0 * dt;
    acc_funds_1 = acc_funds_1 * exp(ret_1 * dt) - fee_1 * acc_funds_1 * dt;
    acc_funds_2 = acc_funds_2 * exp(ret_2 * dt) - fee_2 * acc_funds_2 * dt;
    acc_funds_3 = acc_funds_3 * exp(ret_3 * dt) - fee_3 * acc_funds_3 * dt;
}
```

### Implementation

#### Step 1: Extend Model Context

```fsharp
type AccumulatorMap = {
    Id: int
    Names: string[]
    IndexMap: Map<string, int>
    Inits: Expr[]
    Bodies: Expr[]  // Each body uses AccumArrayRef(id, index)
}

type ModelContext = {
    // ... existing fields ...
    AccumulatorMaps: AccumulatorMap list
}
```

#### Step 2: New Expr Variants

```fsharp
type Expr =
    | // ... existing variants ...
    | AccumArrayRef of mapId: int * index: int  // Runtime: acc_maps[mapId][index]
    | AccumArrayLookup of mapId: int * name: string  // Compile-time: resolved to AccumArrayRef
```

#### Step 3: Name Resolution Pass

Before code generation, resolve all `AccumArrayLookup` to `AccumArrayRef`:

```fsharp
let resolveNames (model: Model) : Model =
    let resolve expr =
        match expr with
        | AccumArrayLookup(mapId, name) ->
            let map = model.AccumulatorMaps.[mapId]
            let index = map.IndexMap.[name]
            AccumArrayRef(mapId, index)
        | other -> other

    // Walk and transform entire expression tree
    model |> transformExprs resolve
```

#### Step 4: Code Generation

```fsharp
let emitAccumArray (map: AccumulatorMap) =
    // Emit declarations
    for i, name in map.Names |> Array.indexed do
        emit $"float acc_{map.Id}_{i} = {emitExpr map.Inits.[i]}; // {name}"

    // Emit updates (inside time loop)
    for i, name in map.Names |> Array.indexed do
        emit $"acc_{map.Id}_{i} = {emitExpr map.Bodies.[i]}; // {name}"
```

### Benefits

| Aspect | String Keys (DSL) | Integer Index (Generated) |
|--------|-------------------|---------------------------|
| Readability | `funds.["LargeCap"]` | `acc_2_0` |
| Safety | Compile-time key check | N/A |
| Performance | N/A | Direct array access |
| Debugging | Meaningful names | Comments in generated code |

### Example: Full Multi-Fund VA Model

```fsharp
let vaModel = model {
    let! dt = scheduleDt schedule

    // Correlated fund returns
    let! zs = correlatedNormals fundCorrelationMatrix

    // Named accumulator map for fund values
    let! funds = evolveMap [
        "LargeCap",    policy.Allocation.["LargeCap"]
        "SmallCap",    policy.Allocation.["SmallCap"]
        "IntlEquity",  policy.Allocation.["IntlEquity"]
        "Bond",        policy.Allocation.["Bond"]
        "MoneyMarket", policy.Allocation.["MoneyMarket"]
    ] (fun funds ->
        // Each fund grows by its return minus fees
        Map.mapi (fun name i av ->
            let ret = fundReturns.[name]
            let vol = fundVols.[name]
            let fee = fundFees.[name]
            av * Expr.exp((ret - 0.5f * vol * vol) * dt + vol * Expr.sqrt(dt) * zs.[i])
               - fee * av * dt
        ) funds
    )

    // Total account value
    let totalAV =
        ["LargeCap"; "SmallCap"; "IntlEquity"; "Bond"; "MoneyMarket"]
        |> List.map (fun name -> funds.[name])
        |> List.reduce (+)

    // GMWB logic using total AV
    let! guaranteeBase = evolve policy.GuaranteeBase (fun gb ->
        Expr.max gb (totalAV * policy.StepUpRate))

    let withdrawal = guaranteeBase * policy.WithdrawalRate * dt
    let benefit = Expr.max (withdrawal - totalAV) 0.0f.C

    let! df = decay rate dt
    let! pvBenefits = evolve 0.0f.C (fun pv -> pv + benefit * df)

    return pvBenefits
}
```

**Generated kernel has no string operations** - all names resolved to indices at compile time.

---

## Summary: Extension Roadmap

| Priority | Extension | Effort | Impact |
|----------|-----------|--------|--------|
| **High** | Forward-mode AD | 4-6 weeks | Greeks 2-4x faster |
| **High** | Adjoint-mode AD | 6-10 weeks | Greeks + all sensitivities |
| **Medium** | Named accumulator maps | 2-3 weeks | Cleaner multi-fund code |
| **Medium** | Multiple outputs | 1-2 weeks | Cash flow attribution |
| **Low** | `bernoulli` primitive | 1-2 days | Cleaner event code |
| **Low** | `poisson` primitive | 1-2 days | Jump processes |

The DSL architecture is sound. These extensions add capability without changing the fundamental design: **Expr AST → C# code generation → ILGPU → GPU kernel**.

---

## AD Mode Options: Dual, HyperDual, Jet, Adjoint

The DSL supports multiple automatic differentiation modes. Users mark inputs as differentiable and select the appropriate mode based on their needs.

### Available Modes

| Mode | What It Tracks | Memory | Use Case |
|------|----------------|--------|----------|
| **Dual** | Value + 1st derivatives | O(n) | Deltas only |
| **HyperDual(diagonal)** | Value + 1st + diagonal 2nd | O(n) | Deltas + individual Gammas |
| **HyperDual(full)** | Value + 1st + all 2nd | O(n²) | Deltas + Gammas + crosses |
| **Jet(k)** | Taylor coefficients to order k | O(nᵏ) | Arbitrary order |
| **Adjoint** | All 1st derivs via backward pass | O(steps) | Many inputs |

### Diagonal vs Full Second-Order Derivatives

Cross-gammas (∂²V/∂Sᵢ∂Sⱼ) depend heavily on correlation assumptions that may not hold in stress scenarios. For most hedging purposes, **diagonal gammas only** (∂²V/∂Sᵢ²) are sufficient:

| Derivative Type | What It Measures | Hedge Instrument | Reliability |
|-----------------|------------------|------------------|-------------|
| Delta (∂V/∂Sᵢ) | Spot sensitivity | Underlying asset | High |
| Diagonal Gamma (∂²V/∂Sᵢ²) | Delta convexity | Options on same asset | High |
| Cross-Gamma (∂²V/∂Sᵢ∂Sⱼ) | Cross-asset convexity | Correlation products | **Low** |

**Why diagonal-only is often preferred:**
- Hedge instruments are univariate (you hedge with the underlying, not correlation swaps)
- Cross-gamma hedging is rare in practice
- Correlation assumptions are unstable in market stress
- P&L attribution is typically per-asset, not cross-asset

### DSL Design: Transparent to User

```fsharp
/// Differentiation mode - user selects based on needs
type DiffMode =
    | Dual                              // 1st order only (fastest)
    | HyperDual of diagonal: bool       // diagonal=true: only ∂²V/∂Sᵢ², diagonal=false: all crosses
    | Jet of order: int                 // Arbitrary order Taylor
    | Adjoint                           // Backward mode (all 1st order, memory efficient)

/// Convenience constructors
module DiffMode =
    let dualOnly = Dual
    let withDiagonalGammas = HyperDual true      // Individual gammas, no crosses
    let withAllGammas = HyperDual false          // Full Hessian including crosses

/// Extended Expr type with differentiable variables
type Expr =
    | Const of float32
    | DiffVar of index: int * value: float32   // NEW: marks differentiable input
    | Add of Expr * Expr
    | Mul of Expr * Expr
    | Exp of Expr
    | Sqrt of Expr
    | Max of Expr * Expr
    | AccumRef of id: int
    // ... existing variants ...

module Expr =
    /// Create differentiable variable
    let diffVar idx value = DiffVar(idx, value)

    /// Shorthand for constants
    let inline c (v: float32) = Const v
```

### User API: Same Model, Different Modes

```fsharp
// ================================================================
// Define model ONCE (no AD logic visible)
// ================================================================
let basketCall (spots: Expr[]) (vols: float32[]) rate strike = model {
    let! zs = correlatedNormals correlation
    let! s0 = gbm zs.[0] (c rate) (c vols.[0]) spots.[0] dt
    let! s1 = gbm zs.[1] (c rate) (c vols.[1]) spots.[1] dt
    let! s2 = gbm zs.[2] (c rate) (c vols.[2]) spots.[2] dt
    let basket = c 0.4f * s0 + c 0.35f * s1 + c 0.25f * s2
    return Expr.max (basket - c strike) (c 0.0f)
}

// ================================================================
// PRICE ONLY (no differentiation)
// ================================================================
let priceOnly = basketCall
    [| c 100.0f; c 100.0f; c 100.0f |]  // Regular constants
    vols rate strike

let prices = Simulation.fold sim priceOnly


// ================================================================
// DELTAS ONLY (Dual mode - fastest)
// ================================================================
let withDeltas = basketCall
    [| diffVar 0 100.0f; diffVar 1 100.0f; diffVar 2 100.0f |]
    vols rate strike

let prices, deltas = Simulation.foldDiff Dual sim withDeltas
// deltas: float32[numSims, 3]


// ================================================================
// DELTAS + DIAGONAL GAMMAS ONLY (recommended for hedging)
// ================================================================
let prices, deltas, gammas =
    Simulation.foldDiff (HyperDual true) sim withDeltas
// gammas: float32[numSims, 3]       - diagonal only: Γ₀₀, Γ₁₁, Γ₂₂
// No cross-gammas computed - avoids correlation dependency


// ================================================================
// DELTAS + ALL GAMMAS including crosses (when you trust correlation)
// ================================================================
let prices, deltas, gammas, crosses =
    Simulation.foldDiff (HyperDual false) sim withDeltas
// gammas: float32[numSims, 3]       - diagonal: Γ₀₀, Γ₁₁, Γ₂₂
// crosses: float32[numSims, 3]      - off-diagonal: Γ₀₁, Γ₀₂, Γ₁₂


// ================================================================
// HIGHER ORDER (Jet mode - e.g., for speed/charm)
// ================================================================
let prices, derivs = Simulation.foldDiff (Jet 3) sim withDeltas
// derivs.[i].[j] = jth derivative w.r.t. input i
// derivs.[0].[1] = delta₀
// derivs.[0].[2] = gamma₀
// derivs.[0].[3] = speed₀ (3rd derivative)


// ================================================================
// MANY INPUTS (Adjoint mode - efficient for 10+ inputs)
// ================================================================
let manyInputsModel = basketCall
    [| for i in 0..19 -> diffVar i spots.[i] |]  // 20 inputs
    vols rate strike

let prices, allDeltas = Simulation.foldDiff Adjoint sim manyInputsModel
// allDeltas: float32[numSims, 20] - all deltas in one backward pass
```

### Mode-Specific Code Generation

The compiler detects `DiffVar` nodes and generates appropriate arithmetic:

```fsharp
module Compiler =

    /// Generate code based on selected mode
    let compile (model: Model) (mode: DiffMode) : GeneratedKernel =
        let diffVars = collectDiffVars model
        let n = diffVars.Length

        match mode with
        | Dual ->
            // Track: value + n first derivatives
            // Per variable: 1 + n floats
            emitDualKernel model n

        | HyperDual diagonal ->
            if diagonal then
                // Track: value + n first + n diagonal second derivatives
                // Per variable: 1 + n + n = 1 + 2n floats
                // NO cross terms - much more efficient
                emitHyperDualDiagonalKernel model n
            else
                // Track: value + n first + n(n+1)/2 second derivatives
                // Per variable: 1 + n + n(n+1)/2 floats
                emitHyperDualFullKernel model n

        | Jet order ->
            // Track: Taylor coefficients up to order k
            // Per variable: binomial(n + order, order) floats
            emitJetKernel model n order

        | Adjoint ->
            // Forward pass stores trajectory
            // Backward pass computes all n derivatives
            emitAdjointKernel model n
```

### Dual Mode: Generated Code

```csharp
// 3 inputs, Dual mode
// Tracks: value + 3 first derivatives per variable

float S0 = spot0;
float dS0_0 = 1.0f, dS0_1 = 0.0f, dS0_2 = 0.0f;  // ∂S₀/∂spotᵢ

for (int t = 0; t < steps; t++) {
    float g = exp(drift + vol * sqrtDt * z);

    // S = S * g  (g is constant w.r.t. spots)
    S0 = S0 * g;
    dS0_0 = dS0_0 * g;
    dS0_1 = dS0_1 * g;
    dS0_2 = dS0_2 * g;
}

// Output: value + 3 deltas
```

### HyperDual Diagonal Mode: Generated Code

```csharp
// 3 inputs, HyperDual(diagonal=true)
// Tracks: value + 3 first + 3 diagonal second derivatives
// NO cross terms - O(n) instead of O(n²)

float S0 = spot0;
float dS0_0 = 1.0f;      // ∂S₀/∂spot₀ (only own derivative)
float d2S0_00 = 0.0f;    // ∂²S₀/∂spot₀² (diagonal gamma only)

// Similar for S1, S2 - each tracks only its own derivatives

for (int t = 0; t < steps; t++) {
    float g = exp(drift + vol * sqrtDt * z);

    // S = S * g (g doesn't depend on spots, so propagation is simple)
    S0 = S0 * g;
    dS0_0 = dS0_0 * g;
    d2S0_00 = d2S0_00 * g;

    // No cross-term arithmetic needed!
}

// Output: value + 3 deltas + 3 diagonal gammas
// Cross-gammas NOT computed - correlation assumptions not required
```

### HyperDual Full Mode: Generated Code

```csharp
// 3 inputs, HyperDual(diagonal=false)
// Tracks: value + 3 first + 6 second derivatives (including crosses)

float S0 = spot0;
float dS0_0 = 1.0f, dS0_1 = 0.0f, dS0_2 = 0.0f;  // All first derivs
float d2S0_00 = 0.0f, d2S0_01 = 0.0f, d2S0_02 = 0.0f;  // All second derivs
float d2S0_11 = 0.0f, d2S0_12 = 0.0f, d2S0_22 = 0.0f;

for (int t = 0; t < steps; t++) {
    float g = exp(drift + vol * sqrtDt * z);

    // S = S * g
    S0 = S0 * g;
    dS0_0 = dS0_0 * g;
    dS0_1 = dS0_1 * g;
    dS0_2 = dS0_2 * g;
    d2S0_00 = d2S0_00 * g;
    d2S0_01 = d2S0_01 * g;
    // ... all 6 second derivatives updated
}

// Output: value + 3 deltas + 3 diagonal gammas + 3 cross-gammas
```

### Jet Mode: Generated Code

```csharp
// 3 inputs, Jet order 3
// Tracks: Taylor coefficients [1, ε₁, ε₂, ε₃, ε₁², ε₁ε₂, ε₁ε₃, ε₂², ε₂ε₃, ε₃², ε₁³, ...]

// For order k with n inputs: binomial(n+k, k) coefficients
// n=3, k=3: binomial(6,3) = 20 coefficients per variable

float S0_coeffs[20];  // Taylor coefficients
S0_coeffs[0] = spot0;  // Constant term
S0_coeffs[1] = 1.0f;   // ε₁ coefficient (∂S₀/∂spot₀)
S0_coeffs[2] = 0.0f;   // ε₂ coefficient
// ... initialize rest to 0

for (int t = 0; t < steps; t++) {
    // Jet multiplication and exp use truncated polynomial arithmetic
    S0_coeffs = jet_mul(S0_coeffs, jet_exp(drift_coeffs + vol_coeffs * z));
}

// Output: coefficients give all derivatives up to order 3
// delta = coeffs[1], gamma = 2*coeffs[4], speed = 6*coeffs[10], etc.
```

### Adjoint Mode: Generated Code

```csharp
// 3 inputs, Adjoint mode
// Forward: store trajectory
// Backward: propagate adjoints

// Forward pass
float[] S0_history = new float[steps + 1];
S0_history[0] = spot0;

for (int t = 0; t < steps; t++) {
    float g = exp(drift + vol * sqrtDt * z[t]);
    S0_history[t + 1] = S0_history[t] * g;
}

float payoff = max(basket - strike, 0);

// Backward pass
float adj_S0 = 0, adj_S1 = 0, adj_S2 = 0;
float adj_payoff = 1.0f;

// Backprop through payoff
if (basket > strike) {
    adj_S0 += adj_payoff * w0;
    adj_S1 += adj_payoff * w1;
    adj_S2 += adj_payoff * w2;
}

// Backprop through time
for (int t = steps - 1; t >= 0; t--) {
    float g = S0_history[t + 1] / S0_history[t];
    adj_S0 = adj_S0 * g;
    // ... similar for S1, S2
}

// adj_S0, adj_S1, adj_S2 are the deltas
```

### Register/Memory Cost by Mode

| Mode | Per-Variable Registers | 3 inputs | 5 inputs | 20 inputs |
|------|------------------------|----------|----------|-----------|
| Dual | 1 + n | 4 | 6 | 21 |
| HyperDual(diagonal) | 1 + 2n | 7 | 11 | 41 |
| HyperDual(full) | 1 + n + n(n+1)/2 | 10 | 21 | 231 |
| Jet(3) | binomial(n+3, 3) | 20 | 56 | 1,771 |
| Adjoint | 1 (+ O(steps) memory) | 1 | 1 | 1 |

**Diagonal vs Full savings:** For 5 inputs, diagonal mode uses 11 registers vs 21 for full - nearly 50% reduction. For 20 inputs, the savings are dramatic: 41 vs 231 registers.

### When to Use Each Mode

| Mode | Best For | Inputs | Derivatives |
|------|----------|--------|-------------|
| **Dual** | Daily delta hedging | ≤ 20 | 1st order only |
| **HyperDual(diagonal)** | Delta + gamma hedging | ≤ 10 | 1st + diagonal 2nd |
| **HyperDual(full)** | Cross-gamma analysis | ≤ 5 | 1st + all 2nd |
| **Jet(k)** | Research, exotic Greeks | ≤ 3 | Up to order k |
| **Adjoint** | Large portfolios | Any | 1st order only |

**Recommendation:** Use `HyperDual(diagonal=true)` for production hedging. Only use `HyperDual(diagonal=false)` when you specifically need cross-gammas and trust your correlation assumptions.

### Mode Recommendation Helper

```fsharp
/// Recommend AD mode based on use case
let recommendMode (numInputs: int) (needGammas: bool) (needCrossGammas: bool) (needHigherOrder: bool) =
    match numInputs, needGammas, needCrossGammas, needHigherOrder with
    | n, false, _, false when n <= 20 -> Dual                    // Deltas only
    | n, true, false, false when n <= 10 -> HyperDual true       // Diagonal gammas (recommended)
    | n, true, true, false when n <= 5 -> HyperDual false        // Full gammas with crosses
    | n, _, _, true when n <= 3 -> Jet 3                         // Higher order, research
    | n, true, _, false when n > 10 ->
        // Many inputs: use Adjoint for deltas, FD for gammas
        printfn "Recommend: Adjoint for deltas + FD for diagonal gammas"
        Adjoint
    | n, false, _, false when n > 20 -> Adjoint                  // Many inputs, deltas only
    | _ -> failwith "Consider hybrid approach"

// Example usage:
recommendMode 5 true false false  // Returns: HyperDual true (diagonal gammas)
recommendMode 5 true true false   // Returns: HyperDual false (with crosses)
recommendMode 20 true false false // Returns: Adjoint (too many for HyperDual)
```

### Hybrid Approach for Many Inputs with Gammas

For portfolios with many assets where gammas are needed:

```fsharp
let computeGreeksHybrid model numInputs =
    // Adjoint for all deltas (efficient)
    let prices, deltas = Simulation.foldDiff Adjoint sim model

    // FD for diagonal gammas only (n bumps, not n²)
    let gammas = Array.init numInputs (fun i ->
        let bumpedUp = bumpInput model i (+bumpSize)
        let bumpedDn = bumpInput model i (-bumpSize)
        let priceUp = Simulation.fold sim bumpedUp |> Array.average
        let priceDn = Simulation.fold sim bumpedDn |> Array.average
        (priceUp - 2.0f * prices.[0] + priceDn) / (bumpSize * bumpSize))

    prices, deltas, gammas
```

### Summary

| Feature | Design |
|---------|--------|
| User marks inputs | `Expr.diffVar idx value` |
| User selects mode | `Simulation.foldDiff mode sim model` |
| Model code | **Unchanged** regardless of mode |
| Operators | Work on all Expr variants uniformly |
| Code generator | Detects `DiffVar`, emits mode-specific arithmetic |
| Flexibility | Dual / HyperDual / Jet / Adjoint |

**The DSL abstracts all derivative logic. Users mark which inputs to differentiate and select the appropriate mode. Model code never changes.**

---

## Example: Greeks for 5-Fund Portfolio (Hybrid AD/FD Approach)

This example demonstrates the recommended approach for computing Greeks on a 5-fund equity portfolio: **AD for deltas, FD for gammas**.

### Model Definition

```fsharp
module Portfolio =
    open Cavere.Core
    open Cavere.Generators

    /// 5-fund VA portfolio with GMWB rider
    let portfolioModel (spots: float32[]) (vols: float32[]) (correlation: float32[,])
                       (rate: float32) (guarantee: float32) (withdrawal: float32)
                       (steps: int) =
        model {
            let dt = Const(1.0f / float32 steps)
            let numFunds = 5

            // Correlated normals for 5 funds
            let! zs = correlatedNormals correlation

            // Evolve each fund
            let! fund0 = gbm zs.[0] (Const rate) (Const vols.[0]) (Const spots.[0]) dt
            let! fund1 = gbm zs.[1] (Const rate) (Const vols.[1]) (Const spots.[1]) dt
            let! fund2 = gbm zs.[2] (Const rate) (Const vols.[2]) (Const spots.[2]) dt
            let! fund3 = gbm zs.[3] (Const rate) (Const vols.[3]) (Const spots.[3]) dt
            let! fund4 = gbm zs.[4] (Const rate) (Const vols.[4]) (Const spots.[4]) dt

            // Total account value
            let totalAV = fund0 + fund1 + fund2 + fund3 + fund4

            // GMWB: guarantee base with step-up
            let! guaranteeBase = evolve (Const guarantee) (fun gb ->
                Expr.max gb totalAV)

            // Withdrawal and benefit
            let withdrawalAmt = Const withdrawal * dt
            let benefit = Expr.max (withdrawalAmt - totalAV) 0.0f.C

            // Discount and accumulate PV of benefits
            let! df = decay (Const rate) dt
            let! pvBenefit = evolve 0.0f.C (fun pv -> pv + benefit * df)

            return pvBenefit
        }
```

### Greeks Computation Module

```fsharp
module Greeks =
    open Cavere.Core

    type PortfolioGreeks = {
        Price: float32
        Deltas: float32[]           // 5 values
        Gammas: float32[]           // 5 values (diagonal)
        CrossGammas: float32[,]     // 5x5 symmetric (10 unique)
        Vega: float32[]             // 5 values
        Rho: float32
    }

    /// Compute all Greeks using hybrid AD/FD approach
    let computeGreeks (baseSpots: float32[]) (vols: float32[]) (correlation: float32[,])
                      (rate: float32) (guarantee: float32) (withdrawal: float32)
                      (steps: int) (numSims: int) : PortfolioGreeks =

        // Helper to build and run model
        let runModel spots =
            let m = Portfolio.portfolioModel spots vols correlation rate guarantee withdrawal steps
            use sim = Simulation.create GPU numSims steps
            Simulation.fold sim m |> Array.average

        // ============================================================
        // APPROACH 1: AD for Deltas (all 5 in one pass)
        // ============================================================
        let price, deltas =
            let m = Portfolio.portfolioModelWithAD baseSpots vols correlation rate guarantee withdrawal steps
            use sim = Simulation.create GPU numSims steps
            let prices, deltaArrays = Simulation.foldWithDeltas sim m
            let avgPrice = Array.average prices
            let avgDeltas = Array.init 5 (fun i ->
                deltaArrays |> Array.averageBy (fun d -> d.[i]))
            avgPrice, avgDeltas

        // ============================================================
        // APPROACH 2: FD for Gammas (simpler, 60 sims is acceptable)
        // ============================================================
        let bumpSize = 0.01f  // 1% bump

        // Diagonal gammas: ∂²V/∂Sᵢ²
        let gammas = Array.init 5 (fun i ->
            let spotsUp = Array.copy baseSpots
            let spotsDn = Array.copy baseSpots
            spotsUp.[i] <- baseSpots.[i] * (1.0f + bumpSize)
            spotsDn.[i] <- baseSpots.[i] * (1.0f - bumpSize)

            let priceUp = runModel spotsUp
            let priceDn = runModel spotsDn
            let h = baseSpots.[i] * bumpSize

            (priceUp - 2.0f * price + priceDn) / (h * h)
        )

        // Cross-gammas: ∂²V/∂Sᵢ∂Sⱼ
        let crossGammas = Array2D.zeroCreate 5 5
        for i in 0..4 do
            crossGammas.[i, i] <- gammas.[i]  // Diagonal
            for j in i+1..4 do
                let hi = baseSpots.[i] * bumpSize
                let hj = baseSpots.[j] * bumpSize

                let bumpBoth upI upJ =
                    let spots = Array.copy baseSpots
                    spots.[i] <- baseSpots.[i] * (1.0f + (if upI then bumpSize else -bumpSize))
                    spots.[j] <- baseSpots.[j] * (1.0f + (if upJ then bumpSize else -bumpSize))
                    runModel spots

                let upUp = bumpBoth true true
                let upDn = bumpBoth true false
                let dnUp = bumpBoth false true
                let dnDn = bumpBoth false false

                let crossGamma = (upUp - upDn - dnUp + dnDn) / (4.0f * hi * hj)
                crossGammas.[i, j] <- crossGamma
                crossGammas.[j, i] <- crossGamma  // Symmetric

        // ============================================================
        // Vegas: FD on volatility (5 bumps)
        // ============================================================
        let vegas = Array.init 5 (fun i ->
            let volsUp = Array.copy vols
            let volsDn = Array.copy vols
            let volBump = 0.01f  // 1 vol point
            volsUp.[i] <- vols.[i] + volBump
            volsDn.[i] <- vols.[i] - volBump

            let priceVolUp =
                let m = Portfolio.portfolioModel baseSpots volsUp correlation rate guarantee withdrawal steps
                use sim = Simulation.create GPU numSims steps
                Simulation.fold sim m |> Array.average

            let priceVolDn =
                let m = Portfolio.portfolioModel baseSpots volsDn correlation rate guarantee withdrawal steps
                use sim = Simulation.create GPU numSims steps
                Simulation.fold sim m |> Array.average

            (priceVolUp - priceVolDn) / (2.0f * volBump)
        )

        // ============================================================
        // Rho: FD on rate (single bump)
        // ============================================================
        let rateBump = 0.0001f  // 1 bp
        let priceRateUp =
            let m = Portfolio.portfolioModel baseSpots vols correlation (rate + rateBump) guarantee withdrawal steps
            use sim = Simulation.create GPU numSims steps
            Simulation.fold sim m |> Array.average
        let priceRateDn =
            let m = Portfolio.portfolioModel baseSpots vols correlation (rate - rateBump) guarantee withdrawal steps
            use sim = Simulation.create GPU numSims steps
            Simulation.fold sim m |> Array.average
        let rho = (priceRateUp - priceRateDn) / (2.0f * rateBump)

        {
            Price = price
            Deltas = deltas
            Gammas = gammas
            CrossGammas = crossGammas
            Vega = vegas
            Rho = rho
        }
```

### AD-Enabled Model (Deltas Only)

```fsharp
module Portfolio =
    /// Model with AD instrumentation for delta computation
    let portfolioModelWithAD (spots: float32[]) (vols: float32[]) (correlation: float32[,])
                              (rate: float32) (guarantee: float32) (withdrawal: float32)
                              (steps: int) =
        modelWithAD {
            let dt = Const(1.0f / float32 steps)

            // Mark spots as differentiable inputs
            let! spot0 = diffInput spots.[0]  // DualExpr with deriv seed
            let! spot1 = diffInput spots.[1]
            let! spot2 = diffInput spots.[2]
            let! spot3 = diffInput spots.[3]
            let! spot4 = diffInput spots.[4]

            let! zs = correlatedNormals correlation

            // GBM with dual numbers (tracks ∂S/∂S₀)
            let! fund0 = gbmDual zs.[0] (Const rate) (Const vols.[0]) spot0 dt
            let! fund1 = gbmDual zs.[1] (Const rate) (Const vols.[1]) spot1 dt
            let! fund2 = gbmDual zs.[2] (Const rate) (Const vols.[2]) spot2 dt
            let! fund3 = gbmDual zs.[3] (Const rate) (Const vols.[3]) spot3 dt
            let! fund4 = gbmDual zs.[4] (Const rate) (Const vols.[4]) spot4 dt

            let totalAV = fund0 + fund1 + fund2 + fund3 + fund4

            let! guaranteeBase = evolveDual (Const guarantee) (fun gb ->
                dualMax gb totalAV)

            let withdrawalAmt = Const withdrawal * dt
            let benefit = dualMax (withdrawalAmt - totalAV.Value) 0.0f.C

            let! df = decay (Const rate) dt
            let! pvBenefit = evolveDual 0.0f.C (fun pv ->
                pv + benefit * df)

            // Return value and all 5 deltas
            return (pvBenefit.Value, [|
                pvBenefit.Derivs.[0]  // ∂V/∂S₀
                pvBenefit.Derivs.[1]  // ∂V/∂S₁
                pvBenefit.Derivs.[2]  // ∂V/∂S₂
                pvBenefit.Derivs.[3]  // ∂V/∂S₃
                pvBenefit.Derivs.[4]  // ∂V/∂S₄
            |])
        }
```

### Generated Kernel (AD Deltas)

```csharp
[Kernel]
public static void PortfolioWithDeltas(
    Index1D idx,
    ArrayView<float> prices,
    ArrayView<float> deltas,  // [idx * 5 + fundId]
    // Inputs
    float spot0, float spot1, float spot2, float spot3, float spot4,
    float vol0, float vol1, float vol2, float vol3, float vol4,
    ArrayView<float> choleskyL,  // Correlation via Cholesky
    float rate, float guarantee, float withdrawal,
    int numSteps,
    ArrayView<XorShift128Plus> rng)
{
    float dt = 1.0f / numSteps;
    float sqrtDt = MathF.Sqrt(dt);

    // Fund values (primal)
    float S0 = spot0, S1 = spot1, S2 = spot2, S3 = spot3, S4 = spot4;

    // Delta accumulators: ∂Sᵢ/∂Sᵢ⁰ (seeded with 1.0 for own spot, 0.0 for others)
    float dS0_dSpot0 = 1.0f, dS0_dSpot1 = 0.0f, dS0_dSpot2 = 0.0f, dS0_dSpot3 = 0.0f, dS0_dSpot4 = 0.0f;
    float dS1_dSpot0 = 0.0f, dS1_dSpot1 = 1.0f, dS1_dSpot2 = 0.0f, dS1_dSpot3 = 0.0f, dS1_dSpot4 = 0.0f;
    float dS2_dSpot0 = 0.0f, dS2_dSpot1 = 0.0f, dS2_dSpot2 = 1.0f, dS2_dSpot3 = 0.0f, dS2_dSpot4 = 0.0f;
    float dS3_dSpot0 = 0.0f, dS3_dSpot1 = 0.0f, dS3_dSpot2 = 0.0f, dS3_dSpot3 = 1.0f, dS3_dSpot4 = 0.0f;
    float dS4_dSpot0 = 0.0f, dS4_dSpot1 = 0.0f, dS4_dSpot2 = 0.0f, dS4_dSpot3 = 0.0f, dS4_dSpot4 = 1.0f;

    // Other accumulators
    float guaranteeBase = guarantee;
    float df = 1.0f;
    float pvBenefit = 0.0f;

    // Derivative of pvBenefit w.r.t. each spot
    float dPV_dSpot0 = 0.0f, dPV_dSpot1 = 0.0f, dPV_dSpot2 = 0.0f, dPV_dSpot3 = 0.0f, dPV_dSpot4 = 0.0f;

    for (int t = 0; t < numSteps; t++)
    {
        // Generate correlated normals
        float z0_indep = GenerateNormal(ref rng[idx]);
        float z1_indep = GenerateNormal(ref rng[idx]);
        float z2_indep = GenerateNormal(ref rng[idx]);
        float z3_indep = GenerateNormal(ref rng[idx]);
        float z4_indep = GenerateNormal(ref rng[idx]);

        // Apply Cholesky: z_corr = L × z_indep
        float z0 = choleskyL[0] * z0_indep;
        float z1 = choleskyL[5] * z0_indep + choleskyL[6] * z1_indep;
        float z2 = choleskyL[10] * z0_indep + choleskyL[11] * z1_indep + choleskyL[12] * z2_indep;
        // ... etc for z3, z4

        // GBM growth factors
        float g0 = MathF.Exp((rate - 0.5f * vol0 * vol0) * dt + vol0 * sqrtDt * z0);
        float g1 = MathF.Exp((rate - 0.5f * vol1 * vol1) * dt + vol1 * sqrtDt * z1);
        float g2 = MathF.Exp((rate - 0.5f * vol2 * vol2) * dt + vol2 * sqrtDt * z2);
        float g3 = MathF.Exp((rate - 0.5f * vol3 * vol3) * dt + vol3 * sqrtDt * z3);
        float g4 = MathF.Exp((rate - 0.5f * vol4 * vol4) * dt + vol4 * sqrtDt * z4);

        // Update fund values
        S0 *= g0; S1 *= g1; S2 *= g2; S3 *= g3; S4 *= g4;

        // Update deltas: ∂Sᵢ/∂Spotⱼ *= gᵢ (chain rule, g doesn't depend on spots)
        dS0_dSpot0 *= g0; dS0_dSpot1 *= g0; dS0_dSpot2 *= g0; dS0_dSpot3 *= g0; dS0_dSpot4 *= g0;
        dS1_dSpot0 *= g1; dS1_dSpot1 *= g1; dS1_dSpot2 *= g1; dS1_dSpot3 *= g1; dS1_dSpot4 *= g1;
        dS2_dSpot0 *= g2; dS2_dSpot1 *= g2; dS2_dSpot2 *= g2; dS2_dSpot3 *= g2; dS2_dSpot4 *= g2;
        dS3_dSpot0 *= g3; dS3_dSpot1 *= g3; dS3_dSpot2 *= g3; dS3_dSpot3 *= g3; dS3_dSpot4 *= g3;
        dS4_dSpot0 *= g4; dS4_dSpot1 *= g4; dS4_dSpot2 *= g4; dS4_dSpot3 *= g4; dS4_dSpot4 *= g4;

        // Total AV and its derivatives
        float totalAV = S0 + S1 + S2 + S3 + S4;
        float dTotalAV_dSpot0 = dS0_dSpot0 + dS1_dSpot0 + dS2_dSpot0 + dS3_dSpot0 + dS4_dSpot0;
        float dTotalAV_dSpot1 = dS0_dSpot1 + dS1_dSpot1 + dS2_dSpot1 + dS3_dSpot1 + dS4_dSpot1;
        float dTotalAV_dSpot2 = dS0_dSpot2 + dS1_dSpot2 + dS2_dSpot2 + dS3_dSpot2 + dS4_dSpot2;
        float dTotalAV_dSpot3 = dS0_dSpot3 + dS1_dSpot3 + dS2_dSpot3 + dS3_dSpot3 + dS4_dSpot3;
        float dTotalAV_dSpot4 = dS0_dSpot4 + dS1_dSpot4 + dS2_dSpot4 + dS3_dSpot4 + dS4_dSpot4;

        // Guarantee base step-up: max(gb, totalAV)
        if (totalAV > guaranteeBase)
        {
            guaranteeBase = totalAV;
            // Derivative follows totalAV when it's the max
        }

        // Benefit calculation
        float withdrawalAmt = withdrawal * dt;
        float benefit = MathF.Max(withdrawalAmt - totalAV, 0.0f);

        // Derivative of benefit: -∂totalAV/∂spot when benefit > 0
        float benefitActive = (withdrawalAmt > totalAV) ? 1.0f : 0.0f;
        float dBenefit_dSpot0 = -benefitActive * dTotalAV_dSpot0;
        float dBenefit_dSpot1 = -benefitActive * dTotalAV_dSpot1;
        float dBenefit_dSpot2 = -benefitActive * dTotalAV_dSpot2;
        float dBenefit_dSpot3 = -benefitActive * dTotalAV_dSpot3;
        float dBenefit_dSpot4 = -benefitActive * dTotalAV_dSpot4;

        // Update discount factor
        df *= MathF.Exp(-rate * dt);

        // Accumulate PV of benefits
        pvBenefit += benefit * df;
        dPV_dSpot0 += dBenefit_dSpot0 * df;
        dPV_dSpot1 += dBenefit_dSpot1 * df;
        dPV_dSpot2 += dBenefit_dSpot2 * df;
        dPV_dSpot3 += dBenefit_dSpot3 * df;
        dPV_dSpot4 += dBenefit_dSpot4 * df;
    }

    // Store results
    prices[idx] = pvBenefit;
    deltas[idx * 5 + 0] = dPV_dSpot0;
    deltas[idx * 5 + 1] = dPV_dSpot1;
    deltas[idx * 5 + 2] = dPV_dSpot2;
    deltas[idx * 5 + 3] = dPV_dSpot3;
    deltas[idx * 5 + 4] = dPV_dSpot4;
}
```

### Usage

```fsharp
// Market data
let spots = [| 100.0f; 50.0f; 75.0f; 120.0f; 80.0f |]
let vols = [| 0.20f; 0.25f; 0.18f; 0.22f; 0.15f |]
let correlation = array2D [|
    [| 1.0f; 0.6f; 0.4f; 0.3f; 0.2f |]
    [| 0.6f; 1.0f; 0.5f; 0.4f; 0.3f |]
    [| 0.4f; 0.5f; 1.0f; 0.6f; 0.4f |]
    [| 0.3f; 0.4f; 0.6f; 1.0f; 0.5f |]
    [| 0.2f; 0.3f; 0.4f; 0.5f; 1.0f |]
|]
let rate = 0.05f
let guarantee = 400.0f  // Total of spots
let withdrawal = 20.0f  // Annual withdrawal

// Compute all Greeks
let greeks = Greeks.computeGreeks spots vols correlation rate guarantee withdrawal 360 100_000

// Output
printfn "Price:       %.4f" greeks.Price
printfn "Deltas:      %A" greeks.Deltas
printfn "Gammas:      %A" greeks.Gammas
printfn "Cross[0,1]:  %.6f" greeks.CrossGammas.[0,1]
printfn "Vegas:       %A" greeks.Vega
printfn "Rho:         %.4f" greeks.Rho
```

### Simulation Count Summary

| Greek Type | Method | Simulations |
|------------|--------|-------------|
| 5 Deltas | AD (single pass) | 1 |
| 5 Gammas | FD | 10 (up/down each) |
| 10 Cross-Gammas | FD | 40 (4 corners each) |
| 5 Vegas | FD | 10 |
| 1 Rho | FD | 2 |
| **Total** | **Hybrid** | **~63 sims** |

With pure FD for deltas too, it would be ~73 sims. **AD saves ~15% and reduces delta variance.**

### Key Implementation Notes

1. **AD deltas share the same random path** — all 5 deltas computed on identical scenarios, eliminating relative noise between them.

2. **FD gammas use fresh paths** — acceptable because gamma is less sensitive to path noise than delta.

3. **Cross-gammas can be parallelized** — all 10 pairs are independent, can run as batch simulation.

4. **Generated code is unrolled** — 5 funds means 5×5=25 derivative accumulators, all in registers.

# Generators

Finance-specific stochastic process builders. References Core only.

## Namespaces

- `Cavere.Generators` — base generators (`Calendar.fs`, `Common.fs`, `Equity.fs`, `Rates.fs`)
- `Cavere.Generators.AAA` — Academy Interest Rate Generator (AIRG) ESG. Files prefixed `AAA.*.fs`, each defines a module in the AAA namespace.

## Generator Signature

All generators follow `... -> ModelCtx -> Expr` (or `Expr[]` for multi-asset). They compose via `let!` in `model { }`:

```fsharp
let! z = normal
let! stock = gbm z rate vol spot dt
```

Writing a new generator: call `normal` and `evolve` inside `fun ctx ->`.

## Calendar.fs

- `businessDays startDate endDate holidays` — builds Schedule from business days (excludes weekends and holidays)

## Common.fs (AutoOpen)

- `decay rate dt` — discount factor accumulator

## Equity.fs (AutoOpen)

- `gbm z rate vol spot dt` — geometric Brownian motion (pass your own normal)
- `gbmLocalVol z surfId rate spot dt` — local vol from 2D surface
- `heston z rate v0 kappa theta xi rho spot dt` — stochastic vol (allocates internal variance normal)
- `multiAssetHeston rate stockCorr volCorr assets dt` — correlated multi-asset Heston

## Rates.fs

- CPU-side: `linearForwards`, `logDiscountForwards` — curve preparation (pure float32[])
- Deterministic: `flat rate`, `curve surfId` — no ModelCtx needed
- Stochastic: `vasicek`, `cir`, `cirpp` — each allocates a normal internally

## AAA Modules

SLV-based economic scenario generator following AIRG methodology:
- **Common** — Parameter record types (LongRateParams, SpreadParams, LogVolParams, EquityFundParams, BondFundParams)
- **Rates** — `slvRateModel` returns `(longRate, shortRate, spread, logVol)` tuple
- **Equity** — `singleEquityFund`, `equityFunds` with leverage effect
- **Bonds** — `singleBondFund`, `bondFunds` with duration/convexity
- **Selection** — Scenario selection (stratified, tail-preserving, moment-matching, CTE-focused)
- **Generator** — Combined model builders

## Conventions

- Generators that take a normal `z` as parameter allow the caller to control correlation (via `correlatedNormals`)
- Generators that allocate normals internally (like `heston`, `vasicek`) manage their own randomness
- CPU-side curve preparation functions are pure `float32[] -> float32[]`, no ModelCtx

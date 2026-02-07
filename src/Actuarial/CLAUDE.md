# Actuarial

Insurance product definitions, deterministic decrements, and policy records. References Core and Generators.

## Namespaces

- `Cavere.Actuarial` — Policy.fs (policyholder, policy record, account structure)
- `Cavere.Actuarial.Decrements` — files prefixed `Decrements.*.fs` (Common, Mortality, Surrender, Withdrawal)
- `Cavere.Actuarial.Product` — files prefixed `Product.*.fs` (Common, Riders, Life, Fixed, FixedIndexed, RILA, Variable)

## Decrements

Deterministic rate-based reductions (not stochastic). Each module provides:
- Table loading: `loadXxxTable (rates: float32[]) : ModelCtx -> int` — registers a Curve1D surface
- Rate lookup: `xxxRate (surfaceId: int) : Expr` — Lookup1D or Interp1D
- Survival accumulation: `survivalProb rate dt : ModelCtx -> Expr` — evolves from 1.0

Key modules:
- **Common** (AutoOpen) — `survivalProb`, `survivalProbAnnual`, `survivalProbContinuous`, `multiDecrementSurvival`, `weightedPV`
- **Mortality** — `loadMortalityTable`, `qx`, `qxFromIssueAge`, joint life (FirstToDie/LastToDie)
- **Surrender** — `loadSurrenderTable`, `persistency`, `cashSurrenderValue`
- **Withdrawal** — `loadWithdrawalRateTable`, free withdrawal amounts, GMWB systematic withdrawals

## Products

Record types defining product features. No simulation logic — just data:
- **Common** — CDSCSchedule, MVAParams, RMDParams, FeeStructure, PremiumMode, BenefitBaseType
- **Riders** — GMDB, GMAB, GMIB, GMWB, GLWB records + RiderPackage
- **Life** — LifeProduct (Term, UL, WholeLife, VUL)
- **Fixed** — FixedAnnuityProduct (Traditional, MYGA), RateCrediting
- **FixedIndexed** — FIAProduct, CreditingStrategy (PointToPoint, MonthlyCap, etc.), IndexAllocation
- **RILA** — RILAProduct, ProtectionType (Buffer, Floor, DualDirection), UpsideStrategy
- **Variable** — VAProduct, SubAccount, SubAccountAllocation

## Policy

- `ProductType` DU — union of all product types for heterogeneous portfolios
- `AccountStructure` DU — `GeneralOnly of float32` | `GeneralAndSeparate of float32 * SubAccountBalance list`
- `Policy` record — Id, Owner, Spouse, Product, IssueDate, Premium, Account, Status
- `Policy.create` — general-account-only (Fixed, FIA, RILA, Life)
- `Policy.createVA` — general + separate account with sub-account allocations

## Conventions

- Product modules define types + helper constructors (e.g., `FixedAnnuityProduct.traditional`, `withMVA`, `withBailout`)
- Decrement functions that need ModelCtx follow the same `ModelCtx -> Expr` pattern as generators
- Pure Expr combinators (no ModelCtx) are preferred where possible (e.g., `qx`, `cashSurrenderValue`, `weightedPV`)

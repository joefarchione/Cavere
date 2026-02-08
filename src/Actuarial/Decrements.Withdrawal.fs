namespace Cavere.Actuarial.Decrements

open Cavere.Core

/// Partial withdrawal modeling for annuity and life products.
/// Withdrawal rates/amounts are indexed by policy duration from issue date.
/// Unlike surrender, partial withdrawals reduce account value but policy stays in force.
module Withdrawal =

    // ── Withdrawal table loading ───────────────────────────────────────

    /// Load a withdrawal rate table (by policy duration) onto the GPU.
    /// Rates represent the fraction of account value withdrawn each period.
    /// Returns a surface ID for use with withdrawalRate.
    let loadWithdrawalRateTable (ratesByDuration: float32[]) : ModelCtx -> int =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(ratesByDuration, ratesByDuration.Length))
            id

    /// Load a withdrawal amount table (by policy duration) onto the GPU.
    /// Amounts represent fixed dollar withdrawals each period.
    let loadWithdrawalAmountTable (amountsByDuration: float32[]) : ModelCtx -> int =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1

            ctx.Surfaces <-
                ctx.Surfaces
                |> Map.add id (Curve1D(amountsByDuration, amountsByDuration.Length))

            id

    // ── Withdrawal rate/amount lookup ──────────────────────────────────

    /// Withdrawal rate lookup by policy duration (TimeIndex).
    let withdrawalRate (surfaceId: int) : Expr = Lookup1D surfaceId

    /// Withdrawal amount lookup by policy duration (TimeIndex).
    let withdrawalAmount (surfaceId: int) : Expr = Lookup1D surfaceId

    /// Constant withdrawal rate (for simplified models).
    let withdrawalRateConstant (rate: float32) : Expr = rate.C

    /// Constant withdrawal amount (for simplified models).
    let withdrawalAmountConstant (amount: float32) : Expr = amount.C

    // ── Account value with withdrawals ─────────────────────────────────

    /// Account value after proportional withdrawal.
    /// New AV = AV * (1 - withdrawalRate)
    let applyProportionalWithdrawal (accountValue: Expr) (withdrawalRate: Expr) : Expr =
        accountValue * (1.0f.C - withdrawalRate)

    /// Account value after fixed dollar withdrawal.
    /// New AV = max(0, AV - withdrawalAmount)
    let applyFixedWithdrawal (accountValue: Expr) (withdrawalAmount: Expr) : Expr =
        Expr.max (accountValue - withdrawalAmount) 0.0f.C

    /// Actual withdrawal taken (may be limited by account value).
    let actualWithdrawal (accountValue: Expr) (requestedWithdrawal: Expr) : Expr =
        Expr.min accountValue requestedWithdrawal

    // ── Withdrawal with free amount ────────────────────────────────────

    /// Many products allow a "free" withdrawal amount (e.g., 10% of AV) without penalty.
    /// Excess withdrawals may incur charges.
    let freeWithdrawalAmount (accountValue: Expr) (freePercent: Expr) : Expr = accountValue * freePercent

    /// Withdrawal charge on excess over free amount.
    let excessWithdrawalCharge (withdrawal: Expr) (freeAmount: Expr) (chargeRate: Expr) : Expr =
        Expr.max (withdrawal - freeAmount) 0.0f.C * chargeRate

    /// Net withdrawal after charge on excess.
    let netWithdrawal (withdrawal: Expr) (freeAmount: Expr) (chargeRate: Expr) : Expr =
        withdrawal - excessWithdrawalCharge withdrawal freeAmount chargeRate

    // ── Cumulative tracking ────────────────────────────────────────────

    /// Cumulative withdrawals over time.
    let cumulativeWithdrawals (withdrawalPerPeriod: Expr) : ModelCtx -> Expr =
        fun ctx -> evolve 0.0f.C (fun total -> total + withdrawalPerPeriod) ctx

    /// Cumulative withdrawal charges over time.
    let cumulativeWithdrawalCharges (chargePerPeriod: Expr) : ModelCtx -> Expr =
        fun ctx -> evolve 0.0f.C (fun total -> total + chargePerPeriod) ctx

    // ── GMWB-style systematic withdrawals ──────────────────────────────

    /// Guaranteed Minimum Withdrawal Benefit: fixed percentage of benefit base.
    /// Withdrawal amount = benefitBase * guaranteedRate (e.g., 5% for life)
    let gmwbWithdrawal (benefitBase: Expr) (guaranteedRate: Expr) : Expr = benefitBase * guaranteedRate

    /// Benefit base that may step up on anniversaries.
    /// Common feature: benefit base = max(benefit base, account value) annually.
    let benefitBaseWithStepUp (currentBase: Expr) (accountValue: Expr) : Expr = Expr.max currentBase accountValue

namespace Cavere.Actuarial.Decrements

open Cavere.Core

/// Surrender (lapse) decrement modeling.
/// Surrender rates are indexed by policy duration from issue date.
module Surrender =

    // ── Surrender table loading ────────────────────────────────────────

    /// Load a surrender rate table (by policy duration) onto the GPU.
    /// Table should be indexed by policy year (0, 1, 2, ...).
    /// Returns a surface ID for use with surrenderRate.
    let loadSurrenderTable (ratesByDuration: float32[]) : ModelCtx -> int =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(ratesByDuration, ratesByDuration.Length))
            id

    // ── Surrender rate lookup ──────────────────────────────────────────

    /// Surrender rate lookup by policy duration (TimeIndex).
    /// Uses direct lookup for integer durations.
    let surrenderRate (surfaceId: int) : Expr = Lookup1D surfaceId

    /// Surrender rate lookup with interpolation for fractional durations.
    let surrenderRateInterp (surfaceId: int) (duration: Expr) : ModelCtx -> Expr =
        fun ctx -> interp1d surfaceId duration ctx

    /// Constant surrender rate (for simplified models).
    let surrenderRateConstant (rate: float32) : Expr = rate.C

    // ── Persistency (survival from surrender) ──────────────────────────

    /// Cumulative probability of not surrendering from issue to current time.
    /// Uses annual surrender rates.
    let persistency (surrenderRate: Expr) : ModelCtx -> Expr = fun ctx -> survivalProbAnnual surrenderRate ctx

    /// Cumulative persistency with sub-annual time steps.
    let persistencySubannual (annualSurrenderRate: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let periodRate = 1.0f.C - Expr.exp (Expr.log (1.0f.C - annualSurrenderRate) * dt)
            Common.survivalProb periodRate 1.0f.C ctx

    // ── Surrender charge ───────────────────────────────────────────────

    /// Load a surrender charge schedule (by policy duration) onto the GPU.
    /// Charges typically decline over time (e.g., 7%, 6%, 5%, 4%, 3%, 2%, 1%, 0%).
    let loadSurrenderChargeTable (chargesByDuration: float32[]) : ModelCtx -> int =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1

            ctx.Surfaces <-
                ctx.Surfaces
                |> Map.add id (Curve1D(chargesByDuration, chargesByDuration.Length))

            id

    /// Surrender charge lookup by policy duration.
    let surrenderCharge (surfaceId: int) : Expr = Lookup1D surfaceId

    /// Cash surrender value: account value minus surrender charge.
    let cashSurrenderValue (accountValue: Expr) (surrenderCharge: Expr) : Expr =
        accountValue * (1.0f.C - surrenderCharge)

    // ── Surrender benefit helpers ──────────────────────────────────────

    /// Expected surrender payout: CSV * surrender rate * persistency to start * discount.
    let expectedSurrenderBenefit (csv: Expr) (surrenderRate: Expr) (persistency: Expr) (df: Expr) : Expr =
        csv * surrenderRate * persistency * df

    /// Cumulative expected surrender payouts over time.
    let cumulativeSurrenderPayouts (csv: Expr) (surrenderRate: Expr) (persistency: Expr) (df: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let periodPayout = expectedSurrenderBenefit csv surrenderRate persistency df
            evolve 0.0f.C (fun total -> total + periodPayout) ctx

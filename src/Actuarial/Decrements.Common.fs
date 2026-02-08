namespace Cavere.Actuarial.Decrements

open Cavere.Core

/// Common types and functions for deterministic decrement modeling.
/// Decrements reduce values by expected rates each period rather than
/// simulating individual events stochastically.
[<AutoOpen>]
module Common =

    // ── Decrement rate sources ─────────────────────────────────────────

    /// Constant decrement rate (same every period)
    let constantRate (rate: float32) : Expr = rate.C

    /// Decrement rate from a 1D table indexed by time step (duration)
    let rateByDuration (surfaceId: int) : Expr = Lookup1D surfaceId

    /// Decrement rate from a 1D table with interpolation
    let rateByTime (surfaceId: int) (t: Expr) : ModelCtx -> Expr = fun ctx -> interp1d surfaceId t ctx

    /// Decrement rate from a 2D surface (e.g., by duration and attained age)
    let rateByDurationAndAge (surfaceId: int) (age: Expr) : ModelCtx -> Expr =
        fun ctx -> interp2d surfaceId TimeIndex age ctx

    // ── Survival probability accumulation ──────────────────────────────

    /// Cumulative survival probability: starts at 1, multiplies by (1 - rate * dt) each step.
    /// Returns an Expr representing the probability of surviving to current time.
    let survivalProb (rate: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx -> evolve 1.0f.C (fun px -> px * (1.0f.C - rate * dt)) ctx

    /// Cumulative survival probability with annual rates (dt = 1 year assumed).
    /// For use with annual mortality/lapse tables where rate is already per-year.
    let survivalProbAnnual (annualRate: Expr) : ModelCtx -> Expr =
        fun ctx -> evolve 1.0f.C (fun px -> px * (1.0f.C - annualRate)) ctx

    /// Cumulative survival using exponential model: exp(-rate * dt) each step.
    /// More accurate for continuous-time decrements.
    let survivalProbContinuous (rate: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx -> evolve 1.0f.C (fun px -> px * Expr.exp (-rate * dt)) ctx

    // ── Multi-decrement survival ───────────────────────────────────────

    /// Combined survival probability from multiple independent decrements.
    /// Each decrement is a (rate, dt) pair.
    let multiDecrementSurvival (decrements: (Expr * Expr) list) : ModelCtx -> Expr =
        fun ctx ->
            let combinedRate = decrements |> List.map (fun (rate, dt) -> rate * dt) |> List.reduce (+)
            evolve 1.0f.C (fun px -> px * (1.0f.C - combinedRate)) ctx

    // ── Weighted value helpers ─────────────────────────────────────────

    /// Apply survival probability to a value (e.g., account value, benefit).
    /// Result represents expected value accounting for probability of being in force.
    let weightedByInforce (value: Expr) (survivalProb: Expr) : Expr = value * survivalProb

    /// Apply survival and discount factor to get present value of expected cash flow.
    let weightedPV (value: Expr) (survivalProb: Expr) (discountFactor: Expr) : Expr =
        value * survivalProb * discountFactor

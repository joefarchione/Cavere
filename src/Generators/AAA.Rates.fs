namespace Cavere.Generators.AAA

open Cavere.Core
open Cavere.Generators.AAA.Common

/// AAA Interest Rate Model — Stochastic Log Volatility (SLV).
/// Three correlated mean-reverting processes:
/// 1. Log of 20-year (long) Treasury rate
/// 2. Spread (short rate excess over long)
/// 3. Log volatility
module Rates =

    // Helper to convert float log result to float32 Const
    let inline logC (x: float32) = Const (float32 (log (float x)))

    // ══════════════════════════════════════════════════════════════════
    // Core SLV Interest Rate Model
    // ══════════════════════════════════════════════════════════════════

    /// Build complete SLV interest rate model.
    /// Returns (longRate, shortRate, spread, logVol) expressions.
    let slvRateModel (p: RateModelParams) : ModelCtx -> Expr * Expr * Expr * Expr =
        fun ctx ->
            // Generate normals for the 3 processes
            let z1 = normal ctx  // Long rate
            let z2 = normal ctx  // Spread
            let z3 = normal ctx  // Log vol

            // Correlate z2 and z3 with z1
            let rho12 = p.RhoLongSpread
            let rho13 = p.RhoLongVol
            let sqrt1MinusRho12Sq = sqrt (1.0f - rho12 * rho12)
            let sqrt1MinusRho13Sq = sqrt (1.0f - rho13 * rho13)
            let z2Corr = rho12.C * z1 + sqrt1MinusRho12Sq.C * z2
            let z3Corr = rho13.C * z1 + sqrt1MinusRho13Sq.C * z3

            // Build log volatility (no dependencies on other state)
            let logVol =
                let beta = p.LogVol.Beta
                let tau = p.LogVol.Tau
                let sigma = p.LogVol.Sigma
                let nu0 = p.LogVol.Nu0
                evolve nu0.C (fun nu ->
                    (1.0f - beta).C * nu + beta.C * tau.C + sigma.C * z3Corr) ctx

            // Build log of long rate
            let logR0 = float32 (log (float p.LongRate.R0))
            let logTau1 = float32 (log (float p.LongRate.Tau))
            let longRateLog =
                let beta = p.LongRate.Beta
                let lb = p.LongRate.LowerBound
                let ub = p.LongRate.UpperBound
                evolve logR0.C (fun logR ->
                    let sigma = Expr.exp logVol
                    let meanRev = (1.0f - beta).C * logR + beta.C * logTau1.C
                    let clamped = Expr.max lb.C (Expr.min ub.C meanRev)
                    clamped + sigma * z1) ctx

            let longRate = Expr.exp longRateLog

            // Build spread
            let spread =
                let beta = p.Spread.Beta
                let tau = p.Spread.Tau
                let sigma = p.Spread.Sigma
                let theta = p.Spread.Theta
                let alpha0 = p.Spread.Alpha0
                evolve alpha0.C (fun alpha ->
                    let meanRev = (1.0f - beta).C * alpha + beta.C * tau.C
                    let ratePower = Expr.exp (theta.C * Expr.log (Expr.max 0.001f.C longRate))
                    meanRev + sigma.C * z2Corr * ratePower) ctx

            // Short rate from long rate and spread
            let shortRate = longRate * Expr.exp spread

            (longRate, shortRate, spread, logVol)

    // ══════════════════════════════════════════════════════════════════
    // Derived Quantities
    // ══════════════════════════════════════════════════════════════════

    /// Discount factor accumulator from short rate.
    let discountFactor (rate: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            evolve 1.0f.C (fun df -> df * Expr.exp (-(rate * dt))) ctx

    /// Accumulated short rate (for path-dependent products).
    let accumulatedRate (rate: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            evolve 0.0f.C (fun accum -> accum + rate * dt) ctx

    /// Nelson-Siegel yield curve interpolation.
    let nelsonSiegelYield (longRate: Expr) (shortRate: Expr) (maturity: float32) (lambda: float32) : Expr =
        let x = maturity / lambda
        let expX = exp (-float x) |> float32
        let factor1 = (1.0f - expX) / x
        shortRate * factor1.C + longRate * (1.0f - factor1).C

    /// Rate change for duration effect.
    let rateChange (rate: Expr) (prevRate: Expr) : Expr =
        rate - prevRate

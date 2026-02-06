namespace Cavere.Generators.AAA

open Cavere.Core
open Cavere.Generators.AAA.Common

/// AAA Bond Fund Model — Returns Tied to Treasury Rates.
/// Bond fund returns are linked to Treasury rates at specific reference
/// maturities with credit spread adjustments.
module Bonds =

    // ══════════════════════════════════════════════════════════════════
    // Bond Return Components
    // ══════════════════════════════════════════════════════════════════

    /// Income return from yield (coupon accrual).
    /// Monthly income = (Treasury yield + credit spread) / 12
    let incomeReturn (treasuryYield: Expr) (creditSpreadBps: float32) : Expr =
        let spreadDecimal = creditSpreadBps / 10000.0f
        (treasuryYield + spreadDecimal.C) / 12.0f.C

    /// Price return from rate change using duration/convexity approximation.
    /// ΔP/P ≈ -D × Δy + 0.5 × C × (Δy)²
    let priceReturn (duration: float32) (convexity: float32) (yieldChange: Expr) : Expr =
        if duration = 0.0f then
            0.0f.C
        else
            let durEffect = -(duration.C * yieldChange)
            let convEffect = 0.5f.C * convexity.C * yieldChange * yieldChange
            durEffect + convEffect

    /// Total bond fund return = income + price return.
    let bondReturn
        (treasuryYield: Expr)
        (yieldChange: Expr)
        (creditSpreadBps: float32)
        (duration: float32)
        (convexity: float32)
        : Expr =
        let income = incomeReturn treasuryYield creditSpreadBps
        let price = priceReturn duration convexity yieldChange
        income + price

    // ══════════════════════════════════════════════════════════════════
    // Single Bond Fund
    // ══════════════════════════════════════════════════════════════════

    /// Build a single bond fund process.
    /// For money market (duration = 0), just accrue at short rate.
    /// For other funds, use duration/convexity for price sensitivity.
    let singleBondFund
        (p: BondFundParams)
        (treasuryYield: Expr)
        (yieldChange: Expr)
        : ModelCtx -> Expr =
        fun ctx ->
            if p.Duration = 0.0f then
                // Money market: simple rate accrual
                evolve p.B0.C (fun b ->
                    let monthlyReturn = treasuryYield / 12.0f.C
                    b * (1.0f.C + monthlyReturn)
                ) ctx
            else
                // Bond fund with duration
                evolve p.B0.C (fun b ->
                    let ret = bondReturn treasuryYield yieldChange p.CreditSpread p.Duration p.Convexity
                    b * (1.0f.C + ret)
                ) ctx

    // ══════════════════════════════════════════════════════════════════
    // Multiple Bond Funds with Yield Curve
    // ══════════════════════════════════════════════════════════════════

    /// Build multiple bond funds.
    /// Each fund references a different point on the yield curve.
    /// longRate is the 20-year rate, shortRate is the 1-year rate.
    /// Uses linear interpolation for intermediate maturities.
    let bondFunds
        (funds: BondFundParams list)
        (longRate: Expr)
        (shortRate: Expr)
        (longRateChange: Expr)
        (shortRateChange: Expr)
        : ModelCtx -> Expr[] =
        fun ctx ->
            funds
            |> List.map (fun fund ->
                // Interpolate yield at reference maturity
                // Simple linear interpolation between 1yr and 20yr
                let weight =
                    if fund.ReferenceMat <= 1.0f then 0.0f
                    elif fund.ReferenceMat >= 20.0f then 1.0f
                    else (fund.ReferenceMat - 1.0f) / 19.0f

                let treasuryYield = (1.0f - weight).C * shortRate + weight.C * longRate
                let yieldChange = (1.0f - weight).C * shortRateChange + weight.C * longRateChange

                singleBondFund fund treasuryYield yieldChange ctx
            )
            |> Array.ofList

    /// Build bond funds using only current rates (approximates rate change from vol).
    /// Uses the rate shock from the SLV model for duration effect.
    let bondFundsWithShock
        (funds: BondFundParams list)
        (longRate: Expr)
        (shortRate: Expr)
        (longRateVol: Expr)
        (z: Expr)
        : ModelCtx -> Expr[] =
        fun ctx ->
            let rateShock = Expr.exp longRateVol * z

            funds
            |> List.map (fun fund ->
                let weight =
                    if fund.ReferenceMat <= 1.0f then 0.0f
                    elif fund.ReferenceMat >= 20.0f then 1.0f
                    else (fund.ReferenceMat - 1.0f) / 19.0f

                let treasuryYield = (1.0f - weight).C * shortRate + weight.C * longRate
                // Scale shock by weight (longer bonds affected more)
                let yieldChange = weight.C * rateShock

                singleBondFund fund treasuryYield yieldChange ctx
            )
            |> Array.ofList

    // ══════════════════════════════════════════════════════════════════
    // Credit Spread Dynamics (optional enhancement)
    // ══════════════════════════════════════════════════════════════════

    /// Stochastic credit spread (mean-reverting).
    /// Can be used for high-yield funds or stressed scenarios.
    let stochasticSpread
        (meanSpreadBps: float32)
        (beta: float32)
        (sigma: float32)
        (z: Expr)
        (initSpreadBps: float32)
        : ModelCtx -> Expr =
        fun ctx ->
            evolve initSpreadBps.C (fun spread ->
                let meanRev = (1.0f - beta).C * spread + beta.C * meanSpreadBps.C
                meanRev + sigma.C * z
            ) ctx

    /// Bond fund with stochastic credit spread.
    let bondFundWithStochasticSpread
        (p: BondFundParams)
        (treasuryYield: Expr)
        (yieldChange: Expr)
        (spreadChange: Expr)
        : ModelCtx -> Expr =
        fun ctx ->
            if p.Duration = 0.0f then
                evolve p.B0.C (fun b ->
                    let monthlyReturn = treasuryYield / 12.0f.C
                    b * (1.0f.C + monthlyReturn)
                ) ctx
            else
                evolve p.B0.C (fun b ->
                    // Total yield change includes spread change
                    let totalYieldChange = yieldChange + spreadChange
                    let ret = bondReturn treasuryYield totalYieldChange p.CreditSpread p.Duration p.Convexity
                    b * (1.0f.C + ret)
                ) ctx

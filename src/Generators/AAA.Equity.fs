namespace Cavere.Generators.AAA

open Cavere.Core
open Cavere.Generators.AAA.Common

/// AAA Equity Model — Stochastic Log Volatility (SLV).
/// Similar to interest rate model but for equity returns.
/// Captures volatility clustering, negative skewness, and fat tails.
module Equity =

    // ══════════════════════════════════════════════════════════════════
    // Single Fund SLV Model
    // ══════════════════════════════════════════════════════════════════

    /// Build single equity fund with SLV.
    /// ln(S_t/S_{t-1}) = μ - σ²/2 + σ*Z_S
    /// where σ = exp(ν_t)
    /// Z_S and Z_ν are correlated with rho (leverage effect)
    let singleEquityFund (p: EquityFundParams) (zReturn: Expr) (zVol: Expr) : ModelCtx -> Expr =
        fun ctx ->
            // Correlate return shock with vol shock (leverage effect)
            let sqrtOneMinusRhoSq = sqrt (1.0f - p.Rho * p.Rho)
            let zCorr = p.Rho.C * zVol + sqrtOneMinusRhoSq.C * zReturn

            // Build log volatility
            let logVol =
                let beta = p.VolBeta
                let tau = p.VolTau
                let sigma = p.VolSigma
                let nu0 = p.Nu0
                evolve nu0.C (fun nu -> (1.0f - beta).C * nu + beta.C * tau.C + sigma.C * zVol) ctx

            // Build price process
            let mu = p.Mu
            let s0 = p.S0

            evolve
                s0.C
                (fun s ->
                    let sigma = Expr.exp logVol
                    let drift = mu.C - 0.5f.C * sigma * sigma
                    s * Expr.exp (drift + sigma * zCorr))
                ctx

    // ══════════════════════════════════════════════════════════════════
    // Multiple Correlated Funds
    // ══════════════════════════════════════════════════════════════════

    /// Build multiple correlated equity funds.
    /// Uses correlation matrix for return shocks.
    /// Each fund has its own independent vol shock (for leverage effect).
    let equityFunds (p: EquityModelParams) : ModelCtx -> Expr[] =
        fun ctx ->
            let n = p.Funds.Length

            // Generate correlated return shocks
            let returnShocks = correlatedNormals p.Correlation ctx

            // Generate independent vol shocks (one per fund)
            let volShocks = Array.init n (fun _ -> normal ctx)

            // Build each fund
            p.Funds
            |> List.mapi (fun i fund ->
                let zReturn = returnShocks.[i]
                let zVol = volShocks.[i]

                // Correlate return shock with vol shock (leverage effect)
                let sqrtOneMinusRhoSq = sqrt (1.0f - fund.Rho * fund.Rho)
                let zCorr = fund.Rho.C * zVol + sqrtOneMinusRhoSq.C * zReturn

                // Build log volatility for this fund
                let logVol =
                    evolve
                        fund.Nu0.C
                        (fun nu ->
                            (1.0f - fund.VolBeta).C * nu
                            + fund.VolBeta.C * fund.VolTau.C
                            + fund.VolSigma.C * zVol)
                        ctx

                // Build price process
                evolve
                    fund.S0.C
                    (fun s ->
                        let sigma = Expr.exp logVol
                        let drift = fund.Mu.C - 0.5f.C * sigma * sigma
                        s * Expr.exp (drift + sigma * zCorr))
                    ctx)
            |> Array.ofList

    // ══════════════════════════════════════════════════════════════════
    // Return Calculations
    // ══════════════════════════════════════════════════════════════════

    /// Compute total return of an equity fund.
    let totalReturn (fundStart: Expr) (fundEnd: Expr) : Expr = fundEnd / fundStart - 1.0f.C

    /// Compute log return of an equity fund.
    let logReturn (fundStart: Expr) (fundEnd: Expr) : Expr = Expr.log fundEnd - Expr.log fundStart

    /// Annualized volatility from log volatility.
    let annualizedVol (logVol: Expr) (monthsPerYear: float32) : Expr = Expr.exp logVol * (sqrt monthsPerYear).C

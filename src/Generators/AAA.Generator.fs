namespace Cavere.Generators.AAA

open Cavere.Core
open Cavere.Generators.AAA.Common
open Cavere.Generators.AAA.Rates
open Cavere.Generators.AAA.Equity
open Cavere.Generators.AAA.Bonds

/// AAA Economic Scenario Generator — Full Implementation.
/// Combines SLV interest rates, SLV equities, and duration-based bond funds.
/// Based on the Academy Interest Rate Generator (AIRG) methodology.
module Generator =

    // ══════════════════════════════════════════════════════════════════
    // Core Generator Functions
    // ══════════════════════════════════════════════════════════════════

    /// Build the full AAA scenario model.
    /// Produces a model that observes rates, equities, and bond funds.
    let buildScenarioModel (p: AAAParams) : Model =
        model {
            // Monthly time step
            let dt = (1.0f / 12.0f).C

            // Build SLV interest rate model
            let! longRate, shortRate, spread, logVol = slvRateModel p.Rates

            // Discount factor from short rate
            let! df = discountFactor shortRate dt

            // Build equity funds (SLV)
            let! equities = equityFunds p.Equity

            // For bond funds, we need rate shock
            let! zBond = normal
            let! bonds = bondFundsWithShock p.Bonds longRate shortRate logVol zBond

            // Observations for analysis
            do! observe "longRate" longRate
            do! observe "shortRate" shortRate
            do! observe "spread" spread
            do! observe "logVol" logVol
            do! observe "df" df

            // Observe each equity fund
            do! iter 0 equities.Length (fun i -> observe ($"equity_{i}") equities.[i])

            // Observe each bond fund
            do! iter 0 bonds.Length (fun i -> observe ($"bond_{i}") bonds.[i])

            // Return discounted portfolio value (equal weighted)
            let totalEquity =
                if equities.Length = 0 then
                    0.0f.C
                else
                    equities |> Array.reduce (+)

            let totalBonds =
                if bonds.Length = 0 then
                    0.0f.C
                else
                    bonds |> Array.reduce (+)

            let numFunds = float32 (equities.Length + bonds.Length)
            return (totalEquity + totalBonds) / numFunds.C * df
        }

    /// Build a rates-only model.
    let buildRatesOnlyModel (p: RateModelParams) : Model =
        model {
            let dt = (1.0f / 12.0f).C
            let! longRate, shortRate, spread, logVol = slvRateModel p
            let! df = discountFactor shortRate dt

            do! observe "longRate" longRate
            do! observe "shortRate" shortRate
            do! observe "spread" spread
            do! observe "logVol" logVol
            do! observe "df" df

            return df
        }

    /// Build an equity-only model.
    let buildEquityOnlyModel (p: EquityModelParams) : Model =
        model {
            let! funds = equityFunds p

            do! iter 0 funds.Length (fun i -> observe ($"fund_{i}") funds.[i])

            return
                if funds.Length = 0 then
                    1.0f.C
                else
                    funds |> Array.reduce (+) |> (fun s -> s / float32 funds.Length)
        }

    /// Build a simple model with just rates and one equity fund (S&P 500).
    let buildSimpleModel (rateP: RateModelParams) (equityP: EquityFundParams) : Model =
        model {
            let dt = (1.0f / 12.0f).C
            let! longRate, shortRate, spread, logVol = slvRateModel rateP
            let! df = discountFactor shortRate dt

            // Single equity fund
            let! zReturn = normal
            let! zVol = normal
            let! equity = singleEquityFund equityP zReturn zVol

            do! observe "longRate" longRate
            do! observe "shortRate" shortRate
            do! observe "df" df
            do! observe "equity" equity

            return equity * df
        }

    // ══════════════════════════════════════════════════════════════════
    // Convenience Functions
    // ══════════════════════════════════════════════════════════════════

    /// Build default AAA model with standard parameters.
    let buildDefaultModel () : Model = buildScenarioModel defaultParams

    /// Build model with custom number of steps (projection months).
    let buildModelWithSteps (steps: int) : Model = buildScenarioModel { defaultParams with Steps = steps }

    // ══════════════════════════════════════════════════════════════════
    // Result Extraction Helpers
    // ══════════════════════════════════════════════════════════════════

    /// Extract long rate from scenario results.
    let extractLongRate (watch: WatchResult) : float32[,] = Watcher.values "longRate" watch

    /// Extract short rate from scenario results.
    let extractShortRate (watch: WatchResult) : float32[,] = Watcher.values "shortRate" watch

    /// Extract discount factors from scenario results.
    let extractDiscountFactors (watch: WatchResult) : float32[,] = Watcher.values "df" watch

    /// Extract equity fund values from scenario results.
    let extractEquity (fundIndex: int) (watch: WatchResult) : float32[,] = Watcher.values ($"equity_{fundIndex}") watch

    /// Extract bond fund values from scenario results.
    let extractBond (fundIndex: int) (watch: WatchResult) : float32[,] = Watcher.values ($"bond_{fundIndex}") watch

    // ══════════════════════════════════════════════════════════════════
    // Statistics Helpers
    // ══════════════════════════════════════════════════════════════════

    /// Compute percentiles across scenarios at each time point.
    let percentiles (data: float32[,]) (pcts: float32[]) : float32[,] =
        let numObs = Array2D.length1 data
        let numScenarios = Array2D.length2 data

        Array2D.init numObs pcts.Length (fun t p ->
            let sorted = [| for s in 0 .. numScenarios - 1 -> data.[t, s] |] |> Array.sort
            let idx = int (pcts.[p] * float32 (numScenarios - 1))
            sorted.[min idx (numScenarios - 1)])

    /// Compute mean across scenarios at each time point.
    let means (data: float32[,]) : float32[] =
        let numObs = Array2D.length1 data
        let numScenarios = Array2D.length2 data

        Array.init numObs (fun t ->
            let mutable sum = 0.0f

            for s in 0 .. numScenarios - 1 do
                sum <- sum + data.[t, s]

            sum / float32 numScenarios)

    /// Compute standard deviation across scenarios at each time point.
    let stdDevs (data: float32[,]) : float32[] =
        let numObs = Array2D.length1 data
        let numScenarios = Array2D.length2 data
        let meanVals = means data

        Array.init numObs (fun t ->
            let mutable sumSq = 0.0f

            for s in 0 .. numScenarios - 1 do
                let diff = data.[t, s] - meanVals.[t]
                sumSq <- sumSq + diff * diff

            sqrt (sumSq / float32 numScenarios))

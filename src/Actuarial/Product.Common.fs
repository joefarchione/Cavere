namespace Cavere.Actuarial.Product

open Cavere.Core

/// Common product features, fees, and adjustments shared across product types.
module Common =

    // ══════════════════════════════════════════════════════════════════
    // CDSC — Contingent Deferred Sales Charge
    // ══════════════════════════════════════════════════════════════════

    /// CDSC schedule: charge percentages by policy year.
    /// Example: [| 7%; 6%; 5%; 4%; 3%; 2%; 1%; 0% |] for 7-year schedule.
    type CDSCSchedule = {
        ChargesByYear: float32[]
    }

    module CDSC =
        let create (charges: float32[]) : CDSCSchedule =
            { ChargesByYear = charges }

        /// Common 7-year declining schedule.
        let standard7Year : CDSCSchedule =
            create [| 0.07f; 0.06f; 0.05f; 0.04f; 0.03f; 0.02f; 0.01f; 0.0f |]

        /// Common 5-year declining schedule.
        let standard5Year : CDSCSchedule =
            create [| 0.05f; 0.04f; 0.03f; 0.02f; 0.01f; 0.0f |]

        /// No surrender charge.
        let none : CDSCSchedule =
            create [| 0.0f |]

        /// Load CDSC schedule onto GPU, returns surface ID.
        let load (schedule: CDSCSchedule) : ModelCtx -> int = fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(schedule.ChargesByYear, schedule.ChargesByYear.Length))
            id

        /// Lookup CDSC rate by policy duration.
        let rate (surfaceId: int) : Expr = Lookup1D surfaceId

        /// Apply CDSC to get cash surrender value.
        let applyCDSC (accountValue: Expr) (cdscRate: Expr) : Expr =
            accountValue * (1.0f.C - cdscRate)

    // ══════════════════════════════════════════════════════════════════
    // MVA — Market Value Adjustment
    // ══════════════════════════════════════════════════════════════════

    /// MVA parameters for interest rate sensitivity adjustment.
    type MVAParams = {
        ReferenceRate: float32      // Rate at issue
        Sensitivity: float32        // MVA sensitivity factor
    }

    module MVA =
        let create (refRate: float32) (sensitivity: float32) : MVAParams =
            { ReferenceRate = refRate; Sensitivity = sensitivity }

        /// Calculate MVA factor based on rate change.
        /// MVA = 1 + sensitivity * (referenceRate - currentRate) * remainingTerm
        let factor (currentRate: Expr) (remainingTerm: Expr) (refRate: float32) (sensitivity: float32) : Expr =
            1.0f.C + sensitivity.C * (refRate.C - currentRate) * remainingTerm

        /// Apply MVA to account value.
        let applyMVA (accountValue: Expr) (mvaFactor: Expr) : Expr =
            accountValue * mvaFactor

    // ══════════════════════════════════════════════════════════════════
    // RMD — Required Minimum Distribution
    // ══════════════════════════════════════════════════════════════════

    /// RMD calculation based on IRS life expectancy tables.
    type RMDParams = {
        StartAge: int               // Age when RMDs begin (72 or 73)
        UseUniformTable: bool       // Uniform Lifetime vs Joint table
    }

    module RMD =
        let create (startAge: int) : RMDParams =
            { StartAge = startAge; UseUniformTable = true }

        /// Standard RMD start age (73 as of SECURE 2.0 for those born 1951-1959).
        let standard : RMDParams = create 73

        /// Load life expectancy factors onto GPU.
        let loadLifeExpectancy (factors: float32[]) : ModelCtx -> int = fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(factors, factors.Length))
            id

        /// RMD amount = Account Value / Life Expectancy Factor
        let rmdAmount (accountValue: Expr) (lifeExpectancyFactor: Expr) : Expr =
            accountValue / lifeExpectancyFactor

        /// Check if RMD applies (age >= startAge).
        let rmdApplies (attainedAge: Expr) (startAge: int) : Expr =
            attainedAge .>= float32 startAge

    // ══════════════════════════════════════════════════════════════════
    // Common Fees
    // ══════════════════════════════════════════════════════════════════

    /// Fee structure for insurance products.
    type FeeStructure = {
        MandE: float32              // Mortality & Expense (annual %)
        Admin: float32              // Administrative fee (annual %)
        FundExpense: float32        // Underlying fund expense (annual %)
        RiderCharges: float32       // Total rider charges (annual %)
    }

    module Fees =
        let create (me: float32) (admin: float32) (fund: float32) (riders: float32) : FeeStructure =
            { MandE = me; Admin = admin; FundExpense = fund; RiderCharges = riders }

        /// Total annual fee rate.
        let totalAnnual (fees: FeeStructure) : float32 =
            fees.MandE + fees.Admin + fees.FundExpense + fees.RiderCharges

        /// No fees.
        let none : FeeStructure =
            create 0.0f 0.0f 0.0f 0.0f

        /// Apply annual fees to account value (continuous deduction).
        let applyFeesContinuous (accountValue: Expr) (annualFeeRate: Expr) (dt: Expr) : Expr =
            accountValue * Expr.exp (-annualFeeRate * dt)

        /// Apply annual fees to account value (periodic deduction).
        let applyFeesPeriodic (accountValue: Expr) (annualFeeRate: Expr) (dt: Expr) : Expr =
            accountValue * (1.0f.C - annualFeeRate * dt)

    // ══════════════════════════════════════════════════════════════════
    // Premium Modes
    // ══════════════════════════════════════════════════════════════════

    type PremiumMode =
        | Single
        | Annual
        | Semiannual
        | Quarterly
        | Monthly

    module PremiumMode =
        let periodsPerYear = function
            | Single -> 1
            | Annual -> 1
            | Semiannual -> 2
            | Quarterly -> 4
            | Monthly -> 12

    // ══════════════════════════════════════════════════════════════════
    // Benefit Base Types (for riders)
    // ══════════════════════════════════════════════════════════════════

    /// How the benefit base is calculated/adjusted.
    type BenefitBaseType =
        | PremiumBased                      // Based on total premiums
        | AccountValueBased                 // Based on current AV
        | HighWaterMark                     // Highest AV on anniversaries
        | RollUp of annualRate: float32     // Compounds at fixed rate
        | GreaterOf of BenefitBaseType * BenefitBaseType

    module BenefitBase =
        /// Update benefit base based on type.
        let update (baseType: BenefitBaseType) (currentBase: Expr) (accountValue: Expr) (premium: Expr) : Expr =
            match baseType with
            | PremiumBased -> premium
            | AccountValueBased -> accountValue
            | HighWaterMark -> Expr.max currentBase accountValue
            | RollUp rate -> currentBase * (1.0f + rate).C
            | GreaterOf (t1, t2) ->
                // Simplified: would need recursive handling
                Expr.max currentBase accountValue

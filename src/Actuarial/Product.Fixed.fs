namespace Cavere.Actuarial.Product

open Cavere.Core

/// Fixed Annuity product definitions.
/// Includes traditional fixed annuities and MYGAs (Multi-Year Guaranteed Annuities).
module Fixed =

    // ══════════════════════════════════════════════════════════════════
    // Fixed Annuity Types
    // ══════════════════════════════════════════════════════════════════

    type FixedAnnuityType =
        | Traditional                       // Declared rate, can change
        | MYGA of guaranteePeriod: int      // Multi-year guaranteed rate

    // ══════════════════════════════════════════════════════════════════
    // Rate Crediting
    // ══════════════════════════════════════════════════════════════════

    type RateCrediting = {
        GuaranteedRate: float32             // Minimum guaranteed rate
        CurrentRate: float32                // Current declared rate
        RenewalRateFormula: RenewalFormula option
    }

    and RenewalFormula =
        | Flat of rate: float32             // Fixed renewal rate
        | SpreadOverIndex of indexName: string * spread: float32
        | MinOfCurrentOrIndex of indexName: string

    module RateCrediting =
        let guaranteed (rate: float32) : RateCrediting =
            { GuaranteedRate = rate; CurrentRate = rate; RenewalRateFormula = None }

        let withCurrent (guaranteed: float32) (current: float32) : RateCrediting =
            { GuaranteedRate = guaranteed; CurrentRate = current; RenewalRateFormula = None }

        /// Credited rate is max of guaranteed and current.
        let effectiveRate (crediting: RateCrediting) : Expr =
            Expr.max crediting.GuaranteedRate.C crediting.CurrentRate.C

    // ══════════════════════════════════════════════════════════════════
    // Fixed Annuity Product Definition
    // ══════════════════════════════════════════════════════════════════

    type FixedAnnuityProduct = {
        Name: string
        ProductType: FixedAnnuityType
        RateCrediting: RateCrediting
        CDSCSchedule: Common.CDSCSchedule
        MVAParams: Common.MVAParams option
        FreeWithdrawalPercent: float32      // Annual free withdrawal (e.g., 10%)
        Fees: Common.FeeStructure
        BailoutRate: float32 option         // Rate below which surrender charge waived
    }

    module FixedAnnuityProduct =
        /// Create a traditional fixed annuity.
        let traditional (name: string) (guaranteed: float32) (current: float32)
                        (cdsc: Common.CDSCSchedule) (freeWithdrawal: float32) : FixedAnnuityProduct =
            {
                Name = name
                ProductType = Traditional
                RateCrediting = RateCrediting.withCurrent guaranteed current
                CDSCSchedule = cdsc
                MVAParams = None
                FreeWithdrawalPercent = freeWithdrawal
                Fees = Common.Fees.none
                BailoutRate = None
            }

        /// Create a MYGA (Multi-Year Guaranteed Annuity).
        let myga (name: string) (guaranteedRate: float32) (years: int)
                 (cdsc: Common.CDSCSchedule) : FixedAnnuityProduct =
            {
                Name = name
                ProductType = MYGA years
                RateCrediting = RateCrediting.guaranteed guaranteedRate
                CDSCSchedule = cdsc
                MVAParams = None
                FreeWithdrawalPercent = 0.10f
                Fees = Common.Fees.none
                BailoutRate = None
            }

        /// Add MVA to product.
        let withMVA (mva: Common.MVAParams) (product: FixedAnnuityProduct) : FixedAnnuityProduct =
            { product with MVAParams = Some mva }

        /// Add bailout provision.
        let withBailout (bailoutRate: float32) (product: FixedAnnuityProduct) : FixedAnnuityProduct =
            { product with BailoutRate = Some bailoutRate }

    // ══════════════════════════════════════════════════════════════════
    // Account Value Accumulation
    // ══════════════════════════════════════════════════════════════════

    module AccountValue =
        /// Simple interest accumulation for fixed annuity.
        let accumulateFixed (rate: Expr) (dt: Expr) (init: Expr) : ModelCtx -> Expr = fun ctx ->
            evolve init (fun av -> av * (1.0f.C + rate * dt)) ctx

        /// Compound interest accumulation.
        let accumulateCompound (rate: Expr) (dt: Expr) (init: Expr) : ModelCtx -> Expr = fun ctx ->
            evolve init (fun av -> av * Expr.exp (rate * dt)) ctx

        /// Account value with fee deduction.
        let accumulateWithFees (rate: Expr) (feeRate: Expr) (dt: Expr) (init: Expr) : ModelCtx -> Expr = fun ctx ->
            let netRate = rate - feeRate
            evolve init (fun av -> av * Expr.exp (netRate * dt)) ctx

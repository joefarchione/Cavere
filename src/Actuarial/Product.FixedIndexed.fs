namespace Cavere.Actuarial.Product

open Cavere.Core

/// Fixed Indexed Annuity (FIA) product definitions.
/// Includes various crediting strategies and index allocation options.
module FixedIndexed =

    // ══════════════════════════════════════════════════════════════════
    // Index Definitions
    // ══════════════════════════════════════════════════════════════════

    type IndexType =
        | SP500
        | Russell2000
        | MSCI_EAFE
        | Nasdaq100
        | Custom of name: string

    // ══════════════════════════════════════════════════════════════════
    // Crediting Strategies
    // ══════════════════════════════════════════════════════════════════

    /// Point-to-point crediting: measures index change from start to end of term.
    type PointToPointParams = {
        Cap: float32 // Maximum credited return
        Floor: float32 // Minimum credited return (usually 0%)
        Participation: float32 // Participation rate (e.g., 100%)
        Spread: float32 // Spread deducted from return
    }

    /// Monthly cap crediting: caps each month's return.
    type MonthlyCapParams = {
        Cap: float32 // Monthly cap
        Floor: float32 // Monthly floor (usually 0%)
    }

    /// Monthly average crediting: averages monthly index values.
    type MonthlyAverageParams = {
        Cap: float32 // Cap on averaged return
        Participation: float32 // Participation rate
    }

    /// Performance trigger: pays fixed rate if index is positive.
    type PerformanceTriggerParams = {
        TriggerRate: float32 // Rate paid if index >= 0
    }

    /// Two-year point-to-point: longer crediting term.
    type MultiYearPointToPointParams = {
        Years: int // Crediting term (2, 3, etc.)
        Cap: float32
        Floor: float32
        Participation: float32
    }

    /// Main crediting strategy discriminated union.
    type CreditingStrategy =
        | PointToPoint of PointToPointParams
        | MonthlyCap of MonthlyCapParams
        | MonthlyAverage of MonthlyAverageParams
        | PerformanceTrigger of PerformanceTriggerParams
        | MultiYearPointToPoint of MultiYearPointToPointParams
        | FixedAccount of guaranteedRate: float32

    module CreditingStrategy =
        /// Standard point-to-point with cap and participation.
        let pointToPoint (cap: float32) (participation: float32) : CreditingStrategy =
            PointToPoint {
                Cap = cap
                Floor = 0.0f
                Participation = participation
                Spread = 0.0f
            }

        /// Point-to-point with spread instead of cap.
        let pointToPointWithSpread (spread: float32) (participation: float32) : CreditingStrategy =
            PointToPoint {
                Cap = 1.0f
                Floor = 0.0f
                Participation = participation
                Spread = spread
            }

        /// Monthly cap strategy.
        let monthlyCap (cap: float32) : CreditingStrategy = MonthlyCap { Cap = cap; Floor = 0.0f }

        /// Monthly average with participation.
        let monthlyAverage (cap: float32) (participation: float32) : CreditingStrategy =
            MonthlyAverage {
                Cap = cap
                Participation = participation
            }

        /// Performance trigger strategy.
        let performanceTrigger (rate: float32) : CreditingStrategy = PerformanceTrigger { TriggerRate = rate }

        /// Fixed account (guaranteed rate).
        let fixedAccount (rate: float32) : CreditingStrategy = FixedAccount rate

    // ══════════════════════════════════════════════════════════════════
    // Index Account Allocation
    // ══════════════════════════════════════════════════════════════════

    /// Allocation to a specific index strategy.
    type IndexAllocation = {
        Index: IndexType
        Strategy: CreditingStrategy
        Allocation: float32 // Percentage allocated (0.0 - 1.0)
    }

    module IndexAllocation =
        let create (index: IndexType) (strategy: CreditingStrategy) (pct: float32) : IndexAllocation = {
            Index = index
            Strategy = strategy
            Allocation = pct
        }

    // ══════════════════════════════════════════════════════════════════
    // FIA Product Definition
    // ══════════════════════════════════════════════════════════════════

    type FIAProduct = {
        Name: string
        IndexAllocations: IndexAllocation list
        CDSCSchedule: Common.CDSCSchedule
        MVAParams: Common.MVAParams option
        FreeWithdrawalPercent: float32
        Fees: Common.FeeStructure
        MinGuaranteedValue: float32 // MGAV as % of premium (e.g., 87.5%)
        BonusPercent: float32 option // Premium bonus (if any)
        BonusVestingYears: int // Years for bonus to vest
        Riders: Riders.RiderPackage // Optional GMxB riders
    }

    module FIAProduct =
        /// Create a basic FIA with single index allocation.
        let create (name: string) (allocation: IndexAllocation) (cdsc: Common.CDSCSchedule) : FIAProduct = {
            Name = name
            IndexAllocations = [ allocation ]
            CDSCSchedule = cdsc
            MVAParams = None
            FreeWithdrawalPercent = 0.10f
            Fees = Common.Fees.none
            MinGuaranteedValue = 0.875f
            BonusPercent = None
            BonusVestingYears = 0
            Riders = Riders.RiderPackage.empty
        }

        /// Create FIA with multiple index allocations.
        let createMultiIndex
            (name: string)
            (allocations: IndexAllocation list)
            (cdsc: Common.CDSCSchedule)
            : FIAProduct =
            {
                Name = name
                IndexAllocations = allocations
                CDSCSchedule = cdsc
                MVAParams = None
                FreeWithdrawalPercent = 0.10f
                Fees = Common.Fees.none
                MinGuaranteedValue = 0.875f
                BonusPercent = None
                BonusVestingYears = 0
                Riders = Riders.RiderPackage.empty
            }

        /// Add premium bonus to FIA.
        let withBonus (bonusPct: float32) (vestingYears: int) (product: FIAProduct) : FIAProduct = {
            product with
                BonusPercent = Some bonusPct
                BonusVestingYears = vestingYears
        }

        /// Add MVA to FIA.
        let withMVA (mva: Common.MVAParams) (product: FIAProduct) : FIAProduct = { product with MVAParams = Some mva }

        /// Set minimum guaranteed account value percentage.
        let withMinGuarantee (pct: float32) (product: FIAProduct) : FIAProduct = {
            product with
                MinGuaranteedValue = pct
        }

        /// Add rider package.
        let withRiders (riders: Riders.RiderPackage) (product: FIAProduct) : FIAProduct = {
            product with
                Riders = riders
        }

        /// Add GLWB rider (common for FIAs).
        let withGLWB (glwb: Riders.GLWB) (product: FIAProduct) : FIAProduct = {
            product with
                Riders = Riders.RiderPackage.withGLWB glwb product.Riders
        }

        /// Add GMDB rider.
        let withGMDB (gmdb: Riders.GMDB) (product: FIAProduct) : FIAProduct = {
            product with
                Riders = Riders.RiderPackage.withGMDB gmdb product.Riders
        }

        /// Total rider charges.
        let totalRiderCharge (product: FIAProduct) : float32 = Riders.RiderPackage.totalAnnualCharge product.Riders

    // ══════════════════════════════════════════════════════════════════
    // Crediting Calculations
    // ══════════════════════════════════════════════════════════════════

    module Crediting =
        /// Point-to-point credited rate calculation.
        /// credit = max(floor, min(cap, (indexReturn - spread) * participation))
        let pointToPointCredit (indexReturn: Expr) (p: PointToPointParams) : Expr =
            let grossReturn = (indexReturn - p.Spread.C) * p.Participation.C
            Expr.max p.Floor.C (Expr.min p.Cap.C grossReturn)

        /// Monthly cap credited rate for single month.
        let monthlyCapCredit (monthReturn: Expr) (p: MonthlyCapParams) : Expr =
            Expr.max p.Floor.C (Expr.min p.Cap.C monthReturn)

        /// Performance trigger credit.
        let performanceTriggerCredit (indexReturn: Expr) (p: PerformanceTriggerParams) : Expr =
            Expr.select (indexReturn .>= 0.0f) p.TriggerRate.C 0.0f.C

        /// Account value accumulation with credited rate.
        let accumulateWithCredit (creditedRate: Expr) (dt: Expr) (init: Expr) : ModelCtx -> Expr =
            fun ctx -> evolve init (fun av -> av * (1.0f.C + creditedRate * dt)) ctx

        /// Index return from start to end values.
        let indexReturn (startValue: Expr) (endValue: Expr) : Expr = (endValue - startValue) / startValue

    // ══════════════════════════════════════════════════════════════════
    // Guaranteed Minimum Values
    // ══════════════════════════════════════════════════════════════════

    module Guarantees =
        /// Minimum Guaranteed Account Value (MGAV).
        /// Typically 87.5% of premium compounded at 1-3%.
        let mgav (premium: Expr) (guaranteeRate: Expr) (dt: Expr) : ModelCtx -> Expr =
            fun ctx -> evolve premium (fun mgav -> mgav * Expr.exp (guaranteeRate * dt)) ctx

        /// Account value floored at MGAV.
        let avWithMgavFloor (accountValue: Expr) (mgav: Expr) : Expr = Expr.max accountValue mgav

        /// Premium bonus applied to account value.
        let applyBonus (premium: Expr) (bonusPercent: float32) : Expr = premium * (1.0f + bonusPercent).C

        /// Vested bonus based on years held.
        /// Common pattern: bonus vests linearly over N years.
        let vestedBonus (totalBonus: Expr) (yearsHeld: Expr) (vestingYears: int) : Expr =
            let vestingFactor = Expr.min (yearsHeld / float32 vestingYears) 1.0f.C
            totalBonus * vestingFactor

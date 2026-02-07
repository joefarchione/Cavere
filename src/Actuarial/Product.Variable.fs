namespace Cavere.Actuarial.Product

open Cavere.Core

/// Variable Annuity (VA) product definitions.
/// Includes sub-account allocations, GMxB riders, and fee structures.
module Variable =

    // ══════════════════════════════════════════════════════════════════
    // Sub-Account Definitions
    // ══════════════════════════════════════════════════════════════════

    type AssetClass =
        | Equity
        | FixedIncome
        | Balanced
        | MoneyMarket
        | International
        | Specialty

    type SubAccount = {
        Name: string
        AssetClass: AssetClass
        ExpenseRatio: float32           // Annual expense ratio
        Volatility: float32             // Expected volatility for risk modeling
    }

    module SubAccount =
        let create (name: string) (assetClass: AssetClass) (expense: float32) (vol: float32) : SubAccount =
            { Name = name; AssetClass = assetClass; ExpenseRatio = expense; Volatility = vol }

        /// Common equity sub-account.
        let equity (name: string) (expense: float32) (vol: float32) : SubAccount =
            create name Equity expense vol

        /// Common bond sub-account.
        let bond (name: string) (expense: float32) (vol: float32) : SubAccount =
            create name FixedIncome expense vol

    type SubAccountAllocation = {
        SubAccount: SubAccount
        Allocation: float32             // Percentage (0.0 - 1.0)
    }

    // ══════════════════════════════════════════════════════════════════
    // Variable Annuity Product Definition
    // ══════════════════════════════════════════════════════════════════

    type VAProduct = {
        Name: string
        SubAccountAllocations: SubAccountAllocation list
        Fees: Common.FeeStructure
        CDSCSchedule: Common.CDSCSchedule option
        FreeWithdrawalPercent: float32
        Riders: Riders.RiderPackage
    }

    module VAProduct =
        /// Create basic VA with no riders.
        let create (name: string) (allocations: SubAccountAllocation list)
                   (fees: Common.FeeStructure) : VAProduct =
            {
                Name = name
                SubAccountAllocations = allocations
                Fees = fees
                CDSCSchedule = None
                FreeWithdrawalPercent = 0.10f
                Riders = Riders.RiderPackage.empty
            }

        /// Add CDSC schedule.
        let withCDSC (cdsc: Common.CDSCSchedule) (product: VAProduct) : VAProduct =
            { product with CDSCSchedule = Some cdsc }

        /// Add rider package.
        let withRiders (riders: Riders.RiderPackage) (product: VAProduct) : VAProduct =
            { product with Riders = riders }

        /// Add GMAB rider.
        let withGMAB (gmab: Riders.GMAB) (product: VAProduct) : VAProduct =
            { product with Riders = Riders.RiderPackage.withGMAB gmab product.Riders }

        /// Add GMIB rider.
        let withGMIB (gmib: Riders.GMIB) (product: VAProduct) : VAProduct =
            { product with Riders = Riders.RiderPackage.withGMIB gmib product.Riders }

        /// Add GMWB rider.
        let withGMWB (gmwb: Riders.GMWB) (product: VAProduct) : VAProduct =
            { product with Riders = Riders.RiderPackage.withGMWB gmwb product.Riders }

        /// Add GLWB rider.
        let withGLWB (glwb: Riders.GLWB) (product: VAProduct) : VAProduct =
            { product with Riders = Riders.RiderPackage.withGLWB glwb product.Riders }

        /// Add GMDB rider.
        let withGMDB (gmdb: Riders.GMDB) (product: VAProduct) : VAProduct =
            { product with Riders = Riders.RiderPackage.withGMDB gmdb product.Riders }

        /// Total rider charges.
        let totalRiderCharge (product: VAProduct) : float32 =
            Riders.RiderPackage.totalAnnualCharge product.Riders

    // ══════════════════════════════════════════════════════════════════
    // Account Value Calculations
    // ══════════════════════════════════════════════════════════════════

    module AccountValue =
        /// Sub-account return based on allocation-weighted returns.
        let weightedReturn (returns: (float32 * Expr) list) : Expr =
            returns
            |> List.map (fun (weight, ret) -> weight.C * ret)
            |> List.reduce (+)

        /// Account value growth with sub-account returns and fees.
        let grow (accountValue: Expr) (subAccountReturn: Expr) (fees: Expr) (dt: Expr) : Expr =
            accountValue * Expr.exp ((subAccountReturn - fees) * dt)

        /// Accumulate AV with fees.
        let accumulateWithFees (return': Expr) (fees: Expr) (dt: Expr) (init: Expr) : ModelCtx -> Expr = fun ctx ->
            evolve init (fun av -> grow av return' fees dt) ctx


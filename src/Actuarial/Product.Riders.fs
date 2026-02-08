namespace Cavere.Actuarial.Product

open Cavere.Core

/// Guaranteed benefit riders shared across annuity products.
/// Includes GMAB, GMIB, GMWB, GLWB, and GMDB.
module Riders =

    // ══════════════════════════════════════════════════════════════════
    // Common Rider Types
    // ══════════════════════════════════════════════════════════════════

    /// How the benefit base grows before withdrawals/annuitization.
    type BenefitBaseGrowth =
        | NoGrowth // Stays at initial value
        | RollUp of annualRate: float32 // Compounds at fixed rate
        | HighWaterMark // Highest AV on anniversaries
        | RollUpWithHighWaterMark of annualRate: float32 // Greater of roll-up or HWM
        | PercentOfPremium of pct: float32 // Fixed percentage of premium

    /// Step-up provisions for benefit bases.
    type StepUpProvision =
        | NoStepUp
        | AnnualStepUp // Step up to AV annually if higher
        | TriennialStepUp // Step up every 3 years
        | QuinquennialStepUp // Step up every 5 years
        | AutomaticStepUp of years: int // Step up every N years

    /// Rider charge structure.
    type RiderCharge = {
        AnnualRate: float32 // Annual charge as % of benefit base
        ChargeBase: ChargeBase // What the charge is based on
        MaxCharge: float32 option // Maximum annual charge (if capped)
    }

    and ChargeBase =
        | AccountValue
        | BenefitBase
        | GreaterOfAVOrBB
        | Premium

    module RiderCharge =
        let onBenefitBase (rate: float32) : RiderCharge = {
            AnnualRate = rate
            ChargeBase = BenefitBase
            MaxCharge = None
        }

        let onAccountValue (rate: float32) : RiderCharge = {
            AnnualRate = rate
            ChargeBase = AccountValue
            MaxCharge = None
        }

        let onGreaterOf (rate: float32) : RiderCharge = {
            AnnualRate = rate
            ChargeBase = GreaterOfAVOrBB
            MaxCharge = None
        }

        let withMaxCharge (maxRate: float32) (charge: RiderCharge) : RiderCharge = {
            charge with
                MaxCharge = Some maxRate
        }

    // ══════════════════════════════════════════════════════════════════
    // GMAB — Guaranteed Minimum Accumulation Benefit
    // ══════════════════════════════════════════════════════════════════

    /// GMAB guarantees a minimum account value after a waiting period.
    type GMAB = {
        GuaranteePercent: float32 // e.g., 100% = return of premium
        WaitingPeriod: int // Years until guarantee applies
        BenefitBaseGrowth: BenefitBaseGrowth
        StepUp: StepUpProvision
        Charge: RiderCharge
    }

    module GMAB =
        /// Return of premium after waiting period.
        let returnOfPremium (waitingYears: int) (charge: RiderCharge) : GMAB = {
            GuaranteePercent = 1.0f
            WaitingPeriod = waitingYears
            BenefitBaseGrowth = NoGrowth
            StepUp = NoStepUp
            Charge = charge
        }

        /// GMAB with roll-up.
        let withRollUp (rate: float32) (waitingYears: int) (charge: RiderCharge) : GMAB = {
            GuaranteePercent = 1.0f
            WaitingPeriod = waitingYears
            BenefitBaseGrowth = RollUp rate
            StepUp = NoStepUp
            Charge = charge
        }

        /// Add step-up provision.
        let withStepUp (stepUp: StepUpProvision) (gmab: GMAB) : GMAB = { gmab with StepUp = stepUp }

        /// Calculate GMAB benefit base.
        let benefitBase
            (premium: Expr)
            (accountValue: Expr)
            (rollUpValue: Expr)
            (highWaterMark: Expr)
            (growth: BenefitBaseGrowth)
            : Expr =
            match growth with
            | NoGrowth -> premium
            | RollUp _ -> rollUpValue
            | HighWaterMark -> highWaterMark
            | RollUpWithHighWaterMark _ -> Expr.max rollUpValue highWaterMark
            | PercentOfPremium pct -> premium * pct.C

        /// GMAB payout: max(0, benefitBase - accountValue) if past waiting period.
        let payout (benefitBase: Expr) (accountValue: Expr) : Expr = Expr.max (benefitBase - accountValue) 0.0f.C

    // ══════════════════════════════════════════════════════════════════
    // GMIB — Guaranteed Minimum Income Benefit
    // ══════════════════════════════════════════════════════════════════

    /// GMIB guarantees a minimum annuitization value.
    type GMIB = {
        WaitingPeriod: int // Years before annuitization allowed
        BenefitBaseGrowth: BenefitBaseGrowth
        PayoutRate: float32 // Guaranteed payout rate at annuitization
        PayoutRateByAge: (int * float32) list option // Age-based payout rates
        StepUp: StepUpProvision
        Charge: RiderCharge
    }

    module GMIB =
        /// Standard GMIB with roll-up.
        let withRollUp (rollUpRate: float32) (waitingYears: int) (payoutRate: float32) (charge: RiderCharge) : GMIB = {
            WaitingPeriod = waitingYears
            BenefitBaseGrowth = RollUp rollUpRate
            PayoutRate = payoutRate
            PayoutRateByAge = None
            StepUp = AnnualStepUp
            Charge = charge
        }

        /// GMIB with age-based payout rates.
        let withAgeBasedPayouts (rates: (int * float32) list) (gmib: GMIB) : GMIB = {
            gmib with
                PayoutRateByAge = Some rates
        }

        /// Guaranteed annual income = benefitBase * payoutRate.
        let guaranteedIncome (benefitBase: Expr) (payoutRate: float32) : Expr = benefitBase * payoutRate.C

        /// GMIB value: present value of guaranteed income stream.
        /// Simplified: assumes level payments for life expectancy.
        let incomeValue (guaranteedIncome: Expr) (lifeExpectancy: Expr) (discountRate: Expr) : Expr =
            // PV of annuity: income * (1 - (1+r)^-n) / r
            let pvFactor = (1.0f.C - Expr.exp (-discountRate * lifeExpectancy)) / discountRate
            guaranteedIncome * pvFactor

    // ══════════════════════════════════════════════════════════════════
    // GMWB — Guaranteed Minimum Withdrawal Benefit
    // ══════════════════════════════════════════════════════════════════

    /// GMWB guarantees withdrawals until benefit base exhausted.
    type GMWB = {
        WithdrawalRate: float32 // Annual withdrawal rate (e.g., 5%)
        BenefitBaseGrowth: BenefitBaseGrowth
        WaitingPeriod: int // Years before withdrawals begin
        StepUp: StepUpProvision
        ExcessWithdrawalPenalty: float32 // Penalty for exceeding guaranteed amount
        Charge: RiderCharge
    }

    module GMWB =
        /// Standard GMWB with roll-up.
        let withRollUp
            (withdrawalRate: float32)
            (rollUpRate: float32)
            (rollUpPeriod: int)
            (charge: RiderCharge)
            : GMWB =
            {
                WithdrawalRate = withdrawalRate
                BenefitBaseGrowth = RollUp rollUpRate
                WaitingPeriod = rollUpPeriod
                StepUp = AnnualStepUp
                ExcessWithdrawalPenalty = 0.0f
                Charge = charge
            }

        /// GMWB with high water mark.
        let withHighWaterMark (withdrawalRate: float32) (charge: RiderCharge) : GMWB = {
            WithdrawalRate = withdrawalRate
            BenefitBaseGrowth = HighWaterMark
            WaitingPeriod = 0
            StepUp = AnnualStepUp
            ExcessWithdrawalPenalty = 0.0f
            Charge = charge
        }

        /// Set excess withdrawal penalty.
        let withExcessPenalty (penalty: float32) (gmwb: GMWB) : GMWB = {
            gmwb with
                ExcessWithdrawalPenalty = penalty
        }

        /// Guaranteed withdrawal amount.
        let guaranteedWithdrawal (benefitBase: Expr) (withdrawalRate: float32) : Expr = benefitBase * withdrawalRate.C

        /// Benefit base after withdrawal (reduced by withdrawal amount).
        let benefitBaseAfterWithdrawal (benefitBase: Expr) (withdrawal: Expr) : Expr =
            Expr.max (benefitBase - withdrawal) 0.0f.C

        /// Excess withdrawal (amount over guaranteed).
        let excessWithdrawal (actualWithdrawal: Expr) (guaranteedWithdrawal: Expr) : Expr =
            Expr.max (actualWithdrawal - guaranteedWithdrawal) 0.0f.C

        /// Benefit base reduction for excess withdrawal (typically dollar-for-dollar or pro-rata).
        let benefitBaseReductionForExcess (benefitBase: Expr) (accountValue: Expr) (excessWithdrawal: Expr) : Expr =
            // Pro-rata reduction: BB reduced by same % as AV
            let reductionRatio = excessWithdrawal / accountValue
            benefitBase * reductionRatio

    // ══════════════════════════════════════════════════════════════════
    // GLWB — Guaranteed Lifetime Withdrawal Benefit
    // ══════════════════════════════════════════════════════════════════

    /// GLWB guarantees withdrawals for life (even if AV exhausted).
    type GLWB = {
        WithdrawalRatesByAge: (int * float32) list // Age-based withdrawal rates
        SingleLifeRate: float32 // Default single life rate
        JointLifeRate: float32 // Rate for joint policies
        BenefitBaseGrowth: BenefitBaseGrowth
        WaitingPeriod: int
        StepUp: StepUpProvision
        BonusProvisions: BonusProvision option
        Charge: RiderCharge
    }

    and BonusProvision =
        | DeferralBonus of ratePerYear: float32 * maxYears: int
        | NoWithdrawalBonus of bonusPercent: float32

    module GLWB =
        /// Standard GLWB with age-based rates.
        let create
            (singleRate: float32)
            (jointRate: float32)
            (rollUpRate: float32)
            (waitingYears: int)
            (charge: RiderCharge)
            : GLWB =
            {
                WithdrawalRatesByAge = []
                SingleLifeRate = singleRate
                JointLifeRate = jointRate
                BenefitBaseGrowth = RollUp rollUpRate
                WaitingPeriod = waitingYears
                StepUp = AnnualStepUp
                BonusProvisions = None
                Charge = charge
            }

        /// Add age-based withdrawal rates.
        let withAgeBasedRates (rates: (int * float32) list) (glwb: GLWB) : GLWB = {
            glwb with
                WithdrawalRatesByAge = rates
        }

        /// Add deferral bonus.
        let withDeferralBonus (ratePerYear: float32) (maxYears: int) (glwb: GLWB) : GLWB = {
            glwb with
                BonusProvisions = Some(DeferralBonus(ratePerYear, maxYears))
        }

        /// Lifetime withdrawal amount (continues even if AV = 0).
        let lifetimeWithdrawal (benefitBase: Expr) (withdrawalRate: float32) : Expr = benefitBase * withdrawalRate.C

        /// Check if GLWB is "in the money" (AV exhausted but withdrawals continue).
        let isInTheMoney (accountValue: Expr) (benefitBase: Expr) : Expr =
            (accountValue .< 0.01f) * (benefitBase .> 0.0f)

    // ══════════════════════════════════════════════════════════════════
    // GMDB — Guaranteed Minimum Death Benefit
    // ══════════════════════════════════════════════════════════════════

    /// GMDB guarantees a minimum death benefit.
    type GMDB = {
        GuaranteeType: GMDBType
        StepUp: StepUpProvision
        MaxAge: int option // Age at which guarantee expires
        Charge: RiderCharge
    }

    and GMDBType =
        | ReturnOfPremium // max(AV, premium)
        | RollUp of annualRate: float32 // max(AV, premium compounded)
        | HighestAnniversaryValue // max(AV, highest anniversary AV)
        | RatchetAndRollUp of annualRate: float32 // Greater of HWM or roll-up
        | EstatePlus of percent: float32 // AV + percent of gains

    module GMDB =
        /// Return of premium GMDB.
        let returnOfPremium (charge: RiderCharge) : GMDB = {
            GuaranteeType = ReturnOfPremium
            StepUp = NoStepUp
            MaxAge = None
            Charge = charge
        }

        /// GMDB with annual roll-up.
        let withRollUp (rate: float32) (charge: RiderCharge) : GMDB = {
            GuaranteeType = RollUp rate
            StepUp = NoStepUp
            MaxAge = None
            Charge = charge
        }

        /// Highest anniversary value GMDB.
        let highestAnniversaryValue (charge: RiderCharge) : GMDB = {
            GuaranteeType = HighestAnniversaryValue
            StepUp = AnnualStepUp
            MaxAge = None
            Charge = charge
        }

        /// GMDB with ratchet and roll-up.
        let ratchetAndRollUp (rate: float32) (charge: RiderCharge) : GMDB = {
            GuaranteeType = RatchetAndRollUp rate
            StepUp = AnnualStepUp
            MaxAge = None
            Charge = charge
        }

        /// Set maximum age for guarantee.
        let withMaxAge (age: int) (gmdb: GMDB) : GMDB = { gmdb with MaxAge = Some age }

        /// Calculate GMDB benefit.
        let deathBenefit
            (accountValue: Expr)
            (premium: Expr)
            (rollUpValue: Expr)
            (highWaterMark: Expr)
            (gmdbType: GMDBType)
            : Expr =
            match gmdbType with
            | ReturnOfPremium -> Expr.max accountValue premium
            | RollUp _ -> Expr.max accountValue rollUpValue
            | HighestAnniversaryValue -> Expr.max accountValue highWaterMark
            | RatchetAndRollUp _ -> Expr.max accountValue (Expr.max rollUpValue highWaterMark)
            | EstatePlus pct ->
                let gains = Expr.max (accountValue - premium) 0.0f.C
                accountValue + gains * pct.C

        /// Net amount at risk for GMDB.
        let netAmountAtRisk (deathBenefit: Expr) (accountValue: Expr) : Expr =
            Expr.max (deathBenefit - accountValue) 0.0f.C

    // ══════════════════════════════════════════════════════════════════
    // Benefit Base Accumulators
    // ══════════════════════════════════════════════════════════════════

    module BenefitBase =
        /// Roll-up benefit base accumulator.
        let rollUp (rate: float32) (dt: Expr) (init: Expr) : ModelCtx -> Expr =
            fun ctx -> evolve init (fun bb -> bb * Expr.exp (rate.C * dt)) ctx

        /// High water mark accumulator (tracks highest AV on anniversaries).
        let highWaterMark (accountValue: Expr) (init: Expr) : ModelCtx -> Expr =
            fun ctx -> evolve init (fun hwm -> Expr.max hwm accountValue) ctx

        /// Combined roll-up with high water mark.
        let rollUpWithHWM (rate: float32) (dt: Expr) (accountValue: Expr) (init: Expr) : ModelCtx -> Expr =
            fun ctx ->
                let rollUpBB = rollUp rate dt init ctx
                let hwmBB = highWaterMark accountValue init ctx
                Expr.max rollUpBB hwmBB

        /// Apply step-up to benefit base.
        let applyStepUp (benefitBase: Expr) (accountValue: Expr) (isAnniversary: Expr) : Expr =
            Expr.select isAnniversary (Expr.max benefitBase accountValue) benefitBase

    // ══════════════════════════════════════════════════════════════════
    // Rider Charge Calculations
    // ══════════════════════════════════════════════════════════════════

    module Charge =
        /// Calculate rider charge base.
        let chargeBase (accountValue: Expr) (benefitBase: Expr) (premium: Expr) (base': ChargeBase) : Expr =
            match base' with
            | AccountValue -> accountValue
            | BenefitBase -> benefitBase
            | GreaterOfAVOrBB -> Expr.max accountValue benefitBase
            | Premium -> premium

        /// Calculate rider charge amount.
        let chargeAmount (chargeBase: Expr) (charge: RiderCharge) (dt: Expr) : Expr =
            let annualCharge = chargeBase * charge.AnnualRate.C
            let periodCharge = annualCharge * dt

            match charge.MaxCharge with
            | Some maxRate -> Expr.min periodCharge (chargeBase * maxRate.C * dt)
            | None -> periodCharge

        /// Apply rider charge to account value.
        let applyCharge (accountValue: Expr) (chargeAmount: Expr) : Expr = Expr.max (accountValue - chargeAmount) 0.0f.C

    // ══════════════════════════════════════════════════════════════════
    // Combined Rider Package
    // ══════════════════════════════════════════════════════════════════

    /// A package of riders attached to a policy.
    type RiderPackage = {
        GMAB: GMAB option
        GMIB: GMIB option
        GMWB: GMWB option
        GLWB: GLWB option
        GMDB: GMDB option
    }

    module RiderPackage =
        let empty: RiderPackage = {
            GMAB = None
            GMIB = None
            GMWB = None
            GLWB = None
            GMDB = None
        }

        let withGMAB (gmab: GMAB) (pkg: RiderPackage) : RiderPackage = { pkg with GMAB = Some gmab }

        let withGMIB (gmib: GMIB) (pkg: RiderPackage) : RiderPackage = { pkg with GMIB = Some gmib }

        let withGMWB (gmwb: GMWB) (pkg: RiderPackage) : RiderPackage = { pkg with GMWB = Some gmwb }

        let withGLWB (glwb: GLWB) (pkg: RiderPackage) : RiderPackage = { pkg with GLWB = Some glwb }

        let withGMDB (gmdb: GMDB) (pkg: RiderPackage) : RiderPackage = { pkg with GMDB = Some gmdb }

        /// Total annual rider charges.
        let totalAnnualCharge (pkg: RiderPackage) : float32 =
            let getRate =
                function
                | Some r -> r.Charge.AnnualRate
                | None -> 0.0f

            [
                pkg.GMAB |> Option.map (fun r -> r.Charge.AnnualRate)
                pkg.GMIB |> Option.map (fun r -> r.Charge.AnnualRate)
                pkg.GMWB |> Option.map (fun r -> r.Charge.AnnualRate)
                pkg.GLWB |> Option.map (fun r -> r.Charge.AnnualRate)
                pkg.GMDB |> Option.map (fun r -> r.Charge.AnnualRate)
            ]
            |> List.choose id
            |> List.sum

namespace Cavere.Actuarial.Product

open Cavere.Core

/// Life insurance product definitions.
module Life =

    // ══════════════════════════════════════════════════════════════════
    // Life Product Types
    // ══════════════════════════════════════════════════════════════════

    type LifeProductType =
        | Term of years: int
        | WholeLife
        | UniversalLife
        | VariableUniversalLife
        | IndexedUniversalLife

    // ══════════════════════════════════════════════════════════════════
    // Death Benefit Options
    // ══════════════════════════════════════════════════════════════════

    type DeathBenefitOption =
        | LevelDeathBenefit                 // Option A: fixed face amount
        | FaceAmountPlusCashValue           // Option B: face + AV
        | ReturnOfPremium                   // Face or total premiums, whichever greater

    module DeathBenefit =
        /// Calculate death benefit based on option.
        let calculate (option: DeathBenefitOption) (faceAmount: Expr) (accountValue: Expr) (totalPremiums: Expr) : Expr =
            match option with
            | LevelDeathBenefit -> faceAmount
            | FaceAmountPlusCashValue -> faceAmount + accountValue
            | ReturnOfPremium -> Expr.max faceAmount totalPremiums

        /// Apply corridor test (DB must be at least AV * corridor factor).
        let applyCorridorTest (deathBenefit: Expr) (accountValue: Expr) (corridorFactor: Expr) : Expr =
            Expr.max deathBenefit (accountValue * corridorFactor)

    // ══════════════════════════════════════════════════════════════════
    // UL Crediting Strategies
    // ══════════════════════════════════════════════════════════════════

    type ULCreditingStrategy =
        | FixedRate of guaranteedRate: float32 * currentRate: float32
        | IndexedCrediting of IndexedCreditingStrategy

    and IndexedCreditingStrategy =
        | PointToPoint of cap: float32 * floor: float32 * participation: float32
        | MonthlyCap of cap: float32 * floor: float32
        | MonthlyAverage of cap: float32 * participation: float32

    // ══════════════════════════════════════════════════════════════════
    // Life Product Definition
    // ══════════════════════════════════════════════════════════════════

    type LifeProduct = {
        Name: string
        ProductType: LifeProductType
        FaceAmount: float32
        DeathBenefitOption: DeathBenefitOption
        CreditingStrategy: ULCreditingStrategy option     // None for term/whole life
        COICharges: float32[]                             // Cost of insurance by age
        Fees: Common.FeeStructure
        CDSCSchedule: Common.CDSCSchedule option
    }

    module LifeProduct =
        /// Create a term life product.
        let term (name: string) (faceAmount: float32) (years: int) (coiByAge: float32[]) : LifeProduct =
            {
                Name = name
                ProductType = Term years
                FaceAmount = faceAmount
                DeathBenefitOption = LevelDeathBenefit
                CreditingStrategy = None
                COICharges = coiByAge
                Fees = Common.Fees.none
                CDSCSchedule = None
            }

        /// Create a universal life product.
        let universalLife (name: string) (faceAmount: float32) (crediting: ULCreditingStrategy)
                         (coiByAge: float32[]) (fees: Common.FeeStructure) : LifeProduct =
            {
                Name = name
                ProductType = UniversalLife
                FaceAmount = faceAmount
                DeathBenefitOption = LevelDeathBenefit
                CreditingStrategy = Some crediting
                COICharges = coiByAge
                Fees = fees
                CDSCSchedule = None
            }

        /// Create an indexed universal life product.
        let indexedUL (name: string) (faceAmount: float32) (indexStrategy: IndexedCreditingStrategy)
                      (coiByAge: float32[]) (fees: Common.FeeStructure) : LifeProduct =
            {
                Name = name
                ProductType = IndexedUniversalLife
                FaceAmount = faceAmount
                DeathBenefitOption = LevelDeathBenefit
                CreditingStrategy = Some (IndexedCrediting indexStrategy)
                COICharges = coiByAge
                Fees = fees
                CDSCSchedule = None
            }

    // ══════════════════════════════════════════════════════════════════
    // COI — Cost of Insurance
    // ══════════════════════════════════════════════════════════════════

    module COI =
        /// Load COI table (by attained age) onto GPU.
        let loadCOITable (coiByAge: float32[]) : ModelCtx -> int = fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(coiByAge, coiByAge.Length))
            id

        /// COI rate lookup by attained age.
        let coiRate (surfaceId: int) (attainedAge: Expr) : ModelCtx -> Expr = fun ctx ->
            interp1d surfaceId attainedAge ctx

        /// COI charge = (DB - AV) * COI rate / 12 (monthly)
        let coiCharge (netAmountAtRisk: Expr) (coiRate: Expr) : Expr =
            netAmountAtRisk * coiRate / 12.0f

        /// Net amount at risk = Death Benefit - Account Value (floored at 0)
        let netAmountAtRisk (deathBenefit: Expr) (accountValue: Expr) : Expr =
            Expr.max (deathBenefit - accountValue) 0.0f.C

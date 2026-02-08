namespace Cavere.Actuarial.Decrements

open Cavere.Core

/// Mortality decrement modeling for life insurance and annuity products.
/// Mortality rates (qx) are indexed by attained age.
/// Supports single life and joint life (first-to-die, last-to-die) policies.
module Mortality =

    // ══════════════════════════════════════════════════════════════════
    // Joint Mortality Types
    // ══════════════════════════════════════════════════════════════════

    /// Type of joint mortality calculation.
    type JointMortalityType =
        | FirstToDie // Benefit paid on first death (both must survive)
        | LastToDie // Benefit paid on last death (at least one survives)

    // ══════════════════════════════════════════════════════════════════
    // Age Calculation
    // ══════════════════════════════════════════════════════════════════

    /// Attained age at current time step.
    /// For annual steps: issueAge + TimeIndex
    let attainedAge (issueAge: Expr) : Expr = issueAge + TimeIndex

    /// Attained age with fractional years using year fraction.
    let attainedAgeFractional (issueAge: Expr) (yearFrac: Expr) : Expr = issueAge + yearFrac

    // ══════════════════════════════════════════════════════════════════
    // Mortality Table Loading
    // ══════════════════════════════════════════════════════════════════

    /// Load a mortality table (qx by age) onto the GPU.
    /// Table should be indexed by integer age (0, 1, 2, ..., maxAge).
    /// Returns a surface ID for use with qxByAge.
    let loadMortalityTable (qxByAge: float32[]) : ModelCtx -> int =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(qxByAge, qxByAge.Length))
            id

    // ══════════════════════════════════════════════════════════════════
    // Single Life Mortality Rate Lookup
    // ══════════════════════════════════════════════════════════════════

    /// Mortality rate lookup by attained age.
    /// Uses interpolation for fractional ages.
    let qx (surfaceId: int) (attainedAge: Expr) : ModelCtx -> Expr = fun ctx -> interp1d surfaceId attainedAge ctx

    /// Mortality rate lookup with issue age as parameter.
    /// Computes attained age internally: issueAge + TimeIndex.
    let qxFromIssueAge (surfaceId: int) (issueAge: Expr) : ModelCtx -> Expr =
        fun ctx -> interp1d surfaceId (issueAge + TimeIndex) ctx

    /// Constant mortality rate (for simplified models).
    let qxConstant (rate: float32) : Expr = rate.C

    // ══════════════════════════════════════════════════════════════════
    // Joint Life Mortality
    // ══════════════════════════════════════════════════════════════════

    /// Joint mortality rate for first-to-die.
    /// qxy = qx + qy - qx * qy (probability at least one dies)
    let qxFirstToDie (qxOwner: Expr) (qxSpouse: Expr) : Expr = qxOwner + qxSpouse - qxOwner * qxSpouse

    /// Joint mortality rate for last-to-die.
    /// qxy = qx * qy (probability both die in the period)
    let qxLastToDie (qxOwner: Expr) (qxSpouse: Expr) : Expr = qxOwner * qxSpouse

    /// Joint mortality rate based on type.
    let qxJoint (jointType: JointMortalityType) (qxOwner: Expr) (qxSpouse: Expr) : Expr =
        match jointType with
        | FirstToDie -> qxFirstToDie qxOwner qxSpouse
        | LastToDie -> qxLastToDie qxOwner qxSpouse

    /// Joint mortality rate lookup from issue ages.
    /// Uses same mortality table for both lives (can use different tables by calling qxFromIssueAge separately).
    let qxJointFromIssueAges
        (surfaceId: int)
        (jointType: JointMortalityType)
        (ownerIssueAge: Expr)
        (spouseIssueAge: Expr)
        : ModelCtx -> Expr =
        fun ctx ->
            let qxOwner = qxFromIssueAge surfaceId ownerIssueAge ctx
            let qxSpouse = qxFromIssueAge surfaceId spouseIssueAge ctx
            qxJoint jointType qxOwner qxSpouse

    /// Joint mortality with separate tables for owner and spouse.
    let qxJointSeparateTables
        (ownerSurfaceId: int)
        (spouseSurfaceId: int)
        (jointType: JointMortalityType)
        (ownerIssueAge: Expr)
        (spouseIssueAge: Expr)
        : ModelCtx -> Expr =
        fun ctx ->
            let qxOwner = qxFromIssueAge ownerSurfaceId ownerIssueAge ctx
            let qxSpouse = qxFromIssueAge spouseSurfaceId spouseIssueAge ctx
            qxJoint jointType qxOwner qxSpouse

    // ══════════════════════════════════════════════════════════════════
    // Single Life Survival Probability
    // ══════════════════════════════════════════════════════════════════

    /// Cumulative probability of surviving from issue to current time.
    /// qx should be the mortality rate at current attained age.
    let survivalProb (qx: Expr) : ModelCtx -> Expr = fun ctx -> survivalProbAnnual qx ctx

    /// Cumulative survival with sub-annual time steps.
    /// Adjusts annual qx to per-period rate.
    let survivalProbSubannual (annualQx: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            // Convert annual to period: 1 - (1-qx)^dt ≈ qx * dt for small qx
            let periodRate = 1.0f.C - Expr.exp (Expr.log (1.0f.C - annualQx) * dt)
            Common.survivalProb periodRate 1.0f.C ctx

    /// Cumulative survival using force of mortality (continuous model).
    let survivalProbContinuous (qx: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let mu = -Expr.log(1.0f.C - qx) // force of mortality from annual qx
            Common.survivalProbContinuous mu dt ctx

    // ══════════════════════════════════════════════════════════════════
    // Joint Life Survival Probability
    // ══════════════════════════════════════════════════════════════════

    /// Joint survival probability for first-to-die.
    /// pxy = px * py (both must survive for policy to remain in force)
    let survivalProbFirstToDie (pxOwner: Expr) (pxSpouse: Expr) : Expr = pxOwner * pxSpouse

    /// Joint survival probability for last-to-die.
    /// pxy = px + py - px * py (at least one must survive)
    let survivalProbLastToDie (pxOwner: Expr) (pxSpouse: Expr) : Expr = pxOwner + pxSpouse - pxOwner * pxSpouse

    /// Cumulative joint survival using mortality rates.
    /// Tracks survival of both lives separately, then combines.
    let survivalProbJoint (jointType: JointMortalityType) (qxOwner: Expr) (qxSpouse: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let pxOwner = survivalProbAnnual qxOwner ctx
            let pxSpouse = survivalProbAnnual qxSpouse ctx

            match jointType with
            | FirstToDie -> survivalProbFirstToDie pxOwner pxSpouse
            | LastToDie -> survivalProbLastToDie pxOwner pxSpouse

    /// Cumulative joint survival from issue ages using same table.
    let survivalProbJointFromIssueAges
        (surfaceId: int)
        (jointType: JointMortalityType)
        (ownerIssueAge: Expr)
        (spouseIssueAge: Expr)
        : ModelCtx -> Expr =
        fun ctx ->
            let qxOwner = qxFromIssueAge surfaceId ownerIssueAge ctx
            let qxSpouse = qxFromIssueAge surfaceId spouseIssueAge ctx
            survivalProbJoint jointType qxOwner qxSpouse ctx

    // ══════════════════════════════════════════════════════════════════
    // Death Benefit Helpers
    // ══════════════════════════════════════════════════════════════════

    /// Expected death benefit: DB * qx * survival to start of period * discount.
    let expectedDeathBenefit (deathBenefit: Expr) (qx: Expr) (survivalProb: Expr) (df: Expr) : Expr =
        deathBenefit * qx * survivalProb * df

    /// Cumulative expected death claims over time.
    let cumulativeDeathClaims (deathBenefit: Expr) (qx: Expr) (survivalProb: Expr) (df: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let periodClaim = expectedDeathBenefit deathBenefit qx survivalProb df
            evolve 0.0f.C (fun total -> total + periodClaim) ctx

    /// Expected death benefit for joint policy.
    let expectedDeathBenefitJoint (deathBenefit: Expr) (qxJoint: Expr) (survivalProbJoint: Expr) (df: Expr) : Expr =
        deathBenefit * qxJoint * survivalProbJoint * df

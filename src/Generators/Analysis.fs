namespace Cavere.Generators

open System
open Cavere.Core

/// Greeks (sensitivities) for a financial model.
type Greeks = {
    Delta: float32
    Gamma: float32
    Vega: float32
    Theta: float32
    Rho: float32
}

/// Known closed-form solutions.
type ClosedFormSolution =
    | BlackScholesCall of spot: float32 * strike: float32 * rate: float32 * vol: float32 * time: float32
    | BlackScholesPut of spot: float32 * strike: float32 * rate: float32 * vol: float32 * time: float32
    | Forward of spot: float32 * rate: float32 * time: float32
    | ZeroCouponBond of rate: float32 * time: float32

/// Result of analyzing a model's structure.
type ModelAnalysis =
    | ClosedForm of solution: ClosedFormSolution
    | RequiresMC of reason: string

/// Closed-form detection and analytical evaluation.
[<RequireQualifiedAccess>]
module Analysis =

    // ══════════════════════════════════════════════════════════════════
    // Normal CDF Approximation (Abramowitz & Stegun)
    // ══════════════════════════════════════════════════════════════════

    let normalCdf (x: float32) : float32 =
        let a1 = 0.254829592f
        let a2 = -0.284496736f
        let a3 = 1.421413741f
        let a4 = -1.453152027f
        let a5 = 1.061405429f
        let p = 0.3275911f
        let sign = if x < 0.0f then -1.0f else 1.0f
        let x' = abs x / MathF.Sqrt(2.0f)
        let t = 1.0f / (1.0f + p * x')

        let y =
            1.0f
            - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x' * x')

        0.5f * (1.0f + sign * y)

    let normalPdf (x: float32) : float32 = MathF.Exp(-0.5f * x * x) / MathF.Sqrt(2.0f * MathF.PI)

    // ══════════════════════════════════════════════════════════════════
    // Closed-Form Evaluation
    // ══════════════════════════════════════════════════════════════════

    /// Evaluate a closed-form solution to get its price.
    let evaluate (solution: ClosedFormSolution) : float32 =
        match solution with
        | BlackScholesCall(s, k, r, sigma, t) ->
            let d1 = (MathF.Log(s / k) + (r + 0.5f * sigma * sigma) * t) / (sigma * MathF.Sqrt(t))
            let d2 = d1 - sigma * MathF.Sqrt(t)
            s * normalCdf d1 - k * MathF.Exp(-r * t) * normalCdf d2

        | BlackScholesPut(s, k, r, sigma, t) ->
            let d1 = (MathF.Log(s / k) + (r + 0.5f * sigma * sigma) * t) / (sigma * MathF.Sqrt(t))
            let d2 = d1 - sigma * MathF.Sqrt(t)
            k * MathF.Exp(-r * t) * normalCdf (-d2) - s * normalCdf (-d1)

        | Forward(s, _, _) -> s

        | ZeroCouponBond(r, t) -> MathF.Exp(-r * t)

    /// Evaluate closed-form Greeks.
    let evaluateGreeks (solution: ClosedFormSolution) : Greeks =
        match solution with
        | BlackScholesCall(s, k, r, sigma, t) ->
            let d1 = (MathF.Log(s / k) + (r + 0.5f * sigma * sigma) * t) / (sigma * MathF.Sqrt(t))
            let d2 = d1 - sigma * MathF.Sqrt(t)

            {
                Delta = normalCdf d1
                Gamma = normalPdf d1 / (s * sigma * MathF.Sqrt(t))
                Vega = s * normalPdf d1 * MathF.Sqrt(t)
                Theta =
                    -(s * normalPdf d1 * sigma) / (2.0f * MathF.Sqrt(t))
                    - r * k * MathF.Exp(-r * t) * normalCdf d2
                Rho = k * t * MathF.Exp(-r * t) * normalCdf d2
            }

        | BlackScholesPut(s, k, r, sigma, t) ->
            let d1 = (MathF.Log(s / k) + (r + 0.5f * sigma * sigma) * t) / (sigma * MathF.Sqrt(t))
            let d2 = d1 - sigma * MathF.Sqrt(t)

            {
                Delta = normalCdf d1 - 1.0f
                Gamma = normalPdf d1 / (s * sigma * MathF.Sqrt(t))
                Vega = s * normalPdf d1 * MathF.Sqrt(t)
                Theta =
                    -(s * normalPdf d1 * sigma) / (2.0f * MathF.Sqrt(t))
                    + r * k * MathF.Exp(-r * t) * normalCdf (-d2)
                Rho = -k * t * MathF.Exp(-r * t) * normalCdf (-d2)
            }

        | Forward(s, r, t) -> {
            Delta = 1.0f
            Gamma = 0.0f
            Vega = 0.0f
            Theta = -r * s
            Rho = t * s
          }

        | ZeroCouponBond(r, t) ->
            {
                Delta = 0.0f
                Gamma = 0.0f
                Vega = 0.0f
                Theta = r * MathF.Exp(-r * t)
                Rho = -t * MathF.Exp(-r * t)
            }

    // ══════════════════════════════════════════════════════════════════
    // Accumulator Pattern Detection
    // ══════════════════════════════════════════════════════════════════

    /// Check if an accumulator follows GBM dynamics:
    /// body = self * exp(drift + vol * sqrt(dt) * z)
    let isGBMAccumulator (id: int) (def: AccumDef) : bool =
        match def.Body with
        // s * exp(something)
        | Mul(AccumRef aid, Exp _) when aid = id -> true
        | _ -> false

    /// Check if an accumulator is a discount factor:
    /// body = self * exp(-rate * dt) or self * exp(neg(rate * dt))
    let isDiscountAccumulator (id: int) (def: AccumDef) : bool =
        match def.Body with
        | Mul(AccumRef aid, Exp(Neg(Mul _))) when aid = id -> true
        | Mul(AccumRef aid, Exp(Mul(Neg _, _))) when aid = id -> true
        | Mul(AccumRef aid, Exp(Mul(_, Neg _))) when aid = id -> true
        | _ -> false

    /// Check if an accumulator computes running statistics.
    let isRunningStatistic (id: int) (def: AccumDef) : bool =
        match def.Body with
        | Max(AccumRef aid, _) when aid = id -> true
        | Min(AccumRef aid, _) when aid = id -> true
        | Add(AccumRef aid, _) when aid = id -> true
        | _ -> false

    /// Try to extract the constant initial value from an accumulator.
    let extractInitialConst (def: AccumDef) : float32 option =
        match def.Init with
        | Const v -> Some v
        | _ -> None

    /// Try to extract the constant rate from a discount factor body:
    /// self * exp(-rate * dt) → rate
    let extractDiscountRate (def: AccumDef) : float32 option =
        match def.Body with
        | Mul(AccumRef _, Exp(Neg(Mul(Const r, _)))) -> Some r
        | Mul(AccumRef _, Exp(Mul(Neg(Const r), _))) -> Some r
        | Mul(AccumRef _, Exp(Mul(Const r, _))) when r < 0.0f -> Some(-r)
        | _ -> None

    /// Try to extract vol and dt from GBM body:
    /// self * exp((rate - 0.5*vol^2)*dt + vol*sqrt(dt)*z)
    /// This is a simplified pattern matcher for common forms.
    let extractGBMVol (def: AccumDef) : float32 option =
        // GBM body: self * exp(drift + vol * sqrt(dt) * z)
        // where drift = (r - 0.5*vol^2)*dt
        // We look for the vol * sqrt(dt) * z pattern
        let rec findVolFromAdd expr =
            match expr with
            | Add(_, Mul(Mul(Const v, Sqrt _), Normal _)) -> Some v
            | Add(_, Mul(Const v, Mul(Sqrt _, Normal _))) -> Some v
            | Add(_, Mul(Mul(Mul(Const v, Sqrt _), Normal _), _)) -> Some v
            | _ -> None

        match def.Body with
        | Mul(AccumRef _, Exp inner) -> findVolFromAdd inner
        | _ -> None

    /// Extract time step dt from a GBM accumulator body.
    let extractDt (def: AccumDef) : float32 option =
        let rec findDt expr =
            match expr with
            | Sqrt(Const dt) -> Some dt
            | Add(_, b) -> findDt b
            | Mul(a, b) ->
                match findDt a with
                | Some dt -> Some dt
                | None -> findDt b
            | _ -> None

        match def.Body with
        | Mul(AccumRef _, Exp inner) -> findDt inner
        | _ -> None

    // ══════════════════════════════════════════════════════════════════
    // Path Dependence Detection
    // ══════════════════════════════════════════════════════════════════

    /// Detect if a model requires MC (is path-dependent).
    let detectPathDependence (model: Model) : bool =
        let usesTimeIdx = Symbolic.containsTimeIndex model.Result
        let usesTimeLookup = Symbolic.containsLookup1D model.Result
        let hasRunningStats = model.Accums |> Map.exists (fun id def -> isRunningStatistic id def)
        let hasObservations = not model.Observers.IsEmpty
        usesTimeIdx || usesTimeLookup || hasRunningStats || hasObservations

    // ══════════════════════════════════════════════════════════════════
    // Closed-Form Pattern Matching
    // ══════════════════════════════════════════════════════════════════

    /// Try to match a model against known closed-form patterns.
    let matchClosedFormPattern (model: Model) : ClosedFormSolution option =
        let normalized = Symbolic.fullySimplify model.Result

        // Helper to get accumulator by ID
        let getAccum id = model.Accums |> Map.tryFind id

        match normalized with
        // ── Pattern 1: European Call — max(S - K, 0) * df ──
        | Mul(Max(Sub(AccumRef sId, Const k), Const 0.0f), AccumRef dfId) ->
            match getAccum sId, getAccum dfId with
            | Some stockDef, Some dfDef when isGBMAccumulator sId stockDef && isDiscountAccumulator dfId dfDef ->
                match
                    extractInitialConst stockDef, extractDiscountRate dfDef, extractGBMVol stockDef, extractDt stockDef
                with
                | Some spot, Some rate, Some vol, Some dt ->
                    let time = float32 model.Accums.Count * dt * float32 (max 1 (model.Accums.Count - 1))
                    // Use steps from surface data or estimate from dt
                    let steps =
                        model.Surfaces
                        |> Map.tryPick (fun _ s ->
                            match s with
                            | Curve1D(_, steps) -> Some steps
                            | Grid2D(_, _, _, steps) -> Some steps)
                        |> Option.defaultValue (int (1.0f / dt))

                    let time = float32 steps * dt
                    Some(BlackScholesCall(spot, k, rate, vol, time))
                | _ -> None
            | _ -> None

        // ── Pattern 2: European Put — max(K - S, 0) * df ──
        | Mul(Max(Sub(Const k, AccumRef sId), Const 0.0f), AccumRef dfId) ->
            match getAccum sId, getAccum dfId with
            | Some stockDef, Some dfDef when isGBMAccumulator sId stockDef && isDiscountAccumulator dfId dfDef ->
                match
                    extractInitialConst stockDef, extractDiscountRate dfDef, extractGBMVol stockDef, extractDt stockDef
                with
                | Some spot, Some rate, Some vol, Some dt ->
                    let steps =
                        model.Surfaces
                        |> Map.tryPick (fun _ s ->
                            match s with
                            | Curve1D(_, steps) -> Some steps
                            | Grid2D(_, _, _, steps) -> Some steps)
                        |> Option.defaultValue (int (1.0f / dt))

                    let time = float32 steps * dt
                    Some(BlackScholesPut(spot, k, rate, vol, time))
                | _ -> None
            | _ -> None

        // ── Pattern 3: Forward — S * df ──
        | Mul(AccumRef sId, AccumRef dfId) ->
            match getAccum sId, getAccum dfId with
            | Some stockDef, Some dfDef when isGBMAccumulator sId stockDef && isDiscountAccumulator dfId dfDef ->
                match extractInitialConst stockDef, extractDiscountRate dfDef, extractDt stockDef with
                | Some spot, Some rate, Some dt ->
                    let steps =
                        model.Surfaces
                        |> Map.tryPick (fun _ s ->
                            match s with
                            | Curve1D(_, steps) -> Some steps
                            | Grid2D(_, _, _, steps) -> Some steps)
                        |> Option.defaultValue (int (1.0f / dt))

                    let time = float32 steps * dt
                    Some(Forward(spot, rate, time))
                | _ -> None
            | _ -> None

        // ── Pattern 4: Zero-coupon bond — df only ──
        | AccumRef dfId ->
            match getAccum dfId with
            | Some dfDef when isDiscountAccumulator dfId dfDef ->
                match extractDiscountRate dfDef, extractDt dfDef with
                | Some rate, Some dt ->
                    let steps =
                        model.Surfaces
                        |> Map.tryPick (fun _ s ->
                            match s with
                            | Curve1D(_, steps) -> Some steps
                            | Grid2D(_, _, _, steps) -> Some steps)
                        |> Option.defaultValue (int (1.0f / dt))

                    let time = float32 steps * dt
                    Some(ZeroCouponBond(rate, time))
                | _ -> None
            | _ -> None

        | _ -> None

    // ══════════════════════════════════════════════════════════════════
    // Model Analysis — Top-Level Entry Point
    // ══════════════════════════════════════════════════════════════════

    /// Analyze a model and determine the best solution method.
    let analyzeModel (model: Model) : ModelAnalysis =
        let pathDependent = detectPathDependence model

        if not pathDependent then
            match matchClosedFormPattern model with
            | Some solution -> ClosedForm solution
            | None -> RequiresMC "No closed-form pattern matched"
        else
            RequiresMC "Path-dependent model"

    // ══════════════════════════════════════════════════════════════════
    // GBM Moment Formulas
    // ══════════════════════════════════════════════════════════════════

    module GBMMoments =
        /// E[S_T] = S_0 * exp(r * T)
        let mean (spot: float32) (rate: float32) (time: float32) : float32 = spot * MathF.Exp(rate * time)

        /// Var[S_T] = S_0^2 * exp(2rT) * (exp(sigma^2 * T) - 1)
        let variance (spot: float32) (rate: float32) (vol: float32) (time: float32) : float32 =
            let s2 = spot * spot
            let exp2rt = MathF.Exp(2.0f * rate * time)
            let expSigma2t = MathF.Exp(vol * vol * time)
            s2 * exp2rt * (expSigma2t - 1.0f)

        /// E[S_T^2] = S_0^2 * exp((2r + sigma^2) * T)
        let secondMoment (spot: float32) (rate: float32) (vol: float32) (time: float32) : float32 =
            spot * spot * MathF.Exp((2.0f * rate + vol * vol) * time)

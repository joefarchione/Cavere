namespace Cavere.Core

/// Forward-mode automatic differentiation via expression transformation.
/// Transforms a Model containing DiffVar nodes into an expanded Model
/// where each accumulator tracks derivatives alongside values.
[<RequireQualifiedAccess>]
module CompilerDiff =

    // ══════════════════════════════════════════════════════════════════
    // DiffVar Collection
    // ══════════════════════════════════════════════════════════════════

    /// Collect all DiffVar indices from an expression tree.
    let rec collectDiffVars (expr: Expr) : Set<int> =
        match expr with
        | DiffVar(idx, _) -> Set.singleton idx
        | Const _ | TimeIndex | Normal _ | Uniform _ | AccumRef _ | Lookup1D _ | BatchRef _ -> Set.empty
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> collectDiffVars a
        | SurfaceAt(_, idx) -> collectDiffVars idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            Set.union (collectDiffVars a) (collectDiffVars b)
        | Select(c, t, f) ->
            collectDiffVars c |> Set.union (collectDiffVars t) |> Set.union (collectDiffVars f)
        | BinSearch(_, _, _, v) -> collectDiffVars v

    /// Collect all DiffVar indices from an entire model.
    let collectModelDiffVars (model: Model) : int[] =
        let fromResult = collectDiffVars model.Result
        let fromAccums =
            model.Accums
            |> Map.toSeq
            |> Seq.collect (fun (_, def) ->
                Set.union (collectDiffVars def.Init) (collectDiffVars def.Body))
            |> Set.ofSeq
        let fromObs =
            model.Observers
            |> List.map (fun o -> collectDiffVars o.Expr)
            |> Set.unionMany
        Set.unionMany [fromResult; fromAccums; fromObs]
        |> Set.toArray
        |> Array.sort

    // ══════════════════════════════════════════════════════════════════
    // Forward-Mode Symbolic Differentiation
    // ══════════════════════════════════════════════════════════════════

    /// Symbolic derivative of an expression with respect to DiffVar index `wrt`.
    /// DiffVar(wrt, _) → 1, DiffVar(other, _) → 0, other nodes follow standard rules.
    /// AccumRef derivatives are represented as AccumRef of a derivative accumulator.
    let rec forwardDiff (expr: Expr) (wrt: int) (accumDerivId: int -> int) : Expr =
        match expr with
        // DiffVar: seed derivative
        | DiffVar(idx, _) -> if idx = wrt then Const 1.0f else Const 0.0f

        // Constants and random variables: zero derivative
        | Const _ | TimeIndex | Normal _ | Uniform _ | Lookup1D _ | BatchRef _ -> Const 0.0f
        | Floor _ -> Const 0.0f
        | SurfaceAt _ -> Const 0.0f
        | BinSearch _ -> Const 0.0f

        // AccumRef: reference the derivative accumulator
        | AccumRef id -> AccumRef(accumDerivId id)

        // Sum rule
        | Add(a, b) -> Add(forwardDiff a wrt accumDerivId, forwardDiff b wrt accumDerivId)
        | Sub(a, b) -> Sub(forwardDiff a wrt accumDerivId, forwardDiff b wrt accumDerivId)

        // Product rule: d(a*b) = da*b + a*db
        | Mul(a, b) ->
            Add(Mul(forwardDiff a wrt accumDerivId, b),
                Mul(a, forwardDiff b wrt accumDerivId))

        // Quotient rule: d(a/b) = (da*b - a*db) / b^2
        | Div(a, b) ->
            Div(Sub(Mul(forwardDiff a wrt accumDerivId, b),
                    Mul(a, forwardDiff b wrt accumDerivId)),
                Mul(b, b))

        // Chain rule for unary functions
        | Exp(a) -> Mul(Exp(a), forwardDiff a wrt accumDerivId)
        | Log(a) -> Mul(Div(Const 1.0f, a), forwardDiff a wrt accumDerivId)
        | Sqrt(a) -> Mul(Div(Const 0.5f, Sqrt(a)), forwardDiff a wrt accumDerivId)
        | Neg(a) -> Neg(forwardDiff a wrt accumDerivId)
        | Abs(a) ->
            let sign = Select(Gt(a, Const 0.0f), Const 1.0f, Const -1.0f)
            Mul(sign, forwardDiff a wrt accumDerivId)

        // Max/Min subgradients
        | Max(a, b) -> Select(Gt(a, b), forwardDiff a wrt accumDerivId, forwardDiff b wrt accumDerivId)
        | Min(a, b) -> Select(Lt(a, b), forwardDiff a wrt accumDerivId, forwardDiff b wrt accumDerivId)

        // Comparison ops: piecewise constant → zero derivative
        | Gt _ | Gte _ | Lt _ | Lte _ -> Const 0.0f

        // Select: differentiate through both branches
        | Select(cond, t, f) ->
            Select(cond, forwardDiff t wrt accumDerivId, forwardDiff f wrt accumDerivId)

    // ══════════════════════════════════════════════════════════════════
    // Model Transformation (Dual Mode)
    // ══════════════════════════════════════════════════════════════════

    /// Transform a model with DiffVar nodes into an expanded model
    /// that computes both values and derivatives.
    /// For each original accumulator and each DiffVar, adds a derivative accumulator.
    /// Returns (expandedModel, numDiffVars) where the result expression
    /// is unchanged but new accumulators carry derivative information.
    let transformDual (model: Model) : Model * int[] =
        let diffVars = collectModelDiffVars model
        let n = diffVars.Length

        if n = 0 then
            model, [||]
        else

        // Map from DiffVar index → position in diffVars array
        let diffVarPos = diffVars |> Array.mapi (fun pos idx -> idx, pos) |> Map.ofArray

        // For each original accumulator ID and each diffVar,
        // create a derivative accumulator at ID: baseId + (1 + pos) * offset
        // We use a simple scheme: original accums are 0..maxAccumId,
        // derivative accums start at maxAccumId + 1
        let maxAccumId =
            if model.Accums.IsEmpty then -1
            else model.Accums |> Map.toSeq |> Seq.map fst |> Seq.max
        let derivAccumBase = maxAccumId + 1

        // Map from (original accum ID, diffVar position) → derivative accum ID
        let derivAccumId (origId: int) (pos: int) : int =
            derivAccumBase + origId * n + pos

        // For forwardDiff, we need: given origAccumId, return derivAccumId for current wrt
        // This is parameterized by the current diffVar position being differentiated

        let mutable newAccums = model.Accums

        let sortedAccums = CompilerCommon.sortAccums model.Accums

        for wrtPos in 0 .. n - 1 do
            let wrtIdx = diffVars.[wrtPos]

            // Build the accumDerivId function for this wrt
            let accumDerivIdFn (origId: int) = derivAccumId origId wrtPos

            for (origId, origDef) in sortedAccums do
                // Derivative of init: differentiate init w.r.t. DiffVar[wrtIdx]
                let dInit = forwardDiff origDef.Init wrtIdx accumDerivIdFn

                // Derivative of body: differentiate body w.r.t. DiffVar[wrtIdx]
                // The body uses AccumRef(origId) for self-reference
                // forwardDiff will map AccumRef(origId) → AccumRef(derivAccumId origId wrtPos)
                let dBody = forwardDiff origDef.Body wrtIdx accumDerivIdFn

                let newId = derivAccumId origId wrtPos
                newAccums <- newAccums |> Map.add newId { Init = dInit; Body = dBody }

        // Create derivative observers for the result
        let dResultExprs =
            diffVars |> Array.mapi (fun pos wrtIdx ->
                let accumDerivIdFn (origId: int) = derivAccumId origId pos
                forwardDiff model.Result wrtIdx accumDerivIdFn)

        // Add derivative observers
        let nextObsSlot = model.Observers.Length
        let derivObservers =
            dResultExprs |> Array.mapi (fun i dExpr ->
                { Name = $"__deriv_{diffVars.[i]}"
                  Expr = dExpr
                  SlotIndex = nextObsSlot + i })
            |> Array.toList

        let expandedModel = {
            model with
                Accums = newAccums
                Observers = model.Observers @ derivObservers
        }

        expandedModel, diffVars

    // ══════════════════════════════════════════════════════════════════
    // Model Transformation (HyperDual Mode — 2nd order)
    // ══════════════════════════════════════════════════════════════════

    /// Transform a model to compute 2nd-order derivatives via nested forward differentiation.
    /// diagonal=true: only d²V/dSi² (gamma); diagonal=false: full d²V/dSi·dSj (cross-gamma).
    let transformHyperDual (diagonal: bool) (model: Model) : Model * int[] =
        let diffVars = collectModelDiffVars model
        let n = diffVars.Length
        if n = 0 then model, [||]
        else

        // Step 1: Apply dual transform for 1st-order derivatives
        let model1, _ = transformDual model

        let maxOrigId =
            if model.Accums.IsEmpty then -1
            else model.Accums |> Map.toSeq |> Seq.map fst |> Seq.max
        let numOrig = model.Accums.Count
        let layer1Base = maxOrigId + 1
        let layer1Id origId pos = layer1Base + origId * n + pos
        let sortedOrigAccums = CompilerCommon.sortAccums model.Accums
        let layer2Base = layer1Base + numOrig * n

        let mutable newAccums = model1.Accums

        if diagonal then
            let layer2DiagId origId pos = layer2Base + origId * n + pos

            for pos in 0 .. n - 1 do
                let wrtIdx = diffVars.[pos]
                for (origId, _) in sortedOrigAccums do
                    let l1Def = model1.Accums.[layer1Id origId pos]
                    let accumDerivIdFn (id: int) =
                        if id <= maxOrigId then layer1Id id pos
                        else layer2DiagId ((id - layer1Base) / n) pos
                    let dInit = forwardDiff l1Def.Init wrtIdx accumDerivIdFn
                    let dBody = forwardDiff l1Def.Body wrtIdx accumDerivIdFn
                    newAccums <- newAccums |> Map.add (layer2DiagId origId pos) { Init = dInit; Body = dBody }

            let nextSlot = model1.Observers.Length
            let layer2Observers =
                diffVars |> Array.mapi (fun pos wrtIdx ->
                    let derivObs = model1.Observers |> List.find (fun o -> o.Name = $"__deriv_{wrtIdx}")
                    let accumDerivIdFn (id: int) =
                        if id <= maxOrigId then layer1Id id pos
                        else layer2DiagId ((id - layer1Base) / n) pos
                    let d2Expr = forwardDiff derivObs.Expr wrtIdx accumDerivIdFn
                    { Name = $"__deriv2_{wrtIdx}"; Expr = d2Expr; SlotIndex = nextSlot + pos })
                |> Array.toList

            { model1 with Accums = newAccums; Observers = model1.Observers @ layer2Observers }, diffVars

        else
            // Full Hessian: d²V/dSi·dSj for all i, j
            let layer2FullId origId i j = layer2Base + origId * n * n + i * n + j

            for i in 0 .. n - 1 do
                for j in 0 .. n - 1 do
                    let wrtIdx = diffVars.[j]
                    for (origId, _) in sortedOrigAccums do
                        let l1Def = model1.Accums.[layer1Id origId i]
                        let accumDerivIdFn (id: int) =
                            if id <= maxOrigId then layer1Id id j
                            else layer2FullId ((id - layer1Base) / n) i j
                        let dInit = forwardDiff l1Def.Init wrtIdx accumDerivIdFn
                        let dBody = forwardDiff l1Def.Body wrtIdx accumDerivIdFn
                        newAccums <- newAccums |> Map.add (layer2FullId origId i j) { Init = dInit; Body = dBody }

            let nextSlot = model1.Observers.Length
            let layer2Observers =
                [| for i in 0 .. n - 1 do
                       for j in 0 .. n - 1 do
                           let wrtIdx = diffVars.[j]
                           let derivObs = model1.Observers |> List.find (fun o -> o.Name = $"__deriv_{diffVars.[i]}")
                           let accumDerivIdFn (id: int) =
                               if id <= maxOrigId then layer1Id id j
                               else layer2FullId ((id - layer1Base) / n) i j
                           let d2Expr = forwardDiff derivObs.Expr wrtIdx accumDerivIdFn
                           { Name = $"__deriv2_{diffVars.[i]}_{diffVars.[j]}"
                             Expr = d2Expr
                             SlotIndex = nextSlot + i * n + j } |]
                |> Array.toList

            { model1 with Accums = newAccums; Observers = model1.Observers @ layer2Observers }, diffVars

    // ══════════════════════════════════════════════════════════════════
    // Partial Derivatives (for Adjoint mode)
    // ══════════════════════════════════════════════════════════════════

    /// Partial derivative of expr w.r.t. AccumRef(targetId).
    /// All other AccumRefs and DiffVars are treated as constants.
    let rec partialDiffAccumRef (expr: Expr) (targetId: int) : Expr =
        match expr with
        | AccumRef id -> if id = targetId then Const 1.0f else Const 0.0f
        | DiffVar _ | Const _ | TimeIndex | Normal _ | Uniform _ | Lookup1D _ | BatchRef _ -> Const 0.0f
        | Floor _ | SurfaceAt _ | BinSearch _ -> Const 0.0f
        | Add(a, b) -> Add(partialDiffAccumRef a targetId, partialDiffAccumRef b targetId)
        | Sub(a, b) -> Sub(partialDiffAccumRef a targetId, partialDiffAccumRef b targetId)
        | Mul(a, b) ->
            Add(Mul(partialDiffAccumRef a targetId, b), Mul(a, partialDiffAccumRef b targetId))
        | Div(a, b) ->
            Div(Sub(Mul(partialDiffAccumRef a targetId, b),
                    Mul(a, partialDiffAccumRef b targetId)), Mul(b, b))
        | Exp a -> Mul(Exp a, partialDiffAccumRef a targetId)
        | Log a -> Mul(Div(Const 1.0f, a), partialDiffAccumRef a targetId)
        | Sqrt a -> Mul(Div(Const 0.5f, Sqrt a), partialDiffAccumRef a targetId)
        | Neg a -> Neg(partialDiffAccumRef a targetId)
        | Abs a ->
            Mul(Select(Gt(a, Const 0.0f), Const 1.0f, Const -1.0f), partialDiffAccumRef a targetId)
        | Max(a, b) -> Select(Gt(a, b), partialDiffAccumRef a targetId, partialDiffAccumRef b targetId)
        | Min(a, b) -> Select(Lt(a, b), partialDiffAccumRef a targetId, partialDiffAccumRef b targetId)
        | Gt _ | Gte _ | Lt _ | Lte _ -> Const 0.0f
        | Select(c, t, f) ->
            Select(c, partialDiffAccumRef t targetId, partialDiffAccumRef f targetId)

    /// Partial derivative of expr w.r.t. DiffVar(targetIdx).
    /// AccumRefs are treated as constants (no chain rule through accums).
    let rec partialDiffDiffVar (expr: Expr) (targetIdx: int) : Expr =
        match expr with
        | DiffVar(idx, _) -> if idx = targetIdx then Const 1.0f else Const 0.0f
        | AccumRef _ | Const _ | TimeIndex | Normal _ | Uniform _ | Lookup1D _ | BatchRef _ -> Const 0.0f
        | Floor _ | SurfaceAt _ | BinSearch _ -> Const 0.0f
        | Add(a, b) -> Add(partialDiffDiffVar a targetIdx, partialDiffDiffVar b targetIdx)
        | Sub(a, b) -> Sub(partialDiffDiffVar a targetIdx, partialDiffDiffVar b targetIdx)
        | Mul(a, b) ->
            Add(Mul(partialDiffDiffVar a targetIdx, b), Mul(a, partialDiffDiffVar b targetIdx))
        | Div(a, b) ->
            Div(Sub(Mul(partialDiffDiffVar a targetIdx, b),
                    Mul(a, partialDiffDiffVar b targetIdx)), Mul(b, b))
        | Exp a -> Mul(Exp a, partialDiffDiffVar a targetIdx)
        | Log a -> Mul(Div(Const 1.0f, a), partialDiffDiffVar a targetIdx)
        | Sqrt a -> Mul(Div(Const 0.5f, Sqrt a), partialDiffDiffVar a targetIdx)
        | Neg a -> Neg(partialDiffDiffVar a targetIdx)
        | Abs a ->
            Mul(Select(Gt(a, Const 0.0f), Const 1.0f, Const -1.0f), partialDiffDiffVar a targetIdx)
        | Max(a, b) -> Select(Gt(a, b), partialDiffDiffVar a targetIdx, partialDiffDiffVar b targetIdx)
        | Min(a, b) -> Select(Lt(a, b), partialDiffDiffVar a targetIdx, partialDiffDiffVar b targetIdx)
        | Gt _ | Gte _ | Lt _ | Lte _ -> Const 0.0f
        | Select(c, t, f) ->
            Select(c, partialDiffDiffVar t targetIdx, partialDiffDiffVar f targetIdx)

    // ══════════════════════════════════════════════════════════════════
    // Convenience: Check if model uses AD
    // ══════════════════════════════════════════════════════════════════

    /// Check if a model contains any DiffVar nodes.
    let hasDiffVars (model: Model) : bool =
        (collectModelDiffVars model).Length > 0

    // ══════════════════════════════════════════════════════════════════
    // AD Method Recommendation
    // ══════════════════════════════════════════════════════════════════

    /// Recommend differentiation method based on model characteristics.
    /// Returns (recommended mode, explanation string).
    let recommend (model: Model) : DiffMode option * string =
        let diffVars = collectModelDiffVars model
        let n = diffVars.Length
        let numAccums = model.Accums.Count

        if n = 0 then
            None,
            "No DiffVar nodes found. Options:\n" +
            "  - Add DiffVar(index, value) to mark differentiable parameters\n" +
            "  - Use finite differences (bump-and-revalue) as a fallback"
        elif n <= 4 then
            Some Dual,
            $"{n} DiffVar(s): Dual (forward mode) recommended.\n" +
            $"  Adds {numAccums * n} derivative accumulators ({numAccums} per DiffVar).\n" +
            "  All derivatives computed in a single forward pass.\n" +
            "  For 2nd-order (gamma): use HyperDual(diagonal=true)."
        elif n <= 10 then
            Some Adjoint,
            $"{n} DiffVars: Adjoint (reverse mode) recommended.\n" +
            $"  Forward mode would add {numAccums * n} accumulators — may hit register pressure.\n" +
            $"  Adjoint uses tape memory: {numAccums} * steps * numScenarios floats.\n" +
            "  Computes all derivatives in one forward + backward pass."
        else
            Some Adjoint,
            $"{n} DiffVars: Adjoint (reverse mode) strongly recommended.\n" +
            $"  Forward mode would need {numAccums * (1 + n)} total accumulators — too many registers.\n" +
            $"  Adjoint tape: {numAccums} * steps * numScenarios floats."

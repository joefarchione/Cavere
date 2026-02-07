namespace Cavere.Core

/// Forward-mode automatic differentiation via expression transformation.
/// Transforms a Model containing Dual/HyperDual nodes into an expanded Model
/// where each accumulator tracks derivatives alongside values.
[<RequireQualifiedAccess>]
module CompilerDiff =

    // ══════════════════════════════════════════════════════════════════
    // DiffVar Collection
    // ══════════════════════════════════════════════════════════════════

    /// Collect all Dual/HyperDual indices from an expression tree.
    let rec collectDiffVars (expr: Expr) : Set<int> =
        match expr with
        | Dual(idx, _, _) | HyperDual(idx, _, _) -> Set.singleton idx
        | Const _ | TimeIndex | Normal _ | Uniform _ | Bernoulli _ | AccumRef _ | Lookup1D _ | BatchRef _ -> Set.empty
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> collectDiffVars a
        | SurfaceAt(_, idx) -> collectDiffVars idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            Set.union (collectDiffVars a) (collectDiffVars b)
        | Select(c, t, f) ->
            collectDiffVars c |> Set.union (collectDiffVars t) |> Set.union (collectDiffVars f)
        | BinSearch(_, _, _, v) -> collectDiffVars v

    /// Collect only HyperDual indices from an expression tree.
    let rec collectHyperDualVars (expr: Expr) : Set<int> =
        match expr with
        | HyperDual(idx, _, _) -> Set.singleton idx
        | Dual _ | Const _ | TimeIndex | Normal _ | Uniform _ | Bernoulli _ | AccumRef _ | Lookup1D _ | BatchRef _ -> Set.empty
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> collectHyperDualVars a
        | SurfaceAt(_, idx) -> collectHyperDualVars idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            Set.union (collectHyperDualVars a) (collectHyperDualVars b)
        | Select(c, t, f) ->
            collectHyperDualVars c |> Set.union (collectHyperDualVars t) |> Set.union (collectHyperDualVars f)
        | BinSearch(_, _, _, v) -> collectHyperDualVars v

    /// Collect name mapping for all Dual/HyperDual nodes in an expression tree.
    let rec private collectDiffVarNames (expr: Expr) : Map<int, string> =
        match expr with
        | Dual(idx, _, name) | HyperDual(idx, _, name) -> Map.ofList [idx, name]
        | Const _ | TimeIndex | Normal _ | Uniform _ | Bernoulli _ | AccumRef _ | Lookup1D _ | BatchRef _ -> Map.empty
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> collectDiffVarNames a
        | SurfaceAt(_, idx) -> collectDiffVarNames idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            let ma = collectDiffVarNames a
            let mb = collectDiffVarNames b
            Map.fold (fun acc k v -> Map.add k v acc) ma mb
        | Select(c, t, f) ->
            let mc = collectDiffVarNames c
            let mt = collectDiffVarNames t
            let mf = collectDiffVarNames f
            Map.fold (fun acc k v -> Map.add k v acc) (Map.fold (fun acc k v -> Map.add k v acc) mc mt) mf
        | BinSearch(_, _, _, v) -> collectDiffVarNames v

    /// Collect name mapping for all Dual/HyperDual nodes in an entire model.
    let collectModelDiffVarNames (model: Model) : Map<int, string> =
        let fromResult = collectDiffVarNames model.Result
        let fromAccums =
            model.Accums
            |> Map.toSeq
            |> Seq.collect (fun (_, def) ->
                [ collectDiffVarNames def.Init; collectDiffVarNames def.Body ])
            |> Seq.fold (fun acc m -> Map.fold (fun a k v -> Map.add k v a) acc m) Map.empty
        let fromObs =
            model.Observers
            |> List.map (fun o -> collectDiffVarNames o.Expr)
            |> List.fold (fun acc m -> Map.fold (fun a k v -> Map.add k v a) acc m) Map.empty
        Map.fold (fun acc k v -> Map.add k v acc)
            (Map.fold (fun acc k v -> Map.add k v acc) fromResult fromAccums) fromObs

    /// Collect all Dual/HyperDual indices from an entire model.
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

    /// Collect only HyperDual indices from an entire model.
    let collectModelHyperDualVars (model: Model) : int[] =
        let fromResult = collectHyperDualVars model.Result
        let fromAccums =
            model.Accums
            |> Map.toSeq
            |> Seq.collect (fun (_, def) ->
                Set.union (collectHyperDualVars def.Init) (collectHyperDualVars def.Body))
            |> Set.ofSeq
        let fromObs =
            model.Observers
            |> List.map (fun o -> collectHyperDualVars o.Expr)
            |> Set.unionMany
        Set.unionMany [fromResult; fromAccums; fromObs]
        |> Set.toArray
        |> Array.sort

    // ══════════════════════════════════════════════════════════════════
    // Forward-Mode Symbolic Differentiation
    // ══════════════════════════════════════════════════════════════════

    /// Symbolic derivative of an expression with respect to Dual/HyperDual index `wrt`.
    /// Dual/HyperDual(wrt, _, _) → 1, Dual/HyperDual(other, _, _) → 0, other nodes follow standard rules.
    /// AccumRef derivatives are represented as AccumRef of a derivative accumulator.
    let rec forwardDiff (expr: Expr) (wrt: int) (accumDerivId: int -> int) : Expr =
        match expr with
        // Dual/HyperDual: seed derivative
        | Dual(idx, _, _) | HyperDual(idx, _, _) -> if idx = wrt then Const 1.0f else Const 0.0f

        // Constants and random variables: zero derivative
        | Const _ | TimeIndex | Normal _ | Uniform _ | Bernoulli _ | Lookup1D _ | BatchRef _ -> Const 0.0f
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

    /// Transform a model with Dual/HyperDual nodes into an expanded model
    /// that computes both values and derivatives.
    /// For each original accumulator and each Dual/HyperDual, adds a derivative accumulator.
    /// Returns (expandedModel, numDiffVars) where the result expression
    /// is unchanged but new accumulators carry derivative information.
    let transformDual (model: Model) : Model * int[] =
        let diffVars = collectModelDiffVars model
        let names = collectModelDiffVarNames model
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
                let name = names |> Map.tryFind diffVars.[i] |> Option.defaultValue (string diffVars.[i])
                { Name = $"d1_{name}"
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
        let names = collectModelDiffVarNames model
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
                    let wrtName = names |> Map.tryFind wrtIdx |> Option.defaultValue (string wrtIdx)
                    let derivObs = model1.Observers |> List.find (fun o -> o.Name = $"d1_{wrtName}")
                    let accumDerivIdFn (id: int) =
                        if id <= maxOrigId then layer1Id id pos
                        else layer2DiagId ((id - layer1Base) / n) pos
                    let d2Expr = forwardDiff derivObs.Expr wrtIdx accumDerivIdFn
                    { Name = $"d2_{wrtName}"; Expr = d2Expr; SlotIndex = nextSlot + pos })
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
                           let iName = names |> Map.tryFind diffVars.[i] |> Option.defaultValue (string diffVars.[i])
                           let jName = names |> Map.tryFind diffVars.[j] |> Option.defaultValue (string diffVars.[j])
                           let derivObs = model1.Observers |> List.find (fun o -> o.Name = $"d1_{iName}")
                           let accumDerivIdFn (id: int) =
                               if id <= maxOrigId then layer1Id id j
                               else layer2FullId ((id - layer1Base) / n) i j
                           let d2Expr = forwardDiff derivObs.Expr wrtIdx accumDerivIdFn
                           { Name = $"d2_{iName}_{jName}"
                             Expr = d2Expr
                             SlotIndex = nextSlot + i * n + j } |]
                |> Array.toList

            { model1 with Accums = newAccums; Observers = model1.Observers @ layer2Observers }, diffVars

    // ══════════════════════════════════════════════════════════════════
    // Selective HyperDual — 2nd order for specified vars only
    // ══════════════════════════════════════════════════════════════════

    /// Apply 2nd-order differentiation only for specified variable indices.
    /// Expects model1 to already have 1st-order derivatives (from transformDual).
    /// hyperVarIndices: indices of HyperDual variables that need 2nd-order.
    let transformHyperDualSelective (hyperVarIndices: int[]) (model1: Model) (originalModel: Model) : Model =
        let allDiffVars = collectModelDiffVars originalModel
        let names = collectModelDiffVarNames originalModel
        let n = allDiffVars.Length
        let hyperSet = Set.ofArray hyperVarIndices

        if hyperVarIndices.Length = 0 then model1
        else

        let maxOrigId =
            if originalModel.Accums.IsEmpty then -1
            else originalModel.Accums |> Map.toSeq |> Seq.map fst |> Seq.max
        let numOrig = originalModel.Accums.Count
        let layer1Base = maxOrigId + 1
        let layer1Id origId pos = layer1Base + origId * n + pos
        let sortedOrigAccums = CompilerCommon.sortAccums originalModel.Accums
        let layer2Base = layer1Base + numOrig * n
        let layer2DiagId origId pos = layer2Base + origId * n + pos

        let mutable newAccums = model1.Accums

        for pos in 0 .. n - 1 do
            let wrtIdx = allDiffVars.[pos]
            if hyperSet.Contains wrtIdx then
                for (origId, _) in sortedOrigAccums do
                    let l1Def = model1.Accums.[layer1Id origId pos]
                    let accumDerivIdFn (id: int) =
                        if id <= maxOrigId then layer1Id id pos
                        else layer2DiagId ((id - layer1Base) / n) pos
                    let dInit = forwardDiff l1Def.Init wrtIdx accumDerivIdFn
                    let dBody = forwardDiff l1Def.Body wrtIdx accumDerivIdFn
                    newAccums <- newAccums |> Map.add (layer2DiagId origId pos) { Init = dInit; Body = dBody }

        let nextSlot = model1.Observers.Length
        let mutable slotIdx = 0
        let layer2Observers =
            [| for pos in 0 .. n - 1 do
                   let wrtIdx = allDiffVars.[pos]
                   if hyperSet.Contains wrtIdx then
                       let wrtName = names |> Map.tryFind wrtIdx |> Option.defaultValue (string wrtIdx)
                       let derivObs = model1.Observers |> List.find (fun o -> o.Name = $"d1_{wrtName}")
                       let accumDerivIdFn (id: int) =
                           if id <= maxOrigId then layer1Id id pos
                           else layer2DiagId ((id - layer1Base) / n) pos
                       let d2Expr = forwardDiff derivObs.Expr wrtIdx accumDerivIdFn
                       let obs = { Name = $"d2_{wrtName}"; Expr = d2Expr; SlotIndex = nextSlot + slotIdx }
                       slotIdx <- slotIdx + 1
                       yield obs |]
            |> Array.toList

        { model1 with Accums = newAccums; Observers = model1.Observers @ layer2Observers }

    // ══════════════════════════════════════════════════════════════════
    // Partial Derivatives (for Adjoint mode)
    // ══════════════════════════════════════════════════════════════════

    /// Partial derivative of expr w.r.t. AccumRef(targetId).
    /// All other AccumRefs and Dual/HyperDual are treated as constants.
    let rec partialDiffAccumRef (expr: Expr) (targetId: int) : Expr =
        match expr with
        | AccumRef id -> if id = targetId then Const 1.0f else Const 0.0f
        | Dual _ | HyperDual _ | Const _ | TimeIndex | Normal _ | Uniform _ | Bernoulli _ | Lookup1D _ | BatchRef _ -> Const 0.0f
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

    /// Partial derivative of expr w.r.t. Dual/HyperDual(targetIdx).
    /// AccumRefs are treated as constants (no chain rule through accums).
    let rec partialDiffDiffVar (expr: Expr) (targetIdx: int) : Expr =
        match expr with
        | Dual(idx, _, _) | HyperDual(idx, _, _) -> if idx = targetIdx then Const 1.0f else Const 0.0f
        | AccumRef _ | Const _ | TimeIndex | Normal _ | Uniform _ | Bernoulli _ | Lookup1D _ | BatchRef _ -> Const 0.0f
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

    /// Check if a model contains any Dual/HyperDual nodes.
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
            "No Dual/HyperDual nodes found. Options:\n" +
            "  - Add Dual(index, value, name) to mark differentiable parameters\n" +
            "  - Use finite differences (bump-and-revalue) as a fallback"
        elif n <= 4 then
            Some DualMode,
            $"{n} Dual/HyperDual(s): Dual (forward mode) recommended.\n" +
            $"  Adds {numAccums * n} derivative accumulators ({numAccums} per variable).\n" +
            "  All derivatives computed in a single forward pass.\n" +
            "  For 2nd-order (gamma): use HyperDual(diagonal=true)."
        elif n <= 10 then
            Some AdjointMode,
            $"{n} Dual/HyperDuals: Adjoint (reverse mode) recommended.\n" +
            $"  Forward mode would add {numAccums * n} accumulators — may hit register pressure.\n" +
            $"  Adjoint uses tape memory: {numAccums} * steps * numScenarios floats.\n" +
            "  Computes all derivatives in one forward + backward pass."
        else
            Some AdjointMode,
            $"{n} Dual/HyperDuals: Adjoint (reverse mode) strongly recommended.\n" +
            $"  Forward mode would need {numAccums * (1 + n)} total accumulators — too many registers.\n" +
            $"  Adjoint tape: {numAccums} * steps * numScenarios floats."

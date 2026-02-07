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
    // Convenience: Check if model uses AD
    // ══════════════════════════════════════════════════════════════════

    /// Check if a model contains any DiffVar nodes.
    let hasDiffVars (model: Model) : bool =
        (collectModelDiffVars model).Length > 0

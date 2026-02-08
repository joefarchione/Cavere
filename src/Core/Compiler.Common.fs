namespace Cavere.Core

open System

/// Surface layout and metadata for GPU memory packing.
type SurfaceLayout = {
    Offsets: Map<int, int>
    TotalSize: int
    Meta: Map<int, SurfaceMeta>
}

and SurfaceMeta =
    | Curve1DMeta of offset: int * count: int * steps: int
    | Grid2DMeta of valOffset: int * timeOffset: int * timeCnt: int * spotOffset: int * spotCnt: int * steps: int

/// Common compiler utilities: surface layout, expression emission, topological sort.
[<RequireQualifiedAccess>]
module CompilerCommon =

    // ══════════════════════════════════════════════════════════════════
    // Surface Layout
    // ══════════════════════════════════════════════════════════════════

    let layoutSurfaces (model: Model) : SurfaceLayout =
        let mutable offset = 0
        let mutable offsets = Map.empty
        let mutable meta = Map.empty

        for kv in model.Surfaces |> Map.toSeq do
            let id, surf = kv
            offsets <- offsets |> Map.add id offset

            match surf with
            | Curve1D(values, steps) ->
                meta <- meta |> Map.add id (Curve1DMeta(offset, values.Length, steps))
                offset <- offset + values.Length
            | Grid2D(values, timeAxis, spotAxis, steps) ->
                let valOff = offset
                offset <- offset + values.Length
                let timeOff = offset
                offset <- offset + timeAxis.Length
                let spotOff = offset
                offset <- offset + spotAxis.Length

                meta <-
                    meta
                    |> Map.add id (Grid2DMeta(valOff, timeOff, timeAxis.Length, spotOff, spotAxis.Length, steps))

        {
            Offsets = offsets
            TotalSize = offset
            Meta = meta
        }

    let packSurfaces (model: Model) (layout: SurfaceLayout) : float32[] =
        let arr = Array.zeroCreate<float32> (max 1 layout.TotalSize)

        for kv in model.Surfaces |> Map.toSeq do
            let id, surf = kv
            let off = layout.Offsets.[id]

            match surf with
            | Curve1D(values, _) -> Array.Copy(values, 0, arr, off, values.Length)
            | Grid2D(values, timeAxis, spotAxis, _) ->
                let mutable pos = off
                Array.Copy(values, 0, arr, pos, values.Length)
                pos <- pos + values.Length
                Array.Copy(timeAxis, 0, arr, pos, timeAxis.Length)
                pos <- pos + timeAxis.Length
                Array.Copy(spotAxis, 0, arr, pos, spotAxis.Length)

        arr

    // ══════════════════════════════════════════════════════════════════
    // Expression Emission
    // ══════════════════════════════════════════════════════════════════

    /// Emit an expression to C#, writing any required preamble statements
    /// (for Cond multi-way conditionals) into the preamble StringBuilder.
    let rec emitExprPreamble
        (layout: SurfaceLayout)
        (preamble: Text.StringBuilder)
        (condCounter: int ref)
        (expr: Expr)
        : string =
        let emit = emitExprPreamble layout preamble condCounter

        match expr with
        | Const v
        | Dual(_, v, _)
        | HyperDual(_, v, _) ->
            if Single.IsInfinity(v) || Single.IsNaN(v) then
                sprintf "%.8ef" v
            else
                sprintf "%.8ff" v
        | TimeIndex -> "t"
        | Normal id -> sprintf "z_%d" id
        | Uniform id -> sprintf "u_%d" id
        | Bernoulli id -> sprintf "b_%d" id
        | AccumRef id -> sprintf "accum_%d" id
        | Add(a, b) -> sprintf "(%s + %s)" (emit a) (emit b)
        | Sub(a, b) -> sprintf "(%s - %s)" (emit a) (emit b)
        | Mul(a, b) -> sprintf "(%s * %s)" (emit a) (emit b)
        | Div(a, b) -> sprintf "(%s / %s)" (emit a) (emit b)
        | Max(a, b) -> sprintf "MathF.Max(%s, %s)" (emit a) (emit b)
        | Min(a, b) -> sprintf "MathF.Min(%s, %s)" (emit a) (emit b)
        | Gt(a, b) -> sprintf "((%s > %s) ? 1.0f : 0.0f)" (emit a) (emit b)
        | Gte(a, b) -> sprintf "((%s >= %s) ? 1.0f : 0.0f)" (emit a) (emit b)
        | Lt(a, b) -> sprintf "((%s < %s) ? 1.0f : 0.0f)" (emit a) (emit b)
        | Lte(a, b) -> sprintf "((%s <= %s) ? 1.0f : 0.0f)" (emit a) (emit b)
        | Select(c, t, f) -> sprintf "((%s > 0.5f) ? %s : %s)" (emit c) (emit t) (emit f)
        | Cond(cases, defaultExpr) ->
            let n = !condCounter
            condCounter := n + 1
            let varName = sprintf "cond_%d" n

            preamble.AppendLine(sprintf "            float %s = %s;" varName (emit defaultExpr))
            |> ignore

            for i, (c, v) in cases |> List.indexed do
                let prefix = if i = 0 then "if" else "else if"

                preamble.AppendLine(sprintf "            %s (%s > 0.5f) %s = %s;" prefix (emit c) varName (emit v))
                |> ignore

            varName
        | Neg a -> sprintf "(-%s)" (emit a)
        | Exp a -> sprintf "MathF.Exp(%s)" (emit a)
        | Log a -> sprintf "MathF.Log(%s)" (emit a)
        | Sqrt a -> sprintf "MathF.Sqrt(%s)" (emit a)
        | Abs a -> sprintf "MathF.Abs(%s)" (emit a)
        | Lookup1D sid ->
            match layout.Meta.[sid] with
            | Curve1DMeta(offset, _, _) -> sprintf "surfaces[%d + t]" offset
            | _ -> failwith "Surface type mismatch: expected Curve1D for Lookup1D"
        | Floor a -> sprintf "MathF.Floor(%s)" (emit a)
        | SurfaceAt(sid, idx) ->
            let baseOffset = layout.Offsets.[sid]
            sprintf "surfaces[%d + (int)(%s)]" baseOffset (emit idx)
        | BatchRef sid ->
            match layout.Meta.[sid] with
            | Curve1DMeta(offset, _, _) -> sprintf "surfaces[%d + batchIdx]" offset
            | _ -> failwith "Surface type mismatch: expected Curve1D for BatchRef"
        | BinSearch(sid, axisOff, axisCnt, value) ->
            let baseOffset = layout.Offsets.[sid]
            sprintf "FindBin(surfaces, %d, %d, %s)" (baseOffset + axisOff) axisCnt (emit value)

    /// Simple expression emission without preamble support (for expressions that cannot contain Cond).
    let emitExpr (layout: SurfaceLayout) (expr: Expr) : string =
        emitExprPreamble layout (Text.StringBuilder()) (ref 0) expr

    // ══════════════════════════════════════════════════════════════════
    // Topological Sort
    // ══════════════════════════════════════════════════════════════════

    let rec collectAccumRefs (expr: Expr) : Set<int> =
        match expr with
        | Const _
        | TimeIndex
        | Normal _
        | Uniform _
        | Bernoulli _
        | Lookup1D _
        | BatchRef _
        | Dual _
        | HyperDual _ -> Set.empty
        | AccumRef id -> Set.singleton id
        | Floor a -> collectAccumRefs a
        | SurfaceAt(_, idx) -> collectAccumRefs idx
        | Add(a, b)
        | Sub(a, b)
        | Mul(a, b)
        | Div(a, b)
        | Max(a, b)
        | Min(a, b)
        | Gt(a, b)
        | Gte(a, b)
        | Lt(a, b)
        | Lte(a, b) -> Set.union (collectAccumRefs a) (collectAccumRefs b)
        | Select(c, t, f) ->
            collectAccumRefs c
            |> Set.union (collectAccumRefs t)
            |> Set.union (collectAccumRefs f)
        | Cond(cases, defaultExpr) ->
            cases
            |> List.fold
                (fun acc (c, v) -> acc |> Set.union (collectAccumRefs c) |> Set.union (collectAccumRefs v))
                (collectAccumRefs defaultExpr)
        | Neg a
        | Exp a
        | Log a
        | Sqrt a
        | Abs a -> collectAccumRefs a
        | BinSearch(_, _, _, v) -> collectAccumRefs v

    // ══════════════════════════════════════════════════════════════════
    // Expression Walking — collect referenced IDs
    // ══════════════════════════════════════════════════════════════════

    let rec collectSurfaceIds (expr: Expr) : Set<int> =
        match expr with
        | Const _
        | TimeIndex
        | Normal _
        | Uniform _
        | Bernoulli _
        | AccumRef _
        | Dual _
        | HyperDual _ -> Set.empty
        | Lookup1D sid
        | BatchRef sid -> Set.singleton sid
        | SurfaceAt(sid, idx) -> collectSurfaceIds idx |> Set.add sid
        | Floor a
        | Neg a
        | Exp a
        | Log a
        | Sqrt a
        | Abs a -> collectSurfaceIds a
        | Add(a, b)
        | Sub(a, b)
        | Mul(a, b)
        | Div(a, b)
        | Max(a, b)
        | Min(a, b)
        | Gt(a, b)
        | Gte(a, b)
        | Lt(a, b)
        | Lte(a, b) -> Set.union (collectSurfaceIds a) (collectSurfaceIds b)
        | Select(c, t, f) ->
            collectSurfaceIds c
            |> Set.union (collectSurfaceIds t)
            |> Set.union (collectSurfaceIds f)
        | Cond(cases, defaultExpr) ->
            cases
            |> List.fold
                (fun acc (c, v) -> acc |> Set.union (collectSurfaceIds c) |> Set.union (collectSurfaceIds v))
                (collectSurfaceIds defaultExpr)
        | BinSearch(sid, _, _, v) -> collectSurfaceIds v |> Set.add sid

    let private collectAllExprs (model: Model) : Expr list = [
        yield model.Result
        for kv in model.Accums |> Map.toSeq do
            let _, def = kv
            yield def.Init
            yield def.Body
        for obs in model.Observers do
            yield obs.Expr
    ]

    // ══════════════════════════════════════════════════════════════════
    // Model Validation
    // ══════════════════════════════════════════════════════════════════

    let validate (model: Model) : unit =
        let exprs = collectAllExprs model
        let referencedSurfaces = exprs |> List.map collectSurfaceIds |> Set.unionMany
        let referencedAccums = exprs |> List.map collectAccumRefs |> Set.unionMany
        let registeredSurfaces = model.Surfaces |> Map.toSeq |> Seq.map fst |> Set.ofSeq
        let registeredAccums = model.Accums |> Map.toSeq |> Seq.map fst |> Set.ofSeq

        let missingSurfaces = Set.difference referencedSurfaces registeredSurfaces
        let missingAccums = Set.difference referencedAccums registeredAccums

        let errors = ResizeArray<string>()

        if not (Set.isEmpty missingSurfaces) then
            errors.Add $"Surface IDs referenced but not registered: {missingSurfaces |> Set.toList}"

        if not (Set.isEmpty missingAccums) then
            errors.Add $"AccumRef IDs referenced but not registered: {missingAccums |> Set.toList}"

        if errors.Count > 0 then
            failwithf "Model validation failed:\n  %s" (errors |> Seq.toList |> String.concat "\n  ")

    let sortAccums (accums: Map<int, AccumDef>) : (int * AccumDef) list =
        let deps =
            accums
            |> Map.map (fun id def ->
                let bodyRefs = collectAccumRefs def.Body
                bodyRefs |> Set.remove id)

        let mutable visited = Set.empty
        let mutable order = []

        let rec visit id =
            if visited |> Set.contains id then
                ()
            else
                visited <- visited |> Set.add id

                match deps |> Map.tryFind id with
                | Some depSet ->
                    for dep in depSet do
                        if accums |> Map.containsKey dep then
                            visit dep
                | None -> ()

                order <- (id, accums.[id]) :: order

        for id in accums |> Map.toSeq |> Seq.map fst do
            visit id

        List.rev order

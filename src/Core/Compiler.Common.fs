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
                meta <- meta |> Map.add id
                    (Grid2DMeta(valOff, timeOff, timeAxis.Length, spotOff, spotAxis.Length, steps))

        { Offsets = offsets; TotalSize = offset; Meta = meta }

    let packSurfaces (model: Model) (layout: SurfaceLayout) : float32[] =
        let arr = Array.zeroCreate<float32> (max 1 layout.TotalSize)
        for kv in model.Surfaces |> Map.toSeq do
            let id, surf = kv
            let off = layout.Offsets.[id]
            match surf with
            | Curve1D(values, _) ->
                Array.Copy(values, 0, arr, off, values.Length)
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

    let rec emitExpr (layout: SurfaceLayout) (expr: Expr) : string =
        match expr with
        | Const v ->
            if Single.IsInfinity(v) || Single.IsNaN(v) then
                sprintf "%.8ef" v
            else
                sprintf "%.8ff" v
        | TimeIndex -> "t"
        | Normal id -> sprintf "z_%d" id
        | Uniform id -> sprintf "u_%d" id
        | AccumRef id -> sprintf "accum_%d" id
        | Add(a, b) -> sprintf "(%s + %s)" (emitExpr layout a) (emitExpr layout b)
        | Sub(a, b) -> sprintf "(%s - %s)" (emitExpr layout a) (emitExpr layout b)
        | Mul(a, b) -> sprintf "(%s * %s)" (emitExpr layout a) (emitExpr layout b)
        | Div(a, b) -> sprintf "(%s / %s)" (emitExpr layout a) (emitExpr layout b)
        | Max(a, b) -> sprintf "MathF.Max(%s, %s)" (emitExpr layout a) (emitExpr layout b)
        | Min(a, b) -> sprintf "MathF.Min(%s, %s)" (emitExpr layout a) (emitExpr layout b)
        | Gt(a, b) -> sprintf "((%s > %s) ? 1.0f : 0.0f)" (emitExpr layout a) (emitExpr layout b)
        | Gte(a, b) -> sprintf "((%s >= %s) ? 1.0f : 0.0f)" (emitExpr layout a) (emitExpr layout b)
        | Lt(a, b) -> sprintf "((%s < %s) ? 1.0f : 0.0f)" (emitExpr layout a) (emitExpr layout b)
        | Lte(a, b) -> sprintf "((%s <= %s) ? 1.0f : 0.0f)" (emitExpr layout a) (emitExpr layout b)
        | Select(c, t, f) -> sprintf "((%s > 0.5f) ? %s : %s)" (emitExpr layout c) (emitExpr layout t) (emitExpr layout f)
        | Neg a -> sprintf "(-%s)" (emitExpr layout a)
        | Exp a -> sprintf "MathF.Exp(%s)" (emitExpr layout a)
        | Log a -> sprintf "MathF.Log(%s)" (emitExpr layout a)
        | Sqrt a -> sprintf "MathF.Sqrt(%s)" (emitExpr layout a)
        | Abs a -> sprintf "MathF.Abs(%s)" (emitExpr layout a)
        | Lookup1D sid ->
            match layout.Meta.[sid] with
            | Curve1DMeta(offset, _, _) ->
                sprintf "surfaces[%d + t]" offset
            | _ -> failwith "Surface type mismatch: expected Curve1D for Lookup1D"
        | Floor a -> sprintf "MathF.Floor(%s)" (emitExpr layout a)
        | SurfaceAt(sid, idx) ->
            let baseOffset = layout.Offsets.[sid]
            sprintf "surfaces[%d + (int)(%s)]" baseOffset (emitExpr layout idx)
        | BatchRef sid ->
            match layout.Meta.[sid] with
            | Curve1DMeta(offset, _, _) ->
                sprintf "surfaces[%d + batchIdx]" offset
            | _ -> failwith "Surface type mismatch: expected Curve1D for BatchRef"
        | BinSearch(sid, axisOff, axisCnt, value) ->
            let baseOffset = layout.Offsets.[sid]
            sprintf "FindBin(surfaces, %d, %d, %s)" (baseOffset + axisOff) axisCnt (emitExpr layout value)

    // ══════════════════════════════════════════════════════════════════
    // Topological Sort
    // ══════════════════════════════════════════════════════════════════

    let rec collectAccumRefs (expr: Expr) : Set<int> =
        match expr with
        | Const _ | TimeIndex | Normal _ | Uniform _ | Lookup1D _ | BatchRef _ -> Set.empty
        | AccumRef id -> Set.singleton id
        | Floor a -> collectAccumRefs a
        | SurfaceAt(_, idx) -> collectAccumRefs idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Max(a, b) | Min(a, b)
        | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            Set.union (collectAccumRefs a) (collectAccumRefs b)
        | Select(c, t, f) ->
            collectAccumRefs c |> Set.union (collectAccumRefs t) |> Set.union (collectAccumRefs f)
        | Neg a | Exp a | Log a | Sqrt a | Abs a -> collectAccumRefs a
        | BinSearch(_, _, _, v) -> collectAccumRefs v

    // ══════════════════════════════════════════════════════════════════
    // Expression Walking — collect referenced IDs
    // ══════════════════════════════════════════════════════════════════

    let rec collectSurfaceIds (expr: Expr) : Set<int> =
        match expr with
        | Const _ | TimeIndex | Normal _ | Uniform _ | AccumRef _ -> Set.empty
        | Lookup1D sid | BatchRef sid -> Set.singleton sid
        | SurfaceAt(sid, idx) -> collectSurfaceIds idx |> Set.add sid
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> collectSurfaceIds a
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Max(a, b) | Min(a, b)
        | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            Set.union (collectSurfaceIds a) (collectSurfaceIds b)
        | Select(c, t, f) ->
            collectSurfaceIds c |> Set.union (collectSurfaceIds t) |> Set.union (collectSurfaceIds f)
        | BinSearch(sid, _, _, v) -> collectSurfaceIds v |> Set.add sid

    let private collectAllExprs (model: Model) : Expr list =
        [ yield model.Result
          for kv in model.Accums |> Map.toSeq do
              let _, def = kv
              yield def.Init
              yield def.Body
          for obs in model.Observers do
              yield obs.Expr ]

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
            accums |> Map.map (fun id def ->
                let bodyRefs = collectAccumRefs def.Body
                bodyRefs |> Set.remove id)
        let mutable visited = Set.empty
        let mutable order = []
        let rec visit id =
            if visited |> Set.contains id then ()
            else
                visited <- visited |> Set.add id
                match deps |> Map.tryFind id with
                | Some depSet ->
                    for dep in depSet do
                        if accums |> Map.containsKey dep then visit dep
                | None -> ()
                order <- (id, accums.[id]) :: order
        for id in accums |> Map.toSeq |> Seq.map fst do
            visit id
        List.rev order

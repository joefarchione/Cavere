namespace Cavere.Core

open System

/// Adjoint (reverse-mode) kernel compilation.
/// Generates a kernel with forward tape + backward adjoint propagation.
/// Use when numDiffVars >> numAccums (e.g. calibration with 10+ parameters).
[<RequireQualifiedAccess>]
module CompilerAdjoint =

    /// Pre-computed partial derivatives for adjoint propagation.
    type AdjointInfo = {
        DiffVars: int[]
        SortedAccums: (int * AccumDef) list
        /// ∂body_j/∂AccumRef(k) for each (j, k) pair
        BodyPartialsAccum: Map<int * int, Expr>
        /// ∂body_j/∂DiffVar(i) for each (j, dvIdx) pair
        BodyPartialsDiffVar: Map<int * int, Expr>
        /// ∂init_j/∂DiffVar(i) for each (j, dvIdx) pair
        InitPartialsDiffVar: Map<int * int, Expr>
        /// ∂result/∂AccumRef(k) for each accum k
        ResultPartialsAccum: Map<int, Expr>
        /// ∂result/∂DiffVar(i) for each DiffVar i
        ResultPartialsDiffVar: Map<int, Expr>
    }

    /// Analyze a model to compute all partial derivatives needed for adjoint mode.
    let computeAdjointInfo (model: Model) : AdjointInfo =
        let diffVars = CompilerDiff.collectModelDiffVars model
        let sorted = CompilerCommon.sortAccums model.Accums
        let accumIds = sorted |> List.map fst

        let bodyPartialsAccum =
            [
                for (jId, jDef) in sorted do
                    for kId in accumIds do
                        (jId, kId), CompilerDiff.partialDiffAccumRef jDef.Body kId
            ]
            |> Map.ofList

        let bodyPartialsDiffVar =
            [
                for (jId, jDef) in sorted do
                    for dvIdx in diffVars do
                        (jId, dvIdx), CompilerDiff.partialDiffDiffVar jDef.Body dvIdx
            ]
            |> Map.ofList

        let initPartialsDiffVar =
            [
                for (jId, jDef) in sorted do
                    for dvIdx in diffVars do
                        (jId, dvIdx), CompilerDiff.partialDiffDiffVar jDef.Init dvIdx
            ]
            |> Map.ofList

        let resultPartialsAccum =
            [
                for aId in accumIds do
                    aId, CompilerDiff.partialDiffAccumRef model.Result aId
            ]
            |> Map.ofList

        let resultPartialsDiffVar =
            [
                for dvIdx in diffVars do
                    dvIdx, CompilerDiff.partialDiffDiffVar model.Result dvIdx
            ]
            |> Map.ofList

        {
            DiffVars = diffVars
            SortedAccums = sorted
            BodyPartialsAccum = bodyPartialsAccum
            BodyPartialsDiffVar = bodyPartialsDiffVar
            InitPartialsDiffVar = initPartialsDiffVar
            ResultPartialsAccum = resultPartialsAccum
            ResultPartialsDiffVar = resultPartialsDiffVar
        }

    // ══════════════════════════════════════════════════════════════════
    // Kernel Emitter
    // ══════════════════════════════════════════════════════════════════

    let private emitFoldAdjointKernel (model: Model) (layout: SurfaceLayout) (info: AdjointInfo) =

        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        let emit expr = CompilerCommon.emitExpr layout expr

        let numAccums = info.SortedAccums.Length
        let numDiffVars = info.DiffVars.Length
        let accumIds = info.SortedAccums |> List.map fst

        line ""
        line "    public static void FoldAdjoint("
        line "        Index1D index,"
        line "        ArrayView1D<float, Stride1D.Dense> output,"
        line "        ArrayView1D<float, Stride1D.Dense> surfaces,"
        line "        ArrayView1D<float, Stride1D.Dense> tape,"
        line "        ArrayView1D<float, Stride1D.Dense> adjointOut,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int indexOffset)"
        line "    {"
        line "        int idx = (int)index;"
        line "        int seed = idx + indexOffset;"

        if numAccums > 0 then
            linef "        int tapeBase = idx * %d * steps;" numAccums

        line "        int t;"
        line ""

        // Declare and init accum variables
        for (aId, aDef) in info.SortedAccums do
            linef "        float accum_%d = %s;" aId (emit aDef.Init)

        if numAccums > 0 then
            line ""
            line "        // Forward pass with tape"
            line "        for (t = 0; t < steps; t++)"
            line "        {"

            // Store in tape before update
            for i, (aId, _) in info.SortedAccums |> List.indexed do
                linef "            tape[tapeBase + %d * steps + t] = accum_%d;" i aId

            CompilerCodegen.emitNormals sb model "seed"
            CompilerCodegen.emitUniforms sb model "seed"
            CompilerCodegen.emitBernoullis sb model "seed"
            CompilerCodegen.emitAccumUpdates sb layout info.SortedAccums

            line "        }"

        line ""
        CompilerCodegen.emitWithPreamble sb layout model.Result (sprintf "        output[idx] = %s;")
        line ""

        // Backward pass — seed adjoints from result
        line "        // Backward pass"

        for aId in accumIds do
            linef "        float adj_a_%d = %s;" aId (emit info.ResultPartialsAccum.[aId])

        for dvIdx in info.DiffVars do
            linef "        float adj_dv_%d = %s;" dvIdx (emit info.ResultPartialsDiffVar.[dvIdx])

        if numAccums > 0 then
            line ""
            line "        for (t = steps - 1; t >= 0; t--)"
            line "        {"

            // Restore accum values from tape
            for i, (aId, _) in info.SortedAccums |> List.indexed do
                linef "            accum_%d = tape[tapeBase + %d * steps + t];" aId i

            // Recompute normals/uniforms/bernoullis (deterministic hash — same seed + t)
            CompilerCodegen.emitNormals sb model "seed"
            CompilerCodegen.emitUniforms sb model "seed"
            CompilerCodegen.emitBernoullis sb model "seed"

            line ""

            // Accumulate DiffVar adjoints: adj_dv_i += sum_j(adj_a_j * ∂body_j/∂DV_i)
            for dvIdx in info.DiffVars do
                for (aId, _) in info.SortedAccums do
                    match info.BodyPartialsDiffVar.[(aId, dvIdx)] with
                    | Const 0.0f -> ()
                    | pd -> linef "            adj_dv_%d += adj_a_%d * %s;" dvIdx aId (emit pd)

            line ""

            // Propagate accum adjoints: adj_a_new = J^T * adj_a_old
            for aId in accumIds do
                linef "            float tmp_a_%d = adj_a_%d;" aId aId

            for kId in accumIds do
                let terms =
                    info.SortedAccums
                    |> List.choose (fun (jId, _) ->
                        match info.BodyPartialsAccum.[(jId, kId)] with
                        | Const 0.0f -> None
                        | pd -> Some $"tmp_a_{jId} * {emit pd}")

                if terms.IsEmpty then
                    linef "            adj_a_%d = 0.0f;" kId
                else
                    linef "            adj_a_%d = %s;" kId (terms |> String.concat " + ")

            line "        }"

        line ""

        // Add contribution from init: adj_dv_i += sum_j(adj_a_j * ∂init_j/∂DV_i)
        for dvIdx in info.DiffVars do
            for (aId, _) in info.SortedAccums do
                match info.InitPartialsDiffVar.[(aId, dvIdx)] with
                | Const 0.0f -> ()
                | pd -> linef "        adj_dv_%d += adj_a_%d * %s;" dvIdx aId (emit pd)

        line ""

        // Write adjoint outputs
        for i, dvIdx in info.DiffVars |> Array.indexed do
            linef "        adjointOut[idx * %d + %d] = adj_dv_%d;" numDiffVars i dvIdx

        line "    }"
        sb.ToString()

    // ══════════════════════════════════════════════════════════════════
    // Source Generation
    // ══════════════════════════════════════════════════════════════════

    let generateSource (model: Model) (layout: SurfaceLayout) (info: AdjointInfo) : string =
        [
            CompilerCodegen.csPreamble
            CompilerCodegen.emitHelpers layout
            emitFoldAdjointKernel model layout info
            "}"
        ]
        |> String.concat "\n"

    // ══════════════════════════════════════════════════════════════════
    // Build Pipeline
    // ══════════════════════════════════════════════════════════════════

    let build (model: Model) : CompiledKernel * AdjointInfo =
        CompilerCommon.validate model
        let layout = CompilerCommon.layoutSurfaces model
        let info = computeAdjointInfo model
        let source = generateSource model layout info
        let assembly = CompilerRegular.compile source
        let kernelType = assembly.GetType("GeneratedKernel")

        {
            Assembly = assembly
            KernelType = kernelType
            SurfaceLayout = layout
            Model = model
        },
        info

    let buildSource (model: Model) : string * SurfaceLayout * AdjointInfo =
        let layout = CompilerCommon.layoutSurfaces model
        let info = computeAdjointInfo model
        let source = generateSource model layout info
        source, layout, info

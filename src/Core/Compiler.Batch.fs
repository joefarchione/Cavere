namespace Cavere.Core

open System

/// Batch kernel compilation: FoldBatch, FoldBatchWatch.
/// All batch elements share the same random scenarios (seeded by scenarioIdx).
[<RequireQualifiedAccess>]
module CompilerBatch =

    // ══════════════════════════════════════════════════════════════════
    // Kernel Emitters
    // ══════════════════════════════════════════════════════════════════

    let private emitFoldBatchKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void FoldBatch("
        line "        Index1D index,"
        line "        ArrayView1D<float, Stride1D.Dense> output,"
        line "        ArrayView1D<float, Stride1D.Dense> surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int numSims, int indexOffset)"
        line "    {"
        line "        int idx = (int)index;"
        line "        int batchIdx = idx / numSims;"
        line "        int scenarioIdx = idx % numSims;"
        line "        int seed = scenarioIdx + indexOffset;"
        line "        int t;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "        for (t = 0; t < steps; t++)"
        line "        {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        line "        }"
        linef "        output[idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "    }"
        sb.ToString()

    let private emitFoldBatchWatchKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void FoldBatchWatch("
        line "        Index1D index,"
        line "        ArrayView1D<float, Stride1D.Dense> output,"
        line "        ArrayView1D<float, Stride1D.Dense> surfaces,"
        line "        ArrayView1D<float, Stride1D.Dense> obsBuffer,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int numSims, int numObs, int interval, int totalThreads, int indexOffset)"
        line "    {"
        line "        int idx = (int)index;"
        line "        int batchIdx = idx / numSims;"
        line "        int scenarioIdx = idx % numSims;"
        line "        int seed = scenarioIdx + indexOffset;"
        line "        int t;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "        for (t = 0; t < steps; t++)"
        line "        {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        CompilerCodegen.emitObserverRecording sb model layout "obsBuffer" "totalThreads"
        line "        }"
        linef "        output[idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "    }"
        sb.ToString()

    // ══════════════════════════════════════════════════════════════════
    // Source Generation
    // ══════════════════════════════════════════════════════════════════

    let generateSource (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) : string =
        [ CompilerCodegen.csPreamble
          CompilerCodegen.emitHelpers layout
          emitFoldBatchKernel model layout sortedAccums
          emitFoldBatchWatchKernel model layout sortedAccums
          "}" ]
        |> String.concat "\n"

    // ══════════════════════════════════════════════════════════════════
    // Build Pipeline
    // ══════════════════════════════════════════════════════════════════

    let build (model: Model) : CompiledKernel =
        CompilerCommon.validate model
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateSource model layout sorted
        let assembly = CompilerRegular.compile source
        let kernelType = assembly.GetType("GeneratedKernel")
        { Assembly = assembly; KernelType = kernelType; SurfaceLayout = layout; Model = model }

    let buildSource (model: Model) : string * SurfaceLayout =
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateSource model layout sorted
        source, layout

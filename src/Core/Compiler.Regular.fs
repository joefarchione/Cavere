namespace Cavere.Core

open System
open System.IO
open System.Reflection
open Microsoft.CodeAnalysis
open Microsoft.CodeAnalysis.CSharp

/// Compiled kernel result.
type CompiledKernel = {
    Assembly: Assembly
    KernelType: Type
    SurfaceLayout: SurfaceLayout
    Model: Model
}

/// Regular (non-batch) kernel compilation: Fold, FoldWatch, Scan.
[<RequireQualifiedAccess>]
module CompilerRegular =

    // ══════════════════════════════════════════════════════════════════
    // Kernel Emitters
    // ══════════════════════════════════════════════════════════════════

    let private emitFoldKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void Fold("
        line "        Index1D index,"
        line "        ArrayView1D<float, Stride1D.Dense> output,"
        line "        ArrayView1D<float, Stride1D.Dense> surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int indexOffset)"
        line "    {"
        line "        int idx = (int)index;"
        line "        int seed = idx + indexOffset;"
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

    let private emitFoldWatchKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void FoldWatch("
        line "        Index1D index,"
        line "        ArrayView1D<float, Stride1D.Dense> output,"
        line "        ArrayView1D<float, Stride1D.Dense> surfaces,"
        line "        ArrayView1D<float, Stride1D.Dense> obsBuffer,"

        line
            "        int steps, int normalCount, int uniformCount, int bernoulliCount, int numSims, int numObs, int interval, int indexOffset)"

        line "    {"
        line "        int idx = (int)index;"
        line "        int seed = idx + indexOffset;"
        line "        int t;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "        for (t = 0; t < steps; t++)"
        line "        {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        CompilerCodegen.emitObserverRecording sb model layout "obsBuffer" "numSims"
        line "        }"
        linef "        output[idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "    }"
        sb.ToString()

    let private emitScanKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void Scan("
        line "        Index1D index,"
        line "        ArrayView1D<float, Stride1D.Dense> output,"
        line "        ArrayView1D<float, Stride1D.Dense> surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int numSims, int indexOffset)"
        line "    {"
        line "        int idx = (int)index;"
        line "        int seed = idx + indexOffset;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "        for (int t = 0; t < steps; t++)"
        line "        {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        linef "            output[t * numSims + idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "        }"
        line "    }"
        sb.ToString()

    let private emitFoldAntitheticKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        let resultExpr = CompilerCommon.emitExpr layout model.Result
        line ""
        line "    public static void FoldAntithetic("
        line "        Index1D index,"
        line "        ArrayView1D<float, Stride1D.Dense> output,"
        line "        ArrayView1D<float, Stride1D.Dense> surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int indexOffset)"
        line "    {"
        line "        int idx = (int)index;"
        line "        int seed = idx + indexOffset;"
        line "        int t;"
        // Path 1: normal z
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "        for (t = 0; t < steps; t++)"
        line "        {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        line "        }"
        linef "        float result1 = %s;" resultExpr
        line ""
        // Path 2: negated z (re-init accums, same seed)
        line "        // Antithetic path: negated normals"

        for (id, def) in sortedAccums do
            linef "        accum_%d = %s;" id (CompilerCommon.emitExpr layout def.Init)

        line ""
        line "        for (t = 0; t < steps; t++)"
        line "        {"
        CompilerCodegen.emitNormalsNegated sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        line "        }"
        linef "        float result2 = %s;" resultExpr
        line ""
        line "        output[idx] = (result1 + result2) * 0.5f;"
        line "    }"
        sb.ToString()

    // ══════════════════════════════════════════════════════════════════
    // Source Generation
    // ══════════════════════════════════════════════════════════════════

    let generateSource (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) : string =
        [
            CompilerCodegen.csPreamble
            CompilerCodegen.emitHelpers layout
            emitFoldKernel model layout sortedAccums
            emitFoldWatchKernel model layout sortedAccums
            emitScanKernel model layout sortedAccums
            "}"
        ]
        |> String.concat "\n"

    // ══════════════════════════════════════════════════════════════════
    // Roslyn Compilation
    // ══════════════════════════════════════════════════════════════════

    let compile (source: string) : Assembly =
        let tree = CSharpSyntaxTree.ParseText(source)

        let refs =
            [|
                typeof<obj>.Assembly.Location
                typeof<MathF>.Assembly.Location
                typeof<Console>.Assembly.Location
                typeof<ILGPU.Index1D>.Assembly.Location
                typeof<ILGPU.Runtime.Accelerator>.Assembly.Location
            |]
            |> Array.append (
                AppDomain.CurrentDomain.GetAssemblies()
                |> Array.filter (fun a -> not a.IsDynamic && not (String.IsNullOrWhiteSpace a.Location))
                |> Array.map (fun a -> a.Location)
            )
            |> Array.distinct
            |> Array.map (fun loc -> MetadataReference.CreateFromFile(loc) :> MetadataReference)

        let opts =
            CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
                .WithAllowUnsafe(true)
                .WithOptimizationLevel(OptimizationLevel.Release)

        let compilation = CSharpCompilation.Create("GeneratedKernel", [| tree |], refs, opts)

        let ms = new MemoryStream()
        let result = compilation.Emit(ms)

        if not result.Success then
            ms.Dispose()

            let errors =
                result.Diagnostics
                |> Seq.filter (fun d -> d.Severity = DiagnosticSeverity.Error)
                |> Seq.map (fun d -> d.ToString())
                |> String.concat "\n"

            failwithf "Roslyn compilation failed:\n%s\n\nSource:\n%s" errors source

        ms.Seek(0L, SeekOrigin.Begin) |> ignore
        let context = System.Runtime.Loader.AssemblyLoadContext("GeneratedKernel", true)
        let asm = context.LoadFromStream(ms)
        ms.Dispose()
        asm

    // ══════════════════════════════════════════════════════════════════
    // Build Pipeline
    // ══════════════════════════════════════════════════════════════════

    let build (model: Model) : CompiledKernel =
        CompilerCommon.validate model
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateSource model layout sorted
        let assembly = compile source
        let kernelType = assembly.GetType("GeneratedKernel")

        {
            Assembly = assembly
            KernelType = kernelType
            SurfaceLayout = layout
            Model = model
        }

    let buildSource (model: Model) : string * SurfaceLayout =
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateSource model layout sorted
        source, layout

    // ── Antithetic variant ──────────────────────────────────────────

    let generateAntitheticSource (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) : string =
        [
            CompilerCodegen.csPreamble
            CompilerCodegen.emitHelpers layout
            emitFoldAntitheticKernel model layout sortedAccums
            "}"
        ]
        |> String.concat "\n"

    let buildAntithetic (model: Model) : CompiledKernel =
        CompilerCommon.validate model
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateAntitheticSource model layout sorted
        let assembly = compile source
        let kernelType = assembly.GetType("GeneratedKernel")

        {
            Assembly = assembly
            KernelType = kernelType
            SurfaceLayout = layout
            Model = model
        }

    let buildAntitheticSource (model: Model) : string * SurfaceLayout =
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateAntitheticSource model layout sorted
        source, layout

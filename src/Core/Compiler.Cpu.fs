namespace Cavere.Core

open System
open System.IO
open System.Reflection
open Microsoft.CodeAnalysis
open Microsoft.CodeAnalysis.CSharp

/// Native CPU kernel compilation: Parallel.For wrappers, no ILGPU dependency.
[<RequireQualifiedAccess>]
module CompilerCpu =

    let private csPreamble =
        CompilerCodegen.dedent
            """
        using System;
        using System.Threading.Tasks;

        public static class GeneratedKernel
        {"""

    // ══════════════════════════════════════════════════════════════════
    // Kernel Emitters
    // ══════════════════════════════════════════════════════════════════

    let private emitFoldKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void Fold("
        line "        float[] output, float[] surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount)"
        line "    {"
        line "        int numScenarios = output.Length;"
        line "        Parallel.For(0, numScenarios, idx => {"
        line "            int seed = idx;"
        line "            int t;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "            for (t = 0; t < steps; t++)"
        line "            {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        line "            }"
        linef "            output[idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "        });"
        line "    }"
        sb.ToString()

    let private emitFoldWatchKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void FoldWatch("
        line "        float[] output, float[] obsBuffer, float[] surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount,"
        line "        int numSims, int numObs, int interval)"
        line "    {"
        line "        Parallel.For(0, numSims, idx => {"
        line "            int seed = idx;"
        line "            int t;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "            for (t = 0; t < steps; t++)"
        line "            {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        CompilerCodegen.emitObserverRecording sb model layout "obsBuffer" "numSims"
        line "            }"
        linef "            output[idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "        });"
        line "    }"
        sb.ToString()

    let private emitScanKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void Scan("
        line "        float[] output, float[] surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int numSims)"
        line "    {"
        line "        Parallel.For(0, numSims, idx => {"
        line "            int seed = idx;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "            for (int t = 0; t < steps; t++)"
        line "            {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        linef "                output[t * numSims + idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "            }"
        line "        });"
        line "    }"
        sb.ToString()

    let private emitFoldBatchKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void FoldBatch("
        line "        float[] output, float[] surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount, int numSims)"
        line "    {"
        line "        int totalThreads = output.Length;"
        line "        Parallel.For(0, totalThreads, idx => {"
        line "            int batchIdx = idx / numSims;"
        line "            int scenarioIdx = idx % numSims;"
        line "            int seed = scenarioIdx;"
        line "            int t;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "            for (t = 0; t < steps; t++)"
        line "            {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        line "            }"
        linef "            output[idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "        });"
        line "    }"
        sb.ToString()

    let private emitFoldBatchWatchKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        line ""
        line "    public static void FoldBatchWatch("
        line "        float[] output, float[] obsBuffer, float[] surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount,"
        line "        int numSims, int numObs, int interval, int totalThreads)"
        line "    {"
        line "        Parallel.For(0, totalThreads, idx => {"
        line "            int batchIdx = idx / numSims;"
        line "            int scenarioIdx = idx % numSims;"
        line "            int seed = scenarioIdx;"
        line "            int t;"
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "            for (t = 0; t < steps; t++)"
        line "            {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        CompilerCodegen.emitObserverRecording sb model layout "obsBuffer" "totalThreads"
        line "            }"
        linef "            output[idx] = %s;" (CompilerCommon.emitExpr layout model.Result)
        line "        });"
        line "    }"
        sb.ToString()

    let private emitFoldAdjointKernel (model: Model) (layout: SurfaceLayout) (info: CompilerAdjoint.AdjointInfo) =

        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        let emit expr = CompilerCommon.emitExpr layout expr

        let numAccums = info.SortedAccums.Length
        let numDiffVars = info.DiffVars.Length
        let accumIds = info.SortedAccums |> List.map fst

        line ""
        line "    public static void FoldAdjoint("
        line "        float[] output, float[] surfaces, float[] tape,"
        line "        float[] adjointOut,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount)"
        line "    {"
        line "        int numScenarios = output.Length;"
        line "        Parallel.For(0, numScenarios, idx => {"
        line "            int seed = idx;"

        if numAccums > 0 then
            linef "            int tapeBase = idx * %d * steps;" numAccums

        line "            int t;"
        line ""

        // Declare and init accum variables
        for (aId, aDef) in info.SortedAccums do
            linef "            float accum_%d = %s;" aId (emit aDef.Init)

        if numAccums > 0 then
            line ""
            line "            // Forward pass with tape"
            line "            for (t = 0; t < steps; t++)"
            line "            {"

            // Store in tape before update
            for i, (aId, _) in info.SortedAccums |> List.indexed do
                linef "                tape[tapeBase + %d * steps + t] = accum_%d;" i aId

            CompilerCodegen.emitNormals sb model "seed"
            CompilerCodegen.emitUniforms sb model "seed"
            CompilerCodegen.emitBernoullis sb model "seed"
            CompilerCodegen.emitAccumUpdates sb layout info.SortedAccums

            line "            }"

        line ""
        linef "            output[idx] = %s;" (emit model.Result)
        line ""

        // Backward pass
        line "            // Backward pass"

        for aId in accumIds do
            linef "            float adj_a_%d = %s;" aId (emit info.ResultPartialsAccum.[aId])

        for dvIdx in info.DiffVars do
            linef "            float adj_dv_%d = %s;" dvIdx (emit info.ResultPartialsDiffVar.[dvIdx])

        if numAccums > 0 then
            line ""
            line "            for (t = steps - 1; t >= 0; t--)"
            line "            {"

            // Restore accum values from tape
            for i, (aId, _) in info.SortedAccums |> List.indexed do
                linef "                accum_%d = tape[tapeBase + %d * steps + t];" aId i

            CompilerCodegen.emitNormals sb model "seed"
            CompilerCodegen.emitUniforms sb model "seed"
            CompilerCodegen.emitBernoullis sb model "seed"

            line ""

            // Accumulate DiffVar adjoints
            for dvIdx in info.DiffVars do
                for (aId, _) in info.SortedAccums do
                    match info.BodyPartialsDiffVar.[(aId, dvIdx)] with
                    | Const 0.0f -> ()
                    | pd -> linef "                adj_dv_%d += adj_a_%d * %s;" dvIdx aId (emit pd)

            line ""

            // Propagate accum adjoints
            for aId in accumIds do
                linef "                float tmp_a_%d = adj_a_%d;" aId aId

            for kId in accumIds do
                let terms =
                    info.SortedAccums
                    |> List.choose (fun (jId, _) ->
                        match info.BodyPartialsAccum.[(jId, kId)] with
                        | Const 0.0f -> None
                        | pd -> Some $"tmp_a_{jId} * {emit pd}")

                if terms.IsEmpty then
                    linef "                adj_a_%d = 0.0f;" kId
                else
                    linef "                adj_a_%d = %s;" kId (terms |> String.concat " + ")

            line "            }"

        line ""

        // Add contribution from init
        for dvIdx in info.DiffVars do
            for (aId, _) in info.SortedAccums do
                match info.InitPartialsDiffVar.[(aId, dvIdx)] with
                | Const 0.0f -> ()
                | pd -> linef "            adj_dv_%d += adj_a_%d * %s;" dvIdx aId (emit pd)

        line ""

        // Write adjoint outputs
        for i, dvIdx in info.DiffVars |> Array.indexed do
            linef "            adjointOut[idx * %d + %d] = adj_dv_%d;" numDiffVars i dvIdx

        line "        });"
        line "    }"
        sb.ToString()

    let private emitFoldAntitheticKernel (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let sb = Text.StringBuilder()
        let line (s: string) = sb.AppendLine(s) |> ignore
        let linef fmt = Printf.kprintf line fmt
        let resultExpr = CompilerCommon.emitExpr layout model.Result
        line ""
        line "    public static void FoldAntithetic("
        line "        float[] output, float[] surfaces,"
        line "        int steps, int normalCount, int uniformCount, int bernoulliCount)"
        line "    {"
        line "        int numScenarios = output.Length;"
        line "        Parallel.For(0, numScenarios, idx => {"
        line "            int seed = idx;"
        line "            int t;"
        // Path 1: normal z
        CompilerCodegen.emitAccumDecls sb layout sortedAccums
        line ""
        line "            for (t = 0; t < steps; t++)"
        line "            {"
        CompilerCodegen.emitNormals sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        line "            }"
        linef "            float result1 = %s;" resultExpr
        line ""
        // Path 2: negated z (re-init accums, same seed)
        line "            // Antithetic path: negated normals"

        for (id, def) in sortedAccums do
            linef "            accum_%d = %s;" id (CompilerCommon.emitExpr layout def.Init)

        line ""
        line "            for (t = 0; t < steps; t++)"
        line "            {"
        CompilerCodegen.emitNormalsNegated sb model "seed"
        CompilerCodegen.emitUniforms sb model "seed"
        CompilerCodegen.emitBernoullis sb model "seed"
        CompilerCodegen.emitAccumUpdates sb layout sortedAccums
        line "            }"
        linef "            float result2 = %s;" resultExpr
        line ""
        line "            output[idx] = (result1 + result2) * 0.5f;"
        line "        });"
        line "    }"
        sb.ToString()

    // ══════════════════════════════════════════════════════════════════
    // Source Generation
    // ══════════════════════════════════════════════════════════════════

    let generateSource (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) : string =
        [
            csPreamble
            CompilerCodegen.emitCpuHelpers layout
            emitFoldKernel model layout sortedAccums
            emitFoldWatchKernel model layout sortedAccums
            emitScanKernel model layout sortedAccums
            "}"
        ]
        |> String.concat "\n"

    let generateBatchSource (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) : string =
        [
            csPreamble
            CompilerCodegen.emitCpuHelpers layout
            emitFoldBatchKernel model layout sortedAccums
            emitFoldBatchWatchKernel model layout sortedAccums
            "}"
        ]
        |> String.concat "\n"

    let generateAdjointSource (model: Model) (layout: SurfaceLayout) (info: CompilerAdjoint.AdjointInfo) : string =
        [
            csPreamble
            CompilerCodegen.emitCpuHelpers layout
            emitFoldAdjointKernel model layout info
            "}"
        ]
        |> String.concat "\n"

    // ══════════════════════════════════════════════════════════════════
    // Roslyn Compilation (no ILGPU references)
    // ══════════════════════════════════════════════════════════════════

    let private stableHash (s: string) =
        let mutable h = 0x811c9dc5u

        for c in s do
            h <- h ^^^ uint32 c
            h <- h * 0x01000193u

        sprintf "%08X" h

    let compile (source: string) : Assembly =
#if DEBUG
        let tempDir = Path.Combine(Path.GetTempPath(), "Cavere")
        Directory.CreateDirectory(tempDir) |> ignore
        let srcPath = Path.Combine(tempDir, $"GeneratedCpuKernel_{stableHash source}.cs")
        File.WriteAllText(srcPath, source)
        System.Diagnostics.Debug.WriteLine($"[Cavere] CPU kernel source: {srcPath}")
        let sourceText = Microsoft.CodeAnalysis.Text.SourceText.From(source, System.Text.Encoding.UTF8)
        let tree = CSharpSyntaxTree.ParseText(sourceText, path = srcPath)
#else
        let tree = CSharpSyntaxTree.ParseText(source)
#endif

        let refs =
            [|
                typeof<obj>.Assembly.Location
                typeof<MathF>.Assembly.Location
                typeof<Console>.Assembly.Location
                typeof<System.Threading.Tasks.Parallel>.Assembly.Location
            |]
            |> Array.append (
                AppDomain.CurrentDomain.GetAssemblies()
                |> Array.filter (fun a -> not a.IsDynamic && not (String.IsNullOrWhiteSpace a.Location))
                |> Array.map (fun a -> a.Location)
            )
            |> Array.distinct
            |> Array.map (fun loc -> MetadataReference.CreateFromFile(loc) :> MetadataReference)

#if DEBUG
        let opts =
            CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
                .WithAllowUnsafe(true)
                .WithOptimizationLevel(OptimizationLevel.Debug)
#else
        let opts =
            CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
                .WithAllowUnsafe(true)
                .WithOptimizationLevel(OptimizationLevel.Release)
#endif

        let compilation = CSharpCompilation.Create("GeneratedCpuKernel", [| tree |], refs, opts)

#if DEBUG
        let ms = new MemoryStream()
        let pdbStream = new MemoryStream()
        let emitOpts = Emit.EmitOptions(debugInformationFormat = Emit.DebugInformationFormat.PortablePdb)
        let result = compilation.Emit(ms, pdbStream, options = emitOpts)
#else
        let ms = new MemoryStream()
        let result = compilation.Emit(ms)
#endif

        if not result.Success then
            ms.Dispose()

            let errors =
                result.Diagnostics
                |> Seq.filter (fun d -> d.Severity = DiagnosticSeverity.Error)
                |> Seq.map (fun d -> d.ToString())
                |> String.concat "\n"

            failwithf "Roslyn CPU compilation failed:\n%s\n\nSource:\n%s" errors source

        ms.Seek(0L, SeekOrigin.Begin) |> ignore
        let context = System.Runtime.Loader.AssemblyLoadContext("GeneratedCpuKernel", true)

#if DEBUG
        pdbStream.Seek(0L, SeekOrigin.Begin) |> ignore
        let asm = context.LoadFromStream(ms, pdbStream)
        pdbStream.Dispose()
#else
        let asm = context.LoadFromStream(ms)
#endif

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

    let buildBatch (model: Model) : CompiledKernel =
        CompilerCommon.validate model
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateBatchSource model layout sorted
        let assembly = compile source
        let kernelType = assembly.GetType("GeneratedKernel")

        {
            Assembly = assembly
            KernelType = kernelType
            SurfaceLayout = layout
            Model = model
        }

    let buildBatchSource (model: Model) : string * SurfaceLayout =
        let layout = CompilerCommon.layoutSurfaces model
        let sorted = CompilerCommon.sortAccums model.Accums
        let source = generateBatchSource model layout sorted
        source, layout

    let buildAdjoint (model: Model) : CompiledKernel * CompilerAdjoint.AdjointInfo =
        CompilerCommon.validate model
        let layout = CompilerCommon.layoutSurfaces model
        let info = CompilerAdjoint.computeAdjointInfo model
        let source = generateAdjointSource model layout info
        let assembly = compile source
        let kernelType = assembly.GetType("GeneratedKernel")

        {
            Assembly = assembly
            KernelType = kernelType
            SurfaceLayout = layout
            Model = model
        },
        info

    let buildAdjointSource (model: Model) : string * SurfaceLayout * CompilerAdjoint.AdjointInfo =
        let layout = CompilerCommon.layoutSurfaces model
        let info = CompilerAdjoint.computeAdjointInfo model
        let source = generateAdjointSource model layout info
        source, layout, info

    // ── Antithetic variant ──────────────────────────────────────────

    let generateAntitheticSource (model: Model) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) : string =
        [
            csPreamble
            CompilerCodegen.emitCpuHelpers layout
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

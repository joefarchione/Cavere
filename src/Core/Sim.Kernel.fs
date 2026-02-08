namespace Cavere.Core

open System.Runtime.CompilerServices

/// Kernel compilation and caching.
/// Compiles Model → CompiledKernel via Roslyn/ILGPU pipeline.
/// Kernels are cached per model instance (weak reference).
module Kernel =

    // ── Caches ───────────────────────────────────────────────────────
    let private regularCache = ConditionalWeakTable<Model, CompiledKernel>()
    let private batchCache = ConditionalWeakTable<Model, CompiledKernel>()
    let private cpuCache = ConditionalWeakTable<Model, CompiledKernel>()
    let private cpuBatchCache = ConditionalWeakTable<Model, CompiledKernel>()

    // ── Auto-expand Dual/HyperDual ────────────────────────────────────

    let private autoExpandDiffVars (m: Model) : Model =
        let allVars = CompilerDiff.collectModelDiffVars m
        let hyperVars = CompilerDiff.collectModelHyperDualVars m

        if allVars.Length = 0 then
            m
        else
            let m1, _ = CompilerDiff.transformDual m

            let expanded =
                if hyperVars.Length = 0 then
                    m1
                else
                    CompilerDiff.transformHyperDualSelective hyperVars m1 m

            {
                expanded with
                    Result = Symbolic.fullySimplify expanded.Result
                    Accums =
                        expanded.Accums
                        |> Map.map (fun _ def -> {
                            Init = Symbolic.fullySimplify def.Init
                            Body = Symbolic.fullySimplify def.Body
                        })
                    Observers =
                        expanded.Observers
                        |> List.map (fun obs -> {
                            obs with
                                Expr = Symbolic.fullySimplify obs.Expr
                        })
            }

    // ── Validation ───────────────────────────────────────────────────

    let private requireSimple (m: Model) =
        if m.BatchSize > 0 then
            failwith "Kernel.compile requires a non-batch model. Use Kernel.compileBatch for models with batchInput."

    let private requireBatch (m: Model) =
        if m.BatchSize = 0 then
            failwith "Kernel.compileBatch requires a batch model. Use Kernel.compile for models without batchInput."

    // ── Regular (non-batch) ──────────────────────────────────────────

    /// Generate C# source for a non-batch model (auto-expands Dual/HyperDual).
    let source (m: Model) : string =
        requireSimple m
        let expanded = autoExpandDiffVars m
        Compiler.buildSource expanded |> fst

    /// Compile a non-batch model to a kernel (auto-expands Dual/HyperDual, cached).
    let compile (m: Model) : CompiledKernel =
        requireSimple m
        let expanded = autoExpandDiffVars m
        regularCache.GetValue(expanded, ConditionalWeakTable<_, _>.CreateValueCallback(Compiler.build))

    // ── Batch ────────────────────────────────────────────────────────

    /// Generate C# source for a batch model.
    let sourceBatch (m: Model) : string =
        requireBatch m
        Compiler.buildBatchSource m |> fst

    /// Compile a batch model to a kernel (cached).
    let compileBatch (m: Model) : CompiledKernel =
        requireBatch m
        batchCache.GetValue(m, ConditionalWeakTable<_, _>.CreateValueCallback(Compiler.buildBatch))

    /// Compile a batch model to a kernel, validating batch size matches.
    let compileBatchFor (m: Model) (numBatch: int) : CompiledKernel =
        requireBatch m

        if m.BatchSize <> numBatch then
            failwithf "Model batch size (%d) does not match expected (%d)" m.BatchSize numBatch

        batchCache.GetValue(m, ConditionalWeakTable<_, _>.CreateValueCallback(Compiler.buildBatch))

    // ── CPU (native, no ILGPU) ─────────────────────────────────────

    /// Generate C# source for native CPU execution (auto-expands Dual/HyperDual).
    let sourceCpu (m: Model) : string =
        requireSimple m
        let expanded = autoExpandDiffVars m
        Compiler.buildCpuSource expanded |> fst

    /// Compile a non-batch model to a native CPU kernel (auto-expands Dual/HyperDual, cached).
    let compileCpu (m: Model) : CompiledKernel =
        requireSimple m
        let expanded = autoExpandDiffVars m
        cpuCache.GetValue(expanded, ConditionalWeakTable<_, _>.CreateValueCallback(Compiler.buildCpu))

    /// Compile a batch model to a native CPU kernel (cached).
    let compileBatchCpu (m: Model) : CompiledKernel =
        requireBatch m
        cpuBatchCache.GetValue(m, ConditionalWeakTable<_, _>.CreateValueCallback(Compiler.buildBatchCpu))

    /// Compile a batch model to a native CPU kernel, validating batch size matches.
    let compileBatchCpuFor (m: Model) (numBatch: int) : CompiledKernel =
        requireBatch m

        if m.BatchSize <> numBatch then
            failwithf "Model batch size (%d) does not match expected (%d)" m.BatchSize numBatch

        cpuBatchCache.GetValue(m, ConditionalWeakTable<_, _>.CreateValueCallback(Compiler.buildBatchCpu))

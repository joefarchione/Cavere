namespace Cavere.Core

open System.Runtime.CompilerServices

/// Kernel compilation and caching.
/// Compiles Model → CompiledKernel via Roslyn/ILGPU pipeline.
/// Kernels are cached per model instance (weak reference).
module Kernel =

    // ── Caches ───────────────────────────────────────────────────────
    let private regularCache = ConditionalWeakTable<Model, CompiledKernel>()
    let private batchCache = ConditionalWeakTable<Model, CompiledKernel>()

    // ── Validation ───────────────────────────────────────────────────

    let private requireSimple (m: Model) =
        if m.BatchSize > 0 then
            failwith "Kernel.compile requires a non-batch model. Use Kernel.compileBatch for models with batchInput."

    let private requireBatch (m: Model) =
        if m.BatchSize = 0 then
            failwith "Kernel.compileBatch requires a batch model. Use Kernel.compile for models without batchInput."

    // ── Regular (non-batch) ──────────────────────────────────────────

    /// Generate C# source for a non-batch model.
    let source (m: Model) : string =
        requireSimple m
        Compiler.buildSource m |> fst

    /// Compile a non-batch model to a kernel (cached).
    let compile (m: Model) : CompiledKernel =
        requireSimple m
        regularCache.GetValue(m, ConditionalWeakTable<_, _>.CreateValueCallback(Compiler.build))

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

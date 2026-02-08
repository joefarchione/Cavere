namespace Cavere.Core

/// Unified compiler module for backwards compatibility.
/// For explicit control, use:
/// - CompilerCommon: surface layout, expression emission, topo sort
/// - CompilerCodegen: C# templates, dynamic emitters
/// - CompilerRegular: Fold, FoldWatch, Scan kernels
/// - CompilerBatch: FoldBatch, FoldBatchWatch kernels
module Compiler =

    // ── Common utilities ──
    let layoutSurfaces = CompilerCommon.layoutSurfaces
    let packSurfaces = CompilerCommon.packSurfaces
    let emitExpr = CompilerCommon.emitExpr
    let sortAccums = CompilerCommon.sortAccums

    // ── Regular (non-batch) compilation ──
    let generateSource = CompilerRegular.generateSource
    let compile = CompilerRegular.compile
    let build = CompilerRegular.build
    let buildSource = CompilerRegular.buildSource

    // ── Batch compilation ──
    let generateBatchSource = CompilerBatch.generateSource
    let buildBatch = CompilerBatch.build
    let buildBatchSource = CompilerBatch.buildSource

    // ── Adjoint compilation ──
    let buildAdjoint = CompilerAdjoint.build
    let buildAdjointSource = CompilerAdjoint.buildSource

    // ── CPU (native, no ILGPU) compilation ──
    let buildCpu = CompilerCpu.build
    let buildCpuSource = CompilerCpu.buildSource
    let buildBatchCpu = CompilerCpu.buildBatch
    let buildBatchCpuSource = CompilerCpu.buildBatchSource
    let buildAdjointCpu = CompilerCpu.buildAdjoint
    let buildAdjointCpuSource = CompilerCpu.buildAdjointSource

    // ── Antithetic compilation ──
    let buildAntithetic = CompilerRegular.buildAntithetic
    let buildAntitheticSource = CompilerRegular.buildAntitheticSource
    let buildAntitheticCpu = CompilerCpu.buildAntithetic
    let buildAntitheticCpuSource = CompilerCpu.buildAntitheticSource

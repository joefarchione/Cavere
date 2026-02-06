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

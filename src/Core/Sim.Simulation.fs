namespace Cavere.Core

open System
open ILGPU
open ILGPU.Runtime

// ════════════════════════════════════════════════════════════════════════════
// Unified Simulation — single, multi-device, or pinned memory
// ════════════════════════════════════════════════════════════════════════════

type internal SimExec =
    | Single of ctx: Context * accel: Accelerator
    | Multi of DeviceSet
    | Pinned of ctx: Context * accel: Accelerator * pool: PinnedPool

type Simulation internal (exec: SimExec, numScenarios: int, steps: int) =
    member _.NumScenarios = numScenarios
    member _.Steps = steps
    member _.Accelerator =
        match exec with
        | Single(_, a) | Pinned(_, a, _) -> a
        | Multi ds -> ds.Accelerators.[0]
    member internal _.Exec = exec

    interface IDisposable with
        member _.Dispose() =
            match exec with
            | Single(ctx, a) -> a.Dispose(); ctx.Dispose()
            | Multi ds -> Device.disposeMulti ds
            | Pinned(ctx, a, pool) -> (pool :> IDisposable).Dispose(); a.Dispose(); ctx.Dispose()

module Simulation =

    let create (deviceType: DeviceType) (numScenarios: int) (steps: int) : Simulation =
        let ctx, accel = Device.create deviceType
        new Simulation(Single(ctx, accel), numScenarios, steps)

    let createMulti (config: MultiDeviceConfig) (numScenarios: int) (steps: int) : Simulation =
        let deviceSet = Device.createMulti config
        new Simulation(Multi deviceSet, numScenarios, steps)

    let createPinned (deviceType: DeviceType) (numScenarios: int) (steps: int) : Simulation =
        let ctx, accel = Device.create deviceType
        let pool = new PinnedPool(accel)
        new Simulation(Pinned(ctx, accel, pool), numScenarios, steps)

    // ── Source export (delegates to Kernel) ───────────────────────────

    let source (m: Model) : string = Kernel.source m

    // ── Compile (delegates to Kernel) ────────────────────────────────

    let compile (m: Model) : CompiledKernel = Kernel.compile m

    // ── Kernel execution ─────────────────────────────────────────────

    let foldKernel (sim: Simulation) (kernel: CompiledKernel) : float32[] =
        match sim.Exec with
        | Single(_, accel) -> Engine.fold accel kernel sim.NumScenarios sim.Steps 0
        | Multi ds -> Engine.foldMulti ds kernel sim.NumScenarios sim.Steps
        | Pinned(_, accel, pool) -> Engine.foldPinned accel pool kernel sim.NumScenarios sim.Steps 0

    let foldWatchKernel (sim: Simulation) (kernel: CompiledKernel) (freq: Frequency) : float32[] * WatchResult =
        let accel =
            match sim.Exec with
            | Single(_, a) | Pinned(_, a, _) -> a
            | Multi _ -> failwith "foldWatch not supported for multi-device simulation"
        let interval = Watcher.intervalOf freq sim.Steps
        let finals, obsBuf = Engine.foldWatch accel kernel sim.NumScenarios sim.Steps interval 0
        let numObs = Watcher.numObs interval sim.Steps
        finals, { Buffer = obsBuf; Observers = kernel.Model.Observers; NumObs = numObs; NumPaths = sim.NumScenarios }

    let scanKernel (sim: Simulation) (kernel: CompiledKernel) : float32[,] =
        let accel =
            match sim.Exec with
            | Single(_, a) | Pinned(_, a, _) -> a
            | Multi _ -> failwith "scan not supported for multi-device simulation"
        Engine.scan accel kernel sim.NumScenarios sim.Steps 0

    // ── Convenience (compile + run) ──────────────────────────────────

    let fold (sim: Simulation) (m: Model) : float32[] =
        foldKernel sim (Kernel.compile m)

    let foldWatch (sim: Simulation) (m: Model) (freq: Frequency) : float32[] * WatchResult =
        foldWatchKernel sim (Kernel.compile m) freq

    let scan (sim: Simulation) (m: Model) : float32[,] =
        scanKernel sim (Kernel.compile m)

    // ── Adjoint execution ───────────────────────────────────────────

    /// Compute value + per-scenario adjoints for all Dual/HyperDual vars via reverse mode.
    /// Returns (values: float32[], adjoints: float32[numScenarios, numDiffVars]).
    let foldAdjoint (sim: Simulation) (m: Model) : float32[] * float32[,] =
        let accel =
            match sim.Exec with
            | Single(_, a) | Pinned(_, a, _) -> a
            | Multi _ -> failwith "foldAdjoint not supported for multi-device simulation"
        let kernel, info = CompilerAdjoint.build m
        Engine.foldAdjoint accel kernel info sim.NumScenarios sim.Steps 0

// ════════════════════════════════════════════════════════════════════════════
// Batch Simulation — for models using batchInput
// ════════════════════════════════════════════════════════════════════════════

type BatchSimulation(deviceType: DeviceType, numBatch: int, numScenarios: int, steps: int) =
    let ctx, accel = Device.create deviceType
    let totalThreads = numBatch * numScenarios

    member _.DeviceType = deviceType
    member _.NumBatch = numBatch
    member _.NumScenarios = numScenarios
    member _.TotalThreads = totalThreads
    member _.Steps = steps
    member _.Accelerator = accel

    interface IDisposable with
        member _.Dispose() =
            accel.Dispose()
            ctx.Dispose()

module BatchSimulation =

    let create (deviceType: DeviceType) (numBatch: int) (numScenarios: int) (steps: int) : BatchSimulation =
        new BatchSimulation(deviceType, numBatch, numScenarios, steps)

    // ── Source export (delegates to Kernel) ───────────────────────────

    let source (m: Model) : string = Kernel.sourceBatch m

    // ── Compile (delegates to Kernel) ────────────────────────────────

    let compile (m: Model) (sim: BatchSimulation) : CompiledKernel =
        Kernel.compileBatchFor m sim.NumBatch

    // ── Kernel execution ─────────────────────────────────────────────

    let foldKernel (sim: BatchSimulation) (kernel: CompiledKernel) : float32[] =
        Engine.foldBatch sim.Accelerator kernel sim.NumBatch sim.NumScenarios sim.Steps 0

    let foldWatchKernel (sim: BatchSimulation) (kernel: CompiledKernel) (freq: Frequency) : float32[] * WatchResult =
        let interval = Watcher.intervalOf freq sim.Steps
        let finals, obsBuf =
            Engine.foldBatchWatch sim.Accelerator kernel sim.NumBatch sim.NumScenarios sim.Steps interval 0
        let numObs = Watcher.numObs interval sim.Steps
        finals, { Buffer = obsBuf; Observers = kernel.Model.Observers; NumObs = numObs; NumPaths = sim.TotalThreads }

    // ── Convenience (compile + run) ──────────────────────────────────

    let fold (sim: BatchSimulation) (m: Model) : float32[] =
        foldKernel sim (Kernel.compileBatchFor m sim.NumBatch)

    let foldWatch (sim: BatchSimulation) (m: Model) (freq: Frequency) : float32[] * WatchResult =
        foldWatchKernel sim (Kernel.compileBatchFor m sim.NumBatch) freq

    let foldMeans (sim: BatchSimulation) (m: Model) : float32[] =
        let raw = fold sim m
        Array.init sim.NumBatch (fun b ->
            let mutable sum = 0.0f
            for i in 0 .. sim.NumScenarios - 1 do
                sum <- sum + raw.[b * sim.NumScenarios + i]
            sum / float32 sim.NumScenarios)

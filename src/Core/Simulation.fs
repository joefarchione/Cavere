namespace Cavere.Core

open System
open System.Runtime.CompilerServices
open ILGPU.Runtime

// ════════════════════════════════════════════════════════════════════════════
// Simple Simulation — for non-batch models
// ════════════════════════════════════════════════════════════════════════════

type Simulation(deviceType: DeviceType, numScenarios: int, steps: int) =
    let ctx, accel = Device.create deviceType

    member _.DeviceType = deviceType
    member _.NumScenarios = numScenarios
    member _.Steps = steps
    member _.Accelerator = accel

    interface IDisposable with
        member _.Dispose() =
            accel.Dispose()
            ctx.Dispose()

module Simulation =

    let private requireSimple (m: Model) =
        if m.BatchSize > 0 then
            failwith "Simulation requires a non-batch model. Use BatchSimulation for models with batchInput."

    let private regularCache = ConditionalWeakTable<Model, CompiledKernel>()

    let create (deviceType: DeviceType) (numScenarios: int) (steps: int) : Simulation =
        new Simulation(deviceType, numScenarios, steps)

    // ── Source export ────────────────────────────────────────────────────

    let source (m: Model) : string =
        requireSimple m
        Compiler.buildSource m |> fst

    // ── Compile once, run many ──────────────────────────────────────────

    let compile (m: Model) : CompiledKernel =
        requireSimple m
        regularCache.GetValue(m, ConditionalWeakTable<_,_>.CreateValueCallback(Compiler.build))

    let foldKernel (sim: Simulation) (kernel: CompiledKernel) : float32[] =
        Engine.fold sim.Accelerator kernel sim.NumScenarios sim.Steps

    let foldWatchKernel (sim: Simulation) (kernel: CompiledKernel) (freq: Frequency) : float32[] * WatchResult =
        let interval = Watcher.intervalOf freq sim.Steps
        let finals, obsBuf = Engine.foldWatch sim.Accelerator kernel sim.NumScenarios sim.Steps interval
        let numObs = Watcher.numObs interval sim.Steps
        finals, { Buffer = obsBuf; Observers = kernel.Model.Observers; NumObs = numObs; NumPaths = sim.NumScenarios }

    let scanKernel (sim: Simulation) (kernel: CompiledKernel) : float32[,] =
        Engine.scan sim.Accelerator kernel sim.NumScenarios sim.Steps

    // ── Convenience (compile + run) ─────────────────────────────────────

    let fold (sim: Simulation) (m: Model) : float32[] =
        foldKernel sim (compile m)

    let foldWatch (sim: Simulation) (m: Model) (freq: Frequency) : float32[] * WatchResult =
        foldWatchKernel sim (compile m) freq

    let scan (sim: Simulation) (m: Model) : float32[,] =
        scanKernel sim (compile m)

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

    let private requireBatch (m: Model) (sim: BatchSimulation) =
        if m.BatchSize = 0 then
            failwith "BatchSimulation requires a batch model. Use Simulation for models without batchInput."
        if m.BatchSize <> sim.NumBatch then
            failwithf "Model batch size (%d) does not match simulation (%d)" m.BatchSize sim.NumBatch

    let create (deviceType: DeviceType) (numBatch: int) (numScenarios: int) (steps: int) : BatchSimulation =
        new BatchSimulation(deviceType, numBatch, numScenarios, steps)

    // ── Source export ────────────────────────────────────────────────────

    let source (m: Model) : string =
        Compiler.buildBatchSource m |> fst

    let private batchCache = ConditionalWeakTable<Model, CompiledKernel>()

    // ── Compile once, run many ──────────────────────────────────────────

    let compile (m: Model) (sim: BatchSimulation) : CompiledKernel =
        requireBatch m sim
        batchCache.GetValue(m, ConditionalWeakTable<_,_>.CreateValueCallback(Compiler.buildBatch))

    let foldKernel (sim: BatchSimulation) (kernel: CompiledKernel) : float32[] =
        Engine.foldBatch sim.Accelerator kernel sim.NumBatch sim.NumScenarios sim.Steps

    let foldWatchKernel (sim: BatchSimulation) (kernel: CompiledKernel) (freq: Frequency) : float32[] * WatchResult =
        let interval = Watcher.intervalOf freq sim.Steps
        let finals, obsBuf =
            Engine.foldBatchWatch sim.Accelerator kernel sim.NumBatch sim.NumScenarios sim.Steps interval
        let numObs = Watcher.numObs interval sim.Steps
        finals, { Buffer = obsBuf; Observers = kernel.Model.Observers; NumObs = numObs; NumPaths = sim.TotalThreads }

    // ── Convenience (compile + run) ─────────────────────────────────────

    let fold (sim: BatchSimulation) (m: Model) : float32[] =
        foldKernel sim (compile m sim)

    let foldWatch (sim: BatchSimulation) (m: Model) (freq: Frequency) : float32[] * WatchResult =
        foldWatchKernel sim (compile m sim) freq

    let foldMeans (sim: BatchSimulation) (m: Model) : float32[] =
        let raw = fold sim m
        Array.init sim.NumBatch (fun b ->
            let mutable sum = 0.0f
            for i in 0 .. sim.NumScenarios - 1 do
                sum <- sum + raw.[b * sim.NumScenarios + i]
            sum / float32 sim.NumScenarios)

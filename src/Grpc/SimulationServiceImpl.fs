namespace Cavere.Grpc

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Grpc.Core
open Cavere.Core

/// Stored compiled kernel with its model metadata.
type StoredKernel = {
    Kernel: CompiledKernel
    Model: Model
    Steps: int
    IsBatch: bool
}

type SimulationServiceImpl() =
    inherit SimulationService.SimulationServiceBase()

    let sessions = ConcurrentDictionary<string, Simulation>()
    let kernels = ConcurrentDictionary<string, StoredKernel>()

    let chunkSize = 10_000

    // ── Helpers ────────────────────────────────────────────────────────

    let packWatchResponse (finals: float32[]) (wr: WatchResult) : WatchResponse =
        let resp = WatchResponse()
        resp.Finals.AddRange(finals)
        for spec in wr.Observers do
            let data = Watcher.values spec.Name wr
            let od = ObserverData(Name = spec.Name, NumObs = wr.NumObs, NumPaths = wr.NumPaths)
            for obs in 0 .. wr.NumObs - 1 do
                for path in 0 .. wr.NumPaths - 1 do
                    od.Values.Add(data.[obs, path])
            resp.Observers.Add(od)
        resp

    let runFold device deviceCount usePinned numScenarios steps (m: Model) =
        if deviceCount > 1 then
            let config = { DeviceType = device; DeviceCount = deviceCount }
            use sim = Simulation.createMulti config numScenarios steps
            Simulation.fold sim m
        elif usePinned then
            use sim = Simulation.createPinned device numScenarios steps
            Simulation.fold sim m
        else
            use sim = Simulation.create device numScenarios steps
            Simulation.fold sim m

    // ── Simple simulation RPCs ────────────────────────────────────────

    override _.Fold(request: SimulationRequest, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            let values = runFold device request.DeviceCount request.UsePinned request.NumScenarios steps m
            let resp = FoldResponse()
            resp.Values.AddRange(values)
            resp)

    override _.FoldWatch(request: WatchRequest, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            let freq = ModelFactory.mapFrequency request.Frequency
            use sim = Simulation.create device request.NumScenarios steps
            let finals, wr = Simulation.foldWatch sim m freq
            packWatchResponse finals wr)

    override _.Scan(request: SimulationRequest, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            use sim = Simulation.create device request.NumScenarios steps
            let data = Simulation.scan sim m
            let resp = ScanResponse(Steps = Array2D.length1 data, NumScenarios = Array2D.length2 data)
            for i in 0 .. Array2D.length1 data - 1 do
                for j in 0 .. Array2D.length2 data - 1 do
                    resp.Values.Add(data.[i, j])
            resp)

    override _.StreamScan(request: SimulationRequest, responseStream: IServerStreamWriter<ScanChunk>, _context: ServerCallContext) =
        task {
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            use sim = Simulation.create device request.NumScenarios steps
            let data = Simulation.scan sim m
            let totalSteps = Array2D.length1 data
            let numScenarios = Array2D.length2 data
            let mutable startStep = 0
            while startStep < totalSteps do
                let count = min chunkSize (totalSteps - startStep)
                let chunk = ScanChunk(StartStep = startStep, StepCount = count, NumScenarios = numScenarios)
                for i in startStep .. startStep + count - 1 do
                    for j in 0 .. numScenarios - 1 do
                        chunk.Values.Add(data.[i, j])
                do! responseStream.WriteAsync(chunk)
                startStep <- startStep + count
        } :> Task

    // ── Batch simulation RPCs ─────────────────────────────────────────

    override _.BatchFold(request: BatchRequest, _context: ServerCallContext) =
        Task.FromResult(
            let batchValues = request.BatchValues |> Seq.toArray
            let m = ModelFactory.buildBatchModel request.Model batchValues
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            let values =
                if request.DeviceCount > 1 then
                    let config = { DeviceType = device; DeviceCount = request.DeviceCount }
                    let deviceSet = Device.createMulti config
                    try
                        let kernel = Kernel.compileBatchFor m batchValues.Length
                        Engine.foldBatchMulti deviceSet kernel batchValues.Length request.NumScenarios steps
                    finally
                        Device.disposeMulti deviceSet
                else
                    use sim = BatchSimulation.create device batchValues.Length request.NumScenarios steps
                    BatchSimulation.fold sim m
            let resp = FoldResponse()
            resp.Values.AddRange(values)
            resp)

    override _.BatchFoldWatch(request: BatchWatchRequest, _context: ServerCallContext) =
        Task.FromResult(
            let batchValues = request.BatchValues |> Seq.toArray
            let m = ModelFactory.buildBatchModel request.Model batchValues
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            let freq = ModelFactory.mapFrequency request.Frequency
            use sim = BatchSimulation.create device batchValues.Length request.NumScenarios steps
            let finals, wr = BatchSimulation.foldWatch sim m freq
            packWatchResponse finals wr)

    override _.BatchFoldMeans(request: BatchRequest, _context: ServerCallContext) =
        Task.FromResult(
            let batchValues = request.BatchValues |> Seq.toArray
            let m = ModelFactory.buildBatchModel request.Model batchValues
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            use sim = BatchSimulation.create device batchValues.Length request.NumScenarios steps
            let values = BatchSimulation.foldMeans sim m
            let resp = FoldResponse()
            resp.Values.AddRange(values)
            resp)

    override _.StreamBatchFold(request: BatchRequest, responseStream: IServerStreamWriter<FoldChunk>, _context: ServerCallContext) =
        task {
            let batchValues = request.BatchValues |> Seq.toArray
            let m = ModelFactory.buildBatchModel request.Model batchValues
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            use sim = BatchSimulation.create device batchValues.Length request.NumScenarios steps
            let values = BatchSimulation.fold sim m
            let mutable startIdx = 0
            while startIdx < values.Length do
                let count = min chunkSize (values.Length - startIdx)
                let chunk = FoldChunk(StartIndex = startIdx, Count = count)
                for i in startIdx .. startIdx + count - 1 do
                    chunk.Values.Add(values.[i])
                do! responseStream.WriteAsync(chunk)
                startIdx <- startIdx + count
        } :> Task

    // ── Automatic differentiation RPCs ────────────────────────────────

    override _.FoldDiff(request: DiffRequest, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            let freq = ModelFactory.mapFrequency request.Frequency
            let mode = ModelFactory.mapDiffMode request.DiffMode
            let transformed =
                match mode with
                | Cavere.Core.DiffMode.DualMode ->
                    let m', _ = CompilerDiff.transformDual m in m'
                | Cavere.Core.DiffMode.HyperDualMode diagonal ->
                    let m', _ = CompilerDiff.transformHyperDual diagonal m in m'
                | _ -> failwith "FoldDiff only supports Dual and HyperDual modes. Use FoldAdjoint for Adjoint mode."
            use sim = Simulation.create device request.NumScenarios steps
            let finals, wr = Simulation.foldWatch sim transformed freq
            packWatchResponse finals wr)

    override _.FoldAdjoint(request: AdjointRequest, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            use sim = Simulation.create device request.NumScenarios steps
            let values, adjoints = Simulation.foldAdjoint sim m
            let numDiffVars = Array2D.length2 adjoints
            let diffVars = CompilerDiff.collectModelDiffVars m
            let resp = AdjointResponse(NumScenarios = request.NumScenarios, NumDiffVars = numDiffVars)
            resp.Values.AddRange(values)
            resp.DiffVarIndices.AddRange(diffVars)
            for s in 0 .. request.NumScenarios - 1 do
                for d in 0 .. numDiffVars - 1 do
                    resp.Adjoints.Add(adjoints.[s, d])
            resp)

    override _.Recommend(request: ModelSpec, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request
            let mode, desc = CompilerDiff.recommend m
            let hasDv = CompilerDiff.hasDiffVars m
            let resp = RecommendResponse(HasDiffVars = hasDv, Description = desc)
            match mode with
            | Some Cavere.Core.DiffMode.DualMode -> resp.RecommendedMode <- Cavere.Grpc.DiffMode.DiffDual
            | Some (Cavere.Core.DiffMode.HyperDualMode true) -> resp.RecommendedMode <- Cavere.Grpc.DiffMode.DiffHyperdualDiag
            | Some (Cavere.Core.DiffMode.HyperDualMode false) -> resp.RecommendedMode <- Cavere.Grpc.DiffMode.DiffHyperdualFull
            | Some Cavere.Core.DiffMode.AdjointMode -> resp.RecommendedMode <- Cavere.Grpc.DiffMode.DiffAdjoint
            | _ -> ()
            resp)

    // ── Kernel management RPCs ──────────────────────────────────────

    override _.CompileKernel(request: CompileKernelRequest, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let kernel, source =
                if request.Batch then
                    Kernel.compileBatch m, Kernel.sourceBatch m
                else
                    Kernel.compile m, Kernel.source m
            let id = Guid.NewGuid().ToString("N")
            kernels.[id] <- { Kernel = kernel; Model = m; Steps = steps; IsBatch = request.Batch }
            KernelResponse(KernelId = id, CsharpSource = source))

    override _.FoldKernel(request: KernelRunRequest, _context: ServerCallContext) =
        Task.FromResult(
            match kernels.TryGetValue(request.KernelId) with
            | false, _ -> failwith $"Kernel not found: {request.KernelId}"
            | true, stored ->
                let device = ModelFactory.mapDeviceType request.Device
                let values =
                    if request.DeviceCount > 1 then
                        let config = { DeviceType = device; DeviceCount = request.DeviceCount }
                        let deviceSet = Device.createMulti config
                        try
                            Engine.foldMulti deviceSet stored.Kernel request.NumScenarios stored.Steps
                        finally
                            Device.disposeMulti deviceSet
                    elif request.UsePinned then
                        let ctx, accel = Device.create device
                        try
                            use pool = new PinnedPool(accel)
                            Engine.foldPinned accel pool stored.Kernel request.NumScenarios stored.Steps 0
                        finally
                            accel.Dispose()
                            ctx.Dispose()
                    else
                        let ctx, accel = Device.create device
                        try
                            Engine.fold accel stored.Kernel request.NumScenarios stored.Steps 0
                        finally
                            accel.Dispose()
                            ctx.Dispose()
                let resp = FoldResponse()
                resp.Values.AddRange(values)
                resp)

    override _.FoldWatchKernel(request: KernelWatchRequest, _context: ServerCallContext) =
        Task.FromResult(
            match kernels.TryGetValue(request.KernelId) with
            | false, _ -> failwith $"Kernel not found: {request.KernelId}"
            | true, stored ->
                let device = ModelFactory.mapDeviceType request.Device
                let freq = ModelFactory.mapFrequency request.Frequency
                use sim = Simulation.create device request.NumScenarios stored.Steps
                let finals, wr = Simulation.foldWatchKernel sim stored.Kernel freq
                packWatchResponse finals wr)

    override _.ScanKernel(request: KernelRunRequest, _context: ServerCallContext) =
        Task.FromResult(
            match kernels.TryGetValue(request.KernelId) with
            | false, _ -> failwith $"Kernel not found: {request.KernelId}"
            | true, stored ->
                let device = ModelFactory.mapDeviceType request.Device
                use sim = Simulation.create device request.NumScenarios stored.Steps
                let data = Simulation.scanKernel sim stored.Kernel
                let resp = ScanResponse(Steps = Array2D.length1 data, NumScenarios = Array2D.length2 data)
                for i in 0 .. Array2D.length1 data - 1 do
                    for j in 0 .. Array2D.length2 data - 1 do
                        resp.Values.Add(data.[i, j])
                resp)

    override _.DestroyKernel(request: KernelId, _context: ServerCallContext) =
        Task.FromResult(
            kernels.TryRemove(request.Id) |> ignore
            Empty())

    // ── Session management ────────────────────────────────────────────

    override _.CreateSession(request: CreateSessionRequest, _context: ServerCallContext) =
        Task.FromResult(
            let device = ModelFactory.mapDeviceType request.Device
            let sim = Simulation.create device request.NumScenarios request.Steps
            let id = Guid.NewGuid().ToString("N")
            sessions.[id] <- sim
            SessionResponse(Id = id))

    override _.DestroySession(request: SessionId, _context: ServerCallContext) =
        Task.FromResult(
            match sessions.TryRemove(request.Id) with
            | true, sim -> (sim :> IDisposable).Dispose()
            | _ -> ()
            Empty())

    // ── Utility ───────────────────────────────────────────────────────

    override _.GetSource(request: ModelSpec, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request
            let src = Simulation.source m
            SourceResponse(CsharpSource = src))

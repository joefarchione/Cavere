namespace Cavere.Grpc

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Grpc.Core
open Cavere.Core

type SimulationServiceImpl() =
    inherit SimulationService.SimulationServiceBase()

    let sessions = ConcurrentDictionary<string, Simulation>()

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

    // ── Simple simulation RPCs ────────────────────────────────────────

    override _.Fold(request: SimulationRequest, _context: ServerCallContext) =
        Task.FromResult(
            let m = ModelFactory.buildModel request.Model
            let steps = ModelFactory.getSteps request.Model
            let device = ModelFactory.mapDeviceType request.Device
            use sim = Simulation.create device request.NumScenarios steps
            let values = Simulation.fold sim m
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
            use sim = BatchSimulation.create device batchValues.Length request.NumScenarios steps
            let values = BatchSimulation.fold sim m
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

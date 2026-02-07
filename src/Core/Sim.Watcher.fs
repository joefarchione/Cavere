namespace Cavere.Core

type Frequency =
    | Terminal
    | Daily
    | Weekly
    | Monthly
    | Quarterly
    | Annually

type WatchResult = {
    Buffer: float32[]
    Observers: ObserverSpec list
    NumObs: int
    NumPaths: int  // total paths in buffer (scenarios for simple, totalThreads for batch)
}

module Watcher =

    let intervalOf (freq: Frequency) (steps: int) : int =
        match freq with
        | Terminal -> steps
        | Daily -> 1
        | Weekly -> max 1 (steps / 52)
        | Monthly -> max 1 (steps / 12)
        | Quarterly -> max 1 (steps / 4)
        | Annually -> steps

    let numObs (interval: int) (steps: int) : int =
        (steps + interval - 1) / interval

    let values (name: string) (wr: WatchResult) : float32[,] =
        let spec = wr.Observers |> List.find (fun o -> o.Name = name)
        let result = Array2D.zeroCreate wr.NumObs wr.NumPaths
        for obs in 0 .. wr.NumObs - 1 do
            for path in 0 .. wr.NumPaths - 1 do
                result.[obs, path] <- wr.Buffer.[(spec.SlotIndex * wr.NumObs + obs) * wr.NumPaths + path]
        result

    let terminals (name: string) (wr: WatchResult) : float32[] =
        let spec = wr.Observers |> List.find (fun o -> o.Name = name)
        let lastObs = wr.NumObs - 1
        Array.init wr.NumPaths (fun path ->
            wr.Buffer.[(spec.SlotIndex * wr.NumObs + lastObs) * wr.NumPaths + path])

    let sliceObs (name: string) (obsIdx: int) (wr: WatchResult) : float32[] =
        let spec = wr.Observers |> List.find (fun o -> o.Name = name)
        Array.init wr.NumPaths (fun path ->
            wr.Buffer.[(spec.SlotIndex * wr.NumObs + obsIdx) * wr.NumPaths + path])

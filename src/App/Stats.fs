namespace Cavere.App

module Stats =
    let mean (arr: float32[]) = Array.average arr

    let stddev (arr: float32[]) =
        let m = mean arr
        arr |> Array.averageBy (fun x -> (x - m) * (x - m)) |> sqrt

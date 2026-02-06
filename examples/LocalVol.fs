// Local Volatility Surface — GBM with time/spot-dependent vol
//
// The volatility is looked up from a 2D surface (time x spot)
// via bilinear interpolation on the GPU.

module Cavere.Examples.LocalVol

open Cavere.Core
open Cavere.Generators

let run () =
    let steps = 252

    let times = [| 0.0f; 0.5f; 1.0f |]
    let spots = [| 80.0f; 100.0f; 120.0f |]
    let vols  = array2D [|
        [| 0.30f; 0.20f; 0.15f |]
        [| 0.28f; 0.19f; 0.14f |]
        [| 0.25f; 0.18f; 0.13f |]
    |]

    let localVolModel = model {
        let dt = (1.0f / 252.0f).C
        let! sid   = surface2d times spots vols steps
        let! z = normal
        let! stock = gbmLocalVol z sid 0.05f.C 100.0f.C dt
        let! df    = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }

    use sim = Simulation.create CPU 100_000 steps
    let results = Simulation.fold sim localVolModel

    printfn "  Simulations: %d" results.Length
    printfn "  Price: %.4f" (Array.average results)

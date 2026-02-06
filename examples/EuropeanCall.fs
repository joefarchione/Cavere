// European Call Option — GBM with constant vol and flat rate
//
// The simplest Monte Carlo pricing example. A single stock evolving
// under geometric Brownian motion, discounted at a flat rate.

module Cavere.Examples.EuropeanCall

open Cavere.Core
open Cavere.Generators

let run () =
    let sched = Schedule.constant (1.0f / 252.0f) 252

    let callModel = model {
        let! dt = scheduleDt sched
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        let! df    = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }

    use sim = Simulation.create CPU 100_000 sched.Steps
    let results = Simulation.fold sim callModel

    printfn "  Simulations: %d" results.Length
    printfn "  Price: %.4f" (Array.average results)

// Nested Stochastic Simulation — conditional expectations
//
// Run an outer simulation to generate stock paths, then for each
// outer path, build an inner model conditioned on the outer stock
// level and compute the conditional expected payoff. Building block
// for exposure profiles, CVA/DVA, and American option pricing.

module Cavere.Examples.NestedSimulation

open Cavere.Core
open Cavere.Generators

let run () =
    let sched = Schedule.constant (1.0f / 252.0f) 252

    let outerModel =
        model {
            let! dt = scheduleDt sched
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            do! observe "stock" stock
            return stock
        }

    let numOuter = 1_000
    use outerSim = Simulation.create CPU numOuter sched.Steps
    let _finals, watch = Simulation.foldWatch outerSim outerModel Quarterly

    let outerStocks = Watcher.sliceObs "stock" 0 watch

    let innerModel =
        model {
            let! dt = scheduleDt sched
            let! stock0 = batchInput outerStocks
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C stock0 dt
            let! df = decay 0.05f.C dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    let numScenarios = 500
    use innerSim = BatchSimulation.create CPU numOuter numScenarios 189
    let expectations = BatchSimulation.foldMeans innerSim innerModel

    printfn "  Outer paths: %d" numOuter
    printfn "  Inner scenarios per batch element: %d" numScenarios
    printfn "  Mean conditional expectation: %.4f" (Array.average expectations)

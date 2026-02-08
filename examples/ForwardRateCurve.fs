// Forward Rate Curve — term-structure-aware pricing
//
// Instead of a flat rate, use a time-varying forward rate curve.
// The rate is interpolated from a 1D surface at each time step,
// affecting both the stock drift and the discount factor.

module Cavere.Examples.ForwardRateCurve

open Cavere.Core
open Cavere.Generators
open Cavere.Generators.Rates

let run () =
    let sched = Schedule.constant (1.0f / 252.0f) 252

    let tenors = [| 0.0f; 0.25f; 0.5f; 1.0f |]
    let zeroRates = [| 0.03f; 0.035f; 0.04f; 0.05f |]
    let fwdRates = linearForwards tenors zeroRates sched.Steps

    let callModel =
        model {
            let! dt = scheduleDt sched
            let! rateSid = surface1d fwdRates sched.Steps
            let! rate = interp1d rateSid TimeIndex
            let! z = normal
            let! stock = gbm z rate 0.20f.C 100.0f.C dt
            let! df = decay rate dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    use sim = Simulation.create CPU 100_000 sched.Steps
    let results = Simulation.fold sim callModel

    printfn "  Short rate: 3%%, Long rate: 5%%"
    printfn "  Simulations: %d" results.Length
    printfn "  Price: %.4f" (Array.average results)

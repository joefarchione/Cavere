// Heston Stochastic Volatility — stochastic on stochastic
//
// Two correlated evolving states: stock price and its instantaneous
// variance. The variance process feeds into the stock process, and
// both compile into a single GPU kernel.

module Cavere.Examples.HestonModel

open Cavere.Core
open Cavere.Generators

let run () =
    let sched = Schedule.constant (1.0f / 252.0f) 252

    let hestonCall = model {
        let! dt = scheduleDt sched
        let! z = normal
        let! stock = heston z
                        0.05f.C              // risk-free rate
                        0.04f.C              // initial variance (v0)
                        1.5f.C               // mean reversion speed (kappa)
                        0.04f.C              // long-run variance (theta)
                        0.3f.C               // vol of vol (xi)
                        (-0.7f).C            // correlation (rho)
                        100.0f.C             // spot
                        dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }

    use sim = Simulation.create CPU 100_000 sched.Steps
    let results = Simulation.fold sim hestonCall

    printfn "  Simulations: %d" results.Length
    printfn "  Price: %.4f" (Array.average results)

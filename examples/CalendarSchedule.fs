// Calendar Schedule — business day aware simulation
//
// Instead of uniform dt = 1/252, use actual calendar dates with
// weekends and holidays excluded. The dt varies per step to reflect
// real elapsed time (3/365 over weekends, 1/365 on normal days).

module Cavere.Examples.CalendarSchedule

open System
open Cavere.Core
open Cavere.Generators

let run () =
    let holidays =
        Set.ofList [
            DateTime(2024, 1, 1)
            DateTime(2024, 1, 15)
            DateTime(2024, 2, 19)
            DateTime(2024, 5, 27)
            DateTime(2024, 6, 19)
            DateTime(2024, 7, 4)
            DateTime(2024, 9, 2)
            DateTime(2024, 11, 28)
            DateTime(2024, 12, 25)
        ]

    let sched = Calendar.businessDays (DateTime(2024, 1, 2)) (DateTime(2024, 12, 31)) holidays

    printfn "  Business days: %d steps" sched.Steps
    printfn "  Total year fraction: %.4f" (Array.last sched.T)
    printfn "  Min dt: %.6f (%.1f calendar days)" (Array.min sched.Dt) (Array.min sched.Dt * 365.0f)
    printfn "  Max dt: %.6f (%.1f calendar days)" (Array.max sched.Dt) (Array.max sched.Dt * 365.0f)

    let callModel =
        model {
            let! dt = scheduleDt sched
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            let! df = decay 0.05f.C dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    use sim = Simulation.create CPU 100_000 sched.Steps
    let results = Simulation.fold sim callModel

    printfn "  Price: %.4f" (Array.average results)

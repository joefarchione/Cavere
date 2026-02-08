module Cavere.Tests.ScheduleTests

open System
open Xunit
open Cavere.Core
open Cavere.Generators

[<Fact>]
let ``Constant schedule has uniform dt`` () =
    let sched = Schedule.constant (1.0f / 252.0f) 252
    Assert.Equal(252, sched.Steps)
    Assert.Equal(252, sched.Dt.Length)
    sched.Dt |> Array.iter (fun v -> Assert.Equal(1.0f / 252.0f, v))

[<Fact>]
let ``Constant schedule yearFrac accumulates correctly`` () =
    let sched = Schedule.constant (1.0f / 252.0f) 252
    Assert.Equal(253, sched.T.Length)
    Assert.Equal(0.0f, sched.T.[0])
    Assert.InRange(sched.T.[252], 0.999f, 1.001f)

[<Fact>]
let ``BusinessDays excludes weekends`` () =
    // Mon 2024-01-01 to Fri 2024-01-12 — 10 weekdays (no holidays)
    let sched = Calendar.businessDays (DateTime(2024, 1, 1)) (DateTime(2024, 1, 12)) Set.empty
    Assert.Equal(9, sched.Steps) // 10 business days = 9 steps

[<Fact>]
let ``BusinessDays excludes holidays`` () =
    // Mon 2024-01-01 to Fri 2024-01-12, with Wed 2024-01-03 as holiday
    let holidays = Set.ofList [ DateTime(2024, 1, 3) ]
    let sched = Calendar.businessDays (DateTime(2024, 1, 1)) (DateTime(2024, 1, 12)) holidays
    Assert.Equal(8, sched.Steps) // 9 business days = 8 steps

[<Fact>]
let ``BusinessDays dt reflects calendar gaps`` () =
    // Mon-Fri week with no holidays: dt should be 1/365 for weekdays, 3/365 for Fri->Mon
    let sched = Calendar.businessDays (DateTime(2024, 1, 1)) (DateTime(2024, 1, 12)) Set.empty
    // Mon->Tue = 1/365
    Assert.InRange(sched.Dt.[0], 0.00273f, 0.00275f)
    // Fri->Mon = 3/365 (index 4: Fri Jan 5 -> Mon Jan 8)
    Assert.InRange(sched.Dt.[4], 0.00821f, 0.00823f)

[<Fact>]
let ``Lookup1D codegen emits direct array access`` () =
    let sched = Schedule.constant (1.0f / 252.0f) 10

    let m =
        model {
            let! dt = scheduleDt sched
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            return stock
        }

    let source, _ = Compiler.buildSource m
    Assert.Contains("surfaces[", source)

[<Fact>]
let ``Roslyn compiles Lookup1D source`` () =
    let sched = Schedule.constant (1.0f / 252.0f) 10

    let m =
        model {
            let! dt = scheduleDt sched
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            return stock
        }

    let kernel = Compiler.build m
    Assert.NotNull(kernel.Assembly)

[<Fact>]
let ``Fold with constant schedule matches const dt`` () =
    let sched = Schedule.constant (1.0f / 252.0f) 252

    let schedModel =
        model {
            let! dt = scheduleDt sched
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            let! df = decay 0.05f.C dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    let constModel =
        model {
            let dt = (1.0f / 252.0f).C
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            let! df = decay 0.05f.C dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    use sim = Simulation.create CPU 50_000 252
    let schedResults = Simulation.fold sim schedModel
    let constResults = Simulation.fold sim constModel
    let schedMean = schedResults |> Array.averageBy float |> float32
    let constMean = constResults |> Array.averageBy float |> float32
    // Tiny float precision difference: Const bakes via sprintf "%.8ff" (truncation),
    // Lookup1D reads full float32 from array. Results should be very close.
    Assert.InRange(float schedMean, float constMean - 0.01, float constMean + 0.01)

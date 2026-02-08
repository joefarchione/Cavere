namespace Cavere.Generators

open System
open Cavere.Core

/// Trading calendar utilities for business-day-aware schedules.
module Calendar =

    /// Build a schedule from business days between start and end dates.
    /// Excludes weekends and specified holidays. Each dt reflects actual
    /// calendar gap (3/365 over weekends, 1/365 on normal days).
    let businessDays (startDate: DateTime) (endDate: DateTime) (holidays: Set<DateTime>) : Schedule =
        let isBusinessDay (d: DateTime) =
            d.DayOfWeek <> DayOfWeek.Saturday
            && d.DayOfWeek <> DayOfWeek.Sunday
            && not (holidays |> Set.contains d.Date)

        let days =
            Seq.initInfinite (fun i -> startDate.AddDays(float i))
            |> Seq.takeWhile (fun d -> d <= endDate)
            |> Seq.filter isBusinessDay
            |> Seq.toArray

        let steps = days.Length - 1
        let dtArr = Array.init steps (fun i -> float32 (days.[i + 1] - days.[i]).TotalDays / 365.0f)

        let yfArr =
            let mutable acc = 0.0f

            Array.init (steps + 1) (fun i ->
                if i = 0 then
                    0.0f
                else
                    acc <- acc + dtArr.[i - 1]
                    acc)

        { Dt = dtArr; T = yfArr; Steps = steps }

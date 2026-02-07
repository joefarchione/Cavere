open Cavere.Core
open Cavere.Generators
open Cavere.App

[<EntryPoint>]
let main _ =
    let sched = Schedule.constant (1.0f / 252.0f) 252
    use sim = Simulation.create CPU 1000 sched.Steps

    // Forward rate curve: rates rise from 3% to 5% over the year
    let fwdRates = Array.init sched.Steps (fun i -> 0.03f + 0.02f * float32 i / float32 (sched.Steps - 1))

    let callModel =
        model {
            let! dt = scheduleDt sched
            let! rateSid = surface1d fwdRates sched.Steps
            let! rate = interp1d rateSid TimeIndex
            let! z = normal
            let! stock = gbm z rate 0.20f.C 100.0f.C dt
            let! df = decay rate dt
            do! observe "stock" stock
            do! observe "df" df
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    let finals, watch = Simulation.foldWatch sim callModel Monthly
    let stockPaths = Watcher.values "stock" watch
    let termDfs = Watcher.terminals "df" watch

    printfn "Discounted European Call (K=100) with forward rate curve:"
    printfn $"  Simulations: %d{finals.Length}"
    printfn $"  Price:  %.4f{Stats.mean finals}"
    printfn $"  StdDev: %.4f{Stats.stddev finals}"

    let header = [ 1..10 ] |> List.map (sprintf "  Sim %d") |> String.concat ""
    printfn $"\nMonthly Stock Prices (10 scenarios):\n       %s{header}"

    [ 0 .. Array2D.length1 stockPaths - 1 ]
    |> List.iter (fun m ->
        let row =
            [ 1..10 ]
            |> List.map (fun s -> sprintf "  %6.1f" stockPaths.[m, s])
            |> String.concat ""

        printfn $"  Month %2d{m + 1}%s{row}")

    printfn $"\nTerminal Discount Factor:\n  Mean: %.6f{Stats.mean termDfs}"
    0

module Cavere.Examples.Program

let private examples =
    [ "european-call",      "European Call (GBM, flat rate)",              EuropeanCall.run
      "local-vol",          "Local Volatility Surface",                   LocalVol.run
      "heston",             "Heston Stochastic Volatility",               HestonModel.run
      "custom-generator",   "Custom Generator (OU + Jump Diffusion)",     CustomGenerator.run
      "forward-curve",      "Forward Rate Curve",                         ForwardRateCurve.run
      "calendar",           "Calendar Schedule (business days)",           CalendarSchedule.run
      "nested",             "Nested Simulation (conditional expectations)", NestedSimulation.run
      "fia",                "Fixed Indexed Annuity (call spread crediting)", FixedIndexedAnnuity.run ]

[<EntryPoint>]
let main argv =
    match argv with
    | [| name |] ->
        match examples |> List.tryFind (fun (n, _, _) -> n = name) with
        | Some (_, title, run) ->
            printfn "%s" title
            run ()
        | None ->
            printfn "Unknown example: %s" name
            printfn "Available: %s" (examples |> List.map (fun (n, _, _) -> n) |> String.concat ", ")
    | _ ->
        examples |> List.iter (fun (_, title, run) ->
            printfn "\n--- %s ---" title
            run ())
    0

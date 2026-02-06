module Cavere.Grpc.Program

open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.DependencyInjection

[<EntryPoint>]
let main args =
    let builder = WebApplication.CreateBuilder(args)
    builder.Services.AddGrpc() |> ignore
    let app = builder.Build()
    app.MapGrpcService<SimulationServiceImpl>() |> ignore
    app.Run()
    0

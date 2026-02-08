namespace Cavere.Core

open System
open ILGPU
open ILGPU.Runtime
open ILGPU.Runtime.CPU
open ILGPU.Runtime.Cuda

type DeviceType =
    | CPU // Native compiled C# with Parallel.For — no ILGPU
    | GPU // ILGPU CUDA
    | Emulated // ILGPU CPU accelerator

type MultiDeviceConfig = {
    DeviceType: DeviceType
    DeviceCount: int
}

type DeviceSet = {
    Context: Context
    Accelerators: Accelerator[]
}

module Device =

    let create (deviceType: DeviceType) : Context * Accelerator =
        match deviceType with
        | CPU -> failwith "CPU mode does not use ILGPU — use Engine.foldCpu directly"
        | Emulated ->
            let ctx =
                Context.Create(fun builder ->
                    builder.CPU() |> ignore
                    builder.EnableAlgorithms() |> ignore)

            let accel = ctx.GetDevice<CPUDevice>(0).CreateAccelerator(ctx)
            ctx, accel
        | GPU ->
            let ctx =
                Context.Create(fun builder ->
                    builder.Cuda() |> ignore
                    builder.EnableAlgorithms() |> ignore)

            let accel = ctx.GetDevice<CudaDevice>(0).CreateAccelerator(ctx)
            ctx, accel

    let cudaDeviceCount () : int =
        try
            use ctx = Context.Create(fun builder -> builder.Cuda() |> ignore)
            ctx.Devices.Length
        with _ ->
            0

    let createMulti (config: MultiDeviceConfig) : DeviceSet =
        match config.DeviceType with
        | CPU -> failwith "CPU mode does not support multi-device — Parallel.For handles threading"
        | Emulated
        | GPU ->
            let ctx =
                Context.Create(fun builder ->
                    match config.DeviceType with
                    | Emulated -> builder.CPU() |> ignore
                    | GPU -> builder.Cuda() |> ignore
                    | CPU -> ()

                    builder.EnableAlgorithms() |> ignore)

            let accels =
                match config.DeviceType with
                | Emulated ->
                    Array.init config.DeviceCount (fun _ -> ctx.GetDevice<CPUDevice>(0).CreateAccelerator(ctx))
                | GPU ->
                    let deviceCount = ctx.Devices.Length

                    Array.init config.DeviceCount (fun i ->
                        ctx.GetDevice<CudaDevice>(i % deviceCount).CreateAccelerator(ctx))
                | CPU -> Array.empty

            { Context = ctx; Accelerators = accels }

    let disposeMulti (ds: DeviceSet) : unit =
        for accel in ds.Accelerators do
            accel.Dispose()

        ds.Context.Dispose()

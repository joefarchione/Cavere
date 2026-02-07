namespace Cavere.Core

open System
open ILGPU
open ILGPU.Runtime
open ILGPU.Runtime.CPU
open ILGPU.Runtime.Cuda

type DeviceType =
    | CPU
    | GPU

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
        let ctx =
            Context.Create(fun builder ->
                match deviceType with
                | CPU -> builder.CPU() |> ignore
                | GPU -> builder.Cuda() |> ignore
                builder.EnableAlgorithms() |> ignore)
        let accel =
            match deviceType with
            | CPU -> ctx.GetDevice<CPUDevice>(0).CreateAccelerator(ctx)
            | GPU -> ctx.GetDevice<CudaDevice>(0).CreateAccelerator(ctx)
        ctx, accel

    let cudaDeviceCount () : int =
        try
            use ctx = Context.Create(fun builder -> builder.Cuda() |> ignore)
            ctx.Devices.Length
        with _ -> 0

    let createMulti (config: MultiDeviceConfig) : DeviceSet =
        let ctx =
            Context.Create(fun builder ->
                match config.DeviceType with
                | CPU -> builder.CPU() |> ignore
                | GPU -> builder.Cuda() |> ignore
                builder.EnableAlgorithms() |> ignore)
        let accels =
            match config.DeviceType with
            | CPU ->
                Array.init config.DeviceCount (fun _ ->
                    ctx.GetDevice<CPUDevice>(0).CreateAccelerator(ctx))
            | GPU ->
                let deviceCount = ctx.Devices.Length
                Array.init config.DeviceCount (fun i ->
                    ctx.GetDevice<CudaDevice>(i % deviceCount).CreateAccelerator(ctx))
        { Context = ctx; Accelerators = accels }

    let disposeMulti (ds: DeviceSet) : unit =
        for accel in ds.Accelerators do
            accel.Dispose()
        ds.Context.Dispose()

namespace Cavere.Core

open ILGPU
open ILGPU.Runtime
open ILGPU.Runtime.CPU
open ILGPU.Runtime.Cuda

type DeviceType =
    | CPU
    | GPU

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

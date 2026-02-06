namespace Cavere.Core

open System
open System.Reflection
open ILGPU
open ILGPU.Runtime

module Engine =

    let private getMethod (kernel: CompiledKernel) (name: string) =
        kernel.KernelType.GetMethod(name, BindingFlags.Public ||| BindingFlags.Static)

    // ── Simple kernels ─────────────────────────────────────────────────

    let fold (accel: Accelerator) (kernel: CompiledKernel) (numScenarios: int) (steps: int) : float32[] =
        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)
        use output = accel.Allocate1D<float32>(numScenarios)

        let methodInfo = getMethod kernel "Fold"
        let k = accel.LoadAutoGroupedKernel(methodInfo)
        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(numScenarios),
            [| output.View :> obj
               surfBuf.View :> obj
               steps :> obj
               kernel.Model.NormalCount :> obj
               kernel.Model.UniformCount :> obj |])
        accel.Synchronize()
        output.GetAsArray1D()

    let foldWatch
        (accel: Accelerator) (kernel: CompiledKernel)
        (numScenarios: int) (steps: int) (interval: int)
        : float32[] * float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)
        use output = accel.Allocate1D<float32>(numScenarios)

        let numObs = (steps + interval - 1) / interval
        let numSlots = kernel.Model.Observers.Length
        let obsBufSize = max 1 (numSlots * numObs * numScenarios)
        use obsBuf = accel.Allocate1D<float32>(obsBufSize)

        let methodInfo = getMethod kernel "FoldWatch"
        let k = accel.LoadAutoGroupedKernel(methodInfo)
        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(numScenarios),
            [| output.View :> obj
               surfBuf.View :> obj
               obsBuf.View :> obj
               steps :> obj
               kernel.Model.NormalCount :> obj
               kernel.Model.UniformCount :> obj
               numScenarios :> obj
               numObs :> obj
               interval :> obj |])
        accel.Synchronize()
        output.GetAsArray1D(), obsBuf.GetAsArray1D()

    let scan (accel: Accelerator) (kernel: CompiledKernel) (numScenarios: int) (steps: int) : float32[,] =
        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)
        use output = accel.Allocate1D<float32>(numScenarios * steps)

        let methodInfo = getMethod kernel "Scan"
        let k = accel.LoadAutoGroupedKernel(methodInfo)
        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(numScenarios),
            [| output.View :> obj
               surfBuf.View :> obj
               steps :> obj
               kernel.Model.NormalCount :> obj
               kernel.Model.UniformCount :> obj
               numScenarios :> obj |])
        accel.Synchronize()

        let flat = output.GetAsArray1D()
        Array2D.init steps numScenarios (fun t s -> flat.[t * numScenarios + s])

    // ── Batch kernels ──────────────────────────────────────────────────

    let foldBatch
        (accel: Accelerator) (kernel: CompiledKernel)
        (numBatch: int) (numScenarios: int) (steps: int)
        : float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)

        let totalThreads = numBatch * numScenarios
        use output = accel.Allocate1D<float32>(totalThreads)

        let methodInfo = getMethod kernel "FoldBatch"
        let k = accel.LoadAutoGroupedKernel(methodInfo)
        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(totalThreads),
            [| output.View :> obj
               surfBuf.View :> obj
               steps :> obj
               kernel.Model.NormalCount :> obj
               kernel.Model.UniformCount :> obj
               numScenarios :> obj |])
        accel.Synchronize()
        output.GetAsArray1D()

    let foldBatchWatch
        (accel: Accelerator) (kernel: CompiledKernel)
        (numBatch: int) (numScenarios: int) (steps: int) (interval: int)
        : float32[] * float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)

        let totalThreads = numBatch * numScenarios
        use output = accel.Allocate1D<float32>(totalThreads)

        let numObs = (steps + interval - 1) / interval
        let numSlots = kernel.Model.Observers.Length
        let obsBufSize = max 1 (numSlots * numObs * totalThreads)
        use obsBuf = accel.Allocate1D<float32>(obsBufSize)

        let methodInfo = getMethod kernel "FoldBatchWatch"
        let k = accel.LoadAutoGroupedKernel(methodInfo)
        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(totalThreads),
            [| output.View :> obj
               surfBuf.View :> obj
               obsBuf.View :> obj
               steps :> obj
               kernel.Model.NormalCount :> obj
               kernel.Model.UniformCount :> obj
               numScenarios :> obj
               numObs :> obj
               interval :> obj
               totalThreads :> obj |])
        accel.Synchronize()
        output.GetAsArray1D(), obsBuf.GetAsArray1D()

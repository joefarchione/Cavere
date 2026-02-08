namespace Cavere.Core

open System
open System.Reflection
open ILGPU
open ILGPU.Runtime

module Engine =

    let private getMethod (kernel: CompiledKernel) (name: string) =
        kernel.KernelType.GetMethod(name, BindingFlags.Public ||| BindingFlags.Static)

    // ── Simple kernels ─────────────────────────────────────────────────

    let fold
        (accel: Accelerator)
        (kernel: CompiledKernel)
        (numScenarios: int)
        (steps: int)
        (indexOffset: int)
        : float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)
        use output = accel.Allocate1D<float32>(numScenarios)

        let methodInfo = getMethod kernel "Fold"
        let k = accel.LoadAutoGroupedKernel(methodInfo)

        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(numScenarios),
            [|
                output.View :> obj
                surfBuf.View :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                indexOffset :> obj
            |]
        )

        accel.Synchronize()
        output.GetAsArray1D()

    let foldWatch
        (accel: Accelerator)
        (kernel: CompiledKernel)
        (numScenarios: int)
        (steps: int)
        (interval: int)
        (indexOffset: int)
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
            [|
                output.View :> obj
                surfBuf.View :> obj
                obsBuf.View :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
                numObs :> obj
                interval :> obj
                indexOffset :> obj
            |]
        )

        accel.Synchronize()
        output.GetAsArray1D(), obsBuf.GetAsArray1D()

    let scan
        (accel: Accelerator)
        (kernel: CompiledKernel)
        (numScenarios: int)
        (steps: int)
        (indexOffset: int)
        : float32[,] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)
        use output = accel.Allocate1D<float32>(numScenarios * steps)

        let methodInfo = getMethod kernel "Scan"
        let k = accel.LoadAutoGroupedKernel(methodInfo)

        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(numScenarios),
            [|
                output.View :> obj
                surfBuf.View :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
                indexOffset :> obj
            |]
        )

        accel.Synchronize()

        let flat = output.GetAsArray1D()
        Array2D.init steps numScenarios (fun t s -> flat.[t * numScenarios + s])

    // ── Batch kernels ──────────────────────────────────────────────────

    let foldBatch
        (accel: Accelerator)
        (kernel: CompiledKernel)
        (numBatch: int)
        (numScenarios: int)
        (steps: int)
        (indexOffset: int)
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
            [|
                output.View :> obj
                surfBuf.View :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
                indexOffset :> obj
            |]
        )

        accel.Synchronize()
        output.GetAsArray1D()

    let foldBatchWatch
        (accel: Accelerator)
        (kernel: CompiledKernel)
        (numBatch: int)
        (numScenarios: int)
        (steps: int)
        (interval: int)
        (indexOffset: int)
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
            [|
                output.View :> obj
                surfBuf.View :> obj
                obsBuf.View :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
                numObs :> obj
                interval :> obj
                totalThreads :> obj
                indexOffset :> obj
            |]
        )

        accel.Synchronize()
        output.GetAsArray1D(), obsBuf.GetAsArray1D()

    // ── Pinned memory variants ───────────────────────────────────────

    let foldPinned
        (accel: Accelerator)
        (pool: PinnedPool)
        (kernel: CompiledKernel)
        (numScenarios: int)
        (steps: int)
        (indexOffset: int)
        : float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let surfPinned = pool.Rent(max 1L (int64 packed.Length))

        try
            let span = surfPinned.Span

            for i in 0 .. packed.Length - 1 do
                span.[i] <- packed.[i]

            use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
            surfBuf.View.CopyFromPageLockedAsync(surfPinned)
            accel.Synchronize()
            use output = accel.Allocate1D<float32>(numScenarios)

            let methodInfo = getMethod kernel "Fold"
            let k = accel.LoadAutoGroupedKernel(methodInfo)

            k.Launch<Index1D>(
                accel.DefaultStream,
                Index1D(numScenarios),
                [|
                    output.View :> obj
                    surfBuf.View :> obj
                    steps :> obj
                    kernel.Model.NormalCount :> obj
                    kernel.Model.UniformCount :> obj
                    kernel.Model.BernoulliCount :> obj
                    indexOffset :> obj
                |]
            )

            accel.Synchronize()

            let resultPinned = pool.Rent(int64 numScenarios)

            try
                output.View.CopyToPageLockedAsync(resultPinned)
                accel.Synchronize()
                resultPinned.GetArray().[.. numScenarios - 1]
            finally
                pool.Return(resultPinned)
        finally
            pool.Return(surfPinned)

    let foldWatchPinned
        (accel: Accelerator)
        (pool: PinnedPool)
        (kernel: CompiledKernel)
        (numScenarios: int)
        (steps: int)
        (interval: int)
        (indexOffset: int)
        : float32[] * float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let surfPinned = pool.Rent(max 1L (int64 packed.Length))

        try
            let span = surfPinned.Span

            for i in 0 .. packed.Length - 1 do
                span.[i] <- packed.[i]

            use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
            surfBuf.View.CopyFromPageLockedAsync(surfPinned)
            accel.Synchronize()
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
                [|
                    output.View :> obj
                    surfBuf.View :> obj
                    obsBuf.View :> obj
                    steps :> obj
                    kernel.Model.NormalCount :> obj
                    kernel.Model.UniformCount :> obj
                    kernel.Model.BernoulliCount :> obj
                    numScenarios :> obj
                    numObs :> obj
                    interval :> obj
                    indexOffset :> obj
                |]
            )

            accel.Synchronize()

            let resultPinned = pool.Rent(int64 numScenarios)
            let obsPinned = pool.Rent(int64 obsBufSize)

            try
                output.View.CopyToPageLockedAsync(resultPinned)
                obsBuf.View.CopyToPageLockedAsync(obsPinned)
                accel.Synchronize()
                resultPinned.GetArray().[.. numScenarios - 1], obsPinned.GetArray().[.. obsBufSize - 1]
            finally
                pool.Return(resultPinned)
                pool.Return(obsPinned)
        finally
            pool.Return(surfPinned)

    // ── Multi-device variants ────────────────────────────────────────

    let foldMulti (deviceSet: DeviceSet) (kernel: CompiledKernel) (numScenarios: int) (steps: int) : float32[] =

        let numDevices = deviceSet.Accelerators.Length
        let scenariosPerDevice = numScenarios / numDevices
        let remainder = numScenarios % numDevices

        let results = Array.zeroCreate<float32[]> numDevices

        // Launch on each device (sequential for now — ILGPU streams are not thread-safe)
        for d in 0 .. numDevices - 1 do
            let accel = deviceSet.Accelerators.[d]
            let count = scenariosPerDevice + (if d < remainder then 1 else 0)
            let offset = d * scenariosPerDevice + (min d remainder)
            results.[d] <- fold accel kernel count steps offset

        Array.concat results

    // ── Adjoint kernel ─────────────────────────────────────────────────

    let foldAdjoint
        (accel: Accelerator)
        (kernel: CompiledKernel)
        (info: CompilerAdjoint.AdjointInfo)
        (numScenarios: int)
        (steps: int)
        (indexOffset: int)
        : float32[] * float32[,] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        use surfBuf = accel.Allocate1D<float32>(max 1 packed.Length)
        surfBuf.View.CopyFromCPU(packed)
        use output = accel.Allocate1D<float32>(numScenarios)

        let numAccums = info.SortedAccums.Length
        let numDiffVars = info.DiffVars.Length
        let tapeSize = max 1 (numScenarios * numAccums * steps)
        use tape = accel.Allocate1D<float32>(tapeSize)
        use adjointOut = accel.Allocate1D<float32>(max 1 (numScenarios * numDiffVars))

        let methodInfo = getMethod kernel "FoldAdjoint"
        let k = accel.LoadAutoGroupedKernel(methodInfo)

        k.Launch<Index1D>(
            accel.DefaultStream,
            Index1D(numScenarios),
            [|
                output.View :> obj
                surfBuf.View :> obj
                tape.View :> obj
                adjointOut.View :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                indexOffset :> obj
            |]
        )

        accel.Synchronize()

        let values = output.GetAsArray1D()
        let adjFlat = adjointOut.GetAsArray1D()
        let adjoints = Array2D.init numScenarios numDiffVars (fun s d -> adjFlat.[s * numDiffVars + d])
        values, adjoints

    // ── Multi-device variants ────────────────────────────────────────

    let foldBatchMulti
        (deviceSet: DeviceSet)
        (kernel: CompiledKernel)
        (numBatch: int)
        (numScenarios: int)
        (steps: int)
        : float32[] =

        let numDevices = deviceSet.Accelerators.Length
        let scenariosPerDevice = numScenarios / numDevices
        let remainder = numScenarios % numDevices

        let results = Array.zeroCreate<float32[]> numDevices

        for d in 0 .. numDevices - 1 do
            let accel = deviceSet.Accelerators.[d]
            let count = scenariosPerDevice + (if d < remainder then 1 else 0)
            let offset = d * scenariosPerDevice + (min d remainder)
            results.[d] <- foldBatch accel kernel numBatch count steps offset

        Array.concat results

    // ── Native CPU execution (no ILGPU) ────────────────────────────

    let foldCpu (kernel: CompiledKernel) (numScenarios: int) (steps: int) : float32[] =
        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let output = Array.zeroCreate<float32> numScenarios
        let methodInfo = getMethod kernel "Fold"

        methodInfo.Invoke(
            null,
            [|
                output :> obj
                packed :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
            |]
        )
        |> ignore

        output

    let foldWatchCpu (kernel: CompiledKernel) (numScenarios: int) (steps: int) (interval: int) : float32[] * float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let output = Array.zeroCreate<float32> numScenarios
        let numObs = (steps + interval - 1) / interval
        let numSlots = kernel.Model.Observers.Length
        let obsBufSize = max 1 (numSlots * numObs * numScenarios)
        let obsBuffer = Array.zeroCreate<float32> obsBufSize
        let methodInfo = getMethod kernel "FoldWatch"

        methodInfo.Invoke(
            null,
            [|
                output :> obj
                obsBuffer :> obj
                packed :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
                numObs :> obj
                interval :> obj
            |]
        )
        |> ignore

        output, obsBuffer

    let scanCpu (kernel: CompiledKernel) (numScenarios: int) (steps: int) : float32[,] =
        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let output = Array.zeroCreate<float32> (numScenarios * steps)
        let methodInfo = getMethod kernel "Scan"

        methodInfo.Invoke(
            null,
            [|
                output :> obj
                packed :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
            |]
        )
        |> ignore

        Array2D.init steps numScenarios (fun t s -> output.[t * numScenarios + s])

    let foldBatchCpu (kernel: CompiledKernel) (numBatch: int) (numScenarios: int) (steps: int) : float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let totalThreads = numBatch * numScenarios
        let output = Array.zeroCreate<float32> totalThreads
        let methodInfo = getMethod kernel "FoldBatch"

        methodInfo.Invoke(
            null,
            [|
                output :> obj
                packed :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
            |]
        )
        |> ignore

        output

    let foldBatchWatchCpu
        (kernel: CompiledKernel)
        (numBatch: int)
        (numScenarios: int)
        (steps: int)
        (interval: int)
        : float32[] * float32[] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let totalThreads = numBatch * numScenarios
        let output = Array.zeroCreate<float32> totalThreads
        let numObs = (steps + interval - 1) / interval
        let numSlots = kernel.Model.Observers.Length
        let obsBufSize = max 1 (numSlots * numObs * totalThreads)
        let obsBuffer = Array.zeroCreate<float32> obsBufSize
        let methodInfo = getMethod kernel "FoldBatchWatch"

        methodInfo.Invoke(
            null,
            [|
                output :> obj
                obsBuffer :> obj
                packed :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
                numScenarios :> obj
                numObs :> obj
                interval :> obj
                totalThreads :> obj
            |]
        )
        |> ignore

        output, obsBuffer

    let foldAdjointCpu
        (kernel: CompiledKernel)
        (info: CompilerAdjoint.AdjointInfo)
        (numScenarios: int)
        (steps: int)
        : float32[] * float32[,] =

        let packed = Compiler.packSurfaces kernel.Model kernel.SurfaceLayout
        let output = Array.zeroCreate<float32> numScenarios
        let numAccums = info.SortedAccums.Length
        let numDiffVars = info.DiffVars.Length
        let tapeSize = max 1 (numScenarios * numAccums * steps)
        let tape = Array.zeroCreate<float32> tapeSize
        let adjointOut = Array.zeroCreate<float32> (max 1 (numScenarios * numDiffVars))
        let methodInfo = getMethod kernel "FoldAdjoint"

        methodInfo.Invoke(
            null,
            [|
                output :> obj
                packed :> obj
                tape :> obj
                adjointOut :> obj
                steps :> obj
                kernel.Model.NormalCount :> obj
                kernel.Model.UniformCount :> obj
                kernel.Model.BernoulliCount :> obj
            |]
        )
        |> ignore

        let adjoints = Array2D.init numScenarios numDiffVars (fun s d -> adjointOut.[s * numDiffVars + d])
        output, adjoints

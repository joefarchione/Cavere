namespace Cavere.Core

open System
open System.Collections.Concurrent
open System.Reflection
open ILGPU
open ILGPU.Runtime

module Engine =

    let private getMethod (kernel: CompiledKernel) (name: string) =
        kernel.KernelType.GetMethod(name, BindingFlags.Public ||| BindingFlags.Static)

    // ── CPU delegate types and cache ─────────────────────────────────

    type private FoldDelegate = delegate of float32[] * float32[] * int * int * int * int -> unit

    type private FoldWatchDelegate =
        delegate of float32[] * float32[] * float32[] * int * int * int * int * int * int * int -> unit

    type private ScanDelegate = delegate of float32[] * float32[] * int * int * int * int * int -> unit
    type private FoldBatchDelegate = delegate of float32[] * float32[] * int * int * int * int * int -> unit

    type private FoldBatchWatchDelegate =
        delegate of float32[] * float32[] * float32[] * int * int * int * int * int * int * int * int -> unit

    type private FoldAdjointDelegate =
        delegate of float32[] * float32[] * float32[] * float32[] * int * int * int * int -> unit

    let private delegateCache = ConcurrentDictionary<struct (Type * string), Delegate>()

    let private getDelegate<'d when 'd :> Delegate> (kernel: CompiledKernel) (name: string) : 'd =
        let key = struct (kernel.KernelType, name)

        let d =
            delegateCache.GetOrAdd(
                key,
                fun _ ->
                    let mi = getMethod kernel name
                    Delegate.CreateDelegate(typeof<'d>, mi)
            )

        d :?> 'd

    // ── Simple kernels ─────────────────────────────────────────────────

    let fold
        (accel: Accelerator)
        (kernel: CompiledKernel)
        (numScenarios: int)
        (steps: int)
        (indexOffset: int)
        : float32[] =

        let packed = kernel.PackedSurfaces
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

        let packed = kernel.PackedSurfaces
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

        let packed = kernel.PackedSurfaces
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

        let packed = kernel.PackedSurfaces
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

        let packed = kernel.PackedSurfaces
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

        let packed = kernel.PackedSurfaces
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

        let packed = kernel.PackedSurfaces
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

        let packed = kernel.PackedSurfaces
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
        let output = Array.zeroCreate<float32> numScenarios
        let d = getDelegate<FoldDelegate> kernel "Fold"

        d.Invoke(
            output,
            kernel.PackedSurfaces,
            steps,
            kernel.Model.NormalCount,
            kernel.Model.UniformCount,
            kernel.Model.BernoulliCount
        )

        output

    let foldWatchCpu (kernel: CompiledKernel) (numScenarios: int) (steps: int) (interval: int) : float32[] * float32[] =
        let output = Array.zeroCreate<float32> numScenarios
        let numObs = (steps + interval - 1) / interval
        let numSlots = kernel.Model.Observers.Length
        let obsBufSize = max 1 (numSlots * numObs * numScenarios)
        let obsBuffer = Array.zeroCreate<float32> obsBufSize
        let d = getDelegate<FoldWatchDelegate> kernel "FoldWatch"

        d.Invoke(
            output,
            obsBuffer,
            kernel.PackedSurfaces,
            steps,
            kernel.Model.NormalCount,
            kernel.Model.UniformCount,
            kernel.Model.BernoulliCount,
            numScenarios,
            numObs,
            interval
        )

        output, obsBuffer

    let scanCpu (kernel: CompiledKernel) (numScenarios: int) (steps: int) : float32[,] =
        let output = Array.zeroCreate<float32> (numScenarios * steps)
        let d = getDelegate<ScanDelegate> kernel "Scan"

        d.Invoke(
            output,
            kernel.PackedSurfaces,
            steps,
            kernel.Model.NormalCount,
            kernel.Model.UniformCount,
            kernel.Model.BernoulliCount,
            numScenarios
        )

        let result = Array2D.zeroCreate<float32> steps numScenarios
        Buffer.BlockCopy(output, 0, result, 0, output.Length * sizeof<float32>)
        result

    let foldBatchCpu (kernel: CompiledKernel) (numBatch: int) (numScenarios: int) (steps: int) : float32[] =
        let totalThreads = numBatch * numScenarios
        let output = Array.zeroCreate<float32> totalThreads
        let d = getDelegate<FoldBatchDelegate> kernel "FoldBatch"

        d.Invoke(
            output,
            kernel.PackedSurfaces,
            steps,
            kernel.Model.NormalCount,
            kernel.Model.UniformCount,
            kernel.Model.BernoulliCount,
            numScenarios
        )

        output

    let foldBatchWatchCpu
        (kernel: CompiledKernel)
        (numBatch: int)
        (numScenarios: int)
        (steps: int)
        (interval: int)
        : float32[] * float32[] =

        let totalThreads = numBatch * numScenarios
        let output = Array.zeroCreate<float32> totalThreads
        let numObs = (steps + interval - 1) / interval
        let numSlots = kernel.Model.Observers.Length
        let obsBufSize = max 1 (numSlots * numObs * totalThreads)
        let obsBuffer = Array.zeroCreate<float32> obsBufSize
        let d = getDelegate<FoldBatchWatchDelegate> kernel "FoldBatchWatch"

        d.Invoke(
            output,
            obsBuffer,
            kernel.PackedSurfaces,
            steps,
            kernel.Model.NormalCount,
            kernel.Model.UniformCount,
            kernel.Model.BernoulliCount,
            numScenarios,
            numObs,
            interval,
            totalThreads
        )

        output, obsBuffer

    let foldAdjointCpu
        (kernel: CompiledKernel)
        (info: CompilerAdjoint.AdjointInfo)
        (numScenarios: int)
        (steps: int)
        : float32[] * float32[,] =

        let output = Array.zeroCreate<float32> numScenarios
        let numAccums = info.SortedAccums.Length
        let numDiffVars = info.DiffVars.Length
        let tapeSize = max 1 (numScenarios * numAccums * steps)
        let tape = Array.zeroCreate<float32> tapeSize
        let adjointOut = Array.zeroCreate<float32> (max 1 (numScenarios * numDiffVars))
        let d = getDelegate<FoldAdjointDelegate> kernel "FoldAdjoint"

        d.Invoke(
            output,
            kernel.PackedSurfaces,
            tape,
            adjointOut,
            steps,
            kernel.Model.NormalCount,
            kernel.Model.UniformCount,
            kernel.Model.BernoulliCount
        )

        let adjoints = Array2D.init numScenarios numDiffVars (fun s dd -> adjointOut.[s * numDiffVars + dd])
        output, adjoints

namespace Cavere.Core

open System
open System.Collections.Concurrent
open ILGPU
open ILGPU.Runtime

/// Pool of page-locked arrays for efficient GPU transfers.
/// Buffers are bucketed by size and reused to avoid repeated allocation.
type PinnedPool(accel: Accelerator) =
    let pools = ConcurrentDictionary<int64, ConcurrentBag<PageLockedArray1D<float32>>>()
    let mutable disposed = false

    member _.Rent(size: int64) : PageLockedArray1D<float32> =
        let bag = pools.GetOrAdd(size, fun _ -> ConcurrentBag<PageLockedArray1D<float32>>())

        match bag.TryTake() with
        | true, buf -> buf
        | _ -> accel.AllocatePageLocked1D<float32>(size)

    member _.Return(buf: PageLockedArray1D<float32>) : unit =
        if not disposed then
            let bag = pools.GetOrAdd(buf.Extent.Size, fun _ -> ConcurrentBag<PageLockedArray1D<float32>>())
            bag.Add(buf)

    interface IDisposable with
        member _.Dispose() =
            disposed <- true

            for kvp in pools do
                for buf in kvp.Value do
                    (buf :> IDisposable).Dispose()

            pools.Clear()

/// GPU transfer utilities using pinned memory and async streams.
module Transfer =

    /// Copy data to a device buffer using pinned memory for faster DMA transfer.
    let copyToDevicePinned (accel: Accelerator) (data: float32[]) : MemoryBuffer1D<float32, Stride1D.Dense> =
        use pinned = accel.AllocatePageLocked1D<float32>(data.LongLength)
        let span = pinned.Span

        for i in 0 .. data.Length - 1 do
            span.[i] <- data.[i]

        let buf = accel.Allocate1D<float32>(data.LongLength)
        buf.View.CopyFromPageLockedAsync(pinned)
        accel.Synchronize()
        buf

    /// Copy device buffer to host using pinned memory for faster DMA transfer.
    let copyFromDevicePinned (accel: Accelerator) (buf: MemoryBuffer1D<float32, Stride1D.Dense>) : float32[] =
        use pinned = accel.AllocatePageLocked1D<float32>(buf.Length)
        buf.View.CopyToPageLockedAsync(pinned)
        accel.Synchronize()
        pinned.GetArray()

    /// Async copy from host pinned array to device buffer via stream.
    let copyToDeviceAsync
        (stream: AcceleratorStream)
        (pinned: PageLockedArray1D<float32>)
        (target: MemoryBuffer1D<float32, Stride1D.Dense>)
        : unit =
        target.View.CopyFromPageLockedAsync(stream, pinned)

    /// Async copy from device buffer to host pinned array via stream.
    let copyFromDeviceAsync
        (stream: AcceleratorStream)
        (source: MemoryBuffer1D<float32, Stride1D.Dense>)
        (pinned: PageLockedArray1D<float32>)
        : unit =
        source.View.CopyToPageLockedAsync(stream, pinned)

# GPU Memory Optimization for Monte Carlo Kernels

This document outlines memory optimization strategies for GPU-based Monte Carlo simulation kernels. Memory access is typically the bottleneck for MC workloads, not compute.

## Memory Access Patterns

| Data | Access Pattern | Size | Optimal Memory Type |
|------|----------------|------|---------------------|
| Accumulators | Per-thread, read/write every step | Small | Registers |
| Random states | Per-thread, read/write every step | ~32 bytes/thread | Registers/Local |
| Surfaces (curves, vols) | All threads read, rarely changes | KB-MB | Texture/Constant |
| Batch inputs (policy data) | Per-batch element, read-only | MB | Texture/Pinned |
| Results | Per-thread, write once | 4 bytes/thread | Global, coalesced |

---

## 1. Texture Memory for Surfaces

Texture memory is ideal for rate curves and volatility surfaces:

- Hardware interpolation (free bilinear interpolation)
- Cached, optimized for 2D spatial locality
- Read-only access pattern

### Current Implementation

```fsharp
// Global memory lookup
let rate = Interp2D(surfaceId, timeIndex, spot)
```

### Optimized Implementation

```fsharp
// ILGPU TextureView for hardware interpolation
let rateTex = accelerator.CreateTexture2D(surfaceData)
let rate = rateTex.Sample(t, s)  // Hardware interpolation
```

### Benefits

- Vol surface lookups become nearly free
- 10-30% speedup for surface-heavy models
- Automatic caching and prefetching

### When to Use

- Volatility surfaces (2D)
- Large rate curves
- Any read-only 2D data with interpolation needs

---

## 2. Constant Memory for Small Curves

For data under 64KB accessed uniformly by all threads:

```fsharp
type KernelConstants = {
    ForwardRates: float32[]   // Constant memory - 252 floats = 1KB
    TimeSteps: float32[]      // Constant memory
    ModelParameters: float32[] // Small parameter arrays
}
```

### Benefits

- Broadcast to all threads in one memory transaction
- Heavily cached (8KB cache per SM)
- 5-10% speedup for rate curve lookups

### When to Use

- Forward rate curves (typically < 1KB)
- Discount factors
- Small parameter arrays shared by all threads
- Data that doesn't change during kernel execution

### Limitations

- Maximum 64KB total constant memory
- Not suitable for per-policy data

---

## 3. Pinned (Page-Locked) Memory for Transfers

### Without Pinning

```
CPU pageable memory → CPU pinned memory (implicit copy) → GPU global memory
```

### With Pinning

```
CPU pinned memory → GPU global memory (direct DMA transfer)
```

### Implementation

```fsharp
// Allocate pinned memory on host
use pinnedInput = accelerator.AllocatePinned1D<float32>(policyData.Length)
pinnedInput.CopyFrom(policyData)

// Async transfer with compute overlap
use stream = accelerator.CreateStream()
stream.CopyToDevice(pinnedInput, deviceBuffer)

// CPU can perform other work while transfer happens
prepareNextBatch()

stream.Synchronize()
```

### Benefits

- 2x faster CPU↔GPU transfers
- Enables async overlap of transfer and compute
- Essential for streaming large datasets

### When to Use

- Large policy data arrays (100k+ policies)
- Frequent CPU↔GPU transfers
- Streaming/chunking workflows for nested stochastic

### Considerations

- Pinned memory is a limited resource
- Allocate once, reuse across batches
- Don't pin more than ~50% of system RAM

---

## 4. Shared Memory for Reductions

Instead of writing all results to global memory, reduce within thread blocks first.

### Standard Approach (Inefficient)

```fsharp
// 100,000 threads each write to global memory
results.[threadId] <- pathValue
// Then CPU aggregates
let mean = Array.average results
```

### Optimized Approach

```fsharp
[<Kernel>]
let reduceKernel (results: ArrayView<float32>) (blockSums: ArrayView<float32>) =
    // Allocate shared memory for block
    let shared = SharedMemory.Allocate1D<float32>(256)
    let tid = Grid.GlobalIndex.X
    let lid = Group.Index.X

    // Load thread result to shared memory
    shared.[lid] <- results.[tid]
    Group.Barrier()

    // Parallel reduction within block
    let mutable stride = 128
    while stride > 0 do
        if lid < stride then
            shared.[lid] <- shared.[lid] + shared.[lid + stride]
        Group.Barrier()
        stride <- stride / 2

    // One global write per block (not per thread)
    if lid = 0 then
        blockSums.[Group.GridIndex.X] <- shared.[0]
```

### Benefits

- 256x fewer global memory writes (for block size 256)
- 10-50% speedup for result-heavy workloads
- Reduced memory bandwidth pressure

### When to Use

- Computing means, variances, percentiles
- Any aggregation over simulation paths
- Nested stochastic inner loop aggregation

---

## 5. Memory Layout for Coalescing

Coalesced memory access means consecutive threads access consecutive memory addresses.

### Bad Layout (Strided Access)

```
results[path, step]

Thread 0 reads results[0, step] → address 0
Thread 1 reads results[1, step] → address 1000  (stride!)
Thread 2 reads results[2, step] → address 2000  (stride!)
```

### Good Layout (Coalesced Access)

```
results[step, path]

Thread 0 reads results[step, 0] → address 0
Thread 1 reads results[step, 1] → address 4    (contiguous!)
Thread 2 reads results[step, 2] → address 8    (contiguous!)
```

### Implementation

```fsharp
type DataLayout =
    | PathMajor   // results[path, step] - bad for GPU
    | StepMajor   // results[step, path] - good for GPU

// Configure layout in simulation options
let config = {
    DataLayout = StepMajor
    // ...
}
```

### Benefits

- 2-5x memory throughput improvement
- Single memory transaction serves entire warp (32 threads)
- Critical for batch input data

### Batch Input Layout

```fsharp
// Bad: policyData[policyId].premium, policyData[policyId].cap, ...
// Struct of Arrays - threads access strided

// Good: premiums[policyId], caps[policyId], ...
// Array of primitives - threads access contiguous
```

---

## Configuration API

```fsharp
type SurfaceMemoryType =
    | Texture    // Large surfaces, interpolation needed
    | Constant   // Small curves < 64KB
    | Global     // Default, no optimization

type DataLayout =
    | PathMajor   // [path, step] - CPU friendly
    | StepMajor   // [step, path] - GPU friendly

type OptimizedSimulationConfig = {
    // Device selection
    Devices: DeviceConfig

    // Transfer optimization
    UsePinnedMemory: bool
    AsyncTransfers: bool

    // Surface memory placement
    SurfaceMemory: SurfaceMemoryType

    // Result handling
    OnDeviceReduction: bool

    // Data layout
    DataLayout: DataLayout

    // Simulation parameters
    NumSims: int
    Steps: int
    Seed: uint64
}

// Example configuration for production
let productionConfig = {
    Devices = MultiGPU [| 0; 1; 2; 3 |]
    UsePinnedMemory = true
    AsyncTransfers = true
    SurfaceMemory = Texture
    OnDeviceReduction = true
    DataLayout = StepMajor
    NumSims = 1_000_000
    Steps = 360
    Seed = 42UL
}
```

---

## Expected Performance Impact

| Optimization | Typical Speedup | Applicability |
|--------------|-----------------|---------------|
| Texture memory for vol surfaces | 10-30% | Surface-heavy models (local vol, Heston) |
| Constant memory for rate curves | 5-10% | All models with term structure |
| Pinned memory | 2x transfer speed | Large batch simulations |
| On-device reduction | 10-50% | Result aggregation |
| Coalesced data layout | 2-5x memory throughput | Batch simulations |

---

## Implementation Priorities

### High Priority (Implement First)

1. **Coalesced data layout** - Largest impact, relatively simple
2. **Pinned memory** - Essential for large policy counts
3. **Texture memory for 2D surfaces** - Big win for local vol models

### Medium Priority

4. **Constant memory for curves** - Modest improvement, low effort
5. **On-device reduction** - Important for nested stochastic

### Lower Priority

6. **Async transfer overlap** - Useful for streaming, more complex

---

## 6. Streaming Results to Disk

For large-scale simulations, results cannot fit in CPU memory. Stream directly from GPU to disk.

### The Problem

```
100k policies × 1000 scenarios × 360 steps × 4 bytes = 144 GB
Can't fit in CPU memory. Must stream.
```

### Solution: Double-Buffered Async Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  GPU    │ →  │ Pinned  │ →  │ Pinned  │ →  │  File   │
│ Compute │    │ Buffer A│    │ Buffer B│    │ (mmap)  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │
     │   Stream 1   │   Stream 2   │    Disk      │
     │   (compute)  │   (transfer) │    I/O       │
     └──────────────┴──────────────┴──────────────┘
              All happening concurrently
```

### Implementation

```fsharp
open System.IO.MemoryMappedFiles
open ILGPU
open ILGPU.Runtime

type StreamingConfig = {
    ChunkSize: int          // Results per chunk (e.g., 10k paths)
    BufferCount: int        // Double buffer = 2
    OutputPath: string
}

let streamResultsToFile (accelerator: Accelerator) (kernel: CompiledKernel)
                        (totalPaths: int) (config: StreamingConfig) =

    let resultsPerPath = 1  // or more if storing paths
    let bytesPerChunk = config.ChunkSize * resultsPerPath * sizeof<float32>
    let totalChunks = (totalPaths + config.ChunkSize - 1) / config.ChunkSize
    let totalBytes = int64 totalPaths * int64 resultsPerPath * 4L

    // Allocate pinned double buffers (only 2 × chunk size in CPU memory)
    let pinnedBuffers =
        Array.init config.BufferCount (fun _ ->
            accelerator.AllocatePinnedArray1D<float32>(config.ChunkSize * resultsPerPath))

    // GPU buffer for compute
    let deviceBuffer = accelerator.Allocate1D<float32>(config.ChunkSize * resultsPerPath)

    // Streams for async operations
    let computeStream = accelerator.CreateStream()
    let transferStream = accelerator.CreateStream()

    // Memory-mapped file for direct writes (no intermediate CPU buffer)
    use mmf = MemoryMappedFile.CreateFromFile(
        config.OutputPath,
        FileMode.Create,
        null,
        totalBytes)
    use accessor = mmf.CreateViewAccessor()

    let mutable writeOffset = 0L
    let mutable currentBuffer = 0

    for chunk in 0 .. totalChunks - 1 do
        let pathOffset = chunk * config.ChunkSize
        let pathsThisChunk = min config.ChunkSize (totalPaths - pathOffset)

        // 1. Launch kernel for this chunk (async)
        kernel.Launch(computeStream, deviceBuffer, pathOffset, pathsThisChunk)

        // 2. Wait for compute, start transfer to pinned (async)
        computeStream.Synchronize()
        transferStream.Copy(deviceBuffer.View, pinnedBuffers.[currentBuffer].View)

        // 3. While transfer happens, write PREVIOUS buffer to file
        if chunk > 0 then
            let prevBuffer = (currentBuffer + config.BufferCount - 1) % config.BufferCount
            let prevChunkSize =
                if chunk = 1 then config.ChunkSize
                else min config.ChunkSize (totalPaths - (chunk - 1) * config.ChunkSize)

            // Direct write to memory-mapped file (no copy)
            accessor.WriteArray(writeOffset, pinnedBuffers.[prevBuffer].GetAsArray1D(), 0, prevChunkSize)
            writeOffset <- writeOffset + int64 prevChunkSize * 4L

        // 4. Swap buffers
        transferStream.Synchronize()
        currentBuffer <- (currentBuffer + 1) % config.BufferCount

    // Write final buffer
    let finalBuffer = (currentBuffer + config.BufferCount - 1) % config.BufferCount
    let finalChunkSize = totalPaths - (totalChunks - 1) * config.ChunkSize
    accessor.WriteArray(writeOffset, pinnedBuffers.[finalBuffer].GetAsArray1D(), 0, finalChunkSize)

    // Cleanup
    computeStream.Dispose()
    transferStream.Dispose()
    deviceBuffer.Dispose()
    for buf in pinnedBuffers do buf.Dispose()
```

### Memory Usage Comparison

```
Traditional:       144 GB (all results in CPU memory)
Double-buffered:   ~80 MB (2 × chunk size + overhead)
                   ────────
                   1800x less memory
```

### Practical Configuration

```fsharp
let config = {
    ChunkSize = 10_000           // 10k paths per chunk
    BufferCount = 2              // Double buffer
    OutputPath = "results.bin"
}

// Memory usage:
// - 2 pinned buffers × 10k × 4 bytes = 80 KB
// - 1 device buffer × 10k × 4 bytes = 40 KB
// Total: ~120 KB (plus overhead)
```

### File Format Considerations

| Format | Write Speed | Read Speed | Size | Queryable |
|--------|-------------|------------|------|-----------|
| Raw binary | Fastest | Fast | 1x | No |
| Memory-mapped | Fastest | Random access | 1x | No |
| Parquet | Slower | Fast + columnar | 0.3-0.5x | Yes |
| HDF5 | Medium | Fast | 0.5-0.8x | Partial |

For streaming writes, raw binary or memory-mapped is fastest. Convert to Parquet in a post-processing step if needed.

### GPUDirect Storage (Advanced)

For NVIDIA GPUs with compatible NVMe SSDs on Linux:

```
┌─────────┐         ┌─────────┐
│  GPU    │ ──────→ │  NVMe   │
│ Memory  │  DMA    │  SSD    │
└─────────┘         └─────────┘
    No CPU involvement at all
```

Requires NVIDIA Magnum IO / GPUDirect Storage. Not available on Windows.

---

## 7. Nested Stochastic (Stochastic-on-Stochastic) Memory Strategy

Nested stochastic simulations have unique memory requirements. The key insight: **you only need terminal values and their average discounted values** - no path storage at all.

### What You Actually Need

| Data | What to Store | Storage |
|------|---------------|---------|
| Outer paths | Monthly snapshots of state variables | Stream to disk |
| Inner simulation | **Nothing** - run and discard | GPU registers only |
| Inner results | Average discounted terminal value | Single float per outer path |

### What You Don't Need

- Full inner paths (never store)
- Inner path intermediate values (never store)
- Individual inner terminal values (reduce on GPU immediately)

### Memory-Efficient Nested Architecture

```
Outer Simulation (1,000 paths × 360 monthly steps):
├── Month 12: snapshot state → run 1,000 inner sims → average discounted terminal → 1 float per outer
├── Month 24: snapshot state → run 1,000 inner sims → average discounted terminal → 1 float per outer
├── ...
└── Month 360: snapshot state → run 1,000 inner sims → average discounted terminal → 1 float per outer

Per valuation date:
- Outer state snapshot: 1,000 paths × ~10 values = 40 KB
- Inner sims running: 1,000 outer × 1,000 inner = 1M paths (GPU only, not stored)
- Inner output: 1,000 floats = 4 KB (average discounted terminal value)
- Inner paths stored: ZERO
```

### Implementation

```fsharp
type NestedSimConfig = {
    OuterPaths: int              // e.g., 1,000
    InnerPaths: int              // e.g., 1,000
    OuterSteps: int              // e.g., 360 (monthly, 30 years)
    InnerSteps: int              // e.g., 120 (monthly, 10 years remaining)
    ValuationFrequency: int      // e.g., 12 (annual valuations)
}

let runNestedStochastic (config: NestedSimConfig) =
    let valuationDates = config.OuterSteps / config.ValuationFrequency

    // Output: average discounted terminal value per outer path per valuation date
    // Size: 1,000 outer × 30 dates × 4 bytes = 120 KB total
    let results = Array2D.zeroCreate<float32> config.OuterPaths valuationDates

    // Outer simulation
    for outerStep in 0 .. config.OuterSteps - 1 do
        outerKernel.Launch(outerPaths, outerStep)

        // At valuation dates, run inner simulation
        if outerStep % config.ValuationFrequency = 0 then
            let valDate = outerStep / config.ValuationFrequency

            // Snapshot outer state (small - just current accumulator values)
            let outerState = extractOuterState(outerPaths)  // 40 KB

            // Inner simulation computes E[discounted terminal value]
            // 1M paths run on GPU, reduced to 1,000 means, nothing else stored
            let avgDiscountedTerminal = runInnerWithReduction(outerState, config.InnerPaths)

            // Store single float per outer path
            results.[*, valDate] <- avgDiscountedTerminal  // 4 KB

    results
```

### On-GPU Reduction: Average Discounted Terminal Value

Each inner path computes a discounted terminal value. These are averaged on-GPU immediately - individual values are never stored:

```fsharp
[<Kernel>]
let innerSimWithReduction
    (outerStates: ArrayView<float32>)    // 1,000 outer path states
    (results: ArrayView<float32>)         // 1,000 output: E[discounted terminal]
    (innerPathsPerOuter: int) =

    let outerIdx = Grid.GlobalIndex.X / innerPathsPerOuter
    let innerIdx = Grid.GlobalIndex.X % innerPathsPerOuter

    // Shared memory for block reduction
    let shared = SharedMemory.Allocate1D<float32>(256)
    let lid = Group.Index.X

    // Run inner path to terminal, compute discounted value
    // All intermediate steps stay in registers - nothing written to global memory
    let mutable stock = outerStates.[outerIdx]
    let mutable df = 1.0f

    for step in 0 .. innerSteps - 1 do
        let z = generateNormal(rngState)
        stock <- stock * exp(drift + vol * sqrt(dt) * z)
        df <- df * exp(-rate * dt)

    // Terminal discounted value (single float, in register)
    let discountedTerminal = max(stock - strike, 0.0f) * df

    // Reduce across inner paths to compute mean
    shared.[lid] <- discountedTerminal
    Group.Barrier()

    let mutable stride = 128
    while stride > 0 do
        if lid < stride then
            shared.[lid] <- shared.[lid] + shared.[lid + stride]
        Group.Barrier()
        stride <- stride / 2

    // One atomic write per block
    if lid = 0 then
        Atomic.Add(&results.[outerIdx], shared.[0] / float32 innerPathsPerOuter)
```

### Memory Comparison: Nested Stochastic

| Approach | Memory Required | Notes |
|----------|-----------------|-------|
| Store all inner paths | 1,000 × 1,000 × 120 × 4 = 480 MB | Never do this |
| Store inner terminals | 1,000 × 1,000 × 4 = 4 MB | Unnecessary |
| **Average on GPU** | **1,000 × 4 = 4 KB** | Correct approach |

**Only the average discounted terminal value matters. Reduce on GPU, transfer 4 KB.**

### Data Flow Summary

```
Inner simulation (per outer path, per valuation date):
┌─────────────────────────────────────────────────────────────────┐
│  1,000 inner paths run in parallel                              │
│  ├── Path 0: stock₀→stock₁→...→stockₜ → discounted terminal    │  Registers
│  ├── Path 1: stock₀→stock₁→...→stockₜ → discounted terminal    │  only
│  ├── ...                                                        │
│  └── Path 999: stock₀→stock₁→...→stockₜ → discounted terminal  │
│                                                                 │
│  Parallel reduction: sum(discounted terminals) / 1000           │  Shared mem
│                                     ↓                           │
│                        Single float: E[discounted terminal]     │  Global mem
└─────────────────────────────────────────────────────────────────┘

Output per valuation date: 1,000 floats (one per outer path)
Output total: 1,000 outer × 30 dates × 4 bytes = 120 KB
```

### What to Store for Regulatory/Audit

| Output | Size | Purpose |
|--------|------|---------|
| Outer path state snapshots | 1,000 × 30 dates × ~10 values = 1.2 MB | Reproducibility |
| E[discounted terminal] per outer path | 1,000 × 30 × 4 = 120 KB | Reserve calculation |
| Random seeds | 1,000 × 8 = 8 KB | Reproducibility |

**Inner paths are not stored. They can be reproduced from outer state + seed if needed for audit.**

---

## 8. Pinned Memory - Detailed Guidance

### What is Pinned Memory?

Normal (pageable) memory can be swapped to disk by the OS. Pinned (page-locked) memory is locked in physical RAM - the OS cannot move or swap it.

### Why GPUs Need It

```
Without pinning:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ CPU Pageable │ →  │ CPU Pinned   │ →  │ GPU Global   │
│ Memory       │    │ (staging)    │    │ Memory       │
└──────────────┘    └──────────────┘    └──────────────┘
                    Hidden copy!         DMA transfer

With pinning:
┌──────────────┐                        ┌──────────────┐
│ CPU Pinned   │ ────────────────────→  │ GPU Global   │
│ Memory       │    Direct DMA          │ Memory       │
└──────────────┘                        └──────────────┘
```

GPU DMA engines can only access physical addresses. Pageable memory might be swapped out, so CUDA silently copies to a pinned staging buffer first.

### When to Use Pinned Memory

| Use Case | Why |
|----------|-----|
| Large transfers (> 1MB) | Eliminates hidden copy overhead |
| Repeated transfers | Allocate once, reuse many times |
| Async overlap | Required for `memcpyAsync` to actually be async |
| Streaming/chunking | Transferring while computing previous chunk |

### When NOT to Use Pinned Memory

| Avoid When | Why |
|------------|-----|
| Small transfers (< 64KB) | Overhead of pinning exceeds benefit |
| One-time transfers | Allocation cost not amortized |
| Memory-constrained host | Pinned memory can't be swapped |
| Many small allocations | Fragmentation, allocation overhead |

### The Downsides (Important)

**1. Limited Resource**

```
Pinned memory = physical RAM only
Regular memory = physical RAM + swap

If you pin 32GB on a 64GB system:
- Only 32GB left for OS, other apps, pageable allocations
- System may become unresponsive
```

**2. Allocation is Expensive**

```fsharp
// SLOW - don't do in hot path
for batch in batches do
    use pinned = accelerator.AllocatePinned1D(size)  // Expensive!
    // ...

// FAST - allocate once, reuse
use pinned = accelerator.AllocatePinned1D(maxSize)  // Once at startup
for batch in batches do
    pinned.CopyFrom(batch)  // Fast
    // ...
```

**3. Memory Pressure on Host**

```
Pinned memory cannot be swapped.
Too much pinned memory → OOM killer → process death
```

**4. No Benefit for Small Transfers**

```
Transfer time = setup + (size / bandwidth)

Small transfer: setup dominates, pinning doesn't help
Large transfer: bandwidth dominates, pinning helps 2x
```

### Rule of Thumb

```fsharp
let shouldPin (transferSize: int64) (transferCount: int) =
    let totalBytes = transferSize * int64 transferCount
    let systemRam = getSystemRam()

    // Pin if:
    // 1. Transfer is large enough to benefit
    // 2. Won't consume too much system RAM
    transferSize > 1_000_000L &&           // > 1MB per transfer
    totalBytes < systemRam / 2L            // < 50% of RAM
```

### Production Pattern

```fsharp
// Allocate pinned buffers once at startup
type PinnedBufferPool(accelerator: Accelerator, bufferSize: int, count: int) =
    let buffers = Array.init count (fun _ ->
        accelerator.AllocatePinnedArray1D<float32>(bufferSize))
    let available = System.Collections.Concurrent.ConcurrentQueue(buffers)

    member _.Rent() =
        match available.TryDequeue() with
        | true, buf -> buf
        | false, _ -> failwith "No buffers available"

    member _.Return(buf) = available.Enqueue(buf)

    interface IDisposable with
        member _.Dispose() = buffers |> Array.iter (fun b -> b.Dispose())

// Usage
use pool = new PinnedBufferPool(accelerator, 100_000, 4)

for batch in batches do
    let buf = pool.Rent()
    buf.CopyFrom(batch)
    // ... use buffer ...
    pool.Return(buf)
```

### Summary

| Aspect | Recommendation |
|--------|----------------|
| When to pin | Large (>1MB), repeated transfers |
| How much | < 50% of system RAM |
| Allocation | Once at startup, reuse via pool |
| Pattern | Double-buffer for streaming |

---

## ILGPU Support

ILGPU already provides:

- Automatic register allocation for local variables (accumulators)
- Basic coalescing for simple access patterns
- Texture2D and Texture3D views
- Shared memory allocation via `SharedMemory.Allocate`
- Pinned memory via `Accelerator.AllocatePinned`
- Multiple streams for async operations

The optimizations above leverage existing ILGPU capabilities - no custom CUDA required.

---

## References

- [ILGPU Documentation](https://ilgpu.net/)
- [CUDA Best Practices Guide - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [NVIDIA GPU Memory Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)

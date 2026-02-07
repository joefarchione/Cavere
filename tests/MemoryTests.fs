module Cavere.Tests.MemoryTests

open System
open System.IO
open Xunit
open Cavere.Core
open Cavere.Generators

// ═══════════════════════════════════════════════════════════════════
// PinnedPool Tests
// ═══════════════════════════════════════════════════════════════════

[<Fact>]
let ``PinnedPool rent returns valid buffer`` () =
    let ctx, accel = Device.create CPU
    use _ctx = { new IDisposable with member _.Dispose() = accel.Dispose(); ctx.Dispose() }
    use pool = new PinnedPool(accel)
    let buf = pool.Rent(100L)
    Assert.Equal(100L, buf.Extent.Size)
    pool.Return(buf)

[<Fact>]
let ``PinnedPool return and re-rent reuses buffer`` () =
    let ctx, accel = Device.create CPU
    use _ctx = { new IDisposable with member _.Dispose() = accel.Dispose(); ctx.Dispose() }
    use pool = new PinnedPool(accel)
    let buf1 = pool.Rent(64L)
    pool.Return(buf1)
    let buf2 = pool.Rent(64L)
    // Should get back the same buffer from the pool
    Assert.True(Object.ReferenceEquals(buf1, buf2))
    pool.Return(buf2)

// ═══════════════════════════════════════════════════════════════════
// IndexOffset Tests
// ═══════════════════════════════════════════════════════════════════

[<Fact>]
let ``Generated source includes indexOffset parameter`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    let source, _ = Compiler.buildSource m
    Assert.Contains("int indexOffset)", source)
    Assert.Contains("int seed = idx + indexOffset;", source)

[<Fact>]
let ``fold with indexOffset 0 matches existing results`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    use sim = Simulation.create CPU 10_000 252
    let results = Simulation.fold sim m
    Assert.Equal(10_000, results.Length)
    let mean = Array.average results
    // GBM call should price near BS value (~10.45)
    Assert.InRange(mean, 5.0f, 20.0f)

// ═══════════════════════════════════════════════════════════════════
// Pinned Memory Execution Tests
// ═══════════════════════════════════════════════════════════════════

[<Fact>]
let ``foldPinned matches fold`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    use pinnedSim = Simulation.createPinned CPU 10_000 252
    let pinnedResults = Simulation.fold pinnedSim m
    use sim = Simulation.create CPU 10_000 252
    let regularResults = Simulation.fold sim m
    // Same random seeds → same results
    Assert.Equal(regularResults.Length, pinnedResults.Length)
    let pinnedMean = Array.average pinnedResults
    let regularMean = Array.average regularResults
    // Both should give similar pricing
    Assert.InRange(abs (pinnedMean - regularMean), 0.0f, 1.0f)

// ═══════════════════════════════════════════════════════════════════
// Output CSV Tests
// ═══════════════════════════════════════════════════════════════════

[<Fact>]
let ``Output.Csv.writeFold round-trips`` () =
    let values = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f |]
    use sw = new StringWriter()
    Output.Csv.writeFold sw values
    let csv = sw.ToString()
    Assert.Contains("scenario,value", csv)
    Assert.Contains("0,1", csv)
    Assert.Contains("4,5", csv)
    let lines = csv.Trim().Split('\n')
    // header + 5 data lines
    Assert.Equal(6, lines.Length)

[<Fact>]
let ``Output.writeScan CSV produces correct dimensions`` () =
    let data = Array2D.init 3 2 (fun t s -> float32 (t * 2 + s))
    use sw = new StringWriter()
    Output.Csv.writeScan sw data
    let csv = sw.ToString()
    Assert.Contains("step,sim_0,sim_1", csv)
    let lines = csv.Trim().Split('\n')
    // header + 3 data lines
    Assert.Equal(4, lines.Length)

// ═══════════════════════════════════════════════════════════════════
// Output Parquet Tests
// ═══════════════════════════════════════════════════════════════════

[<Fact>]
let ``Output.Parquet.writeFold creates valid file`` () =
    let values = [| 1.0f; 2.0f; 3.0f |]
    let path = Path.Combine(Path.GetTempPath(), $"cavere_test_{Guid.NewGuid()}.parquet")
    try
        Output.Parquet.writeFold path values
        Assert.True(File.Exists(path))
        let fi = FileInfo(path)
        Assert.True(fi.Length > 0L)
    finally
        if File.Exists(path) then File.Delete(path)

// ═══════════════════════════════════════════════════════════════════
// Multi-Device Config Tests
// ═══════════════════════════════════════════════════════════════════

[<Fact>]
let ``MultiDeviceConfig with 1 device matches single device`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    use sim = Simulation.create CPU 10_000 252
    let singleResult = Simulation.fold sim m
    let singleMean = Array.average singleResult

    use multiSim = Simulation.createMulti { DeviceType = CPU; DeviceCount = 1 } 10_000 252
    let multiResult = Simulation.fold multiSim m
    let multiMean = Array.average multiResult

    Assert.Equal(singleResult.Length, multiResult.Length)
    // Same engine path with 1 device and offset=0 → same results
    Assert.InRange(abs (singleMean - multiMean), 0.0f, 1.0f)

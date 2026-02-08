module Cavere.Tests.CompilerTests

open Xunit
open Cavere.Core
open Cavere.Generators

[<Fact>]
let ``Model CE allocates correct IDs`` () =
    let m =
        model {
            let dt = (1.0f / 252.0f).C
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            let! df = decay 0.05f.C dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    Assert.Equal(1, m.NormalCount)
    Assert.Equal(2, m.Accums.Count)
    Assert.Equal(Const 100.0f, m.Accums.[0].Init)
    Assert.Equal(Const 1.0f, m.Accums.[1].Init)

[<Fact>]
let ``Topological sort orders accums by dependency`` () =
    // accum_1 depends on accum_0
    let accums =
        Map.ofList [
            0, { Init = 0.05f.C; Body = 0.05f.C }
            1,
            {
                Init = 100.0f.C
                Body = AccumRef 1 * Expr.exp (AccumRef 0 * 0.01f.C)
            }
        ]

    let sorted = Compiler.sortAccums accums
    Assert.Equal(0, fst sorted.[0])
    Assert.Equal(1, fst sorted.[1])

[<Fact>]
let ``Generated source compiles with Roslyn`` () =
    let m =
        model {
            let dt = (1.0f / 252.0f).C
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            let! df = decay 0.05f.C dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    let source, _ = Compiler.buildSource m
    // Verify it contains expected structure
    Assert.Contains("public static void Fold(", source)
    Assert.Contains("public static void FoldWatch(", source)
    Assert.Contains("public static void Scan(", source)
    Assert.Contains("float accum_0", source)
    Assert.Contains("float accum_1", source)
    Assert.Contains("float z_0", source)

[<Fact>]
let ``Roslyn compiles generated source successfully`` () =
    let m =
        model {
            let dt = (1.0f / 252.0f).C
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            let! df = decay 0.05f.C dt
            return Expr.max (stock - 100.0f) 0.0f.C * df
        }

    let kernel = Compiler.build m
    Assert.NotNull(kernel.Assembly)
    Assert.NotNull(kernel.KernelType)
    Assert.Equal("GeneratedKernel", kernel.KernelType.Name)

[<Fact>]
let ``Constant model fold returns correct value`` () =
    let m = model { return 42.0f.C }
    use sim = Simulation.create CPU 100 10
    let results = Simulation.fold sim m
    Assert.Equal(100, results.Length)

    for v in results do
        Assert.Equal(42.0f, v)

[<Fact>]
let ``Surface1d generates interpolation code`` () =
    let m =
        model {
            let! sid = surface1d [| 0.03f; 0.04f; 0.05f |] 252
            let! result = interp1d sid TimeIndex
            return result
        }

    let source, _ = Compiler.buildSource m
    Assert.Contains("MathF.Floor(", source)
    Assert.Contains("surfaces[", source)

[<Fact>]
let ``Surface2d generates interpolation code`` () =
    let m =
        model {
            let! sid =
                surface2d
                    [| 0.0f; 1.0f |]
                    [| 50.0f; 150.0f |]
                    (array2D [| [| 0.20f; 0.20f |]; [| 0.20f; 0.20f |] |])
                    252

            let! z = normal
            let! stock = gbmLocalVol z sid 0.05f.C 100.0f.C (1.0f / 252.0f).C
            return stock
        }

    let source, _ = Compiler.buildSource m
    // interp2d uses Select chains for bin-finding and SurfaceAt for lookups
    Assert.Contains("?", source)
    Assert.Contains("surfaces[", source)

[<Fact>]
let ``Select generates branchless ternary`` () =
    let m =
        model {
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C (1.0f / 252.0f).C
            let floor = 90.0f.C
            return Expr.select (stock .< floor) floor stock
        }

    let source, _ = Compiler.buildSource m
    // Select with comparison emits clean bool: (a < b) ? x : y
    Assert.Contains("?", source)
    // No double-encode: condition goes directly into ternary
    Assert.DoesNotContain("> 0.5f)", source)

[<Fact>]
let ``Select fold returns correct values`` () =
    let m = model { return Expr.select (50.0f.C .> 100.0f.C) 1.0f.C 0.0f.C }
    use sim = Simulation.create CPU 100 10
    let results = Simulation.fold sim m
    results |> Array.iter (fun v -> Assert.Equal(0.0f, v))

    let m2 = model { return Expr.select (200.0f.C .> 100.0f.C) 1.0f.C 0.0f.C }
    let results2 = Simulation.fold sim m2
    results2 |> Array.iter (fun v -> Assert.Equal(1.0f, v))

// ── Batch kernel tests ─────────────────────────────────────────────

[<Fact>]
let ``FoldBatch source contains batchIdx and surfaces lookup`` () =
    let m =
        model {
            let! p = batchInput [| 1.0f; 2.0f |]
            let! x = evolve 0.0f.C (fun x -> x + p)
            return x
        }

    let source, _ = Compiler.buildBatchSource m
    Assert.Contains("public static void FoldBatch(", source)
    Assert.Contains("surfaces[", source)
    Assert.Contains("int batchIdx = idx / numSims;", source)

[<Fact>]
let ``FoldBatch source uses seed derived from scenarioIdx + indexOffset`` () =
    let m =
        model {
            let! p = batchInput [| 1.0f; 2.0f |]
            let! z = normal
            let! x = evolve 0.0f.C (fun x -> x + p * z)
            return x
        }

    let source, _ = Compiler.buildBatchSource m
    Assert.Contains("int seed = scenarioIdx + indexOffset;", source)
    Assert.Contains("seed * 6364136", source)
    Assert.DoesNotContain("public static void Fold(", source)

[<Fact>]
let ``Roslyn compiles FoldBatch successfully`` () =
    let m =
        model {
            let! p = batchInput [| 1.0f; 2.0f |]
            let! x = evolve 0.0f.C (fun x -> x + p)
            return x
        }

    let kernel = Compiler.buildBatch m
    Assert.NotNull(kernel.Assembly)
    Assert.NotNull(kernel.KernelType)

    let batchMethod =
        kernel.KernelType.GetMethod(
            "FoldBatch",
            System.Reflection.BindingFlags.Public ||| System.Reflection.BindingFlags.Static
        )

    Assert.NotNull(batchMethod)

[<Fact>]
let ``Batch fold produces correct means`` () =
    // Model: accum starts at premium (batch), adds rate (batch) each step
    // After `steps` steps: premium + rate * steps
    let premiums = [| 10.0f; 20.0f; 30.0f |]
    let rates = [| 1.0f; 2.0f; 3.0f |]

    let m =
        model {
            let! premium = batchInput premiums
            let! rate = batchInput rates
            let! x = evolve premium (fun x -> x + rate)
            return x
        }

    let steps = 5
    let numScenarios = 100
    use sim = BatchSimulation.create CPU 3 numScenarios steps
    let means = BatchSimulation.foldMeans sim m
    Assert.Equal(3, means.Length)

    for i in 0..2 do
        let expected = premiums.[i] + rates.[i] * float32 steps
        Assert.InRange(means.[i], expected - 0.01f, expected + 0.01f)

[<Fact>]
let ``Batch fold shares random scenarios across batch elements`` () =
    // Two batch elements with same params should get identical results
    let m =
        model {
            let! scale = batchInput [| 1.0f; 1.0f |]
            let! z = normal
            let! x = evolve 0.0f.C (fun x -> x + scale * z)
            return x
        }

    let numScenarios = 1000
    use sim = BatchSimulation.create CPU 2 numScenarios 10
    let raw = BatchSimulation.fold sim m

    for i in 0 .. numScenarios - 1 do
        Assert.Equal(raw.[i], raw.[numScenarios + i])

// ── Validation tests ─────────────────────────────────────────────

[<Fact>]
let ``Validate rejects dangling surface ID`` () =
    let badModel = {
        Result = Lookup1D 99
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0
        UniformCount = 0
        BernoulliCount = 0
        BatchSize = 0
    }

    let ex = Assert.Throws<exn>(fun () -> Compiler.build badModel |> ignore)
    Assert.Contains("Surface IDs", ex.Message)
    Assert.Contains("99", ex.Message)

[<Fact>]
let ``Validate rejects dangling AccumRef`` () =
    let badModel = {
        Result = AccumRef 5
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0
        UniformCount = 0
        BernoulliCount = 0
        BatchSize = 0
    }

    let ex = Assert.Throws<exn>(fun () -> Compiler.build badModel |> ignore)
    Assert.Contains("AccumRef IDs", ex.Message)
    Assert.Contains("5", ex.Message)

[<Fact>]
let ``Validate passes for well-formed model`` () =
    let m =
        model {
            let dt = (1.0f / 252.0f).C
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
            return stock
        }
    // Should not throw
    Compiler.build m |> ignore

[<Fact>]
let ``Bernoulli model compiles and generates b_0`` () =
    let m =
        model {
            let! b = bernoulli
            let! x = evolve 0.0f.C (fun x -> x + b)
            return x
        }

    Assert.Equal(1, m.BernoulliCount)
    let source, _ = Compiler.buildSource m
    Assert.Contains("float b_0", source)
    Assert.Contains("bernoulliCount", source)

// ── Condition combinator tests ─────────────────────────────────────

[<Fact>]
let ``And/Or/Not generate clean boolean C#`` () =
    let m =
        model {
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C (1.0f / 252.0f).C
            let inRange = (stock .> 80.0f.C) .&& (stock .< 120.0f.C)
            return Expr.select inRange stock 0.0f.C
        }

    let source, _ = Compiler.buildSource m
    // Should emit clean && instead of nested ternaries
    Assert.Contains("&&", source)
    // Select should use direct bool, not > 0.5f wrapper
    Assert.DoesNotContain("> 0.5f)", source)

[<Fact>]
let ``Or generates || in C#`` () =
    let m =
        model {
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C (1.0f / 252.0f).C
            let trigger = (stock .< 80.0f.C) .|| (stock .> 120.0f.C)
            return Expr.select trigger 1.0f.C 0.0f.C
        }

    let source, _ = Compiler.buildSource m
    Assert.Contains("||", source)

[<Fact>]
let ``Not generates ! in C#`` () =
    let m =
        model {
            let! z = normal
            let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C (1.0f / 252.0f).C
            let notExpired = Expr.not' (TimeIndex .> 200.0f.C)
            return Expr.select notExpired stock 0.0f.C
        }

    let source, _ = Compiler.buildSource m
    Assert.Contains("!", source)

[<Fact>]
let ``Condition combinators fold correctly`` () =
    // And: both true → true, otherwise false
    let m1 = model { return Expr.select ((100.0f.C .> 50.0f.C) .&& (200.0f.C .> 150.0f.C)) 1.0f.C 0.0f.C }
    use sim = Simulation.create CPU 100 1
    let r1 = Simulation.fold sim m1
    r1 |> Array.iter (fun v -> Assert.Equal(1.0f, v))

    // And: one false → false
    let m2 = model { return Expr.select ((100.0f.C .> 50.0f.C) .&& (200.0f.C .< 150.0f.C)) 1.0f.C 0.0f.C }
    let r2 = Simulation.fold sim m2
    r2 |> Array.iter (fun v -> Assert.Equal(0.0f, v))

    // Or: one true → true
    let m3 = model { return Expr.select ((100.0f.C .< 50.0f.C) .|| (200.0f.C .> 150.0f.C)) 1.0f.C 0.0f.C }
    let r3 = Simulation.fold sim m3
    r3 |> Array.iter (fun v -> Assert.Equal(1.0f, v))

    // Not: negation
    let m4 = model { return Expr.select (Expr.not' (100.0f.C .> 50.0f.C)) 1.0f.C 0.0f.C }
    let r4 = Simulation.fold sim m4
    r4 |> Array.iter (fun v -> Assert.Equal(0.0f, v))

[<Fact>]
let ``Bool arithmetic: condition * float works`` () =
    // (100 > 50) is 1.0f, so 1.0f * 42.0f = 42.0f
    let m1 = model { return (100.0f.C .> 50.0f.C) * 42.0f.C }
    use sim = Simulation.create CPU 100 1
    let r1 = Simulation.fold sim m1
    r1 |> Array.iter (fun v -> Assert.Equal(42.0f, v))

    // (100 < 50) is 0.0f, so 0.0f * 42.0f = 0.0f
    let m2 = model { return (100.0f.C .< 50.0f.C) * 42.0f.C }
    let r2 = Simulation.fold sim m2
    r2 |> Array.iter (fun v -> Assert.Equal(0.0f, v))

[<Fact>]
let ``Bernoulli fold produces 0 or 1 values`` () =
    let m =
        model {
            let! b = bernoulli
            let! x = evolve 0.0f.C (fun _ -> b)
            return x
        }

    use sim = Simulation.create CPU 1000 1
    let results = Simulation.fold sim m

    for v in results do
        Assert.True(v = 0.0f || v = 1.0f, $"Expected 0 or 1, got {v}")

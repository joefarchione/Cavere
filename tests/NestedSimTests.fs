module Cavere.Tests.NestedSimTests

open Xunit
open Cavere.Core
open Cavere.Generators

[<Fact>]
let ``FoldBatch codegen contains FoldBatch method`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! stock0 = batchInput [| 100.0f; 110.0f |]
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C stock0 dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    let source, _ = Compiler.buildBatchSource m
    Assert.Contains("public static void FoldBatch(", source)
    Assert.Contains("int batchIdx = idx / numSims;", source)
    Assert.Contains("surfaces[", source)

[<Fact>]
let ``Roslyn compiles FoldBatch successfully`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! stock0 = batchInput [| 100.0f; 110.0f |]
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C stock0 dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    let kernel = Compiler.buildBatch m
    Assert.NotNull(kernel.Assembly)
    Assert.NotNull(kernel.KernelType)
    let foldBatchMethod =
        kernel.KernelType.GetMethod("FoldBatch",
            System.Reflection.BindingFlags.Public ||| System.Reflection.BindingFlags.Static)
    Assert.NotNull(foldBatchMethod)

[<Fact>]
let ``Identity accum with batchInput returns batch values`` () =
    // Model: accum_0 starts from batchInput, body = self, result = accum_0
    // With 0 steps, the time loop doesn't execute, so result = init from batchInput
    let m = model {
        let! x0 = batchInput [| 10.0f; 20.0f; 30.0f |]
        let! x = evolve x0 (fun self -> self)
        return x
    }
    use sim = BatchSimulation.create CPU 3 1 0
    let raw = BatchSimulation.fold sim m
    // numScenarios=1, so each batch element has 1 scenario, init comes from batchInput
    Assert.Equal(3, raw.Length)
    Assert.Equal(10.0f, raw.[0])
    Assert.Equal(20.0f, raw.[1])
    Assert.Equal(30.0f, raw.[2])

[<Fact>]
let ``Watcher sliceObs matches manual extraction from values`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        do! observe "stock" stock
        return stock
    }
    use sim = Simulation.create CPU 200 252
    let _, watch = Simulation.foldWatch sim m Quarterly
    let allVals = Watcher.values "stock" watch
    let obsIdx = 1
    let sliced = Watcher.sliceObs "stock" obsIdx watch
    Assert.Equal(200, sliced.Length)
    for i in 0 .. 199 do
        Assert.Equal(allVals.[obsIdx, i], sliced.[i])

[<Fact>]
let ``End-to-end outer foldWatch into inner BatchSimulation.foldMeans`` () =
    let dt = (1.0f / 252.0f).C
    let outerModel = model {
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        do! observe "stock" stock
        return stock
    }
    let numOuter = 500
    use outerSim = Simulation.create CPU numOuter 252
    let _, watch = Simulation.foldWatch outerSim outerModel Quarterly

    // Get outer stock values at first observation date
    let outerStocks = Watcher.sliceObs "stock" 0 watch

    // Inner model: continue GBM from outer stock level, price a call
    let innerModel = model {
        let! stock0 = batchInput outerStocks
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C stock0 dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }

    let numScenarios = 200
    use innerSim = BatchSimulation.create CPU numOuter numScenarios 189  // ~3/4 year remaining
    let batchMeans = BatchSimulation.foldMeans innerSim innerModel

    // Basic sanity: right shape, non-negative, finite
    Assert.Equal(numOuter, batchMeans.Length)
    batchMeans |> Array.iter (fun v ->
        Assert.True(System.Single.IsFinite(v), sprintf "Non-finite value: %f" v)
        Assert.True(v >= 0.0f, sprintf "Negative conditional expectation: %f" v))
    // At least some should be positive (stock likely > 100 for some paths)
    let positiveCount = batchMeans |> Array.filter (fun v -> v > 0.0f) |> Array.length
    Assert.True(positiveCount > 0, "Expected at least some positive conditional expectations")

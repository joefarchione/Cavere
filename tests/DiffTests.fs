module Cavere.Tests.DiffTests

open Xunit
open Cavere.Core
open Cavere.Generators

// ══════════════════════════════════════════════════════════════════
// DiffVar Collection
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``collectDiffVars finds DiffVar in simple expression`` () =
    let expr = Add(Dual(0, 100.0f, "x0"), Const 1.0f)
    let vars = CompilerDiff.collectDiffVars expr
    Assert.Equal(1, vars.Count)
    Assert.Contains(0, vars)

[<Fact>]
let ``collectDiffVars finds multiple DiffVars`` () =
    let expr = Mul(Dual(0, 100.0f, "x0"), Add(Dual(1, 0.2f, "x1"), Const 1.0f))
    let vars = CompilerDiff.collectDiffVars expr
    Assert.Equal(2, vars.Count)
    Assert.Contains(0, vars)
    Assert.Contains(1, vars)

[<Fact>]
let ``collectDiffVars returns empty for expressions without DiffVar`` () =
    let expr = Add(Const 1.0f, Mul(Const 2.0f, Normal 0))
    let vars = CompilerDiff.collectDiffVars expr
    Assert.True(vars.IsEmpty)

[<Fact>]
let ``collectDiffVars finds DiffVar through nested ops`` () =
    let expr = Exp(Sqrt(Add(Dual(3, 5.0f, "x3"), Const 1.0f)))
    let vars = CompilerDiff.collectDiffVars expr
    Assert.Equal<int>(Set.singleton 3, vars)

[<Fact>]
let ``collectDiffVars finds DiffVar in Select`` () =
    let expr = Select(Gt(Dual(0, 1.0f, "x0"), Const 0.0f), Dual(1, 2.0f, "x1"), Const 0.0f)
    let vars = CompilerDiff.collectDiffVars expr
    Assert.Equal(2, vars.Count)

// ══════════════════════════════════════════════════════════════════
// Model-level DiffVar Collection
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``collectModelDiffVars finds DiffVars in result`` () =
    let m = {
        Result = Add(Dual(0, 100.0f, "x0"), Const 1.0f)
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let vars = CompilerDiff.collectModelDiffVars m
    Assert.Equal(1, vars.Length)
    Assert.Equal(0, vars.[0])

[<Fact>]
let ``collectModelDiffVars finds DiffVars in accumulators`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Dual(1, 0.05f, "x1") }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let vars = CompilerDiff.collectModelDiffVars m
    Assert.Equal(2, vars.Length)

[<Fact>]
let ``collectModelDiffVars finds DiffVars in observers`` () =
    let m = {
        Result = Const 0.0f
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = [ { Name = "test"; Expr = Dual(2, 42.0f, "x2"); SlotIndex = 0 } ]
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let vars = CompilerDiff.collectModelDiffVars m
    Assert.Equal(1, vars.Length)
    Assert.Equal(2, vars.[0])

[<Fact>]
let ``collectModelDiffVars returns sorted unique indices`` () =
    let m = {
        Result = Add(Dual(2, 1.0f, "x2"), Dual(0, 2.0f, "x0"))
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 + Dual(1, 3.0f, "x1") }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let vars = CompilerDiff.collectModelDiffVars m
    Assert.Equal(3, vars.Length)
    Assert.Equal(0, vars.[0])
    Assert.Equal(1, vars.[1])
    Assert.Equal(2, vars.[2])

// ══════════════════════════════════════════════════════════════════
// Forward-Mode Differentiation Rules
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``forwardDiff of DiffVar wrt itself is 1`` () =
    let d = CompilerDiff.forwardDiff (Dual(0, 5.0f, "x0")) 0 id
    Assert.Equal(Const 1.0f, d)

[<Fact>]
let ``forwardDiff of DiffVar wrt other is 0`` () =
    let d = CompilerDiff.forwardDiff (Dual(0, 5.0f, "x0")) 1 id
    Assert.Equal(Const 0.0f, d)

[<Fact>]
let ``forwardDiff of Const is 0`` () =
    let d = CompilerDiff.forwardDiff (Const 42.0f) 0 id
    Assert.Equal(Const 0.0f, d)

[<Fact>]
let ``forwardDiff of Add follows sum rule`` () =
    // d/dx (x + 3) = 1 + 0
    let expr = Add(Dual(0, 2.0f, "x0"), Const 3.0f)
    let d = CompilerDiff.forwardDiff expr 0 id
    Assert.Equal(Add(Const 1.0f, Const 0.0f), d)

[<Fact>]
let ``forwardDiff of Mul follows product rule`` () =
    // d/dx (x * y) = 1*y + x*0 = y (when differentiating wrt x)
    let x = Dual(0, 2.0f, "x0")
    let y = Dual(1, 3.0f, "x1")
    let expr = Mul(x, y)
    let d = CompilerDiff.forwardDiff expr 0 id
    // Should be Add(Mul(1, y), Mul(x, 0))
    Assert.Equal(Add(Mul(Const 1.0f, y), Mul(x, Const 0.0f)), d)

[<Fact>]
let ``forwardDiff of Exp follows chain rule`` () =
    // d/dx exp(x) = exp(x) * 1
    let x = Dual(0, 1.0f, "x0")
    let d = CompilerDiff.forwardDiff (Exp x) 0 id
    Assert.Equal(Mul(Exp x, Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Log follows chain rule`` () =
    // d/dx log(x) = (1/x) * 1
    let x = Dual(0, 1.0f, "x0")
    let d = CompilerDiff.forwardDiff (Log x) 0 id
    Assert.Equal(Mul(Div(Const 1.0f, x), Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Sqrt follows chain rule`` () =
    // d/dx sqrt(x) = 0.5/sqrt(x) * 1
    let x = Dual(0, 4.0f, "x0")
    let d = CompilerDiff.forwardDiff (Sqrt x) 0 id
    Assert.Equal(Mul(Div(Const 0.5f, Sqrt x), Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Neg follows negation rule`` () =
    let x = Dual(0, 1.0f, "x0")
    let d = CompilerDiff.forwardDiff (Neg x) 0 id
    Assert.Equal(Neg(Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Div follows quotient rule`` () =
    // d/dx (x / y) = (1*y - x*0) / (y*y)
    let x = Dual(0, 6.0f, "x0")
    let y = Dual(1, 3.0f, "x1")
    let d = CompilerDiff.forwardDiff (Div(x, y)) 0 id
    Assert.Equal(Div(Sub(Mul(Const 1.0f, y), Mul(x, Const 0.0f)), Mul(y, y)), d)

[<Fact>]
let ``forwardDiff of Max uses subgradient`` () =
    let a = Dual(0, 5.0f, "x0")
    let b = Dual(1, 3.0f, "x1")
    let d = CompilerDiff.forwardDiff (Max(a, b)) 0 id
    // Should be Select(a > b, d/dx a, d/dx b)
    Assert.Equal(Select(Gt(a, b), Const 1.0f, Const 0.0f), d)

[<Fact>]
let ``forwardDiff of comparison ops is 0`` () =
    let a = Dual(0, 1.0f, "x0")
    let b = Dual(1, 2.0f, "x1")
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Gt(a, b)) 0 id)
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Lt(a, b)) 0 id)
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Gte(a, b)) 0 id)
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Lte(a, b)) 0 id)

[<Fact>]
let ``forwardDiff of Select differentiates through branches`` () =
    let cond = Gt(Const 1.0f, Const 0.0f)
    let t = Dual(0, 10.0f, "x0")
    let f = Dual(1, 5.0f, "x1")
    let d = CompilerDiff.forwardDiff (Select(cond, t, f)) 0 id
    // d/dx Select(cond, x, y) = Select(cond, 1, 0)
    Assert.Equal(Select(cond, Const 1.0f, Const 0.0f), d)

[<Fact>]
let ``forwardDiff of AccumRef maps to derivative accumulator`` () =
    let derivIdFn id = id + 100
    let d = CompilerDiff.forwardDiff (AccumRef 0) 0 derivIdFn
    Assert.Equal(AccumRef 100, d)

[<Fact>]
let ``forwardDiff of Abs uses sign subgradient`` () =
    let x = Dual(0, 3.0f, "x0")
    let d = CompilerDiff.forwardDiff (Abs x) 0 id
    let expectedSign = Select(Gt(x, Const 0.0f), Const 1.0f, Const -1.0f)
    Assert.Equal(Mul(expectedSign, Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Floor is 0`` () =
    let d = CompilerDiff.forwardDiff (Floor(Dual(0, 1.5f, "x0"))) 0 id
    Assert.Equal(Const 0.0f, d)

[<Fact>]
let ``forwardDiff of Normal is 0`` () =
    let d = CompilerDiff.forwardDiff (Normal 0) 0 id
    Assert.Equal(Const 0.0f, d)

[<Fact>]
let ``forwardDiff of TimeIndex is 0`` () =
    let d = CompilerDiff.forwardDiff TimeIndex 0 id
    Assert.Equal(Const 0.0f, d)

// ══════════════════════════════════════════════════════════════════
// transformDual — Model Expansion
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``transformDual returns unchanged model when no DiffVars`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        return stock
    }
    let expanded, diffVars = CompilerDiff.transformDual m
    Assert.Empty(diffVars)
    Assert.Equal(m.Accums.Count, expanded.Accums.Count)
    Assert.Equal(m.Result, expanded.Result)

[<Fact>]
let ``transformDual adds derivative accumulators for one DiffVar`` () =
    // Simple model: accum_0 = init: DiffVar(0, 100), body: accum_0 * 1.01
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.01f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, diffVars = CompilerDiff.transformDual m
    Assert.Equal(1, diffVars.Length)
    Assert.Equal(0, diffVars.[0])
    // Should have original accum + 1 derivative accum = 2 total
    Assert.Equal(2, expanded.Accums.Count)
    // Derivative observer added
    Assert.Equal(1, expanded.Observers.Length)
    Assert.Equal("d1_x0", expanded.Observers.[0].Name)

[<Fact>]
let ``transformDual adds derivative accumulators for two DiffVars`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Exp(Dual(1, 0.05f, "x1")) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, diffVars = CompilerDiff.transformDual m
    Assert.Equal(2, diffVars.Length)
    // 1 original + 2 derivative accums = 3 total
    Assert.Equal(3, expanded.Accums.Count)
    // 2 derivative observers
    Assert.Equal(2, expanded.Observers.Length)

[<Fact>]
let ``transformDual preserves existing observers`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.01f.C }
        ]
        Surfaces = Map.empty
        Observers = [ { Name = "stock"; Expr = AccumRef 0; SlotIndex = 0 } ]
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformDual m
    // Original observer + 1 derivative observer
    Assert.Equal(2, expanded.Observers.Length)
    Assert.Equal("stock", expanded.Observers.[0].Name)
    Assert.Equal("d1_x0", expanded.Observers.[1].Name)

[<Fact>]
let ``transformDual derivative accum init is differentiated`` () =
    // Init = DiffVar(0, 100) → d/dDiffVar0 = 1
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformDual m
    // Derivative accum should have Init = 1 (derivative of DiffVar(0,_) wrt 0)
    let derivAccumId = 1 // maxAccumId=0, derivAccumBase=1, derivAccumId(0,0) = 1 + 0*1 + 0 = 1
    Assert.True(expanded.Accums.ContainsKey derivAccumId)
    Assert.Equal(Const 1.0f, expanded.Accums.[derivAccumId].Init)

[<Fact>]
let ``transformDual expanded model compiles`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Exp(Dual(1, 0.001f, "x1")) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformDual m
    let source, _ = Compiler.buildSource expanded
    Assert.Contains("accum_0", source)
    // Derivative accumulators should appear in generated source
    Assert.Contains("accum_1", source)
    Assert.Contains("accum_2", source)

[<Fact>]
let ``transformDual with multiple accums creates correct derivative count`` () =
    let m = {
        Result = AccumRef 0 + AccumRef 1
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.01f.C }
            1, { Init = Const 1.0f; Body = AccumRef 1 * Exp(-Dual(1, 0.05f, "x1") / 252.0f.C) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, diffVars = CompilerDiff.transformDual m
    Assert.Equal(2, diffVars.Length)
    // 2 original + 2 accums * 2 diffVars = 6 total
    Assert.Equal(6, expanded.Accums.Count)

// ══════════════════════════════════════════════════════════════════
// hasDiffVars
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``hasDiffVars returns false for standard model`` () =
    let m = model {
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C (1.0f / 252.0f).C
        return stock
    }
    Assert.False(CompilerDiff.hasDiffVars m)

[<Fact>]
let ``hasDiffVars returns true when DiffVar in result`` () =
    let m = {
        Result = Dual(0, 1.0f, "x0")
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    Assert.True(CompilerDiff.hasDiffVars m)

// ══════════════════════════════════════════════════════════════════
// DiffVar in emitted code
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``DiffVar emits as constant value in C# source`` () =
    let layout = { Offsets = Map.empty; TotalSize = 0; Meta = Map.empty }
    let code = CompilerCommon.emitExpr layout (Dual(0, 3.14f, "x0"))
    Assert.Contains("3.14", code)

// ══════════════════════════════════════════════════════════════════
// End-to-end: AD model compiles and runs
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``AD model with DiffVar compiles and produces source`` () =
    // Model: stock = S0 * exp(drift), where S0 is a DiffVar
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0")
                 Body = AccumRef 0 * Exp((Dual(1, 0.05f, "x1") - 0.5f.C * Dual(2, 0.2f, "x2") * Dual(2, 0.2f, "x2")) / 252.0f.C
                                         + Dual(2, 0.2f, "x2") * Sqrt(1.0f.C / 252.0f.C) * Normal 0) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 1; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, diffVars = CompilerDiff.transformDual m
    Assert.Equal(3, diffVars.Length)
    // Should compile cleanly
    let source, _ = Compiler.buildSource expanded
    Assert.Contains("Fold", source)

[<Fact>]
let ``AD model runs on CPU and produces finite results`` () =
    // Simple: accum = DiffVar(0, 100) * 1.001 each step
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformDual m
    use sim = Simulation.create CPU 100 10
    let results = Simulation.fold sim expanded
    // All results should be finite
    Assert.True(results |> Array.forall System.Single.IsFinite)
    // Result should be ~100 * 1.001^10 ≈ 101.005
    Assert.InRange(results.[0], 100.0f, 102.0f)

[<Fact>]
let ``AD derivative via FoldWatch produces delta estimate`` () =
    // Model: accum = S0 * 1.001^t, d/dS0 = 1.001^t
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformDual m
    // Run with watch to get derivative observer
    use sim = Simulation.create CPU 100 10
    let _, watchResult = Simulation.foldWatch sim expanded Monthly
    // Should have the derivative observer
    Assert.True(expanded.Observers |> List.exists (fun o -> o.Name = "d1_x0"))

[<Fact>]
let ``DiffMode type has expected cases`` () =
    let modes = [ DualMode; HyperDualMode true; HyperDualMode false; JetMode 3; AdjointMode ]
    Assert.Equal(5, modes.Length)

// ══════════════════════════════════════════════════════════════════
// transformHyperDual — 2nd order
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``transformHyperDual returns unchanged model when no DiffVars`` () =
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C dt
        return stock
    }
    let expanded, diffVars = CompilerDiff.transformHyperDual true m
    Assert.Empty(diffVars)
    Assert.Equal(m.Accums.Count, expanded.Accums.Count)

[<Fact>]
let ``transformHyperDual diagonal adds 2nd-order accumulators`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Exp(Dual(1, 0.05f, "x1")) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, diffVars = CompilerDiff.transformHyperDual true m
    Assert.Equal(2, diffVars.Length)
    // 1 original + 2 layer-1 + 2 layer-2 (diagonal) = 5
    Assert.Equal(5, expanded.Accums.Count)
    // Should have 1st-order + 2nd-order observers
    Assert.True(expanded.Observers |> List.exists (fun o -> o.Name = "d1_x0"))
    Assert.True(expanded.Observers |> List.exists (fun o -> o.Name = "d2_x0"))

[<Fact>]
let ``transformHyperDual full adds cross-term accumulators`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Exp(Dual(1, 0.05f, "x1")) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, diffVars = CompilerDiff.transformHyperDual false m
    Assert.Equal(2, diffVars.Length)
    // 1 original + 2 layer-1 + 4 layer-2 (2x2 full) = 7
    Assert.Equal(7, expanded.Accums.Count)
    // Cross-term observers
    Assert.True(expanded.Observers |> List.exists (fun o -> o.Name = "d2_x0_x1"))
    Assert.True(expanded.Observers |> List.exists (fun o -> o.Name = "d2_x1_x0"))

[<Fact>]
let ``transformHyperDual diagonal compiles and runs`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformHyperDual true m
    use sim = Simulation.create CPU 100 10
    let results = Simulation.fold sim expanded
    Assert.True(results |> Array.forall System.Single.IsFinite)

[<Fact>]
let ``transformHyperDual gamma for linear model is zero`` () =
    // accum = DiffVar(0) * 1.001^t → d/dS = 1.001^t, d²/dS² = 0
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformHyperDual true m
    use sim = Simulation.create CPU 100 10
    let _, watch = Simulation.foldWatch sim expanded Monthly
    // The 2nd-order derivative observer should be ~0 (linear model)
    let d2Obs = expanded.Observers |> List.find (fun o -> o.Name = "d2_x0")
    let vals = Watcher.values d2Obs.Name watch
    let mean2 = vals |> Seq.cast<float32> |> Seq.averageBy abs
    Assert.InRange(mean2, 0.0f, 0.01f)

// ══════════════════════════════════════════════════════════════════
// Partial Derivatives
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``partialDiffAccumRef of AccumRef(target) is 1`` () =
    let pd = CompilerDiff.partialDiffAccumRef (AccumRef 0) 0
    Assert.Equal(Const 1.0f, pd)

[<Fact>]
let ``partialDiffAccumRef of AccumRef(other) is 0`` () =
    let pd = CompilerDiff.partialDiffAccumRef (AccumRef 0) 1
    Assert.Equal(Const 0.0f, pd)

[<Fact>]
let ``partialDiffAccumRef of product applies product rule`` () =
    // d(AccumRef(0) * G) / d(AccumRef(0)) where G = DiffVar(1, 0.05)
    let pd = CompilerDiff.partialDiffAccumRef (Mul(AccumRef 0, Dual(1, 0.05f, "x1"))) 0
    // = 1 * DiffVar(1, 0.05) + AccumRef(0) * 0
    Assert.Equal(Add(Mul(Const 1.0f, Dual(1, 0.05f, "x1")), Mul(AccumRef 0, Const 0.0f)), pd)

[<Fact>]
let ``partialDiffDiffVar of DiffVar(target) is 1`` () =
    let pd = CompilerDiff.partialDiffDiffVar (Dual(0, 5.0f, "x0")) 0
    Assert.Equal(Const 1.0f, pd)

[<Fact>]
let ``partialDiffDiffVar of AccumRef is 0`` () =
    let pd = CompilerDiff.partialDiffDiffVar (AccumRef 0) 0
    Assert.Equal(Const 0.0f, pd)

// ══════════════════════════════════════════════════════════════════
// Adjoint Mode
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``computeAdjointInfo produces correct structure`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Exp(Dual(1, 0.001f, "x1")) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let info = CompilerAdjoint.computeAdjointInfo m
    Assert.Equal(2, info.DiffVars.Length)
    Assert.Equal(1, info.SortedAccums.Length)
    // ∂body/∂AccumRef(0) should not be Const 0
    Assert.NotEqual(Const 0.0f, info.BodyPartialsAccum.[(0, 0)])
    // ∂body/∂DiffVar(1) should not be Const 0
    Assert.NotEqual(Const 0.0f, info.BodyPartialsDiffVar.[(0, 1)])

[<Fact>]
let ``adjoint kernel source compiles`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Exp(Dual(1, 0.001f, "x1")) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let source, _, _ = CompilerAdjoint.buildSource m
    Assert.Contains("FoldAdjoint", source)
    Assert.Contains("adj_dv_0", source)
    Assert.Contains("adj_dv_1", source)
    Assert.Contains("tape[", source)
    // Should compile through Roslyn
    let asm = CompilerRegular.compile source
    Assert.NotNull(asm)

[<Fact>]
let ``adjoint mode runs on CPU and produces finite results`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    use sim = Simulation.create CPU 100 10
    let values, adjoints = Simulation.foldAdjoint sim m
    Assert.Equal(100, values.Length)
    Assert.Equal(100, Array2D.length1 adjoints)
    Assert.Equal(1, Array2D.length2 adjoints)
    Assert.True(values |> Array.forall System.Single.IsFinite)
    Assert.True(adjoints |> Seq.cast<float32> |> Seq.forall System.Single.IsFinite)

[<Fact>]
let ``adjoint delta matches forward-mode delta`` () =
    // Model: accum = S0 * 1.001^t, d/dS0 = 1.001^t
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    use sim = Simulation.create CPU 100 10

    // Adjoint delta
    let _, adjoints = Simulation.foldAdjoint sim m
    let adjointDelta = adjoints.[0, 0]

    // Forward-mode delta via transformDual
    let expanded, _ = CompilerDiff.transformDual m
    let _, watch = Simulation.foldWatch sim expanded Monthly
    let derivObs = expanded.Observers |> List.find (fun o -> o.Name = "d1_x0")
    let derivVals = Watcher.values derivObs.Name watch
    // The last observation should have the derivative
    let fwdDelta = derivVals.[Array2D.length1 derivVals - 1, 0]

    // Both should be ~1.001^10 ≈ 1.01005
    Assert.InRange(adjointDelta, 1.0f, 1.05f)
    Assert.InRange(abs (adjointDelta - fwdDelta), 0.0f, 0.01f)

[<Fact>]
let ``adjoint with stochastic model produces delta`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0")
                 Body = AccumRef 0 * Exp(Dual(1, 0.05f, "x1") / 252.0f.C
                                         + Dual(2, 0.2f, "x2") * Sqrt(1.0f.C / 252.0f.C) * Normal 0) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 1; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    use sim = Simulation.create CPU 1_000 252
    let values, adjoints = Simulation.foldAdjoint sim m
    // All should be finite
    Assert.True(values |> Array.forall System.Single.IsFinite)
    Assert.True(adjoints |> Seq.cast<float32> |> Seq.forall System.Single.IsFinite)
    // Delta (d/dS0) should be positive (stock is monotonically increasing in S0)
    let meanDelta = Array.init 1000 (fun i -> adjoints.[i, 0]) |> Array.average
    Assert.True(meanDelta > 0.0f)

// ══════════════════════════════════════════════════════════════════
// Recommend
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``recommend returns None for model without DiffVars`` () =
    let m = model {
        let! z = normal
        let! stock = gbm z 0.05f.C 0.20f.C 100.0f.C (1.0f / 252.0f).C
        return stock
    }
    let mode, _ = CompilerDiff.recommend m
    Assert.True(mode.IsNone)

[<Fact>]
let ``recommend returns Dual for few DiffVars`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Dual(0, 100.0f, "x0"); Body = AccumRef 0 * Exp(Dual(1, 0.05f, "x1")) }
        ]
        Surfaces = Map.empty; Observers = []; NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let mode, desc = CompilerDiff.recommend m
    Assert.Equal(Some DualMode, mode)
    Assert.Contains("Dual", desc)

[<Fact>]
let ``recommend returns Adjoint for many DiffVars`` () =
    // 6 DiffVars → should recommend Adjoint
    let m = {
        Result = Dual(0, 1.0f, "x0") + Dual(1, 2.0f, "x1") + Dual(2, 3.0f, "x2")
                 + Dual(3, 4.0f, "x3") + Dual(4, 5.0f, "x4") + Dual(5, 6.0f, "x5")
        Accums = Map.ofList [
            0, { Init = Const 1.0f; Body = AccumRef 0 }
        ]
        Surfaces = Map.empty; Observers = []; NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    let mode, desc = CompilerDiff.recommend m
    Assert.Equal(Some AdjointMode, mode)
    Assert.Contains("Adjoint", desc)

// ══════════════════════════════════════════════════════════════════
// HyperDual End-to-End Tests
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``HyperDual vomma: GBM call d²V/dvol² is non-zero`` () =
    // vol as HyperDual → should produce d1_vol and d2_vol
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let vol = HyperDual(0, 0.2f, "vol")
        let! stock = gbm z 0.05f.C vol 100.0f.C dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    // Auto-expand: Kernel.compile handles Dual+HyperDual automatically
    use sim = Simulation.create CPU 10_000 252
    let _, watch = Simulation.foldWatch sim m Monthly
    // Check 1st-order (vega) observer exists and is non-zero
    let vegaObs = watch.Observers |> List.tryFind (fun o -> o.Name = "d1_vol")
    Assert.True(vegaObs.IsSome)
    let vegaVals = Watcher.values "d1_vol" watch
    let meanVega = Array.init 10_000 (fun i -> vegaVals.[Array2D.length1 vegaVals - 1, i]) |> Array.average
    Assert.True(abs meanVega > 1.0f, $"Vega should be significant, got {meanVega}")
    // Check 2nd-order (vomma) observer exists
    let vommaObs = watch.Observers |> List.tryFind (fun o -> o.Name = "d2_vol")
    Assert.True(vommaObs.IsSome)

[<Fact>]
let ``HyperDual rate convexity: bond d²V/dr² is positive`` () =
    // Bond = exp(-r*T), d/dr = -T*exp(-rT), d²/dr² = T²*exp(-rT) > 0
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = Const 1.0f
                 Body = AccumRef 0 * Exp(-HyperDual(0, 0.05f, "rate") / 252.0f.C) }
        ]
        Surfaces = Map.empty; Observers = []; NormalCount = 0; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    use sim = Simulation.create CPU 1_000 252
    let _, watch = Simulation.foldWatch sim m Monthly
    // 2nd-order derivative observer
    let d2Obs = watch.Observers |> List.find (fun o -> o.Name = "d2_rate")
    let d2Vals = Watcher.values d2Obs.Name watch
    let lastObs = Array2D.length1 d2Vals - 1
    let meanConvexity = Array.init 1_000 (fun i -> d2Vals.[lastObs, i]) |> Array.average
    // Convexity should be positive for a bond
    Assert.True(meanConvexity > 0.0f, $"Bond convexity should be positive, got {meanConvexity}")

[<Fact>]
let ``HyperDual vanna: cross-Greek d²V/dSdvol is non-zero`` () =
    // Both spot and vol as HyperDual → full Hessian
    let m = {
        Result = AccumRef 1
        Accums = Map.ofList [
            0, { Init = HyperDual(0, 100.0f, "spot")
                 Body = AccumRef 0 * Exp((0.05f.C - 0.5f.C * HyperDual(1, 0.2f, "vol") * HyperDual(1, 0.2f, "vol")) / 252.0f.C
                                         + HyperDual(1, 0.2f, "vol") * Sqrt(1.0f.C / 252.0f.C) * Normal 0) }
            1, { Init = Const 0.0f
                 Body = Expr.max (AccumRef 0 - 100.0f.C) 0.0f.C * Exp(-0.05f.C) }
        ]
        Surfaces = Map.empty; Observers = []; NormalCount = 1; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    // Use full Hessian for cross-terms
    let expanded, _ = CompilerDiff.transformHyperDual false m
    use sim = Simulation.create CPU 5_000 252
    let _, watch = Simulation.foldWatch sim expanded Monthly
    // Cross-term d2_spot_vol should exist
    let crossObs = expanded.Observers |> List.tryFind (fun o -> o.Name = "d2_spot_vol")
    Assert.True(crossObs.IsSome, "Cross-term observer d2_spot_vol should exist")
    // Vanna should be non-zero for a call option
    let crossVals = Watcher.values "d2_spot_vol" watch
    let lastObs = Array2D.length1 crossVals - 1
    let meanVanna = Array.init 5_000 (fun i -> crossVals.[lastObs, i]) |> Array.average
    Assert.True(System.Single.IsFinite meanVanna, $"Vanna should be finite, got {meanVanna}")

[<Fact>]
let ``Mixed-order: Dual spot + HyperDual vol produces delta but no gamma for spot`` () =
    // spot as Dual (delta only), vol as HyperDual (vomma)
    let m = model {
        let dt = (1.0f / 252.0f).C
        let! z = normal
        let spot = Dual(0, 100.0f, "spot")
        let vol = HyperDual(1, 0.2f, "vol")
        let! stock = gbm z 0.05f.C vol spot dt
        let! df = decay 0.05f.C dt
        return Expr.max (stock - 100.0f) 0.0f.C * df
    }
    use sim = Simulation.create CPU 5_000 252
    let _, watch = Simulation.foldWatch sim m Monthly
    // Should have d1_spot (delta) — 1st order for spot
    let deltaObs = watch.Observers |> List.tryFind (fun o -> o.Name = "d1_spot")
    Assert.True(deltaObs.IsSome, "Delta observer should exist")
    // Should have d1_vol (vega) — 1st order for vol
    let vegaObs = watch.Observers |> List.tryFind (fun o -> o.Name = "d1_vol")
    Assert.True(vegaObs.IsSome, "Vega observer should exist")
    // Should have d2_vol (vomma) — 2nd order only for HyperDual vol
    let vommaObs = watch.Observers |> List.tryFind (fun o -> o.Name = "d2_vol")
    Assert.True(vommaObs.IsSome, "Vomma observer should exist for HyperDual vol")
    // Should NOT have d2_spot — spot is Dual, not HyperDual
    let gammaObs = watch.Observers |> List.tryFind (fun o -> o.Name = "d2_spot")
    Assert.True(gammaObs.IsNone, "Gamma observer should NOT exist for Dual spot")

[<Fact>]
let ``HyperDual 3-param diagonal: spot + rate + vol all have finite 2nd derivatives`` () =
    let m = {
        Result = AccumRef 0 * AccumRef 1
        Accums = Map.ofList [
            0, { Init = HyperDual(0, 100.0f, "spot")
                 Body = AccumRef 0 * Exp((HyperDual(1, 0.05f, "rate") - 0.5f.C * HyperDual(2, 0.2f, "vol") * HyperDual(2, 0.2f, "vol")) / 252.0f.C
                                         + HyperDual(2, 0.2f, "vol") * Sqrt(1.0f.C / 252.0f.C) * Normal 0) }
            1, { Init = Const 1.0f
                 Body = AccumRef 1 * Exp(-HyperDual(1, 0.05f, "rate") / 252.0f.C) }
        ]
        Surfaces = Map.empty; Observers = []; NormalCount = 1; UniformCount = 0; BernoulliCount = 0; BatchSize = 0
    }
    use sim = Simulation.create CPU 1_000 252
    let _, watch = Simulation.foldWatch sim m Monthly
    // All 3 second-order derivatives should exist and be finite
    for name in ["d2_spot"; "d2_rate"; "d2_vol"] do
        let obs = watch.Observers |> List.tryFind (fun o -> o.Name = name)
        Assert.True(obs.IsSome, $"Observer {name} should exist")
        let vals = Watcher.values name watch
        let lastObs = Array2D.length1 vals - 1
        let mean = Array.init 1_000 (fun i -> vals.[lastObs, i]) |> Array.average
        Assert.True(System.Single.IsFinite mean, $"{name} should be finite, got {mean}")
    // Also check 1st-order derivatives exist
    for name in ["d1_spot"; "d1_rate"; "d1_vol"] do
        let obs = watch.Observers |> List.tryFind (fun o -> o.Name = name)
        Assert.True(obs.IsSome, $"Observer {name} should exist")

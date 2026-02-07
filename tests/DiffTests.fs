module Cavere.Tests.DiffTests

open Xunit
open Cavere.Core
open Cavere.Generators

// ══════════════════════════════════════════════════════════════════
// DiffVar Collection
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``collectDiffVars finds DiffVar in simple expression`` () =
    let expr = Add(DiffVar(0, 100.0f), Const 1.0f)
    let vars = CompilerDiff.collectDiffVars expr
    Assert.Equal(1, vars.Count)
    Assert.Contains(0, vars)

[<Fact>]
let ``collectDiffVars finds multiple DiffVars`` () =
    let expr = Mul(DiffVar(0, 100.0f), Add(DiffVar(1, 0.2f), Const 1.0f))
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
    let expr = Exp(Sqrt(Add(DiffVar(3, 5.0f), Const 1.0f)))
    let vars = CompilerDiff.collectDiffVars expr
    Assert.Equal<int>(Set.singleton 3, vars)

[<Fact>]
let ``collectDiffVars finds DiffVar in Select`` () =
    let expr = Select(Gt(DiffVar(0, 1.0f), Const 0.0f), DiffVar(1, 2.0f), Const 0.0f)
    let vars = CompilerDiff.collectDiffVars expr
    Assert.Equal(2, vars.Count)

// ══════════════════════════════════════════════════════════════════
// Model-level DiffVar Collection
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``collectModelDiffVars finds DiffVars in result`` () =
    let m = {
        Result = Add(DiffVar(0, 100.0f), Const 1.0f)
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
    }
    let vars = CompilerDiff.collectModelDiffVars m
    Assert.Equal(1, vars.Length)
    Assert.Equal(0, vars.[0])

[<Fact>]
let ``collectModelDiffVars finds DiffVars in accumulators`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * DiffVar(1, 0.05f) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
    }
    let vars = CompilerDiff.collectModelDiffVars m
    Assert.Equal(2, vars.Length)

[<Fact>]
let ``collectModelDiffVars finds DiffVars in observers`` () =
    let m = {
        Result = Const 0.0f
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = [ { Name = "test"; Expr = DiffVar(2, 42.0f); SlotIndex = 0 } ]
        NormalCount = 0; UniformCount = 0; BatchSize = 0
    }
    let vars = CompilerDiff.collectModelDiffVars m
    Assert.Equal(1, vars.Length)
    Assert.Equal(2, vars.[0])

[<Fact>]
let ``collectModelDiffVars returns sorted unique indices`` () =
    let m = {
        Result = Add(DiffVar(2, 1.0f), DiffVar(0, 2.0f))
        Accums = Map.ofList [
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 + DiffVar(1, 3.0f) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
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
    let d = CompilerDiff.forwardDiff (DiffVar(0, 5.0f)) 0 id
    Assert.Equal(Const 1.0f, d)

[<Fact>]
let ``forwardDiff of DiffVar wrt other is 0`` () =
    let d = CompilerDiff.forwardDiff (DiffVar(0, 5.0f)) 1 id
    Assert.Equal(Const 0.0f, d)

[<Fact>]
let ``forwardDiff of Const is 0`` () =
    let d = CompilerDiff.forwardDiff (Const 42.0f) 0 id
    Assert.Equal(Const 0.0f, d)

[<Fact>]
let ``forwardDiff of Add follows sum rule`` () =
    // d/dx (x + 3) = 1 + 0
    let expr = Add(DiffVar(0, 2.0f), Const 3.0f)
    let d = CompilerDiff.forwardDiff expr 0 id
    Assert.Equal(Add(Const 1.0f, Const 0.0f), d)

[<Fact>]
let ``forwardDiff of Mul follows product rule`` () =
    // d/dx (x * y) = 1*y + x*0 = y (when differentiating wrt x)
    let x = DiffVar(0, 2.0f)
    let y = DiffVar(1, 3.0f)
    let expr = Mul(x, y)
    let d = CompilerDiff.forwardDiff expr 0 id
    // Should be Add(Mul(1, y), Mul(x, 0))
    Assert.Equal(Add(Mul(Const 1.0f, y), Mul(x, Const 0.0f)), d)

[<Fact>]
let ``forwardDiff of Exp follows chain rule`` () =
    // d/dx exp(x) = exp(x) * 1
    let x = DiffVar(0, 1.0f)
    let d = CompilerDiff.forwardDiff (Exp x) 0 id
    Assert.Equal(Mul(Exp x, Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Log follows chain rule`` () =
    // d/dx log(x) = (1/x) * 1
    let x = DiffVar(0, 1.0f)
    let d = CompilerDiff.forwardDiff (Log x) 0 id
    Assert.Equal(Mul(Div(Const 1.0f, x), Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Sqrt follows chain rule`` () =
    // d/dx sqrt(x) = 0.5/sqrt(x) * 1
    let x = DiffVar(0, 4.0f)
    let d = CompilerDiff.forwardDiff (Sqrt x) 0 id
    Assert.Equal(Mul(Div(Const 0.5f, Sqrt x), Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Neg follows negation rule`` () =
    let x = DiffVar(0, 1.0f)
    let d = CompilerDiff.forwardDiff (Neg x) 0 id
    Assert.Equal(Neg(Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Div follows quotient rule`` () =
    // d/dx (x / y) = (1*y - x*0) / (y*y)
    let x = DiffVar(0, 6.0f)
    let y = DiffVar(1, 3.0f)
    let d = CompilerDiff.forwardDiff (Div(x, y)) 0 id
    Assert.Equal(Div(Sub(Mul(Const 1.0f, y), Mul(x, Const 0.0f)), Mul(y, y)), d)

[<Fact>]
let ``forwardDiff of Max uses subgradient`` () =
    let a = DiffVar(0, 5.0f)
    let b = DiffVar(1, 3.0f)
    let d = CompilerDiff.forwardDiff (Max(a, b)) 0 id
    // Should be Select(a > b, d/dx a, d/dx b)
    Assert.Equal(Select(Gt(a, b), Const 1.0f, Const 0.0f), d)

[<Fact>]
let ``forwardDiff of comparison ops is 0`` () =
    let a = DiffVar(0, 1.0f)
    let b = DiffVar(1, 2.0f)
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Gt(a, b)) 0 id)
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Lt(a, b)) 0 id)
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Gte(a, b)) 0 id)
    Assert.Equal(Const 0.0f, CompilerDiff.forwardDiff (Lte(a, b)) 0 id)

[<Fact>]
let ``forwardDiff of Select differentiates through branches`` () =
    let cond = Gt(Const 1.0f, Const 0.0f)
    let t = DiffVar(0, 10.0f)
    let f = DiffVar(1, 5.0f)
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
    let x = DiffVar(0, 3.0f)
    let d = CompilerDiff.forwardDiff (Abs x) 0 id
    let expectedSign = Select(Gt(x, Const 0.0f), Const 1.0f, Const -1.0f)
    Assert.Equal(Mul(expectedSign, Const 1.0f), d)

[<Fact>]
let ``forwardDiff of Floor is 0`` () =
    let d = CompilerDiff.forwardDiff (Floor(DiffVar(0, 1.5f))) 0 id
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
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * 1.01f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
    }
    let expanded, diffVars = CompilerDiff.transformDual m
    Assert.Equal(1, diffVars.Length)
    Assert.Equal(0, diffVars.[0])
    // Should have original accum + 1 derivative accum = 2 total
    Assert.Equal(2, expanded.Accums.Count)
    // Derivative observer added
    Assert.Equal(1, expanded.Observers.Length)
    Assert.Equal("__deriv_0", expanded.Observers.[0].Name)

[<Fact>]
let ``transformDual adds derivative accumulators for two DiffVars`` () =
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * Exp(DiffVar(1, 0.05f)) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
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
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * 1.01f.C }
        ]
        Surfaces = Map.empty
        Observers = [ { Name = "stock"; Expr = AccumRef 0; SlotIndex = 0 } ]
        NormalCount = 0; UniformCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformDual m
    // Original observer + 1 derivative observer
    Assert.Equal(2, expanded.Observers.Length)
    Assert.Equal("stock", expanded.Observers.[0].Name)
    Assert.Equal("__deriv_0", expanded.Observers.[1].Name)

[<Fact>]
let ``transformDual derivative accum init is differentiated`` () =
    // Init = DiffVar(0, 100) → d/dDiffVar0 = 1
    let m = {
        Result = AccumRef 0
        Accums = Map.ofList [
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
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
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * Exp(DiffVar(1, 0.001f)) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
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
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * 1.01f.C }
            1, { Init = Const 1.0f; Body = AccumRef 1 * Exp(-DiffVar(1, 0.05f) / 252.0f.C) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
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
        Result = DiffVar(0, 1.0f)
        Accums = Map.empty
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
    }
    Assert.True(CompilerDiff.hasDiffVars m)

// ══════════════════════════════════════════════════════════════════
// DiffVar in emitted code
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``DiffVar emits as constant value in C# source`` () =
    let layout = { Offsets = Map.empty; TotalSize = 0; Meta = Map.empty }
    let code = CompilerCommon.emitExpr layout (DiffVar(0, 3.14f))
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
            0, { Init = DiffVar(0, 100.0f)
                 Body = AccumRef 0 * Exp((DiffVar(1, 0.05f) - 0.5f.C * DiffVar(2, 0.2f) * DiffVar(2, 0.2f)) / 252.0f.C
                                         + DiffVar(2, 0.2f) * Sqrt(1.0f.C / 252.0f.C) * Normal 0) }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 1; UniformCount = 0; BatchSize = 0
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
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
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
            0, { Init = DiffVar(0, 100.0f); Body = AccumRef 0 * 1.001f.C }
        ]
        Surfaces = Map.empty
        Observers = []
        NormalCount = 0; UniformCount = 0; BatchSize = 0
    }
    let expanded, _ = CompilerDiff.transformDual m
    // Run with watch to get derivative observer
    use sim = Simulation.create CPU 100 10
    let _, watchResult = Simulation.foldWatch sim expanded Monthly
    // Should have the derivative observer
    Assert.True(expanded.Observers |> List.exists (fun o -> o.Name = "__deriv_0"))

[<Fact>]
let ``DiffMode type has expected cases`` () =
    // Just ensure the type compiles with all cases
    let modes = [ Dual; HyperDual true; HyperDual false; Jet 3; Adjoint ]
    Assert.Equal(5, modes.Length)

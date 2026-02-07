namespace Cavere.Tests

open Xunit
open Cavere.Core

module SymbolicTests =

    // ══════════════════════════════════════════════════════════════════
    // Simplification Tests
    // ══════════════════════════════════════════════════════════════════

    [<Fact>]
    let ``simplify: add zero is identity`` () =
        let expr = Add(Const 0.0f, Const 5.0f)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 5.0f, result)

    [<Fact>]
    let ``simplify: multiply by zero`` () =
        let expr = Mul(Normal 0, Const 0.0f)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 0.0f, result)

    [<Fact>]
    let ``simplify: multiply by one is identity`` () =
        let expr = Mul(Const 1.0f, Normal 0)
        let result = Symbolic.simplify expr
        Assert.Equal(Normal 0, result)

    [<Fact>]
    let ``simplify: constant folding addition`` () =
        let expr = Add(Const 3.0f, Const 4.0f)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 7.0f, result)

    [<Fact>]
    let ``simplify: constant folding multiplication`` () =
        let expr = Mul(Const 3.0f, Const 4.0f)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 12.0f, result)

    [<Fact>]
    let ``simplify: double negation`` () =
        let expr = Neg(Neg(Normal 0))
        let result = Symbolic.simplify expr
        Assert.Equal(Normal 0, result)

    [<Fact>]
    let ``simplify: subtract self is zero`` () =
        let expr = Sub(AccumRef 0, AccumRef 0)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 0.0f, result)

    [<Fact>]
    let ``simplify: divide self is one`` () =
        let expr = Div(AccumRef 0, AccumRef 0)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 1.0f, result)

    [<Fact>]
    let ``simplify: exp of log`` () =
        let expr = Exp(Log(Normal 0))
        let result = Symbolic.simplify expr
        Assert.Equal(Normal 0, result)

    [<Fact>]
    let ``simplify: log of exp`` () =
        let expr = Log(Exp(Normal 0))
        let result = Symbolic.simplify expr
        Assert.Equal(Normal 0, result)

    [<Fact>]
    let ``simplify: exp of zero`` () =
        let expr = Exp(Const 0.0f)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 1.0f, result)

    [<Fact>]
    let ``simplify: nested constant folding`` () =
        // (2 + 3) * (1 + 0) = 5 * 1 = 5
        let expr = Mul(Add(Const 2.0f, Const 3.0f), Add(Const 1.0f, Const 0.0f))
        let result = Symbolic.fullySimplify expr
        Assert.Equal(Const 5.0f, result)

    [<Fact>]
    let ``simplify: select with constant true condition`` () =
        let expr = Select(Const 1.0f, Const 10.0f, Const 20.0f)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 10.0f, result)

    [<Fact>]
    let ``simplify: select with constant false condition`` () =
        let expr = Select(Const 0.0f, Const 10.0f, Const 20.0f)
        let result = Symbolic.simplify expr
        Assert.Equal(Const 20.0f, result)

    [<Fact>]
    let ``simplify: constant multiplication merge`` () =
        // 3 * (4 * x) = 12 * x
        let expr = Mul(Const 3.0f, Mul(Const 4.0f, Normal 0))
        let result = Symbolic.fullySimplify expr
        Assert.Equal(Mul(Const 12.0f, Normal 0), result)

    // ══════════════════════════════════════════════════════════════════
    // Node Counting Tests
    // ══════════════════════════════════════════════════════════════════

    [<Fact>]
    let ``countNodes: leaf nodes are 1`` () =
        Assert.Equal(1, Symbolic.countNodes (Const 1.0f))
        Assert.Equal(1, Symbolic.countNodes (Normal 0))
        Assert.Equal(1, Symbolic.countNodes TimeIndex)

    [<Fact>]
    let ``countNodes: binary op is 1 + children`` () =
        let expr = Add(Const 1.0f, Const 2.0f)
        Assert.Equal(3, Symbolic.countNodes expr)

    [<Fact>]
    let ``countNodes: nested expression`` () =
        // exp(a + b) = 3 nodes (exp, add, a) + b = but actually it's 1(exp) + 1(add) + 1(a) + 1(b)
        let expr = Exp(Add(Normal 0, Const 1.0f))
        Assert.Equal(4, Symbolic.countNodes expr)

    // ══════════════════════════════════════════════════════════════════
    // Symbolic Differentiation Tests
    // ══════════════════════════════════════════════════════════════════

    [<Fact>]
    let ``diff: constant has zero derivative`` () =
        let result = Symbolic.diff (Const 5.0f) 0
        Assert.Equal(Const 0.0f, result)

    [<Fact>]
    let ``diff: sum rule`` () =
        // d(Const(3) + Const(4))/dx = 0 + 0 = 0
        let expr = Add(Const 3.0f, Const 4.0f)
        let result = Symbolic.diff expr 0
        Assert.Equal(Const 0.0f, result)

    [<Fact>]
    let ``diff: product rule with constants`` () =
        // d(Const(3) * Const(4))/dx = 0*4 + 3*0 = 0
        let expr = Mul(Const 3.0f, Const 4.0f)
        let result = Symbolic.diff expr 0
        Assert.Equal(Const 0.0f, result)

    [<Fact>]
    let ``diff: exp chain rule`` () =
        // d(exp(Const(2)))/dx = exp(Const(2)) * 0 = 0
        let expr = Exp(Const 2.0f)
        let result = Symbolic.diff expr 0
        Assert.Equal(Const 0.0f, result)

    [<Fact>]
    let ``diff: negation`` () =
        // d(-Const(3))/dx = -0 = 0
        let expr = Neg(Const 3.0f)
        let result = Symbolic.diff expr 0
        Assert.Equal(Const 0.0f, result)

    [<Fact>]
    let ``simplify reduces node count`` () =
        // (0 + x) * (1 * y) should simplify to x * y
        let expr = Mul(Add(Const 0.0f, Normal 0), Mul(Const 1.0f, Normal 1))
        let simplified = Symbolic.fullySimplify expr
        Assert.True(Symbolic.countNodes simplified <= Symbolic.countNodes expr)

    // ══════════════════════════════════════════════════════════════════
    // Tree Utility Tests
    // ══════════════════════════════════════════════════════════════════

    [<Fact>]
    let ``containsTimeIndex: finds TimeIndex`` () =
        let expr = Add(TimeIndex, Const 1.0f)
        Assert.True(Symbolic.containsTimeIndex expr)

    [<Fact>]
    let ``containsTimeIndex: false when absent`` () =
        let expr = Add(Normal 0, Const 1.0f)
        Assert.False(Symbolic.containsTimeIndex expr)

    [<Fact>]
    let ``containsLookup1D: finds Lookup1D`` () =
        let expr = Add(Lookup1D 0, Const 1.0f)
        Assert.True(Symbolic.containsLookup1D expr)

    [<Fact>]
    let ``containsAccumRef: finds AccumRef`` () =
        let expr = Mul(AccumRef 2, Const 1.0f)
        Assert.True(Symbolic.containsAccumRef expr 2)
        Assert.False(Symbolic.containsAccumRef expr 3)

    // ══════════════════════════════════════════════════════════════════
    // Analysis Tests
    // ══════════════════════════════════════════════════════════════════

    [<Fact>]
    let ``normalCdf: standard values`` () =
        Assert.InRange(Analysis.normalCdf 0.0f, 0.499f, 0.501f)
        Assert.InRange(Analysis.normalCdf 1.0f, 0.840f, 0.842f)
        Assert.InRange(Analysis.normalCdf -1.0f, 0.158f, 0.160f)

    [<Fact>]
    let ``evaluate: Black-Scholes call`` () =
        // BS call with S=100, K=100, r=0.05, vol=0.2, T=1
        // Known approximate value: ~10.45
        let price = Analysis.evaluate (BlackScholesCall(100.0f, 100.0f, 0.05f, 0.20f, 1.0f))
        Assert.InRange(price, 10.0f, 11.0f)

    [<Fact>]
    let ``evaluate: Black-Scholes put`` () =
        // BS put with S=100, K=100, r=0.05, vol=0.2, T=1
        // Known approximate value: ~5.57
        let price = Analysis.evaluate (BlackScholesPut(100.0f, 100.0f, 0.05f, 0.20f, 1.0f))
        Assert.InRange(price, 5.0f, 6.0f)

    [<Fact>]
    let ``evaluate: put-call parity`` () =
        let s, k, r, v, t = 100.0f, 100.0f, 0.05f, 0.20f, 1.0f
        let call = Analysis.evaluate (BlackScholesCall(s, k, r, v, t))
        let put = Analysis.evaluate (BlackScholesPut(s, k, r, v, t))
        // C - P = S - K*exp(-rT)
        let parity = s - k * System.MathF.Exp(-r * t)
        Assert.InRange(call - put, parity - 0.01f, parity + 0.01f)

    [<Fact>]
    let ``evaluate: zero-coupon bond`` () =
        let price = Analysis.evaluate (ZeroCouponBond(0.05f, 1.0f))
        let expected = System.MathF.Exp(-0.05f)
        Assert.InRange(price, expected - 0.001f, expected + 0.001f)

    [<Fact>]
    let ``evaluate: forward`` () =
        let price = Analysis.evaluate (Forward(100.0f, 0.05f, 1.0f))
        Assert.Equal(100.0f, price)

    [<Fact>]
    let ``evaluateGreeks: BS call delta is positive`` () =
        let greeks = Analysis.evaluateGreeks (BlackScholesCall(100.0f, 100.0f, 0.05f, 0.20f, 1.0f))
        Assert.InRange(greeks.Delta, 0.5f, 0.8f)

    [<Fact>]
    let ``evaluateGreeks: BS call gamma is positive`` () =
        let greeks = Analysis.evaluateGreeks (BlackScholesCall(100.0f, 100.0f, 0.05f, 0.20f, 1.0f))
        Assert.True(greeks.Gamma > 0.0f)

    [<Fact>]
    let ``isGBMAccumulator: detects GBM pattern`` () =
        // body = self * exp(something)
        let def = { Init = Const 100.0f; Body = Mul(AccumRef 0, Exp(Add(Const 0.001f, Normal 0))) }
        Assert.True(Analysis.isGBMAccumulator 0 def)

    [<Fact>]
    let ``isGBMAccumulator: rejects non-GBM`` () =
        let def = { Init = Const 100.0f; Body = Add(AccumRef 0, Const 1.0f) }
        Assert.False(Analysis.isGBMAccumulator 0 def)

    [<Fact>]
    let ``isDiscountAccumulator: detects discount pattern`` () =
        // body = self * exp(-rate * dt)
        let def = { Init = Const 1.0f; Body = Mul(AccumRef 0, Exp(Neg(Mul(Const 0.05f, Const 0.004f)))) }
        Assert.True(Analysis.isDiscountAccumulator 0 def)

    [<Fact>]
    let ``detectPathDependence: no time or observers`` () =
        let m = {
            Result = Max(Sub(AccumRef 0, Const 100.0f), Const 0.0f)
            Accums = Map.ofList [0, { Init = Const 100.0f; Body = Mul(AccumRef 0, Exp(Const 0.01f)) }]
            Surfaces = Map.empty; Observers = []; NormalCount = 0; UniformCount = 0; BatchSize = 0
        }
        Assert.False(Analysis.detectPathDependence m)

    [<Fact>]
    let ``detectPathDependence: with observer`` () =
        let m = {
            Result = AccumRef 0
            Accums = Map.ofList [0, { Init = Const 100.0f; Body = Mul(AccumRef 0, Exp(Const 0.01f)) }]
            Surfaces = Map.empty
            Observers = [{ Name = "stock"; Expr = AccumRef 0; SlotIndex = 0 }]
            NormalCount = 0; UniformCount = 0; BatchSize = 0
        }
        Assert.True(Analysis.detectPathDependence m)

    [<Fact>]
    let ``GBMMoments: mean of GBM`` () =
        let expected = 100.0f * System.MathF.Exp(0.05f * 1.0f)
        let actual = Analysis.GBMMoments.mean 100.0f 0.05f 1.0f
        Assert.InRange(actual, expected - 0.01f, expected + 0.01f)

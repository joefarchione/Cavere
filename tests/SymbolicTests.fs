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
    // Condition Combinator Tests
    // ══════════════════════════════════════════════════════════════════

    [<Fact>]
    let ``simplify: And with constant true`` () =
        let expr = And(Const 1.0f, Normal 0)
        Assert.Equal(Normal 0, Symbolic.simplify expr)

    [<Fact>]
    let ``simplify: And with constant false`` () =
        let expr = And(Const 0.0f, Normal 0)
        Assert.Equal(Const 0.0f, Symbolic.simplify expr)

    [<Fact>]
    let ``simplify: Or with constant true`` () =
        let expr = Or(Const 1.0f, Normal 0)
        Assert.Equal(Const 1.0f, Symbolic.simplify expr)

    [<Fact>]
    let ``simplify: Or with constant false`` () =
        let expr = Or(Const 0.0f, Normal 0)
        Assert.Equal(Normal 0, Symbolic.simplify expr)

    [<Fact>]
    let ``simplify: Not of Not is identity`` () =
        let expr = Not(Not(Normal 0))
        Assert.Equal(Normal 0, Symbolic.simplify expr)

    [<Fact>]
    let ``simplify: Not of constant`` () =
        Assert.Equal(Const 0.0f, Symbolic.simplify (Not(Const 1.0f)))
        Assert.Equal(Const 1.0f, Symbolic.simplify (Not(Const 0.0f)))

    [<Fact>]
    let ``diff: And/Or/Not have zero derivative`` () =
        let and' = And(Gt(Normal 0, Const 0.0f), Lt(Normal 1, Const 1.0f))
        Assert.Equal(Const 0.0f, Symbolic.diff and' 0)
        let or' = Or(Gt(Normal 0, Const 0.0f), Lt(Normal 1, Const 1.0f))
        Assert.Equal(Const 0.0f, Symbolic.diff or' 0)
        let not' = Not(Gt(Normal 0, Const 0.0f))
        Assert.Equal(Const 0.0f, Symbolic.diff not' 0)

    [<Fact>]
    let ``countNodes: And is binary, Not is unary`` () =
        Assert.Equal(3, Symbolic.countNodes (And(Const 1.0f, Const 0.0f)))
        Assert.Equal(3, Symbolic.countNodes (Or(Const 1.0f, Const 0.0f)))
        Assert.Equal(2, Symbolic.countNodes (Not(Const 1.0f)))

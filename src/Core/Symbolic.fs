namespace Cavere.Core

/// Expression simplification and symbolic differentiation.
[<RequireQualifiedAccess>]
module Symbolic =

    // ══════════════════════════════════════════════════════════════════
    // Expression Simplification
    // ══════════════════════════════════════════════════════════════════

    /// Single-pass algebraic simplification.
    let rec simplify (expr: Expr) : Expr =
        match expr with
        // ── Addition identities ──
        | Add(Const 0.0f, x) -> simplify x
        | Add(x, Const 0.0f) -> simplify x
        | Add(Const a, Const b) -> Const(a + b)

        // ── Subtraction identities ──
        | Sub(x, Const 0.0f) -> simplify x
        | Sub(Const 0.0f, x) -> Neg(simplify x) |> simplify
        | Sub(Const a, Const b) -> Const(a - b)
        | Sub(x, y) when x = y -> Const 0.0f

        // ── Multiplication identities ──
        | Mul(Const 0.0f, _) -> Const 0.0f
        | Mul(_, Const 0.0f) -> Const 0.0f
        | Mul(Const 1.0f, x) -> simplify x
        | Mul(x, Const 1.0f) -> simplify x
        | Mul(Const a, Const b) -> Const(a * b)
        | Mul(Const a, Mul(Const b, x)) -> simplify (Mul(Const(a * b), x))
        | Mul(Neg a, Neg b) -> simplify (Mul(a, b))

        // ── Division identities ──
        | Div(x, Const 1.0f) -> simplify x
        | Div(Const 0.0f, _) -> Const 0.0f
        | Div(Const a, Const b) when b <> 0.0f -> Const(a / b)
        | Div(x, y) when x = y -> Const 1.0f

        // ── Exponential / log identities ──
        | Exp(Const 0.0f) -> Const 1.0f
        | Exp(Log x) -> simplify x
        | Log(Const 1.0f) -> Const 0.0f
        | Log(Exp x) -> simplify x

        // ── Sqrt identities ──
        | Sqrt(Const 0.0f) -> Const 0.0f
        | Sqrt(Const 1.0f) -> Const 1.0f

        // ── Double negation ──
        | Neg(Neg x) -> simplify x
        | Neg(Const c) -> Const(-c)

        // ── Abs identities ──
        | Abs(Const c) -> Const(abs c)
        | Abs(Abs x) -> Abs(simplify x)

        // ── Max / Min with constants ──
        | Max(Const a, Const b) -> Const(max a b)
        | Min(Const a, Const b) -> Const(min a b)

        // ── Floor of constant ──
        | Floor(Const c) -> Const(floor c)

        // ── Select with constant condition ──
        | Select(Const c, t, _) when c > 0.5f -> simplify t
        | Select(Const c, _, f) when c <= 0.5f -> simplify f

        // ── Recursive simplification for binary ops ──
        | Add(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then simplify (Add(a', b')) else Add(a', b')
        | Sub(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then simplify (Sub(a', b')) else Sub(a', b')
        | Mul(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then simplify (Mul(a', b')) else Mul(a', b')
        | Div(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then simplify (Div(a', b')) else Div(a', b')
        | Max(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then simplify (Max(a', b')) else Max(a', b')
        | Min(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then simplify (Min(a', b')) else Min(a', b')
        | Gt(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then Gt(a', b') else expr
        | Gte(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then Gte(a', b') else expr
        | Lt(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then Lt(a', b') else expr
        | Lte(a, b) ->
            let a', b' = simplify a, simplify b
            if a' <> a || b' <> b then Lte(a', b') else expr

        // ── Recursive simplification for ternary ops ──
        | Select(c, t, f) ->
            let c', t', f' = simplify c, simplify t, simplify f
            if c' <> c || t' <> t || f' <> f then simplify (Select(c', t', f'))
            else Select(c', t', f')

        // ── Recursive simplification for unary ops ──
        | Neg a ->
            let a' = simplify a
            if a' <> a then simplify (Neg a') else Neg a'
        | Exp a ->
            let a' = simplify a
            if a' <> a then simplify (Exp a') else Exp a'
        | Log a ->
            let a' = simplify a
            if a' <> a then simplify (Log a') else Log a'
        | Sqrt a ->
            let a' = simplify a
            if a' <> a then simplify (Sqrt a') else Sqrt a'
        | Abs a ->
            let a' = simplify a
            if a' <> a then simplify (Abs a') else Abs a'
        | Floor a ->
            let a' = simplify a
            if a' <> a then simplify (Floor a') else Floor a'

        // ── Leaf nodes and others pass through ──
        | _ -> expr

    /// Apply simplification until fixed point (max 100 iterations).
    let fullySimplify (expr: Expr) : Expr =
        let mutable current = expr
        let mutable prev = Const System.Single.NaN // sentinel
        let mutable iterations = 0
        while current <> prev && iterations < 100 do
            prev <- current
            current <- simplify current
            iterations <- iterations + 1
        current

    // ══════════════════════════════════════════════════════════════════
    // Node Counting
    // ══════════════════════════════════════════════════════════════════

    /// Count AST nodes in an expression.
    let rec countNodes (expr: Expr) : int =
        match expr with
        | Const _ | TimeIndex | Normal _ | Uniform _ | AccumRef _ | Lookup1D _ | BatchRef _ -> 1
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> 1 + countNodes a
        | SurfaceAt(_, idx) -> 1 + countNodes idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            1 + countNodes a + countNodes b
        | Select(c, t, f) -> 1 + countNodes c + countNodes t + countNodes f
        | BinSearch(_, _, _, v) -> 1 + countNodes v

    // ══════════════════════════════════════════════════════════════════
    // Symbolic Differentiation
    // ══════════════════════════════════════════════════════════════════

    /// Symbolic derivative of Expr w.r.t. a DiffVar index.
    /// DiffVar nodes must be introduced by wrapping Const values:
    ///   Use `DiffVar(idx, value)` where differentiation is needed.
    /// Non-DiffVar constants differentiate to 0.
    let rec symbolicDiff (expr: Expr) (wrt: int) : Expr =
        match expr with
        // ── Leaf nodes ──
        | Const _ -> Const 0.0f
        | TimeIndex -> Const 0.0f
        | Normal _ -> Const 0.0f
        | Uniform _ -> Const 0.0f
        | Lookup1D _ -> Const 0.0f
        | BatchRef _ -> Const 0.0f
        | AccumRef _ -> Const 0.0f // AccumRef derivatives need special handling at model level
        | Floor _ -> Const 0.0f // piecewise constant, derivative is 0 a.e.

        // ── Sum rule ──
        | Add(a, b) -> Add(symbolicDiff a wrt, symbolicDiff b wrt)

        // ── Difference rule ──
        | Sub(a, b) -> Sub(symbolicDiff a wrt, symbolicDiff b wrt)

        // ── Product rule ──
        | Mul(a, b) ->
            Add(Mul(symbolicDiff a wrt, b), Mul(a, symbolicDiff b wrt))

        // ── Quotient rule ──
        | Div(a, b) ->
            Div(Sub(Mul(symbolicDiff a wrt, b), Mul(a, symbolicDiff b wrt)),
                Mul(b, b))

        // ── Chain rule for unary ──
        | Exp(a) -> Mul(Exp(a), symbolicDiff a wrt)
        | Log(a) -> Mul(Div(Const 1.0f, a), symbolicDiff a wrt)
        | Sqrt(a) -> Mul(Div(Const 0.5f, Sqrt(a)), symbolicDiff a wrt)
        | Neg(a) -> Neg(symbolicDiff a wrt)
        | Abs(a) ->
            // Sign function via select
            let sign = Select(Gt(a, Const 0.0f), Const 1.0f, Const -1.0f)
            Mul(sign, symbolicDiff a wrt)

        // ── Max/Min subgradients ──
        | Max(a, b) -> Select(Gt(a, b), symbolicDiff a wrt, symbolicDiff b wrt)
        | Min(a, b) -> Select(Lt(a, b), symbolicDiff a wrt, symbolicDiff b wrt)

        // ── Comparison operators (piecewise constant) ──
        | Gt _ | Gte _ | Lt _ | Lte _ -> Const 0.0f

        // ── Select (assume condition independent of wrt) ──
        | Select(cond, t, f) ->
            Select(cond, symbolicDiff t wrt, symbolicDiff f wrt)

        // ── Surface lookups: treat as opaque ──
        | SurfaceAt _ -> Const 0.0f
        | BinSearch _ -> Const 0.0f

    /// Differentiate and simplify.
    let diff (expr: Expr) (wrt: int) : Expr =
        symbolicDiff expr wrt |> fullySimplify

    // ══════════════════════════════════════════════════════════════════
    // Expression Tree Utilities
    // ══════════════════════════════════════════════════════════════════

    /// Check if expression contains a specific node type.
    let rec containsTimeIndex (expr: Expr) : bool =
        match expr with
        | TimeIndex -> true
        | Const _ | Normal _ | Uniform _ | AccumRef _ | Lookup1D _ | BatchRef _ -> false
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> containsTimeIndex a
        | SurfaceAt(_, idx) -> containsTimeIndex idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            containsTimeIndex a || containsTimeIndex b
        | Select(c, t, f) -> containsTimeIndex c || containsTimeIndex t || containsTimeIndex f
        | BinSearch(_, _, _, v) -> containsTimeIndex v

    let rec containsLookup1D (expr: Expr) : bool =
        match expr with
        | Lookup1D _ -> true
        | Const _ | TimeIndex | Normal _ | Uniform _ | AccumRef _ | BatchRef _ -> false
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> containsLookup1D a
        | SurfaceAt(_, idx) -> containsLookup1D idx
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            containsLookup1D a || containsLookup1D b
        | Select(c, t, f) -> containsLookup1D c || containsLookup1D t || containsLookup1D f
        | BinSearch(_, _, _, v) -> containsLookup1D v

    let rec containsAccumRef (expr: Expr) (id: int) : bool =
        match expr with
        | AccumRef aid -> aid = id
        | Const _ | TimeIndex | Normal _ | Uniform _ | Lookup1D _ | BatchRef _ -> false
        | Floor a | Neg a | Exp a | Log a | Sqrt a | Abs a -> containsAccumRef a id
        | SurfaceAt(_, idx) -> containsAccumRef idx id
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b)
        | Max(a, b) | Min(a, b) | Gt(a, b) | Gte(a, b) | Lt(a, b) | Lte(a, b) ->
            containsAccumRef a id || containsAccumRef b id
        | Select(c, t, f) -> containsAccumRef c id || containsAccumRef t id || containsAccumRef f id
        | BinSearch(_, _, _, v) -> containsAccumRef v id

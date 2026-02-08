namespace Cavere.Core

type Expr =
    | Const of float32
    | TimeIndex
    | Normal of id: int
    | Uniform of id: int
    | Bernoulli of id: int
    | AccumRef of id: int
    | Lookup1D of surfaceId: int
    | Floor of Expr
    | SurfaceAt of surfaceId: int * index: Expr
    | Add of Expr * Expr
    | Sub of Expr * Expr
    | Mul of Expr * Expr
    | Div of Expr * Expr
    | Max of Expr * Expr
    | Min of Expr * Expr
    | Gt of Expr * Expr
    | Gte of Expr * Expr
    | Lt of Expr * Expr
    | Lte of Expr * Expr
    | And of Expr * Expr
    | Or of Expr * Expr
    | Not of Expr
    | Select of cond: Expr * ifTrue: Expr * ifFalse: Expr
    | Neg of Expr
    | Exp of Expr
    | Log of Expr
    | Sqrt of Expr
    | Abs of Expr
    | BatchRef of surfaceId: int
    | BinSearch of surfaceId: int * axisOff: int * axisCnt: int * value: Expr
    | Dual of index: int * value: float32 * name: string
    | HyperDual of index: int * value: float32 * name: string

    // Binary: Expr op Expr
    static member (+)(a: Expr, b: Expr) = Add(a, b)
    static member (-)(a: Expr, b: Expr) = Sub(a, b)
    static member (*)(a: Expr, b: Expr) = Mul(a, b)
    static member (/)(a: Expr, b: Expr) = Div(a, b)

    // Comparison: Expr .op Expr
    static member (.>)(a: Expr, b: Expr) = Gt(a, b)
    static member (.>=)(a: Expr, b: Expr) = Gte(a, b)
    static member (.<)(a: Expr, b: Expr) = Lt(a, b)
    static member (.<=)(a: Expr, b: Expr) = Lte(a, b)

    // Logical: Expr .op Expr (for combining conditions)
    static member (.&&)(a: Expr, b: Expr) = And(a, b)
    static member (.||)(a: Expr, b: Expr) = Or(a, b)

    // Comparison: mixed Expr / float32
    static member (.>)(a: Expr, b: float32) = Gt(a, Const b)
    static member (.>)(a: float32, b: Expr) = Gt(Const a, b)
    static member (.>=)(a: Expr, b: float32) = Gte(a, Const b)
    static member (.>=)(a: float32, b: Expr) = Gte(Const a, b)
    static member (.<)(a: Expr, b: float32) = Lt(a, Const b)
    static member (.<)(a: float32, b: Expr) = Lt(Const a, b)
    static member (.<=)(a: Expr, b: float32) = Lte(a, Const b)
    static member (.<=)(a: float32, b: Expr) = Lte(Const a, b)

    // Unary
    static member (~-)(a: Expr) = Neg(a)

    // Mixed: Expr op float32
    static member (+)(a: Expr, b: float32) = Add(a, Const b)
    static member (+)(a: float32, b: Expr) = Add(Const a, b)
    static member (-)(a: Expr, b: float32) = Sub(a, Const b)
    static member (-)(a: float32, b: Expr) = Sub(Const a, b)
    static member (*)(a: Expr, b: float32) = Mul(a, Const b)
    static member (*)(a: float32, b: Expr) = Mul(Const a, b)
    static member (/)(a: Expr, b: float32) = Div(a, Const b)
    static member (/)(a: float32, b: Expr) = Div(Const a, b)

/// Differentiation mode for automatic differentiation.
type DiffMode =
    | DualMode // 1st order only (fastest)
    | HyperDualMode of diagonal: bool // diagonal=true: only d²V/dSi², diagonal=false: all crosses
    | JetMode of order: int // Arbitrary order Taylor
    | AdjointMode // Backward mode (all 1st order, memory efficient)

module Expr =
    let exp e = Exp e
    let log e = Log e
    let sqrt e = Sqrt e
    let abs e = Abs e
    let max a b = Max(a, b)
    let min a b = Min(a, b)
    let select cond ifTrue ifFalse = Select(cond, ifTrue, ifFalse)
    let not' e = Not e
    let and' a b = And(a, b)
    let or' a b = Or(a, b)
    let clip lo hi x = Max(lo, Min(hi, x))
    let floor e = Floor e
    let surfaceAt sid index = SurfaceAt(sid, index)
    let dual name idx value = Dual(idx, value, name)
    let hyperDual name idx value = HyperDual(idx, value, name)

[<AutoOpen>]
module ExprExtensions =
    type System.Single with
        member x.C = Const x

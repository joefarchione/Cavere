# Plan: Multi-way Conditional (Cond) Expr Case

## Problem

Deeply nested `Select` chains cause ILGPU stack overflow due to recursive SSA building.
The `Select` case emits as ternary `(cond > 0.5f) ? a : b`, and nesting 20+ levels deep
crashes ILGPU's IR builder. The `binSearchThreshold = 10` workaround in `findBin`
uses `BinSearch` (a C# while-loop) for large grids, but users writing their own
multi-way conditionals still face this risk.

## Decision

**Keep `Select` for simple two-way conditionals** (it's branchless, good GPU perf).
**Add `Cond` for multi-way conditionals** that emits as flat if/else-if statements.
**Lower `binSearchThreshold` to 0** so all interp2d bin-finding uses BinSearch loops.

## New Expr Case

```fsharp
| Cond of cases: (Expr * Expr) list * defaultExpr: Expr
```

DSL helper:
```fsharp
let cond (cases: (Expr * Expr) list) (defaultExpr: Expr) = Cond(cases, defaultExpr)
```

User API:
```fsharp
// Instead of deeply nested Select:
Expr.cond [
    x .> 3.0f, resultD
    x .> 2.0f, resultC
    x .> 1.0f, resultB
] resultA  // default
```

## Emission Strategy

Cond emits as flat if/else-if with a temp variable:
```csharp
float cond_0 = defaultExpr;
if (c1 > 0.5f) cond_0 = v1;
else if (c2 > 0.5f) cond_0 = v2;
// ... then cond_0 is used in the containing expression
```

This is O(1) nesting depth for ILGPU regardless of case count.

Implementation: `emitExprPreamble` function writes preamble statements to a
separate StringBuilder, returns the variable reference. Call sites flush
preamble before the main expression line.

## AD Handling

Forward-mode: differentiate through each branch expression independently.
```fsharp
forwardDiff(Cond(cases, default)) =
    Cond(cases |> List.map (fun (c, v) -> (c, forwardDiff v)), forwardDiff default)
```
Conditions are not differentiated (piecewise constant).

Adjoint: same pattern as Select — partial derivatives through each branch.

## Files Changed

1. `Dsl.Expr.fs` — Add Cond DU case + Expr.cond helper
2. `Compiler.Common.fs` — Add emitExprPreamble, update emitExpr, update walkers
3. `Compiler.Codegen.fs` — Use preamble emission in accum/observer/result emitters
4. `Compiler.Regular.fs` — Flush preamble before result lines
5. `Compiler.Batch.fs` — Same
6. `Compiler.Adjoint.fs` — Same
7. `Compiler.Diff.fs` — Add Cond to all diff/walker functions
8. `Ast.Symbolic.fs` — Add Cond to simplify, countNodes, etc.
9. `Dsl.Model.fs` — Lower binSearchThreshold to 0
10. Tests — Verify Cond compilation and execution

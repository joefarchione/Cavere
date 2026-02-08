# Core

Domain-agnostic expression DSL, compiler pipeline, and GPU engine. No finance concepts here.

## Namespace

All files use `namespace Cavere.Core`. Files are organized by prefix:

- **Dsl.** — Expression types, schedule, model builder (`Dsl.Expr.fs`, `Dsl.Schedule.fs`, `Dsl.Model.fs`)
- **Ast.** — Expression walkers, simplification (`Ast.Symbolic.fs`)
- **Compiler.** — Code generation pipeline (`Compiler.Common.fs`, `Compiler.Codegen.fs`, `Compiler.Regular.fs`, `Compiler.Batch.fs`, `Compiler.Diff.fs`, `Compiler.Adjoint.fs`, `Compiler.Cpu.fs`, `Compiler.fs`)
- **Sim.** — Device management, kernel launch, simulation (`Sim.Device.fs`, `Sim.Transfer.fs`, `Sim.Engine.fs`, `Sim.Watcher.fs`, `Sim.Kernel.fs`, `Sim.Simulation.fs`, `Sim.Output.fs`)

## Key Types

- `Expr` — Discriminated union (27 cases). Operator overloads on Expr and mixed Expr/float32. Extension `.C` on float32 converts to `Const`.
- `Model` — Immutable record output of `model { }` CE. Contains Result, Accums, Surfaces, Observers, counts.
- `ModelCtx` — Mutable builder context used only inside `ModelBuilder.Run`. Allocates IDs for normals, uniforms, accums, surfaces.
- `ModelBuilder` — CE with `Return`, `Bind`, `Zero`, `For`, `Combine`, `Delay`, `Run`. `Run` creates a fresh `ModelCtx`, evaluates, extracts `Model`.
- `Schedule` — `{ Dt: float32[]; T: float32[]; Steps: int }` for non-uniform time grids.

## DSL Primitives (ModelDsl, AutoOpen)

All primitives have signature `... -> ModelCtx -> 'T` and compose via `let!` in the CE:

- `normal` / `uniform` / `bernoulli` — allocate fresh random variable ID
- `evolve init body` — accumulator (AccumRef), body receives self-reference
- `surface1d` / `surface2d` — register lookup data, return surface ID
- `schedule` — register dt + year-fraction curves, return `(Expr * Expr)`
- `batchInput` — per-batch data vector as `BatchRef`
- `observe` — register observer for path extraction
- `iter` — loop inside CE
- `correlatedNormals` — Cholesky on CPU, bakes L coefficients as Const nodes
- `cholesky` — pure CPU-side matrix decomposition

## Compiler Pipeline

1. `layoutSurfaces` → `SurfaceLayout` (offsets into packed float32[])
2. `sortAccums` → topological sort by AccumRef dependencies
3. `generateSource` → C# class string with kernel methods
4. `compile` → Roslyn in-memory assembly
5. `build` / `buildBatch` → `CompiledKernel` ready for launch

Kernel variants: Fold, FoldWatch, Scan (regular); FoldBatch, FoldBatchWatch (batch).

## Conventions

- Expr operators (`+`, `-`, `*`, `/`) work on Expr-Expr and Expr-float32
- Comparison operators `.>`, `.>=`, `.<`, `.<=` return Expr (not bool) for use in `Expr.select`
- `Expr.select cond thenExpr elseExpr` — GPU-side conditional (no branching)
- `Expr.clip lo hi x` — clamp
- All numeric values are `float32`

# Tests

xUnit test suite (37 tests). References Core and Generators.

## Running

```bash
dotnet test                          # all tests
dotnet test --filter "FullyQualifiedName~Compiler"  # specific module
```

## Test Modules

- **CompilerTests** — Model CE ID allocation, topological sort, Roslyn compilation, generated C# verification
- **PricingTests** — Black-Scholes analytical comparison (call/put pricing convergence)
- **RatesTests** — Rate model mean-reversion, forward curve consistency
- **NestedSimTests** — Nested simulation (inner/outer loop) patterns
- **ScheduleTests** — Schedule construction, business day calendars

## Conventions

- Use `[<Fact>]` for individual tests
- Namespace: `Cavere.Tests.<ModuleName>`
- Build models with `model { }`, run with `Simulation.create CPU simCount steps`
- Use `Assert.InRange` for numerical tolerance checks
- Use ``[<Fact>] let ``descriptive name with backticks`` () =`` for test naming
- Keep tests focused — one assertion per concept

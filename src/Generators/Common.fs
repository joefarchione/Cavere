namespace Cavere.Generators

open Cavere.Core

[<AutoOpen>]
module Common =

    /// Discount factor accumulator: df(t) = exp(-∫r dt).
    let decay (rate: Expr) (dt: Expr) : ModelCtx -> Expr =
        fun ctx -> evolve 1.0f.C (fun df -> df * Expr.exp (-rate * dt)) ctx

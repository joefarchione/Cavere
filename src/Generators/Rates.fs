namespace Cavere.Generators

open Cavere.Core

module Rates =

    // ── CPU-side curve preparation ──

    let private lerp (xs: float32[]) (ys: float32[]) (x: float32) : float32 =
        let n = xs.Length

        if n = 1 || x <= xs.[0] then
            ys.[0]
        elif x >= xs.[n - 1] then
            ys.[n - 1]
        else
            let lo = [| 0 .. n - 2 |] |> Array.findBack (fun i -> xs.[i] <= x)
            let frac = (x - xs.[lo]) / (xs.[lo + 1] - xs.[lo])
            ys.[lo] + frac * (ys.[lo + 1] - ys.[lo])

    let linearForwards (tenors: float32[]) (zeroRates: float32[]) (steps: int) : float32[] =
        let dt = 1.0f / float32 steps

        Array.init steps (fun i ->
            let t0 = float32 i * dt
            let t1 = float32 (i + 1) * dt
            let zr0 = lerp tenors zeroRates t0
            let zr1 = lerp tenors zeroRates t1
            if i = 0 then zr1 else (zr1 * t1 - zr0 * t0) / dt)

    let logDiscountForwards (tenors: float32[]) (forwardRates: float32[]) (steps: int) : float32[] =
        let logDfs =
            tenors
            |> Array.mapi (fun i t ->
                if i = 0 then
                    0.0f
                else
                    [| 1..i |]
                    |> Array.sumBy (fun j -> -forwardRates.[j - 1] * (tenors.[j] - tenors.[j - 1])))

        let dt = 1.0f / float32 steps

        Array.init steps (fun i ->
            let ld0 = lerp tenors logDfs (float32 i * dt)
            let ld1 = lerp tenors logDfs (float32 (i + 1) * dt)
            -(ld1 - ld0) / dt)

    // ── Deterministic rate primitives (no ModelCtx needed) ──

    let flat (rate: float32) : Expr = rate.C

    let curve (surfId: int) : ModelCtx -> Expr = fun ctx -> interp1d surfId TimeIndex ctx

    // ── Stochastic rate models ──

    let vasicek (kappa: Expr) (theta: Expr) (sigma: Expr) (r0: float32) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let z = normal ctx
            evolve r0.C (fun r -> r + kappa * (theta - r) * dt + sigma * Expr.sqrt dt * z) ctx

    let cir (kappa: Expr) (theta: Expr) (sigma: Expr) (r0: float32) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let z = normal ctx

            evolve
                r0.C
                (fun r ->
                    let rp = Expr.max r 0.0f.C
                    r + kappa * (theta - rp) * dt + sigma * Expr.sqrt (rp * dt) * z)
                ctx

    let cirpp (kappa: Expr) (theta: Expr) (sigma: Expr) (x0: float32) (shiftSurfId: int) (dt: Expr) : ModelCtx -> Expr =
        fun ctx ->
            let z = normal ctx

            let x =
                evolve
                    x0.C
                    (fun x ->
                        let xp = Expr.max x 0.0f.C
                        x + kappa * (theta - xp) * dt + sigma * Expr.sqrt (xp * dt) * z)
                    ctx

            x + interp1d shiftSurfId TimeIndex ctx

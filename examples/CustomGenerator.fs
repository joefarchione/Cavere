// Custom Generator — building your own stochastic process
//
// Generators are just functions that call `normal` and `evolve`.
// There is nothing special about the built-in gbm/heston/cir —
// you can write your own in a few lines.

module Cavere.Examples.CustomGenerator

open Cavere.Core

// Ornstein-Uhlenbeck mean-reverting process:
//   dx = kappa * (theta - x) * dt + sigma * dW
let ornsteinUhlenbeck
    (kappa: Expr) (theta: Expr) (sigma: Expr)
    (x0: float32) (dt: Expr)
    : ModelCtx -> Expr = fun ctx ->
    let z = normal ctx
    evolve x0.C (fun x ->
        x + kappa * (theta - x) * dt + sigma * Expr.sqrt dt * z) ctx

// Jump-diffusion (Merton-style):
//   dS/S = (r - 0.5*vol^2 - lambda*k)*dt + vol*dW + J*dN
// where dN is Poisson with intensity lambda, and J ~ N(muJ, sigmaJ)
let jumpDiffusion
    (rate: Expr) (vol: Expr) (lambda: Expr) (muJ: Expr) (sigmaJ: Expr)
    (spot: float32) (dt: Expr)
    : ModelCtx -> Expr = fun ctx ->
    let z = normal ctx       // diffusion shock
    let zJ = normal ctx      // jump size shock
    let u = uniform ctx      // uniform for jump timing
    let jumpProb = lambda * dt
    let jumpSize = muJ + sigmaJ * zJ
    let k = Expr.exp (muJ + 0.5f * sigmaJ * sigmaJ) - 1.0f  // compensator
    evolve spot.C (fun price ->
        let drift = (rate - 0.5f * vol * vol - lambda * k) * dt
        let diffusion = vol * Expr.sqrt dt * z
        let hasJump = u .< jumpProb
        let jump = Expr.select hasJump jumpSize 0.0f.C
        price * Expr.exp (drift + diffusion + jump)) ctx

let run () =
    let sched = Schedule.constant (1.0f / 252.0f) 252

    let customModel = model {
        let! dt = scheduleDt sched
        let! rate  = ornsteinUhlenbeck 0.5f.C 0.05f.C 0.01f.C 0.03f dt
        // Jump-diffusion with lambda=1 (1 jump/year), mean jump size -5%, jump vol 10%
        let! stock = jumpDiffusion rate 0.25f.C 1.0f.C (-0.05f).C 0.10f.C 100.0f dt
        return stock
    }

    use sim = Simulation.create CPU 10_000 sched.Steps
    let results = Simulation.fold sim customModel

    printfn "  Simulations: %d" results.Length
    printfn "  Mean terminal: %.2f" (Array.average results)

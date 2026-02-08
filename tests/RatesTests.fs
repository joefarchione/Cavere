module Cavere.Tests.RatesTests

open Xunit
open Cavere.Core
open Cavere.Generators

let mean (arr: float32[]) = arr |> Array.averageBy float |> float

// ── Forward curve tests ──

[<Fact>]
let ``Flat zero curve produces constant forwards`` () =
    let tenors = [| 0.0f; 0.5f; 1.0f |]
    let rates = [| 0.05f; 0.05f; 0.05f |]
    let fwds = Rates.linearForwards tenors rates 252
    fwds |> Array.iter (fun f -> Assert.InRange(float f, 0.049, 0.051))

[<Fact>]
let ``Linear and log-discount agree for flat curve`` () =
    let tenors = [| 0.0f; 0.25f; 0.5f; 1.0f |]
    let rate = 0.05f
    let rates = tenors |> Array.map (fun _ -> rate)
    let linFwds = Rates.linearForwards tenors rates 252
    let logFwds = Rates.logDiscountForwards tenors rates 252
    Array.iter2 (fun l d -> Assert.InRange(float (abs (l - d)), 0.0, 0.002)) linFwds logFwds

// ── Forward curve in model ──

[<Fact>]
let ``Forward curve discount matches flat rate`` () =
    let steps = 252
    let rate = 0.05f
    let fwds = Rates.linearForwards [| 0.0f; 1.0f |] [| rate; rate |] steps

    let curveModel =
        model {
            let dt = (1.0f / float32 steps).C
            let! sid = surface1d fwds steps
            let! r = Rates.curve sid
            let! df = decay r dt
            return df
        }

    let flatModel =
        model {
            let dt = (1.0f / float32 steps).C
            let! df = decay (Rates.flat rate) dt
            return df
        }

    use sim = Simulation.create CPU 1000 steps
    let curveDf = Simulation.fold sim curveModel |> mean
    let flatDf = Simulation.fold sim flatModel |> mean
    Assert.InRange(abs (curveDf - flatDf), 0.0, 0.001)

// ── Stochastic rate models ──

[<Fact>]
let ``Vasicek mean matches analytical`` () =
    let kappa, theta, r0 = 1.0f, 0.05f, 0.10f
    let expected = float theta + float (r0 - theta) * exp (-float kappa * 1.0)

    let m =
        model {
            let dt = (1.0f / 252.0f).C
            let! r = Rates.vasicek kappa.C theta.C 0.01f.C r0 dt
            return r
        }

    use sim = Simulation.create CPU 50_000 252
    let finals = Simulation.fold sim m
    Assert.InRange(mean finals, expected - 0.005, expected + 0.005)

[<Fact>]
let ``CIR mean matches analytical`` () =
    let kappa, theta, r0 = 1.0f, 0.05f, 0.10f
    let expected = float theta + float (r0 - theta) * exp (-float kappa * 1.0)

    let m =
        model {
            let dt = (1.0f / 252.0f).C
            let! r = Rates.cir kappa.C theta.C 0.05f.C r0 dt
            return r
        }

    use sim = Simulation.create CPU 50_000 252
    let finals = Simulation.fold sim m
    Assert.InRange(mean finals, expected - 0.005, expected + 0.005)

[<Fact>]
let ``CIR++ adds deterministic shift`` () =
    let m =
        model {
            let dt = (1.0f / 252.0f).C
            let shift = Array.create 252 0.02f
            let! sid = surface1d shift 252
            let! r = Rates.cirpp 1.0f.C 0.03f.C 0.05f.C 0.03f sid dt
            return r
        }

    let source, _ = Compiler.buildSource m
    Assert.Contains("MathF.Floor(", source)
    use sim = Simulation.create CPU 50_000 252
    let finals = Simulation.fold sim m
    // CIR base reverts to 0.03, shift adds 0.02 → ~0.05
    Assert.InRange(mean finals, 0.04, 0.06)

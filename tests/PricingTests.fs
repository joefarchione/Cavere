module Cavere.Tests.PricingTests

open Xunit
open Cavere.Core
open Cavere.Generators

// --- Black-Scholes analytical formula ---

let normalCdf x =
    let a1 = 0.254829592
    let a2 = -0.284496736
    let a3 = 1.421413741
    let a4 = -1.453152027
    let a5 = 1.061405429
    let p = 0.3275911

    let sign = if x < 0.0 then -1.0 else 1.0
    let x = abs x / sqrt 2.0
    let t = 1.0 / (1.0 + p * x)
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp (-x * x)
    0.5 * (1.0 + sign * y)

let bsCall s k r sigma t =
    let d1 = (log (s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt t)
    let d2 = d1 - sigma * sqrt t
    s * normalCdf d1 - k * exp (-r * t) * normalCdf d2

let mean (arr: float32[]) = arr |> Array.averageBy float |> float

// --- Tests ---

let makeCallModel (spot: float32) (strike: float32) (rate: float32) (vol: float32) (steps: int) =
    model {
        let dt = (1.0f / float32 steps).C
        let! z = normal
        let! stock = gbm z rate.C vol.C spot.C dt
        let! df = decay rate.C dt
        return Expr.max (stock - strike.C) 0.0f.C * df
    }

let makeLocalVolCallModel (spot: float32) (strike: float32) (rate: float32) (vol: float32) (steps: int) =
    model {
        let dt = (1.0f / float32 steps).C
        let! sid = surface2d [| 0.0f; 1.0f |] [| 0.0f; 1e6f |]
                      (array2D [| [| vol; vol |]; [| vol; vol |] |]) steps
        let! z = normal
        let! stock = gbmLocalVol z sid rate.C spot.C dt
        let! df = decay rate.C dt
        return Expr.max (stock - strike.C) 0.0f.C * df
    }

[<Fact>]
let ``GBM matches Black-Scholes for ATM call`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 100_000

    let m = makeCallModel spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps
    let finals = Simulation.fold sim m
    let mcPrice = mean finals

    let analyticalPrice = bsCall (float spot) (float strike) (float rate) (float vol) 1.0
    Assert.InRange(mcPrice, analyticalPrice - 0.5, analyticalPrice + 0.5)

[<Fact>]
let ``GBM matches Black-Scholes for OTM call`` () =
    let spot = 100.0f
    let strike = 120.0f
    let vol = 0.25f
    let rate = 0.03f
    let steps = 252
    let numSims = 100_000

    let m = makeCallModel spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps
    let finals = Simulation.fold sim m
    let mcPrice = mean finals

    let analyticalPrice = bsCall (float spot) (float strike) (float rate) (float vol) 1.0
    Assert.InRange(mcPrice, analyticalPrice - 0.5, analyticalPrice + 0.5)

[<Fact>]
let ``GBM matches Black-Scholes for ITM call`` () =
    let spot = 100.0f
    let strike = 80.0f
    let vol = 0.30f
    let rate = 0.05f
    let steps = 252
    let numSims = 100_000

    let m = makeCallModel spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps
    let finals = Simulation.fold sim m
    let mcPrice = mean finals

    let analyticalPrice = bsCall (float spot) (float strike) (float rate) (float vol) 1.0
    Assert.InRange(mcPrice, analyticalPrice - 1.0, analyticalPrice + 1.0)

[<Fact>]
let ``Flat local vol and GBM produce same price`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 50_000

    let gbmModel = makeCallModel spot strike rate vol steps
    let lvModel = makeLocalVolCallModel spot strike rate vol steps

    use sim = Simulation.create CPU numSims steps

    let gbmFinals = Simulation.fold sim gbmModel
    let gbmPrice = mean gbmFinals

    let lvFinals = Simulation.fold sim lvModel
    let lvPrice = mean lvFinals

    Assert.InRange(abs (gbmPrice - lvPrice), 0.0, 0.5)

// --- Dupire local vol helpers ---

/// Implied vol smile: mild put skew around ATM vol of 0.20
let impliedVol (k: float) = 0.20 + 0.10 * (100.0 / k - 1.0)

/// BS call price (double precision for Dupire numerics)
let bsCallPrice (s: float) (k: float) (r: float) (sigma: float) (t: float) =
    let d1 = (log (s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt t)
    let d2 = d1 - sigma * sqrt t
    s * normalCdf d1 - k * exp (-r * t) * normalCdf d2

/// Dupire local vol from implied vol smile via finite differences on BS prices.
/// σ²_LV(K, T) = [∂C/∂T + r·K·∂C/∂K] / [½·K²·∂²C/∂K²]
let dupireLocalVol (s0: float) (r: float) (k: float) (t: float) =
    let dT = 0.001
    let dK = 0.50
    let sigma = impliedVol k
    let cUp = bsCallPrice s0 k r sigma (t + dT)
    let cDn = bsCallPrice s0 k r sigma (t - dT)
    let dCdT = (cUp - cDn) / (2.0 * dT)

    let sigUp = impliedVol (k + dK)
    let sigDn = impliedVol (k - dK)
    let cKUp = bsCallPrice s0 (k + dK) r sigUp t
    let cKDn = bsCallPrice s0 (k - dK) r sigDn t
    let cK   = bsCallPrice s0 k r sigma t
    let dCdK = (cKUp - cKDn) / (2.0 * dK)
    let d2CdK2 = (cKUp - 2.0 * cK + cKDn) / (dK * dK)

    let numerator = dCdT + r * k * dCdK
    let denominator = 0.5 * k * k * d2CdK2
    if denominator < 1e-12 then sigma  // fallback to implied vol
    else sqrt (numerator / denominator)

[<Fact>]
let ``Local vol with Dupire surface reproduces BS smile`` () =
    let s0 = 100.0
    let rate = 0.05
    let steps = 252
    let numSims = 20_000

    // Build Dupire local vol grid (small grid to avoid deep ILGPU expression trees)
    let timePoints = [| 0.1; 0.5; 1.0 |]
    let spotPoints = [| 60.0; 80.0; 100.0; 120.0; 140.0 |]

    let localVolGrid =
        array2D [|
            for t in timePoints ->
                [| for s in spotPoints -> float32 (dupireLocalVol s0 rate s t) |]
        |]

    let timeAxisF32 = timePoints |> Array.map float32
    let spotAxisF32 = spotPoints |> Array.map float32

    // Build model returning terminal stock price (no payoff)
    let stockModel = model {
        let dt = (1.0f / float32 steps).C
        let! sid = surface2d timeAxisF32 spotAxisF32 localVolGrid steps
        let! z = normal
        let! stock = gbmLocalVol z sid (float32 rate).C (float32 s0).C dt
        return stock
    }

    // Run simulation
    use sim = Simulation.create CPU numSims steps
    let terminals = Simulation.fold sim stockModel

    // Check at three strikes
    let df = exp (-rate * 1.0)
    let testStrikes = [| 90.0; 100.0; 110.0 |]

    for k in testStrikes do
        let mcPrice =
            terminals
            |> Array.averageBy (fun st -> max (st - float32 k) 0.0f |> float)
            |> fun p -> p * df

        let bsPrice = bsCallPrice s0 k rate (impliedVol k) 1.0
        Assert.InRange(mcPrice, bsPrice - 2.0, bsPrice + 2.0)

[<Fact>]
let ``Local vol with large Dupire grid uses BinSearch`` () =
    let s0 = 100.0
    let rate = 0.05
    let steps = 252
    let numSims = 20_000

    // Large grid (15 time x 20 spot) — triggers BinSearch for both axes
    let timePoints = Array.init 15 (fun i -> 0.05 + float i * 0.07)
    let spotPoints = Array.init 20 (fun i -> 50.0 + float i * 5.0)

    let localVolGrid =
        array2D [|
            for t in timePoints ->
                [| for s in spotPoints -> float32 (dupireLocalVol s0 rate s t) |]
        |]

    let timeAxisF32 = timePoints |> Array.map float32
    let spotAxisF32 = spotPoints |> Array.map float32

    let stockModel = model {
        let dt = (1.0f / float32 steps).C
        let! sid = surface2d timeAxisF32 spotAxisF32 localVolGrid steps
        let! z = normal
        let! stock = gbmLocalVol z sid (float32 rate).C (float32 s0).C dt
        return stock
    }

    use sim = Simulation.create CPU numSims steps
    let terminals = Simulation.fold sim stockModel

    let df = exp (-rate * 1.0)
    let testStrikes = [| 90.0; 100.0; 110.0 |]

    for k in testStrikes do
        let mcPrice =
            terminals
            |> Array.averageBy (fun st -> max (st - float32 k) 0.0f |> float)
            |> fun p -> p * df

        let bsPrice = bsCallPrice s0 k rate (impliedVol k) 1.0
        Assert.InRange(mcPrice, bsPrice - 2.0, bsPrice + 2.0)

module Cavere.Tests.PricingTests

open System.Numerics
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
        let! sid = surface2d [| 0.0f; 1.0f |] [| 0.0f; 1e6f |] (array2D [| [| vol; vol |]; [| vol; vol |] |]) steps
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
    let cK = bsCallPrice s0 k r sigma t
    let dCdK = (cKUp - cKDn) / (2.0 * dK)
    let d2CdK2 = (cKUp - 2.0 * cK + cKDn) / (dK * dK)

    let numerator = dCdT + r * k * dCdK
    let denominator = 0.5 * k * k * d2CdK2

    if denominator < 1e-12 then
        sigma // fallback to implied vol
    else
        sqrt (numerator / denominator)

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
            for t in timePoints -> [| for s in spotPoints -> float32 (dupireLocalVol s0 rate s t) |]
        |]

    let timeAxisF32 = timePoints |> Array.map float32
    let spotAxisF32 = spotPoints |> Array.map float32

    // Build model returning terminal stock price (no payoff)
    let stockModel =
        model {
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
            for t in timePoints -> [| for s in spotPoints -> float32 (dupireLocalVol s0 rate s t) |]
        |]

    let timeAxisF32 = timePoints |> Array.map float32
    let spotAxisF32 = spotPoints |> Array.map float32

    let stockModel =
        model {
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

// ══════════════════════════════════════════════════════════════════
// Heston Semi-Analytical Formula (Heston 1993)
// ══════════════════════════════════════════════════════════════════

/// Heston characteristic function call price.
/// Uses the original Heston (1993) two-integral formulation.
let hestonCall
    (s0: float)
    (k: float)
    (r: float)
    (v0: float)
    (kappa: float)
    (theta: float)
    (xi: float)
    (rho: float)
    (t: float)
    : float =
    let a = kappa * theta
    let x = log s0
    let lnK = log k
    let ci = Complex.ImaginaryOne

    let computeP (uj: float) (bj: float) =
        let integrand (phi: float) =
            if phi < 1e-10 then
                0.0
            else
                let u = Complex(phi, 0.0)
                let bjC = Complex(bj, 0.0)
                let rhoXiC = Complex(rho * xi, 0.0)
                let xi2C = Complex(xi * xi, 0.0)
                let ujC = Complex(uj, 0.0)

                let d =
                    Complex.Sqrt(
                        (bjC - rhoXiC * ci * u) * (bjC - rhoXiC * ci * u)
                        + xi2C * (u * u - Complex(2.0, 0.0) * ujC * ci * u)
                    )

                let g = (bjC - rhoXiC * ci * u - d) / (bjC - rhoXiC * ci * u + d)
                let edT = Complex.Exp(-d * Complex(t, 0.0))

                let cj =
                    Complex(r, 0.0) * ci * u * Complex(t, 0.0)
                    + Complex(a / (xi * xi), 0.0)
                      * ((bjC - rhoXiC * ci * u - d) * Complex(t, 0.0)
                         - Complex(2.0, 0.0) * Complex.Log((Complex.One - g * edT) / (Complex.One - g)))

                let dj =
                    (bjC - rhoXiC * ci * u - d) / xi2C * (Complex.One - edT)
                    / (Complex.One - g * edT)

                let fj = Complex.Exp(cj + dj * Complex(v0, 0.0) + ci * u * Complex(x, 0.0))
                (fj * Complex.Exp(-ci * u * Complex(lnK, 0.0)) / (ci * u)).Real

        let maxPhi = 200.0
        let n = 10000
        let h = maxPhi / float n
        let mutable sum = 0.0

        for j in 1 .. n - 1 do
            sum <- sum + integrand (float j * h)

        sum <- sum + 0.5 * integrand maxPhi
        0.5 + sum * h / System.Math.PI

    let p1 = computeP 0.5 (kappa - rho * xi)
    let p2 = computeP -0.5 kappa
    s0 * p1 - k * exp (-r * t) * p2

// ══════════════════════════════════════════════════════════════════
// Heston MC vs Semi-Analytical
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``Heston MC matches semi-analytical for ATM call`` () =
    let s0 = 100.0f
    let k = 100.0f
    let r = 0.05f
    let v0 = 0.04f
    let kappa = 2.0f
    let theta = 0.04f
    let xi = 0.5f
    let rho = -0.7f
    let steps = 252
    let numSims = 200_000

    let m =
        model {
            let dt = (1.0f / float32 steps).C
            let! z = normal
            let! stock = heston z r.C v0.C kappa.C theta.C xi.C rho.C s0.C dt
            let! df = decay r.C dt
            return Expr.max (stock - k.C) 0.0f.C * df
        }

    use sim = Simulation.create CPU numSims steps
    let finals = Simulation.fold sim m
    let mcPrice = mean finals

    let analyticalPrice =
        hestonCall (float s0) (float k) (float r) (float v0) (float kappa) (float theta) (float xi) (float rho) 1.0

    Assert.InRange(mcPrice, analyticalPrice - 1.0, analyticalPrice + 1.0)

[<Fact>]
let ``Heston MC matches semi-analytical for OTM call`` () =
    let s0 = 100.0f
    let k = 120.0f
    let r = 0.05f
    let v0 = 0.04f
    let kappa = 2.0f
    let theta = 0.04f
    let xi = 0.3f
    let rho = -0.5f
    let steps = 252
    let numSims = 200_000

    let m =
        model {
            let dt = (1.0f / float32 steps).C
            let! z = normal
            let! stock = heston z r.C v0.C kappa.C theta.C xi.C rho.C s0.C dt
            let! df = decay r.C dt
            return Expr.max (stock - k.C) 0.0f.C * df
        }

    use sim = Simulation.create CPU numSims steps
    let finals = Simulation.fold sim m
    let mcPrice = mean finals

    let analyticalPrice =
        hestonCall (float s0) (float k) (float r) (float v0) (float kappa) (float theta) (float xi) (float rho) 1.0

    Assert.InRange(mcPrice, analyticalPrice - 1.0, analyticalPrice + 1.0)

// ══════════════════════════════════════════════════════════════════
// AD Greeks vs Black-Scholes Analytical
// ══════════════════════════════════════════════════════════════════

let normalPdfD (x: float) = exp (-0.5 * x * x) / sqrt (2.0 * System.Math.PI)

let bsDelta (s: float) (k: float) (r: float) (sigma: float) (t: float) =
    let d1 = (log (s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt t)
    normalCdf d1

let bsVega (s: float) (k: float) (r: float) (sigma: float) (t: float) =
    let d1 = (log (s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt t)
    s * normalPdfD d1 * sqrt t

[<Fact>]
let ``AD forward-mode delta matches Black-Scholes delta`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 200_000

    let m =
        model {
            let dt = (1.0f / float32 steps).C
            let! z = normal
            let! stock = gbm z rate.C vol.C (Dual(0, spot, "spot")) dt
            let! df = decay rate.C dt
            return Expr.max (stock - strike.C) 0.0f.C * df
        }

    use sim = Simulation.create CPU numSims steps
    let _, watch = Simulation.foldWatch sim m Monthly
    let deltas = Watcher.terminals "d1_spot" watch
    let mcDelta = deltas |> Array.averageBy float |> float

    let analyticalDelta = bsDelta (float spot) (float strike) (float rate) (float vol) 1.0
    Assert.InRange(mcDelta, analyticalDelta - 0.02, analyticalDelta + 0.02)

[<Fact>]
let ``AD forward-mode vega matches Black-Scholes vega`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 200_000

    let m =
        model {
            let dt = (1.0f / float32 steps).C
            let! z = normal
            let! stock = gbm z rate.C (Dual(0, vol, "vol")) spot.C dt
            let! df = decay rate.C dt
            return Expr.max (stock - strike.C) 0.0f.C * df
        }

    use sim = Simulation.create CPU numSims steps
    let _, watch = Simulation.foldWatch sim m Monthly
    let vegas = Watcher.terminals "d1_vol" watch
    let mcVega = vegas |> Array.averageBy float |> float

    let analyticalVega = bsVega (float spot) (float strike) (float rate) (float vol) 1.0
    Assert.InRange(mcVega, analyticalVega - 2.0, analyticalVega + 2.0)

[<Fact>]
let ``AD adjoint delta matches Black-Scholes delta`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 200_000

    let m =
        model {
            let dt = (1.0f / float32 steps).C
            let! z = normal
            let! stock = gbm z rate.C vol.C (Dual(0, spot, "spot")) dt
            let! df = decay rate.C dt
            return Expr.max (stock - strike.C) 0.0f.C * df
        }

    use sim = Simulation.create CPU numSims steps
    let _, adjoints = Simulation.foldAdjoint sim m
    let mcDelta = Array.init numSims (fun i -> adjoints.[i, 0]) |> Array.averageBy float |> float

    let analyticalDelta = bsDelta (float spot) (float strike) (float rate) (float vol) 1.0
    Assert.InRange(mcDelta, analyticalDelta - 0.02, analyticalDelta + 0.02)

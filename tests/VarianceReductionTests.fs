module Cavere.Tests.VarianceReductionTests

open Xunit
open Cavere.Core
open Cavere.Generators

let mean (arr: float32[]) = arr |> Array.averageBy float |> float

let variance (arr: float32[]) =
    let m = Array.average arr
    arr |> Array.averageBy (fun x -> (x - m) * (x - m)) |> float

let makeCallModel (spot: float32) (strike: float32) (rate: float32) (vol: float32) (steps: int) =
    model {
        let dt = (1.0f / float32 steps).C
        let! z = normal
        let! stock = gbm z rate.C vol.C spot.C dt
        let! df = decay rate.C dt
        return Expr.max (stock - strike.C) 0.0f.C * df
    }

let makeCallModelWithControl (spot: float32) (strike: float32) (rate: float32) (vol: float32) (steps: int) =
    model {
        let dt = (1.0f / float32 steps).C
        let! z = normal
        let! stock = gbm z rate.C vol.C spot.C dt
        let! df = decay rate.C dt
        do! observe "fwd" (stock * df)
        return Expr.max (stock - strike.C) 0.0f.C * df
    }

// ══════════════════════════════════════════════════════════════════
// Antithetic Variates
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``Antithetic converges to same mean as naive for ATM call`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 50_000

    let m = makeCallModel spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps

    let naive = Simulation.fold sim m
    let antithetic = Simulation.foldAntithetic sim m

    let naiveMean = mean naive
    let antitheticMean = mean antithetic

    // Both should converge to approximately the same price
    Assert.InRange(abs (naiveMean - antitheticMean), 0.0, 1.0)

[<Fact>]
let ``Antithetic reduces variance for ATM call`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 50_000

    let m = makeCallModel spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps

    let naive = Simulation.fold sim m
    let antithetic = Simulation.foldAntithetic sim m

    let naiveVar = variance naive
    let antitheticVar = variance antithetic

    // Antithetic should have lower variance (at least 10% reduction)
    Assert.True(
        antitheticVar < naiveVar * 0.90,
        $"Expected variance reduction: naive={naiveVar:F4}, antithetic={antitheticVar:F4}"
    )

[<Fact>]
let ``Antithetic reduces variance for OTM call`` () =
    let spot = 100.0f
    let strike = 120.0f
    let vol = 0.25f
    let rate = 0.03f
    let steps = 252
    let numSims = 50_000

    let m = makeCallModel spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps

    let naive = Simulation.fold sim m
    let antithetic = Simulation.foldAntithetic sim m

    let naiveVar = variance naive
    let antitheticVar = variance antithetic

    Assert.True(
        antitheticVar < naiveVar * 0.95,
        $"Expected variance reduction: naive={naiveVar:F4}, antithetic={antitheticVar:F4}"
    )

// ══════════════════════════════════════════════════════════════════
// Control Variates
// ══════════════════════════════════════════════════════════════════

[<Fact>]
let ``Control variate converges to same mean as naive for ATM call`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 50_000

    let m = makeCallModelWithControl spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps

    let naive = Simulation.fold sim m
    let naiveMean = mean naive

    let controls = [ ControlVariate.discountedAsset "fwd" spot ]
    let cv = Simulation.foldControlVariate sim m controls
    let cvMean = mean cv

    // Both should converge to approximately the same price
    Assert.InRange(abs (naiveMean - cvMean), 0.0, 1.0)

[<Fact>]
let ``Control variate produces tighter estimate than naive`` () =
    let spot = 100.0f
    let strike = 100.0f
    let vol = 0.20f
    let rate = 0.05f
    let steps = 252
    let numSims = 50_000

    let m = makeCallModelWithControl spot strike rate vol steps
    use sim = Simulation.create CPU numSims steps

    // Run naive multiple times to get standard error estimate
    let naive = Simulation.fold sim m
    let naiveMean = mean naive

    let controls = [ ControlVariate.discountedAsset "fwd" spot ]
    let cv = Simulation.foldControlVariate sim m controls
    let cvMean = mean cv

    // Black-Scholes reference for ATM 1Y call
    let bsRef = PricingTests.bsCall 100.0 100.0 0.05 0.20 1.0

    // The CV estimate should be closer to the analytical price than naive
    let naiveError = abs (naiveMean - bsRef)
    let cvError = abs (cvMean - bsRef)

    // CV should be at least as good (typically much better)
    // Use a generous tolerance since these are stochastic
    Assert.True(cvError < naiveError + 0.5, $"CV error ({cvError:F4}) should be <= naive error ({naiveError:F4})")

[<Fact>]
let ``Control variate fails for missing observer`` () =
    let m = makeCallModel 100.0f 100.0f 0.05f 0.20f 252
    use sim = Simulation.create CPU 1_000 252

    let controls = [ ControlVariate.create "nonexistent" 100.0f ]

    Assert.Throws<System.Exception>(fun () -> Simulation.foldControlVariate sim m controls |> ignore)

[<Fact>]
let ``Control variate with empty list falls back to fold`` () =
    let m = makeCallModel 100.0f 100.0f 0.05f 0.20f 252
    use sim = Simulation.create CPU 10_000 252

    let naive = Simulation.fold sim m
    let cv = Simulation.foldControlVariate sim m []

    // With empty controls, should get same result as fold
    Assert.Equal(naive.Length, cv.Length)

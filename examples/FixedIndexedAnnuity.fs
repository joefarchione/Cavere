// Fixed Indexed Annuity — batch kernel
//
// Annual point-to-point crediting on equity index returns. Each policy
// earns clip(0, cap, indexReturn) per year. Account value compounds
// the credited rates over the policy term.
//
// All policies run in a SINGLE kernel launch via BatchSimulation. Premium,
// cap, and term vectors are passed into the model as batch inputs —
// each thread indexes its policy's values via batchIdx. The kernel
// runs maxTerm steps; shorter-term policies freeze via Select guards.

module Cavere.Examples.FixedIndexedAnnuity

open Cavere.Core
open Cavere.Generators

// ── Domain types ───────────────────────────────────────────────────

type CreditingStrategy = CallSpread of cap: float32

type Product = {
    Name: string
    Strategy: CreditingStrategy
    Term: int
}

type Policy = {
    Id: string
    Product: Product
    Premium: float32
}

// ── Generator ──────────────────────────────────────────────────────

/// Annual call spread accumulator with per-batch cap and term.
/// Freezes account value after term via Select guard.
let callSpreadAccumBatch (rate: Expr) (vol: Expr) (cap: Expr) (term: Expr) (init: Expr) (dt: Expr) : ModelCtx -> Expr =
    fun ctx ->
        let z = normal ctx
        let growth = Expr.exp ((rate - 0.5f * vol * vol) * dt + vol * Expr.sqrt dt * z)
        let credited = Expr.clip 0.0f.C cap (growth - 1.0f)
        evolve init (fun av -> Expr.select (TimeIndex .< term) (av * (1.0f.C + credited)) av) ctx

/// Discount factor that freezes after term.
let decayUntil (rate: Expr) (term: Expr) (dt: Expr) : ModelCtx -> Expr =
    fun ctx -> evolve 1.0f.C (fun df -> Expr.select (TimeIndex .< term) (df * Expr.exp (-rate * dt)) df) ctx

// ── Model builder ────────────────────────────────────────────────

/// Batch model — data vectors are baked into the model at build time.
let buildFIAModel (rate: float32) (vol: float32) (premiums: float32[]) (caps: float32[]) (terms: float32[]) : Model =
    model {
        let dt = 1.0f.C
        let! premium = batchInput premiums
        let! cap = batchInput caps
        let! term = batchInput terms
        let! av = callSpreadAccumBatch rate.C vol.C cap term premium dt
        let! df = decayUntil rate.C term dt
        do! observe "av" av
        return av * df
    }

// ── Mock data ──────────────────────────────────────────────────────

let products = [
    {
        Name = "Growth Plus 7"
        Strategy = CallSpread 0.065f
        Term = 7
    }
    {
        Name = "Income Builder 10"
        Strategy = CallSpread 0.055f
        Term = 10
    }
    {
        Name = "Shield 5"
        Strategy = CallSpread 0.080f
        Term = 5
    }
]

let policies = [
    { Id = "POL-001"; Product = products.[0]; Premium = 100_000.0f }
    { Id = "POL-002"; Product = products.[0]; Premium = 250_000.0f }
    { Id = "POL-003"; Product = products.[1]; Premium = 150_000.0f }
    { Id = "POL-004"; Product = products.[1]; Premium = 75_000.0f }
    { Id = "POL-005"; Product = products.[2]; Premium = 500_000.0f }
]

// ── Runner ─────────────────────────────────────────────────────────

let run () =
    let rate = 0.04f
    let vol = 0.18f
    let numScenarios = 50_000

    printfn ""
    printfn "  %-8s  %-20s  %10s  %5s  %5s  %12s" "Policy" "Product" "Premium" "Term" "Cap" "PV Interest"
    printfn "  %s" (System.String('-', 72))

    let maxTerm = policies |> List.map _.Product.Term |> List.max

    let pols = policies |> Array.ofList
    let premiums = pols |> Array.map (fun p -> p.Premium)
    let caps = pols |> Array.map (fun p -> let (CallSpread c) = p.Product.Strategy in c)
    let terms = pols |> Array.map (fun p -> float32 p.Product.Term)

    let m = buildFIAModel rate vol premiums caps terms

    use sim = BatchSimulation.create CPU policies.Length numScenarios maxTerm
    let _finals, watch = BatchSimulation.foldWatch sim m Monthly

    // Compute means from finals
    let means = BatchSimulation.foldMeans sim m

    policies
    |> List.iteri (fun i policy ->
        let (CallSpread cap) = policy.Product.Strategy
        let pvInterest = means.[i] - policy.Premium * exp (-rate * float32 policy.Product.Term)

        printfn
            "  %-8s  %-20s  %10.0f  %5d  %4.1f%%  %12.2f"
            policy.Id
            policy.Product.Name
            policy.Premium
            policy.Product.Term
            (cap * 100.0f)
            pvInterest)

    // Path display for first policy (batch 0 = threads 0..numScenarios-1)
    printfn ""
    printfn "  Account value paths (POL-001, first 5 paths):"
    let avPaths = Watcher.values "av" watch

    printfn "  %5s  %s" "Year" (String.concat "  " [ for p in 0..4 -> sprintf "%12s" (sprintf "Path %d" (p + 1)) ])

    for yr in 0 .. maxTerm - 1 do
        let vals = [ for p in 0..4 -> sprintf "%12.0f" avPaths.[yr, p] ]
        printfn "  %5d  %s" (yr + 1) (String.concat "  " vals)

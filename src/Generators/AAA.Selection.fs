namespace Cavere.Generators.AAA

/// AAA Scenario Selection — Reduce large scenario sets to representative subsets.
/// Preserves statistical properties (tails, moments) for efficient actuarial projections.
module Selection =

    // ══════════════════════════════════════════════════════════════════
    // Utility Functions
    // ══════════════════════════════════════════════════════════════════

    /// Argsort: return indices that would sort the array.
    let argsort (arr: float32[]) : int[] =
        arr
        |> Array.mapi (fun i v -> (i, v))
        |> Array.sortBy snd
        |> Array.map fst

    /// Compute percentile value (0.0 to 1.0).
    let percentile (sorted: float32[]) (p: float32) : float32 =
        let idx = int (p * float32 (sorted.Length - 1))
        sorted.[min idx (sorted.Length - 1)]

    /// Compute mean of array.
    let mean (arr: float32[]) : float32 =
        if arr.Length = 0 then 0.0f
        else Array.sum arr / float32 arr.Length

    /// Compute variance of array.
    let variance (arr: float32[]) : float32 =
        if arr.Length < 2 then 0.0f
        else
            let m = mean arr
            arr |> Array.map (fun x -> (x - m) * (x - m)) |> Array.sum |> fun s -> s / float32 arr.Length

    /// Compute skewness of array.
    let skewness (arr: float32[]) : float32 =
        if arr.Length < 3 then 0.0f
        else
            let m = mean arr
            let v = variance arr
            if v < 1e-10f then 0.0f
            else
                let s = sqrt v
                arr |> Array.map (fun x -> ((x - m) / s) ** 3.0f) |> Array.sum |> fun sum -> sum / float32 arr.Length

    /// Compute kurtosis of array (excess kurtosis, normal = 0).
    let kurtosis (arr: float32[]) : float32 =
        if arr.Length < 4 then 0.0f
        else
            let m = mean arr
            let v = variance arr
            if v < 1e-10f then 0.0f
            else
                let s = sqrt v
                arr |> Array.map (fun x -> ((x - m) / s) ** 4.0f) |> Array.sum |> fun sum -> sum / float32 arr.Length - 3.0f

    // ══════════════════════════════════════════════════════════════════
    // Stratified Selection
    // ══════════════════════════════════════════════════════════════════

    /// Stratified selection by percentile buckets.
    /// bucketEdges: e.g., [| 0.0f; 0.05f; 0.25f; 0.50f; 0.75f; 0.95f; 1.0f |] for 6 buckets
    /// perBucket: how many scenarios to select from each bucket
    /// Returns indices of selected scenarios.
    let stratifiedSelect (metric: float32[]) (bucketEdges: float32[]) (perBucket: int) : int[] =
        let n = metric.Length
        let sortedIdx = argsort metric

        // For each bucket, find the range of indices and sample
        let numBuckets = bucketEdges.Length - 1
        [| for b in 0 .. numBuckets - 1 do
            let startPct = bucketEdges.[b]
            let endPct = bucketEdges.[b + 1]
            let startIdx = int (startPct * float32 n)
            let endIdx = min n (int (endPct * float32 n))
            let bucketSize = endIdx - startIdx

            if bucketSize > 0 then
                // Evenly space selections within bucket
                let step = max 1 (bucketSize / perBucket)
                for i in 0 .. min perBucket bucketSize - 1 do
                    yield sortedIdx.[startIdx + i * step]
        |]

    /// Default bucket edges for tail-preserving stratification.
    /// Buckets: 0-5% (left tail), 5-25%, 25-50%, 50-75%, 75-95%, 95-100% (right tail)
    let defaultBucketEdges = [| 0.0f; 0.05f; 0.25f; 0.50f; 0.75f; 0.95f; 1.0f |]

    // ══════════════════════════════════════════════════════════════════
    // Tail-Preserving Selection
    // ══════════════════════════════════════════════════════════════════

    /// Select scenarios preserving both tails.
    /// worstN: number of scenarios from worst (lowest metric values)
    /// bestN: number of scenarios from best (highest metric values)
    /// middleN: number of scenarios from the middle
    /// Returns indices of selected scenarios.
    let tailPreservingSelect (metric: float32[]) (worstN: int) (bestN: int) (middleN: int) : int[] =
        let n = metric.Length
        let sortedIdx = argsort metric

        let worst = sortedIdx.[0 .. min worstN n - 1]
        let best = sortedIdx.[max 0 (n - bestN) .. n - 1]

        // Select middle evenly spaced
        let middleStart = worstN
        let middleEnd = n - bestN
        let middleRange = middleEnd - middleStart
        let middle =
            if middleRange <= 0 || middleN <= 0 then [||]
            else
                let step = max 1 (middleRange / middleN)
                [| for i in 0 .. min middleN middleRange - 1 -> sortedIdx.[middleStart + i * step] |]

        Array.concat [| worst; middle; best |] |> Array.distinct

    /// CTE-focused selection: heavily weight the left tail for CTE calculations.
    /// cteLevel: e.g., 0.70 for CTE70
    /// tailCount: scenarios in the CTE tail
    /// otherCount: scenarios outside the tail
    let cteSelect (metric: float32[]) (cteLevel: float32) (tailCount: int) (otherCount: int) : int[] =
        let n = metric.Length
        let sortedIdx = argsort metric
        let tailSize = int ((1.0f - cteLevel) * float32 n)

        // Select all or most from the tail
        let tail = sortedIdx.[0 .. min tailCount tailSize - 1]

        // Select evenly from the rest
        let restStart = tailSize
        let restRange = n - tailSize
        let rest =
            if restRange <= 0 || otherCount <= 0 then [||]
            else
                let step = max 1 (restRange / otherCount)
                [| for i in 0 .. min otherCount restRange - 1 -> sortedIdx.[restStart + i * step] |]

        Array.concat [| tail; rest |] |> Array.distinct

    // ══════════════════════════════════════════════════════════════════
    // Moment Matching Selection
    // ══════════════════════════════════════════════════════════════════

    /// Greedy moment-matching selection.
    /// Iteratively selects scenarios that best match target moments.
    /// targetCount: number of scenarios to select
    /// Returns indices and optional weights (equal weights for simplicity).
    let momentMatchingSelect (metric: float32[]) (targetCount: int) : int[] =
        let n = metric.Length
        if targetCount >= n then [| 0 .. n - 1 |]
        else
            // Target moments from full set
            let targetMean = mean metric
            let targetVar = variance metric
            let targetSkew = skewness metric

            // Greedy selection
            let selected = ResizeArray<int>()
            let remaining = ResizeArray<int>([| 0 .. n - 1 |])

            // Start with median scenario
            let sortedIdx = argsort metric
            let medianIdx = sortedIdx.[n / 2]
            selected.Add(medianIdx)
            remaining.Remove(medianIdx) |> ignore

            while selected.Count < targetCount && remaining.Count > 0 do
                // Find scenario that best improves moment matching
                let currentVals = [| for i in selected -> metric.[i] |]
                let mutable bestIdx = remaining.[0]
                let mutable bestError = System.Single.MaxValue

                for idx in remaining do
                    let testVals = Array.append currentVals [| metric.[idx] |]
                    let testMean = mean testVals
                    let testVar = variance testVals
                    let testSkew = skewness testVals
                    let error =
                        abs (testMean - targetMean) / (abs targetMean + 0.001f) +
                        abs (testVar - targetVar) / (abs targetVar + 0.001f) +
                        abs (testSkew - targetSkew) / (abs targetSkew + 1.0f)
                    if error < bestError then
                        bestError <- error
                        bestIdx <- idx

                selected.Add(bestIdx)
                remaining.Remove(bestIdx) |> ignore

            selected.ToArray()

    // ══════════════════════════════════════════════════════════════════
    // Multi-Metric Selection
    // ══════════════════════════════════════════════════════════════════

    /// Combined metric for multi-factor selection.
    /// weights: weights for each metric (should sum to 1)
    let combinedMetric (metrics: float32[][]) (weights: float32[]) : float32[] =
        let n = metrics.[0].Length
        Array.init n (fun i ->
            Array.zip metrics weights
            |> Array.sumBy (fun (m, w) -> w * m.[i]))

    /// Multi-metric stratified selection.
    /// Uses combined metric for stratification.
    let multiMetricSelect
        (equityMetric: float32[])
        (rateMetric: float32[])
        (equityWeight: float32)
        (bucketEdges: float32[])
        (perBucket: int)
        : int[] =
        let combined = combinedMetric [| equityMetric; rateMetric |] [| equityWeight; 1.0f - equityWeight |]
        stratifiedSelect combined bucketEdges perBucket

    // ══════════════════════════════════════════════════════════════════
    // Validation & Diagnostics
    // ══════════════════════════════════════════════════════════════════

    /// Compute selection quality metrics.
    type SelectionQuality = {
        FullMean: float32
        SelectedMean: float32
        FullVariance: float32
        SelectedVariance: float32
        FullSkewness: float32
        SelectedSkewness: float32
        FullKurtosis: float32
        SelectedKurtosis: float32
        MeanError: float32
        VarianceError: float32
    }

    /// Evaluate the quality of a scenario selection.
    let evaluateSelection (fullMetric: float32[]) (selectedIndices: int[]) : SelectionQuality =
        let selected = [| for i in selectedIndices -> fullMetric.[i] |]

        let fullMean = mean fullMetric
        let selMean = mean selected
        let fullVar = variance fullMetric
        let selVar = variance selected

        {
            FullMean = fullMean
            SelectedMean = selMean
            FullVariance = fullVar
            SelectedVariance = selVar
            FullSkewness = skewness fullMetric
            SelectedSkewness = skewness selected
            FullKurtosis = kurtosis fullMetric
            SelectedKurtosis = kurtosis selected
            MeanError = abs (selMean - fullMean) / (abs fullMean + 0.001f)
            VarianceError = abs (selVar - fullVar) / (abs fullVar + 0.001f)
        }

    /// Print selection quality summary.
    let printQuality (q: SelectionQuality) : unit =
        printfn "Selection Quality:"
        printfn $"  Mean:     Full={q.FullMean:F4}  Selected={q.SelectedMean:F4}  Error={q.MeanError * 100.0f:F2}%%"
        printfn $"  Variance: Full={q.FullVariance:F4}  Selected={q.SelectedVariance:F4}  Error={q.VarianceError * 100.0f:F2}%%"
        printfn $"  Skewness: Full={q.FullSkewness:F4}  Selected={q.SelectedSkewness:F4}"
        printfn $"  Kurtosis: Full={q.FullKurtosis:F4}  Selected={q.SelectedKurtosis:F4}"

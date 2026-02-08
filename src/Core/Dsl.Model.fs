namespace Cavere.Core

type AccumDef = { Init: Expr; Body: Expr }

type SurfaceData =
    | Curve1D of values: float32[] * steps: int
    | Grid2D of values: float32[] * timeAxis: float32[] * spotAxis: float32[] * steps: int

type ObserverSpec = {
    Name: string
    Expr: Expr
    SlotIndex: int
}

type Model = {
    Result: Expr
    Accums: Map<int, AccumDef>
    Surfaces: Map<int, SurfaceData>
    Observers: ObserverSpec list
    NormalCount: int
    UniformCount: int
    BernoulliCount: int
    BatchSize: int
}

type ModelCtx = {
    mutable NextNormalId: int
    mutable NextUniformId: int
    mutable NextBernoulliId: int
    mutable NextAccumId: int
    mutable NextSurfaceId: int
    mutable BatchSize: int
    mutable Accums: Map<int, AccumDef>
    mutable Surfaces: Map<int, SurfaceData>
    mutable Observers: ObserverSpec list
}

type ModelBuilder() =
    member _.Return(x: Expr) : ModelCtx -> Expr = fun _ -> x

    member _.Bind(f: ModelCtx -> 'T, g: 'T -> ModelCtx -> 'R) : ModelCtx -> 'R = fun ctx -> (g (f ctx)) ctx

    member _.Zero() : ModelCtx -> unit = fun _ -> ()

    member _.For(sequence: seq<'T>, body: 'T -> ModelCtx -> unit) : ModelCtx -> unit =
        fun ctx ->
            for item in sequence do
                body item ctx

    member _.Combine(a: ModelCtx -> unit, b: ModelCtx -> 'T) : ModelCtx -> 'T =
        fun ctx ->
            a ctx
            b ctx

    member _.Delay(f: unit -> ModelCtx -> 'T) : ModelCtx -> 'T = fun ctx -> f () ctx

    member _.Run(f: ModelCtx -> Expr) : Model =
        let ctx = {
            NextNormalId = 0
            NextUniformId = 0
            NextBernoulliId = 0
            NextAccumId = 0
            NextSurfaceId = 0
            BatchSize = 0
            Accums = Map.empty
            Surfaces = Map.empty
            Observers = []
        }

        let result = f ctx

        {
            Result = result
            Accums = ctx.Accums
            Surfaces = ctx.Surfaces
            Observers = List.rev ctx.Observers
            NormalCount = ctx.NextNormalId
            UniformCount = ctx.NextUniformId
            BernoulliCount = ctx.NextBernoulliId
            BatchSize = ctx.BatchSize
        }

[<AutoOpen>]
module ModelDsl =

    let model = ModelBuilder()

    let normal: ModelCtx -> Expr =
        fun ctx ->
            let id = ctx.NextNormalId
            ctx.NextNormalId <- id + 1
            Normal id

    let uniform: ModelCtx -> Expr =
        fun ctx ->
            let id = ctx.NextUniformId
            ctx.NextUniformId <- id + 1
            Uniform id

    let bernoulli: ModelCtx -> Expr =
        fun ctx ->
            let id = ctx.NextBernoulliId
            ctx.NextBernoulliId <- id + 1
            Bernoulli id

    let evolve (init: Expr) (body: Expr -> Expr) : ModelCtx -> Expr =
        fun ctx ->
            let id = ctx.NextAccumId
            ctx.NextAccumId <- id + 1
            let self = AccumRef id
            ctx.Accums <- ctx.Accums |> Map.add id { Init = init; Body = body self }
            self

    let surface1d (values: float32[]) (steps: int) : ModelCtx -> int =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(values, steps))
            id

    let surface2d (times: float32[]) (spots: float32[]) (vols: float32[,]) (steps: int) : ModelCtx -> int =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1

            let flat =
                Array.init (Array2D.length1 vols * Array2D.length2 vols) (fun i ->
                    vols.[i / Array2D.length2 vols, i % Array2D.length2 vols])

            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Grid2D(flat, times, spots, steps))
            id

    let scheduleDt (sched: Schedule) : ModelCtx -> Expr =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(sched.Dt, sched.Steps))
            Lookup1D id

    let scheduleT (sched: Schedule) : ModelCtx -> Expr =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(sched.T, sched.Steps))
            Lookup1D id

    let schedule (sched: Schedule) : ModelCtx -> Expr * Expr =
        fun ctx ->
            let dt = scheduleDt sched ctx
            let t = scheduleT sched ctx
            dt, t

    let batchInput (values: float32[]) : ModelCtx -> Expr =
        fun ctx ->
            let id = ctx.NextSurfaceId
            ctx.NextSurfaceId <- id + 1
            ctx.Surfaces <- ctx.Surfaces |> Map.add id (Curve1D(values, values.Length))
            ctx.BatchSize <- values.Length
            BatchRef id

    let observe (name: string) (expr: Expr) : ModelCtx -> unit =
        fun ctx ->
            let slot = ctx.Observers.Length

            ctx.Observers <-
                {
                    Name = name
                    Expr = expr
                    SlotIndex = slot
                }
                :: ctx.Observers

    /// Iterate from start to endExclusive (exclusive), running body for each index.
    /// Use inside model { } CE with do!
    /// Example: do! iter 0 arr.Length (fun i -> observe (sprintf "item_%d" i) arr.[i])
    let iter (start: int) (endExclusive: int) (body: int -> ModelCtx -> unit) : ModelCtx -> unit =
        fun ctx ->
            for i in start .. endExclusive - 1 do
                body i ctx

    let interp1d (surfaceId: int) (t: Expr) : ModelCtx -> Expr =
        fun ctx ->
            match ctx.Surfaces.[surfaceId] with
            | Curve1D(values, steps) ->
                let count = values.Length
                let pos = t * (float32 (count - 1)).C / (float32 (max 1 (steps - 1))).C
                let lo = Expr.clip 0.0f.C (float32 (max 0 (count - 2))).C (Floor pos)
                let frac = pos - lo
                let v0 = SurfaceAt(surfaceId, lo)
                let v1 = SurfaceAt(surfaceId, lo + 1.0f.C)
                v0 + (v1 - v0) * frac
            | Grid2D _ -> failwith "interp1d requires Curve1D surface"

    let private binSearchThreshold = 0

    let private findBin (surfaceId: int) (axisOff: int) (axisCnt: int) (value: Expr) =
        if axisCnt <= binSearchThreshold then
            [ 1 .. axisCnt - 1 ]
            |> List.fold
                (fun acc i ->
                    let axisVal = SurfaceAt(surfaceId, (float32 axisOff + float32 i).C)
                    Expr.select (value .>= axisVal) (float32 i).C acc)
                0.0f.C
            |> fun lo -> Expr.clip 0.0f.C (float32 (max 0 (axisCnt - 2))).C lo
        else
            BinSearch(surfaceId, axisOff, axisCnt, value)

    let interp2d (surfaceId: int) (t: Expr) (s: Expr) : ModelCtx -> Expr =
        fun ctx ->
            match ctx.Surfaces.[surfaceId] with
            | Grid2D(values, timeAxis, spotAxis, steps) ->
                let timeCnt = timeAxis.Length
                let spotCnt = spotAxis.Length
                let valSize = values.Length
                let timeAxisOff = valSize
                let spotAxisOff = valSize + timeCnt

                let tNorm = t / (float32 (max 1 (steps - 1))).C

                // Find time bin
                let tLo = findBin surfaceId timeAxisOff timeCnt tNorm

                let tLoVal = SurfaceAt(surfaceId, (float32 timeAxisOff).C + tLo)
                let tHiVal = SurfaceAt(surfaceId, (float32 timeAxisOff).C + tLo + 1.0f.C)
                let tSpan = tHiVal - tLoVal
                let tFrac = Expr.select (tSpan .> 0.0f) (Expr.clip 0.0f.C 1.0f.C ((tNorm - tLoVal) / tSpan)) 0.0f.C

                // Find spot bin
                let sLo = findBin surfaceId spotAxisOff spotCnt s

                let sLoVal = SurfaceAt(surfaceId, (float32 spotAxisOff).C + sLo)
                let sHiVal = SurfaceAt(surfaceId, (float32 spotAxisOff).C + sLo + 1.0f.C)
                let sSpan = sHiVal - sLoVal
                let sFrac = Expr.select (sSpan .> 0.0f) (Expr.clip 0.0f.C 1.0f.C ((s - sLoVal) / sSpan)) 0.0f.C

                // Bilinear interpolation
                let spotCntF = (float32 spotCnt).C
                let v00 = SurfaceAt(surfaceId, tLo * spotCntF + sLo)
                let v01 = SurfaceAt(surfaceId, tLo * spotCntF + sLo + 1.0f.C)
                let v10 = SurfaceAt(surfaceId, (tLo + 1.0f.C) * spotCntF + sLo)
                let v11 = SurfaceAt(surfaceId, (tLo + 1.0f.C) * spotCntF + sLo + 1.0f.C)

                let v0 = v00 + (v01 - v00) * sFrac
                let v1 = v10 + (v11 - v10) * sFrac
                v0 + (v1 - v0) * tFrac
            | Curve1D _ -> failwith "interp2d requires Grid2D surface"

    let dual (name: string) (idx: int) (value: float32) : ModelCtx -> Expr = fun _ -> Dual(idx, value, name)

    let hyperDual (name: string) (idx: int) (value: float32) : ModelCtx -> Expr = fun _ -> HyperDual(idx, value, name)

    /// Cholesky decomposition of a symmetric positive-definite matrix.
    /// Returns lower triangular matrix L where A = L * L^T.
    let cholesky (a: float32[,]) : float32[,] =
        let n = Array2D.length1 a
        let l = Array2D.zeroCreate n n

        for i in 0 .. n - 1 do
            for j in 0..i do
                let mutable sum = 0.0f

                for k in 0 .. j - 1 do
                    sum <- sum + l.[i, k] * l.[j, k]

                if i = j then
                    l.[i, j] <- sqrt (a.[i, i] - sum)
                else
                    l.[i, j] <- (a.[i, j] - sum) / l.[j, j]

        l

    /// Generate N correlated standard normals from a correlation matrix.
    /// Uses Cholesky decomposition at model build time.
    /// Returns an array of N Expr values representing correlated normals.
    let correlatedNormals (correlation: float32[,]) : ModelCtx -> Expr[] =
        fun ctx ->
            let n = Array2D.length1 correlation
            let choleskyL = cholesky correlation

            // Allocate N independent normals
            let independentNormals = Array.init n (fun _ -> normal ctx)

            // Compute correlated normals: Z_corr[i] = sum_j L[i,j] * Z_indep[j]
            Array.init n (fun i ->
                seq { 0..i }
                |> Seq.filter (fun j -> abs choleskyL.[i, j] > 1e-10f)
                |> Seq.map (fun j -> choleskyL.[i, j].C * independentNormals.[j])
                |> Seq.reduce (+))

module Model =
    /// Identity pipe for adjoint mode. Dual/HyperDual markers are read directly
    /// by foldAdjoint — no expansion needed. Use: m |> Model.adjoint |> Simulation.foldAdjoint sim
    let adjoint (m: Model) : Model = m

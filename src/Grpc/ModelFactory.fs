module Cavere.Grpc.ModelFactory

open Cavere.Core
open Cavere.Generators
open Google.Protobuf.Collections

// ════════════════════════════════════════════════════════════════════════════
// Proto enum → F# DU mapping
// ════════════════════════════════════════════════════════════════════════════

let mapDeviceType (d: Cavere.Grpc.DeviceType) : Cavere.Core.DeviceType =
    match d with
    | Cavere.Grpc.DeviceType.Gpu -> Cavere.Core.DeviceType.GPU
    | Cavere.Grpc.DeviceType.Emulated -> Cavere.Core.DeviceType.Emulated
    | _ -> Cavere.Core.DeviceType.CPU

let mapFrequency (f: Cavere.Grpc.Frequency) : Cavere.Core.Frequency =
    match f with
    | Cavere.Grpc.Frequency.Daily -> Cavere.Core.Frequency.Daily
    | Cavere.Grpc.Frequency.Weekly -> Cavere.Core.Frequency.Weekly
    | Cavere.Grpc.Frequency.Monthly -> Cavere.Core.Frequency.Monthly
    | Cavere.Grpc.Frequency.Quarterly -> Cavere.Core.Frequency.Quarterly
    | Cavere.Grpc.Frequency.Annually -> Cavere.Core.Frequency.Annually
    | _ -> Cavere.Core.Frequency.Terminal

let mapDiffMode (d: Cavere.Grpc.DiffMode) : Cavere.Core.DiffMode =
    match d with
    | Cavere.Grpc.DiffMode.DiffHyperdualDiag -> Cavere.Core.DiffMode.HyperDualMode true
    | Cavere.Grpc.DiffMode.DiffHyperdualFull -> Cavere.Core.DiffMode.HyperDualMode false
    | Cavere.Grpc.DiffMode.DiffAdjoint -> Cavere.Core.DiffMode.AdjointMode
    | _ -> Cavere.Core.DiffMode.DualMode

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

let private applyPayoff (payoff: PayoffSpec) (rate: Expr) (dt: Expr) (spot: Expr) : ModelCtx -> Expr =
    fun ctx ->
        let raw =
            if isNull payoff then
                spot
            else
                match payoff.Type with
                | PayoffType.PayoffCall -> Expr.max (spot - payoff.Strike.C) 0.0f.C
                | PayoffType.PayoffPut -> Expr.max (payoff.Strike.C - spot) 0.0f.C
                | _ -> spot

        if not (isNull payoff) && payoff.Discounted then
            let df = decay rate dt ctx
            raw * df
        else
            raw

let private toFloatArray (rf: RepeatedField<float32>) : float32[] = rf |> Seq.toArray

let private unflattenMatrix (flat: RepeatedField<float32>) (n: int) : float32[,] =
    Array2D.init n n (fun i j -> flat.[i * n + j])

// ════════════════════════════════════════════════════════════════════════════
// Template model builders
// ════════════════════════════════════════════════════════════════════════════

let private buildGBM (m: GBMModel) (observers: RepeatedField<string>) (batchValues: float32[] option) : Model =
    model {
        let dt = (1.0f / float32 m.Steps).C
        let rate = m.Rate.C
        let! z = normal
        let! stock = gbm z rate m.Vol.C m.Spot.C dt

        let! result =
            match batchValues with
            | Some bv ->
                fun ctx ->
                    let strike = batchInput bv ctx

                    let raw =
                        if isNull m.Payoff then
                            stock
                        else
                            match m.Payoff.Type with
                            | PayoffType.PayoffCall -> Expr.max (stock - strike) 0.0f.C
                            | PayoffType.PayoffPut -> Expr.max (strike - stock) 0.0f.C
                            | _ -> stock

                    if not (isNull m.Payoff) && m.Payoff.Discounted then
                        let df = decay rate dt ctx
                        raw * df
                    else
                        raw
            | None -> applyPayoff m.Payoff rate dt stock

        do!
            fun ctx ->
                for name in observers do
                    observe name stock ctx

        return result
    }

let private buildGBMLocalVol
    (m: GBMLocalVolModel)
    (observers: RepeatedField<string>)
    (_batchValues: float32[] option)
    : Model =
    let vs = m.VolSurface
    let times = toFloatArray vs.TimeAxis
    let spots = toFloatArray vs.SpotAxis
    let flatVals = toFloatArray vs.Values
    let vols = Array2D.init times.Length spots.Length (fun i j -> flatVals.[i * spots.Length + j])

    model {
        let dt = (1.0f / float32 m.Steps).C
        let rate = m.Rate.C
        let! surfId = surface2d times spots vols m.Steps
        let! z = normal
        let! stock = gbmLocalVol z surfId rate m.Spot.C dt
        let! result = applyPayoff m.Payoff rate dt stock

        do!
            fun ctx ->
                for name in observers do
                    observe name stock ctx

        return result
    }

let private buildHeston (m: HestonModel) (observers: RepeatedField<string>) (batchValues: float32[] option) : Model =
    model {
        let dt = (1.0f / float32 m.Steps).C
        let rate = m.Rate.C
        let! z = normal
        let! stock = heston z rate m.V0.C m.Kappa.C m.Theta.C m.Xi.C m.Rho.C m.Spot.C dt

        let! result =
            match batchValues with
            | Some bv ->
                fun ctx ->
                    let strike = batchInput bv ctx

                    let raw =
                        if isNull m.Payoff then
                            stock
                        else
                            match m.Payoff.Type with
                            | PayoffType.PayoffCall -> Expr.max (stock - strike) 0.0f.C
                            | PayoffType.PayoffPut -> Expr.max (strike - stock) 0.0f.C
                            | _ -> stock

                    if not (isNull m.Payoff) && m.Payoff.Discounted then
                        let df = decay rate dt ctx
                        raw * df
                    else
                        raw
            | None -> applyPayoff m.Payoff rate dt stock

        do!
            fun ctx ->
                for name in observers do
                    observe name stock ctx

        return result
    }

let private buildVasicek (m: VasicekModel) (observers: RepeatedField<string>) (_batchValues: float32[] option) : Model =
    model {
        let dt = (1.0f / float32 m.Steps).C
        let! rate = Rates.vasicek m.Kappa.C m.Theta.C m.Sigma.C m.R0 dt

        do!
            fun ctx ->
                for name in observers do
                    observe name rate ctx

        return rate
    }

let private buildCIR (m: CIRModel) (observers: RepeatedField<string>) (_batchValues: float32[] option) : Model =
    model {
        let dt = (1.0f / float32 m.Steps).C
        let! rate = Rates.cir m.Kappa.C m.Theta.C m.Sigma.C m.R0 dt

        do!
            fun ctx ->
                for name in observers do
                    observe name rate ctx

        return rate
    }

let private buildCIRPP (m: CIRPPModel) (observers: RepeatedField<string>) (_batchValues: float32[] option) : Model =
    let tenors = toFloatArray m.ForwardTenors
    let rates = toFloatArray m.ForwardRates
    let fwds = Rates.linearForwards tenors rates m.Steps

    model {
        let dt = (1.0f / float32 m.Steps).C
        let! shiftSurfId = surface1d fwds m.Steps
        let! rate = Rates.cirpp m.Kappa.C m.Theta.C m.Sigma.C m.X0 shiftSurfId dt

        do!
            fun ctx ->
                for name in observers do
                    observe name rate ctx

        return rate
    }

let private buildMultiAssetHeston
    (m: MultiAssetHestonModel)
    (observers: RepeatedField<string>)
    (_batchValues: float32[] option)
    : Model =
    let n = m.Assets.Count
    let stockCorr = unflattenMatrix m.StockCorrelation n
    let volCorr = unflattenMatrix m.VolCorrelation n

    let assets =
        m.Assets
        |> Seq.map (fun a -> {
            Equity.HestonAsset.Spot = a.Spot.C
            V0 = a.V0.C
            Kappa = a.Kappa.C
            Theta = a.Theta.C
            Xi = a.Xi.C
            Rho = a.Rho
        })
        |> Seq.toArray

    model {
        let dt = (1.0f / float32 m.Steps).C
        let rate = m.Rate.C
        let! stocks = multiAssetHeston rate stockCorr volCorr assets dt

        do!
            fun ctx ->
                for i in 0 .. n - 1 do
                    for name in observers do
                        observe (sprintf "%s_%d" name i) stocks.[i] ctx

        let! result =
            fun ctx ->
                let mutable acc = 0.0f.C

                for i in 0 .. n - 1 do
                    let payoff = if i < m.Payoffs.Count then m.Payoffs.[i] else null
                    let r = applyPayoff payoff rate dt stocks.[i] ctx
                    acc <- acc + r

                acc

        return result
    }

// ════════════════════════════════════════════════════════════════════════════
// Custom model — ExprNode → Expr deserialization
// ════════════════════════════════════════════════════════════════════════════

let rec exprFromProto (node: ExprNode) : Expr =
    match node.ExprCase with
    | ExprNode.ExprOneofCase.ConstVal -> Const node.ConstVal
    | ExprNode.ExprOneofCase.TimeIndex -> TimeIndex
    | ExprNode.ExprOneofCase.NormalId -> Normal node.NormalId
    | ExprNode.ExprOneofCase.UniformId -> Uniform node.UniformId
    | ExprNode.ExprOneofCase.AccumRefId -> AccumRef node.AccumRefId
    | ExprNode.ExprOneofCase.Lookup1DId -> Lookup1D node.Lookup1DId
    | ExprNode.ExprOneofCase.BatchRefId -> BatchRef node.BatchRefId
    | ExprNode.ExprOneofCase.Add -> Add(exprFromProto node.Add.Left, exprFromProto node.Add.Right)
    | ExprNode.ExprOneofCase.Sub -> Sub(exprFromProto node.Sub.Left, exprFromProto node.Sub.Right)
    | ExprNode.ExprOneofCase.Mul -> Mul(exprFromProto node.Mul.Left, exprFromProto node.Mul.Right)
    | ExprNode.ExprOneofCase.Div -> Div(exprFromProto node.Div.Left, exprFromProto node.Div.Right)
    | ExprNode.ExprOneofCase.Max -> Max(exprFromProto node.Max.Left, exprFromProto node.Max.Right)
    | ExprNode.ExprOneofCase.Min -> Min(exprFromProto node.Min.Left, exprFromProto node.Min.Right)
    | ExprNode.ExprOneofCase.Gt -> Gt(exprFromProto node.Gt.Left, exprFromProto node.Gt.Right)
    | ExprNode.ExprOneofCase.Gte -> Gte(exprFromProto node.Gte.Left, exprFromProto node.Gte.Right)
    | ExprNode.ExprOneofCase.Lt -> Lt(exprFromProto node.Lt.Left, exprFromProto node.Lt.Right)
    | ExprNode.ExprOneofCase.Lte -> Lte(exprFromProto node.Lte.Left, exprFromProto node.Lte.Right)
    | ExprNode.ExprOneofCase.Select ->
        Select(exprFromProto node.Select.Cond, exprFromProto node.Select.IfTrue, exprFromProto node.Select.IfFalse)
    | ExprNode.ExprOneofCase.Neg -> Neg(exprFromProto node.Neg)
    | ExprNode.ExprOneofCase.Exp -> Exp(exprFromProto node.Exp)
    | ExprNode.ExprOneofCase.Log -> Log(exprFromProto node.Log)
    | ExprNode.ExprOneofCase.Sqrt -> Sqrt(exprFromProto node.Sqrt)
    | ExprNode.ExprOneofCase.Abs -> Abs(exprFromProto node.Abs)
    | ExprNode.ExprOneofCase.Floor -> Floor(exprFromProto node.Floor)
    | ExprNode.ExprOneofCase.SurfaceAt -> SurfaceAt(node.SurfaceAt.SurfaceId, exprFromProto node.SurfaceAt.Index)
    | ExprNode.ExprOneofCase.BinSearch ->
        let bs = node.BinSearch
        BinSearch(bs.SurfaceId, bs.AxisOff, bs.AxisCnt, exprFromProto bs.Value)
    | ExprNode.ExprOneofCase.Dual -> Dual(node.Dual.Index, node.Dual.Value, node.Dual.Name)
    | ExprNode.ExprOneofCase.HyperDual -> HyperDual(node.HyperDual.Index, node.HyperDual.Value, node.HyperDual.Name)
    | _ -> failwith $"Unknown ExprNode case: {node.ExprCase}"

let private buildCustom (m: CustomModel) : Model =
    let accums =
        m.Accums
        |> Seq.map (fun a ->
            a.Id,
            {
                Init = exprFromProto a.Init
                Body = exprFromProto a.Body
            })
        |> Map.ofSeq

    let surfaces =
        m.Surfaces
        |> Seq.map (fun s ->
            let data =
                match s.DataCase with
                | SurfaceDef.DataOneofCase.Curve1D -> Curve1D(toFloatArray s.Curve1D.Values, s.Curve1D.Steps)
                | SurfaceDef.DataOneofCase.Grid2D ->
                    let g = s.Grid2D
                    Grid2D(toFloatArray g.Values, toFloatArray g.TimeAxis, toFloatArray g.SpotAxis, g.Steps)
                | _ -> failwith $"Unknown SurfaceDef case: {s.DataCase}"

            s.Id, data)
        |> Map.ofSeq

    {
        Result = exprFromProto m.Result
        Accums = accums
        Surfaces = surfaces
        Observers = []
        NormalCount = m.NormalCount
        UniformCount = m.UniformCount
        BernoulliCount = 0
        BatchSize = 0
    }

// ════════════════════════════════════════════════════════════════════════════
// Top-level dispatcher
// ════════════════════════════════════════════════════════════════════════════

let private buildModelInner (spec: ModelSpec) (batchValues: float32[] option) : Model =
    let observers = spec.Observers

    match spec.ModelCase with
    | ModelSpec.ModelOneofCase.Gbm -> buildGBM spec.Gbm observers batchValues
    | ModelSpec.ModelOneofCase.GbmLocalVol -> buildGBMLocalVol spec.GbmLocalVol observers batchValues
    | ModelSpec.ModelOneofCase.Heston -> buildHeston spec.Heston observers batchValues
    | ModelSpec.ModelOneofCase.Vasicek -> buildVasicek spec.Vasicek observers batchValues
    | ModelSpec.ModelOneofCase.Cir -> buildCIR spec.Cir observers batchValues
    | ModelSpec.ModelOneofCase.Cirpp -> buildCIRPP spec.Cirpp observers batchValues
    | ModelSpec.ModelOneofCase.MultiAssetHeston -> buildMultiAssetHeston spec.MultiAssetHeston observers batchValues
    | ModelSpec.ModelOneofCase.Custom -> buildCustom spec.Custom
    | _ -> failwith $"Unknown ModelSpec case: {spec.ModelCase}"

let buildModel (spec: ModelSpec) : Model = buildModelInner spec None

let buildBatchModel (spec: ModelSpec) (batchValues: float32[]) : Model = buildModelInner spec (Some batchValues)

let getSteps (spec: ModelSpec) : int =
    match spec.ModelCase with
    | ModelSpec.ModelOneofCase.Gbm -> spec.Gbm.Steps
    | ModelSpec.ModelOneofCase.GbmLocalVol -> spec.GbmLocalVol.Steps
    | ModelSpec.ModelOneofCase.Heston -> spec.Heston.Steps
    | ModelSpec.ModelOneofCase.Vasicek -> spec.Vasicek.Steps
    | ModelSpec.ModelOneofCase.Cir -> spec.Cir.Steps
    | ModelSpec.ModelOneofCase.Cirpp -> spec.Cirpp.Steps
    | ModelSpec.ModelOneofCase.MultiAssetHeston -> spec.MultiAssetHeston.Steps
    | ModelSpec.ModelOneofCase.Custom -> spec.Custom.Steps
    | _ -> failwith $"Unknown ModelSpec case: {spec.ModelCase}"

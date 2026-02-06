namespace Cavere.Generators

open Cavere.Core

[<AutoOpen>]
module Equity =

    /// GBM process. Pass a normal from `normal` or `correlatedNormals`.
    let gbm (z: Expr) (rate: Expr) (vol: Expr) (spot: Expr) (dt: Expr) : ModelCtx -> Expr = fun ctx ->
        evolve spot (fun price ->
            let drift = rate - 0.5f * vol * vol
            price * Expr.exp (drift * dt + vol * Expr.sqrt dt * z)) ctx

    /// Local vol GBM. Pass a normal from `normal` or `correlatedNormals`.
    let gbmLocalVol (z: Expr) (surfId: int) (rate: Expr) (spot: Expr) (dt: Expr) : ModelCtx -> Expr = fun ctx ->
        evolve spot (fun price ->
            let vol = interp2d surfId TimeIndex price ctx
            let drift = rate - 0.5f * vol * vol
            price * Expr.exp (drift * dt + vol * Expr.sqrt dt * z)) ctx

    /// Heston stochastic volatility process.
    /// Pass z (stock normal) from `normal` or `correlatedNormals`.
    /// Variance normal is generated internally and correlated via rho.
    ///
    /// For multi-asset Heston with correlated stocks:
    /// 1. let! zs = correlatedNormals stockCorrelationMatrix
    /// 2. let! stock1 = heston zs.[0] rate v0 kappa theta xi rho spot1 dt
    /// 3. let! stock2 = heston zs.[1] rate v0 kappa theta xi rho spot2 dt
    let heston
        (z: Expr) (rate: Expr) (v0: Expr) (kappa: Expr) (theta: Expr)
        (xi: Expr) (rho: Expr) (spot: Expr) (dt: Expr)
        : ModelCtx -> Expr = fun ctx ->
        let zV = normal ctx  // independent variance normal
        let sqrtDt = Expr.sqrt dt
        let dw1 = sqrtDt * z
        let dw2 = sqrtDt * (rho * z + Expr.sqrt (1.0f.C - rho * rho) * zV)
        let v = evolve v0 (fun v ->
            let vp = Expr.max v 0.0f.C
            v + kappa * (theta - vp) * dt + xi * Expr.sqrt vp * dw2) ctx
        evolve spot (fun s ->
            let vp = Expr.max v 0.0f.C
            s * Expr.exp ((rate - 0.5f * vp) * dt + Expr.sqrt vp * dw1)) ctx

    /// Per-asset Heston parameters for multi-asset model.
    type HestonAsset = {
        Spot: Expr
        V0: Expr
        Kappa: Expr
        Theta: Expr
        Xi: Expr
        Rho: float32  // Stock-variance correlation (must be float32 for Cholesky)
    }

    /// Multi-asset Heston with full correlation structure.
    ///
    /// Correlation structure:
    /// - Corr(dW1_i, dW1_j) = stockCorrelation[i,j] (inter-asset equity correlation)
    /// - Corr(dW2_i, dW2_j) = volCorrelation[i,j] (inter-asset vol correlation)
    /// - Corr(dW1_i, dW2_i) = rho_i (each asset's stock-vol correlation)
    /// - Corr(dW1_i, dW2_j) = 0 for i≠j (no cross stock-vol between different assets)
    ///
    /// Returns an array of stock price Exprs, one per asset.
    let multiAssetHeston
        (rate: Expr)
        (stockCorrelation: float32[,])
        (volCorrelation: float32[,])
        (assets: HestonAsset[])
        (dt: Expr)
        : ModelCtx -> Expr[] = fun ctx ->
        let n = assets.Length
        let sqrtDt = Expr.sqrt dt

        // Build 2N×2N correlation matrix:
        // [ stockCorr   cross    ]
        // [ cross'      volCorr  ]
        // where cross[i,i] = rho_i, cross[i,j] = 0 for i≠j
        let fullCorr = Array2D.init (2 * n) (2 * n) (fun i j ->
            if i < n && j < n then
                // Stock-stock block
                stockCorrelation.[i, j]
            elif i >= n && j >= n then
                // Vol-vol block
                volCorrelation.[i - n, j - n]
            elif i < n && j >= n then
                // Stock-vol block: only diagonal (same asset) has rho
                if i = j - n then assets.[i].Rho else 0.0f
            else
                // Vol-stock block (transpose of above)
                if j = i - n then assets.[j].Rho else 0.0f)

        // Generate 2N correlated normals using Cholesky
        let choleskyL = cholesky fullCorr
        let independentNormals = Array.init (2 * n) (fun _ -> normal ctx)

        let correlatedNormals =
            Array.init (2 * n) (fun i ->
                seq { 0 .. i }
                |> Seq.filter (fun j -> abs choleskyL.[i, j] > 1e-10f)
                |> Seq.map (fun j -> choleskyL.[i, j].C * independentNormals.[j])
                |> Seq.fold (+) 0.0f.C)

        let stockNormals = correlatedNormals.[0 .. n - 1]
        let volNormals = correlatedNormals.[n .. 2 * n - 1]

        // Build each asset's Heston process
        Array.init n (fun i ->
            let a = assets.[i]
            let dw1 = sqrtDt * stockNormals.[i]
            let dw2 = sqrtDt * volNormals.[i]

            // Variance process (CIR-like)
            let v = evolve a.V0 (fun v ->
                let vp = Expr.max v 0.0f.C
                v + a.Kappa * (a.Theta - vp) * dt + a.Xi * Expr.sqrt vp * dw2) ctx

            // Stock process
            evolve a.Spot (fun s ->
                let vp = Expr.max v 0.0f.C
                s * Expr.exp ((rate - 0.5f.C * vp) * dt + Expr.sqrt vp * dw1)) ctx)


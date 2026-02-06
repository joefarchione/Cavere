namespace Cavere.Generators.AAA

open Cavere.Core

/// AAA Economic Scenario Generator — Common Types and Parameters.
/// Based on the Academy Interest Rate Generator (AIRG) Stochastic Log Volatility model.
module Common =

    // ══════════════════════════════════════════════════════════════════
    // Interest Rate Model Parameters (SLV)
    // ══════════════════════════════════════════════════════════════════

    /// Parameters for the 20-year long rate process.
    type LongRateParams = {
        /// Mean reversion speed (β₁)
        Beta: float32
        /// Target long rate (τ₁) - the Mean Reversion Parameter (MRP)
        Tau: float32
        /// Lower bound on log rate (λ_L)
        LowerBound: float32
        /// Upper bound on log rate (λ_U)
        UpperBound: float32
        /// Steepness parameter (Ψ) - reduces likelihood of curve inversion
        Psi: float32
        /// Initial 20-year rate
        R0: float32
    }

    /// Parameters for the spread process (short rate excess over long).
    type SpreadParams = {
        /// Mean reversion speed (β₂)
        Beta: float32
        /// Target spread (τ₂)
        Tau: float32
        /// Volatility (σ₂)
        Sigma: float32
        /// Tilt parameter (Φ) - reduces inversions
        Phi: float32
        /// Power parameter (θ) for rate dependence
        Theta: float32
        /// Initial spread
        Alpha0: float32
    }

    /// Parameters for the log volatility process.
    type LogVolParams = {
        /// Mean reversion speed (β₃)
        Beta: float32
        /// Target log volatility (τ₃)
        Tau: float32
        /// Volatility of volatility (σ₃)
        Sigma: float32
        /// Initial log volatility
        Nu0: float32
    }

    /// Complete interest rate model parameters.
    type RateModelParams = {
        LongRate: LongRateParams
        Spread: SpreadParams
        LogVol: LogVolParams
        /// Correlation between long rate and spread shocks
        RhoLongSpread: float32
        /// Correlation between long rate and vol shocks
        RhoLongVol: float32
    }

    // ══════════════════════════════════════════════════════════════════
    // Equity Model Parameters (SLV)
    // ══════════════════════════════════════════════════════════════════

    /// Parameters for equity fund SLV model.
    type EquityFundParams = {
        /// Fund name
        Name: string
        /// Mean log return (μ)
        Mu: float32
        /// Mean reversion speed for log volatility
        VolBeta: float32
        /// Target log volatility
        VolTau: float32
        /// Volatility of volatility
        VolSigma: float32
        /// Correlation between return and vol shocks (leverage effect)
        Rho: float32
        /// Initial fund value
        S0: float32
        /// Initial log volatility
        Nu0: float32
    }

    /// Parameters for multiple equity funds.
    type EquityModelParams = {
        Funds: EquityFundParams list
        /// Correlation matrix between fund returns
        Correlation: float32[,]
    }

    // ══════════════════════════════════════════════════════════════════
    // Bond Fund Parameters
    // ══════════════════════════════════════════════════════════════════

    /// Bond fund type.
    type BondType =
        | MoneyMarket
        | ShortTerm
        | Intermediate
        | LongTerm
        | HighYield

    /// Parameters for a bond fund.
    type BondFundParams = {
        Name: string
        BondType: BondType
        /// Reference maturity in years (for Treasury linkage)
        ReferenceMat: float32
        /// Credit spread in basis points
        CreditSpread: float32
        /// Duration for price sensitivity
        Duration: float32
        /// Convexity for price sensitivity
        Convexity: float32
        /// Initial fund value
        B0: float32
    }

    // ══════════════════════════════════════════════════════════════════
    // Combined AAA Parameters
    // ══════════════════════════════════════════════════════════════════

    /// Full AAA scenario generator parameters.
    type AAAParams = {
        Rates: RateModelParams
        Equity: EquityModelParams
        Bonds: BondFundParams list
        /// Monthly time steps
        Steps: int
    }

    // ══════════════════════════════════════════════════════════════════
    // Default Parameters (based on AIRG calibration)
    // ══════════════════════════════════════════════════════════════════

    /// Default long rate parameters.
    let defaultLongRate = {
        Beta = 0.00509f        // Monthly mean reversion
        Tau = 0.0325f          // 3.25% long-term target (MRP)
        LowerBound = -3.0f     // ~5% floor
        UpperBound = 0.5f      // ~165% cap
        Psi = 0.25f            // Steepness parameter
        R0 = 0.03f             // 3% initial 20-year rate
    }

    /// Default spread parameters.
    let defaultSpread = {
        Beta = 0.02685f        // Monthly mean reversion
        Tau = 0.0f             // Target spread (normally negative)
        Sigma = 0.04148f       // Spread volatility
        Phi = 0.0002f          // Tilt parameter
        Theta = 0.25f          // Rate power dependence
        Alpha0 = -0.005f       // Initial spread (-50bps)
    }

    /// Default log volatility parameters.
    let defaultLogVol = {
        Beta = 0.03f           // Vol mean reversion
        Tau = -2.5f            // Target log vol (~8.2%)
        Sigma = 0.10f          // Vol of vol
        Nu0 = -2.5f            // Initial log vol
    }

    /// Default rate model parameters.
    let defaultRateParams = {
        LongRate = defaultLongRate
        Spread = defaultSpread
        LogVol = defaultLogVol
        RhoLongSpread = 0.0f
        RhoLongVol = -0.2f
    }

    /// Default S&P 500 equity fund parameters.
    let defaultSP500 = {
        Name = "S&P 500"
        Mu = 0.0055f           // ~6.6% annual return
        VolBeta = 0.08f        // Vol mean reversion
        VolTau = -2.0f         // Target log vol (~13.5%)
        VolSigma = 0.15f       // Vol of vol
        Rho = -0.3f            // Leverage effect
        S0 = 1.0f
        Nu0 = -2.0f
    }

    /// Default international equity parameters.
    let defaultIntlEquity = {
        Name = "International"
        Mu = 0.005f            // ~6% annual return
        VolBeta = 0.06f
        VolTau = -1.8f         // Higher vol than S&P
        VolSigma = 0.18f
        Rho = -0.25f
        S0 = 1.0f
        Nu0 = -1.8f
    }

    /// Default equity model with two funds.
    let defaultEquityParams = {
        Funds = [ defaultSP500; defaultIntlEquity ]
        Correlation = array2D [| [| 1.0f; 0.7f |]; [| 0.7f; 1.0f |] |]
    }

    /// Default money market fund.
    let defaultMoneyMarket = {
        Name = "Money Market"
        BondType = MoneyMarket
        ReferenceMat = 0.25f
        CreditSpread = 0.0f
        Duration = 0.0f
        Convexity = 0.0f
        B0 = 1.0f
    }

    /// Default intermediate bond fund.
    let defaultIntermediateBond = {
        Name = "Intermediate Bond"
        BondType = Intermediate
        ReferenceMat = 5.0f
        CreditSpread = 50.0f   // 50 bps
        Duration = 4.5f
        Convexity = 25.0f
        B0 = 1.0f
    }

    /// Default long bond fund.
    let defaultLongBond = {
        Name = "Long Bond"
        BondType = LongTerm
        ReferenceMat = 20.0f
        CreditSpread = 75.0f   // 75 bps
        Duration = 12.0f
        Convexity = 180.0f
        B0 = 1.0f
    }

    /// Default complete AAA parameters (30 year monthly projection).
    let defaultParams = {
        Rates = defaultRateParams
        Equity = defaultEquityParams
        Bonds = [ defaultMoneyMarket; defaultIntermediateBond; defaultLongBond ]
        Steps = 360
    }


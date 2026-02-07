namespace Cavere.Actuarial.Product

open Cavere.Core

/// Registered Index-Linked Annuity (RILA) product definitions.
/// Also known as Buffer Annuities or Structured Annuities.
/// Unlike FIAs, RILAs expose policyholder to some downside risk in exchange for higher upside.
module RILA =

    // ══════════════════════════════════════════════════════════════════
    // Protection Types
    // ══════════════════════════════════════════════════════════════════

    /// Buffer: insurer absorbs first X% of losses.
    type BufferParams = {
        BufferLevel: float32            // e.g., 10% = insurer absorbs first 10% loss
    }

    /// Floor: policyholder protected below floor.
    type FloorParams = {
        FloorLevel: float32             // e.g., -10% = losses capped at 10%
    }

    /// Dual direction: buffer on downside, participation on upside.
    type DualDirectionParams = {
        DownsideBuffer: float32
        UpsideParticipation: float32
    }

    /// Protection mechanism discriminated union.
    type ProtectionType =
        | Buffer of BufferParams
        | Floor of FloorParams
        | DualDirection of DualDirectionParams
        | NoProtection                  // Full market exposure

    module ProtectionType =
        /// Standard buffer (e.g., 10%, 15%, 20%).
        let buffer (level: float32) : ProtectionType =
            Buffer { BufferLevel = level }

        /// Floor protection.
        let floor (level: float32) : ProtectionType =
            Floor { FloorLevel = level }

        /// Dual direction with buffer and participation.
        let dualDirection (buffer: float32) (participation: float32) : ProtectionType =
            DualDirection { DownsideBuffer = buffer; UpsideParticipation = participation }

    // ══════════════════════════════════════════════════════════════════
    // Upside Strategies
    // ══════════════════════════════════════════════════════════════════

    type UpsideStrategy =
        | Cap of maxReturn: float32
        | ParticipationRate of rate: float32
        | Uncapped                      // Full upside participation

    module UpsideStrategy =
        let cap (maxReturn: float32) : UpsideStrategy = Cap maxReturn
        let participation (rate: float32) : UpsideStrategy = ParticipationRate rate
        let uncapped : UpsideStrategy = Uncapped

    // ══════════════════════════════════════════════════════════════════
    // Crediting Term
    // ══════════════════════════════════════════════════════════════════

    type CreditingTerm =
        | OneYear
        | TwoYear
        | ThreeYear
        | SixYear

    module CreditingTerm =
        let years = function
            | OneYear -> 1
            | TwoYear -> 2
            | ThreeYear -> 3
            | SixYear -> 6

    // ══════════════════════════════════════════════════════════════════
    // RILA Strategy (combines protection + upside)
    // ══════════════════════════════════════════════════════════════════

    type RILAStrategy = {
        Index: FixedIndexed.IndexType
        Protection: ProtectionType
        Upside: UpsideStrategy
        Term: CreditingTerm
        Allocation: float32
    }

    module RILAStrategy =
        let create (index: FixedIndexed.IndexType) (protection: ProtectionType)
                   (upside: UpsideStrategy) (term: CreditingTerm) (allocation: float32) : RILAStrategy =
            { Index = index; Protection = protection; Upside = upside; Term = term; Allocation = allocation }

        /// Common 10% buffer with cap.
        let buffer10WithCap (index: FixedIndexed.IndexType) (cap: float32) (term: CreditingTerm) : RILAStrategy =
            { Index = index
              Protection = ProtectionType.buffer 0.10f
              Upside = UpsideStrategy.cap cap
              Term = term
              Allocation = 1.0f }

        /// 20% buffer with uncapped upside.
        let buffer20Uncapped (index: FixedIndexed.IndexType) (term: CreditingTerm) : RILAStrategy =
            { Index = index
              Protection = ProtectionType.buffer 0.20f
              Upside = Uncapped
              Term = term
              Allocation = 1.0f }

    // ══════════════════════════════════════════════════════════════════
    // RILA Product Definition
    // ══════════════════════════════════════════════════════════════════

    type RILAProduct = {
        Name: string
        Strategies: RILAStrategy list
        CDSCSchedule: Common.CDSCSchedule
        FreeWithdrawalPercent: float32
        Fees: Common.FeeStructure
        StepUpFrequency: int option     // Years between rate resets (if any)
        Riders: Riders.RiderPackage     // Optional GMxB riders
    }

    module RILAProduct =
        /// Create RILA with single strategy.
        let create (name: string) (strategy: RILAStrategy)
                   (cdsc: Common.CDSCSchedule) : RILAProduct =
            {
                Name = name
                Strategies = [ strategy ]
                CDSCSchedule = cdsc
                FreeWithdrawalPercent = 0.10f
                Fees = Common.Fees.none
                StepUpFrequency = None
                Riders = Riders.RiderPackage.empty
            }

        /// Create RILA with multiple strategies.
        let createMultiStrategy (name: string) (strategies: RILAStrategy list)
                                (cdsc: Common.CDSCSchedule) : RILAProduct =
            {
                Name = name
                Strategies = strategies
                CDSCSchedule = cdsc
                FreeWithdrawalPercent = 0.10f
                Fees = Common.Fees.none
                StepUpFrequency = None
                Riders = Riders.RiderPackage.empty
            }

        /// Add step-up feature.
        let withStepUp (years: int) (product: RILAProduct) : RILAProduct =
            { product with StepUpFrequency = Some years }

        /// Add rider package.
        let withRiders (riders: Riders.RiderPackage) (product: RILAProduct) : RILAProduct =
            { product with Riders = riders }

        /// Add GLWB rider.
        let withGLWB (glwb: Riders.GLWB) (product: RILAProduct) : RILAProduct =
            { product with Riders = Riders.RiderPackage.withGLWB glwb product.Riders }

        /// Add GMDB rider.
        let withGMDB (gmdb: Riders.GMDB) (product: RILAProduct) : RILAProduct =
            { product with Riders = Riders.RiderPackage.withGMDB gmdb product.Riders }

        /// Total rider charges.
        let totalRiderCharge (product: RILAProduct) : float32 =
            Riders.RiderPackage.totalAnnualCharge product.Riders

    // ══════════════════════════════════════════════════════════════════
    // RILA Crediting Calculations
    // ══════════════════════════════════════════════════════════════════

    module Crediting =
        /// Apply buffer protection to index return.
        /// Loss beyond buffer is passed through to policyholder.
        let applyBuffer (indexReturn: Expr) (bufferLevel: float32) : Expr =
            // If return > 0: full upside
            // If return > -buffer: no loss
            // If return < -buffer: loss = return + buffer
            Expr.select (indexReturn .>= 0.0f) indexReturn
                (Expr.select (indexReturn .>= (-bufferLevel).C) 0.0f.C
                    (indexReturn + bufferLevel.C))

        /// Apply floor protection to index return.
        let applyFloor (indexReturn: Expr) (floorLevel: float32) : Expr =
            Expr.max floorLevel.C indexReturn

        /// Apply cap to upside.
        let applyCap (indexReturn: Expr) (cap: float32) : Expr =
            Expr.min cap.C indexReturn

        /// Apply participation rate.
        let applyParticipation (indexReturn: Expr) (rate: float32) : Expr =
            indexReturn * rate.C

        /// Full RILA crediting calculation.
        let rilaCredit (indexReturn: Expr) (protection: ProtectionType) (upside: UpsideStrategy) : Expr =
            // Apply protection first
            let protected' =
                match protection with
                | Buffer p -> applyBuffer indexReturn p.BufferLevel
                | Floor p -> applyFloor indexReturn p.FloorLevel
                | DualDirection p ->
                    let buffered = applyBuffer indexReturn p.DownsideBuffer
                    Expr.select (indexReturn .>= 0.0f) (indexReturn * p.UpsideParticipation.C) buffered
                | NoProtection -> indexReturn

            // Apply upside limit
            match upside with
            | Cap maxReturn -> Expr.select (protected' .>= 0.0f) (Expr.min maxReturn.C protected') protected'
            | ParticipationRate rate -> Expr.select (protected' .>= 0.0f) (protected' * rate.C) protected'
            | Uncapped -> protected'

        /// Account value accumulation with RILA crediting.
        let accumulateRILA (creditedReturn: Expr) (accountValue: Expr) : Expr =
            accountValue * (1.0f.C + creditedReturn)


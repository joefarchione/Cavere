namespace Cavere.Actuarial

open System
open Cavere.Actuarial.Product

/// Policy module containing policyholder information and product assignment.
module Policy =

    // ══════════════════════════════════════════════════════════════════
    // Policyholder Information
    // ══════════════════════════════════════════════════════════════════

    type Gender =
        | Male
        | Female

    type Policyholder = {
        Id: Guid
        DateOfBirth: DateTime
        Gender: Gender
        IsSmoker: bool
        State: string // State of residence (for tax/regulatory)
    }

    module Policyholder =
        let create (dob: DateTime) (gender: Gender) : Policyholder = {
            Id = Guid.NewGuid()
            DateOfBirth = dob
            Gender = gender
            IsSmoker = false
            State = ""
        }

        let createWithId (id: Guid) (dob: DateTime) (gender: Gender) : Policyholder = {
            Id = id
            DateOfBirth = dob
            Gender = gender
            IsSmoker = false
            State = ""
        }

        let withSmokerStatus (isSmoker: bool) (p: Policyholder) : Policyholder = { p with IsSmoker = isSmoker }

        let withState (state: string) (p: Policyholder) : Policyholder = { p with State = state }

        /// Calculate attained age as of a given date.
        let attainedAge (asOf: DateTime) (p: Policyholder) : int =
            let age = asOf.Year - p.DateOfBirth.Year
            if asOf < p.DateOfBirth.AddYears(age) then age - 1 else age

        /// Calculate issue age (age at policy issue).
        let issueAge (issueDate: DateTime) (p: Policyholder) : int = attainedAge issueDate p

    // ══════════════════════════════════════════════════════════════════
    // Product Assignment (Discriminated Union)
    // ══════════════════════════════════════════════════════════════════

    /// Union of all product types that can be assigned to a policy.
    type ProductType =
        | LifeProduct of Life.LifeProduct
        | FixedAnnuity of Fixed.FixedAnnuityProduct
        | FixedIndexedAnnuity of FixedIndexed.FIAProduct
        | RILA of RILA.RILAProduct
        | VariableAnnuity of Variable.VAProduct

    module ProductType =
        /// Get the product name.
        let name =
            function
            | LifeProduct p -> p.Name
            | FixedAnnuity p -> p.Name
            | FixedIndexedAnnuity p -> p.Name
            | RILA p -> p.Name
            | VariableAnnuity p -> p.Name

        /// Get the CDSC schedule if applicable.
        let cdscSchedule =
            function
            | LifeProduct p -> p.CDSCSchedule
            | FixedAnnuity p -> Some p.CDSCSchedule
            | FixedIndexedAnnuity p -> Some p.CDSCSchedule
            | RILA p -> Some p.CDSCSchedule
            | VariableAnnuity p -> p.CDSCSchedule

        /// Get the fee structure.
        let fees =
            function
            | LifeProduct p -> p.Fees
            | FixedAnnuity p -> p.Fees
            | FixedIndexedAnnuity p -> p.Fees
            | RILA p -> p.Fees
            | VariableAnnuity p -> p.Fees

    // ══════════════════════════════════════════════════════════════════
    // Account Structure
    // ══════════════════════════════════════════════════════════════════

    type SubAccountBalance = {
        SubAccount: Variable.SubAccount
        Units: float32
        Value: float32
    }

    type AccountStructure =
        | GeneralOnly of value: float32
        | GeneralAndSeparate of general: float32 * subAccounts: SubAccountBalance list

    module AccountStructure =
        let totalValue =
            function
            | GeneralOnly v -> v
            | GeneralAndSeparate(ga, subs) -> ga + (subs |> List.sumBy _.Value)

        let generalValue =
            function
            | GeneralOnly v -> v
            | GeneralAndSeparate(ga, _) -> ga

        let subAccountValues =
            function
            | GeneralOnly _ -> []
            | GeneralAndSeparate(_, subs) -> subs

        let mapGeneral (f: float32 -> float32) =
            function
            | GeneralOnly v -> GeneralOnly(f v)
            | GeneralAndSeparate(ga, subs) -> GeneralAndSeparate(f ga, subs)

    // ══════════════════════════════════════════════════════════════════
    // Policy Record
    // ══════════════════════════════════════════════════════════════════

    type Policy = {
        Id: int
        Owner: Policyholder
        Spouse: Policyholder option
        Product: ProductType
        IssueDate: DateTime
        Premium: float32
        Account: AccountStructure
        Status: PolicyStatus
    }

    and PolicyStatus =
        | Active
        | Lapsed
        | Surrendered
        | Matured
        | DeathClaim
        | Annuitized

    module Policy =
        /// Create a general-account-only policy (Fixed, FIA, RILA, Life).
        let create
            (id: int)
            (owner: Policyholder)
            (product: ProductType)
            (issueDate: DateTime)
            (premium: float32)
            : Policy =
            {
                Id = id
                Owner = owner
                Spouse = None
                Product = product
                IssueDate = issueDate
                Premium = premium
                Account = GeneralOnly premium
                Status = Active
            }

        /// Create a VA policy with general + separate account allocations.
        let createVA
            (id: int)
            (owner: Policyholder)
            (product: Variable.VAProduct)
            (issueDate: DateTime)
            (premium: float32)
            (generalPortion: float32)
            : Policy =
            let separatePortion = premium - generalPortion

            let subAccounts =
                product.SubAccountAllocations
                |> List.map (fun alloc ->
                    let value = separatePortion * alloc.Allocation

                    {
                        SubAccount = alloc.SubAccount
                        Units = value
                        Value = value
                    })

            {
                Id = id
                Owner = owner
                Spouse = None
                Product = VariableAnnuity product
                IssueDate = issueDate
                Premium = premium
                Account = GeneralAndSeparate(generalPortion, subAccounts)
                Status = Active
            }

        /// Add spouse to policy.
        let withSpouse (spouse: Policyholder) (policy: Policy) : Policy = { policy with Spouse = Some spouse }

        /// Update account structure.
        let withAccount (account: AccountStructure) (policy: Policy) : Policy = { policy with Account = account }

        /// Total account value across all accounts.
        let totalValue (policy: Policy) : float32 = AccountStructure.totalValue policy.Account

        /// Update policy status.
        let withStatus (status: PolicyStatus) (policy: Policy) : Policy = { policy with Status = status }

        /// Calculate policy duration in years as of a given date.
        let durationYears (asOf: DateTime) (policy: Policy) : float32 =
            let span = asOf - policy.IssueDate
            float32 span.TotalDays / 365.25f

        /// Calculate policy duration in whole years (for CDSC lookup).
        let durationYearsInt (asOf: DateTime) (policy: Policy) : int = int (durationYears asOf policy)

        /// Get owner's attained age as of a given date.
        let ownerAttainedAge (asOf: DateTime) (policy: Policy) : int = Policyholder.attainedAge asOf policy.Owner

        /// Get owner's issue age.
        let ownerIssueAge (policy: Policy) : int = Policyholder.issueAge policy.IssueDate policy.Owner

        /// Check if policy is in force.
        let isInForce (policy: Policy) : bool = policy.Status = Active

        /// Get product name.
        let productName (policy: Policy) : string = ProductType.name policy.Product

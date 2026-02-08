namespace Cavere.Core

/// Shape of a tensor, known at DSL build time.
type Shape =
    | Scalar
    | Vec of int
    | Mat of int * int

    member this.ElementCount =
        match this with
        | Scalar -> 1
        | Vec n -> n
        | Mat(m, n) -> m * n

/// Tensor expression tree for linear algebra operations.
/// Operates over Shape-typed tensors of float32, composable with scalar Expr.
type TensorExpr =
    // Leaf nodes
    | TConst of Shape * float32[]
    | TZeros of Shape
    | TIdentity of int
    | TInput of id: int * Shape

    // Element-wise unary
    | TMap of (Expr -> Expr) * TensorExpr
    | TNeg of TensorExpr
    | TExp of TensorExpr
    | TLog of TensorExpr
    | TSqrt of TensorExpr
    | TAbs of TensorExpr

    // Element-wise binary (shapes must match)
    | TAdd of TensorExpr * TensorExpr
    | TSub of TensorExpr * TensorExpr
    | TMul of TensorExpr * TensorExpr
    | TDiv of TensorExpr * TensorExpr
    | TScale of Expr * TensorExpr

    // Linear algebra
    | TMatMul of TensorExpr * TensorExpr
    | TTranspose of TensorExpr
    | TDot of TensorExpr * TensorExpr

    // Reductions
    | TSum of TensorExpr
    | TRowSum of TensorExpr
    | TColSum of TensorExpr

    // Access
    | TElement of TensorExpr * int * int
    | TRow of TensorExpr * int
    | TCol of TensorExpr * int
    | TDiag of TensorExpr
    | TFromDiag of TensorExpr

    // Construction
    | TStack of TensorExpr list
    | TConcat of TensorExpr list
    | TReshape of Shape * TensorExpr

    static member (+)(a: TensorExpr, b: TensorExpr) = TAdd(a, b)
    static member (-)(a: TensorExpr, b: TensorExpr) = TSub(a, b)
    static member (*)(a: TensorExpr, b: TensorExpr) = TMul(a, b)
    static member (/)(a: TensorExpr, b: TensorExpr) = TDiv(a, b)
    static member (~-)(a: TensorExpr) = TNeg a

module TensorExpr =
    /// Infer the shape of a TensorExpr. Shapes are known at build time.
    let rec shape (t: TensorExpr) : Shape =
        match t with
        | TConst(s, _) -> s
        | TZeros s -> s
        | TIdentity n -> Mat(n, n)
        | TInput(_, s) -> s

        // Element-wise unary: preserve shape
        | TMap(_, a) -> shape a
        | TNeg a -> shape a
        | TExp a -> shape a
        | TLog a -> shape a
        | TSqrt a -> shape a
        | TAbs a -> shape a

        // Element-wise binary: shapes must match (take left)
        | TAdd(a, _) -> shape a
        | TSub(a, _) -> shape a
        | TMul(a, _) -> shape a
        | TDiv(a, _) -> shape a
        | TScale(_, a) -> shape a

        // Linear algebra
        | TMatMul(a, b) ->
            match shape a, shape b with
            | Mat(m, _), Mat(_, p) -> Mat(m, p)
            | Mat(m, _), Vec _ -> Vec m
            | Vec _, Vec _ -> Scalar // row-vector × col-vector
            | _ -> failwith "TMatMul: incompatible shapes"
        | TTranspose a ->
            match shape a with
            | Mat(m, n) -> Mat(n, m)
            | Vec n -> Mat(1, n) // column vec → row vec
            | Scalar -> Scalar
        | TDot _ -> Scalar

        // Reductions
        | TSum _ -> Scalar
        | TRowSum a ->
            match shape a with
            | Mat(m, _) -> Vec m
            | _ -> failwith "TRowSum: requires matrix"
        | TColSum a ->
            match shape a with
            | Mat(_, n) -> Vec n
            | _ -> failwith "TColSum: requires matrix"

        // Access
        | TElement _ -> Scalar
        | TRow(a, _) ->
            match shape a with
            | Mat(_, n) -> Vec n
            | _ -> failwith "TRow: requires matrix"
        | TCol(a, _) ->
            match shape a with
            | Mat(m, _) -> Vec m
            | _ -> failwith "TCol: requires matrix"
        | TDiag a ->
            match shape a with
            | Mat(m, n) -> Vec(min m n)
            | _ -> failwith "TDiag: requires matrix"
        | TFromDiag a ->
            match shape a with
            | Vec n -> Mat(n, n)
            | _ -> failwith "TFromDiag: requires vector"

        // Construction
        | TStack exprs ->
            match exprs with
            | [] -> failwith "TStack: empty list"
            | first :: _ ->
                match shape first with
                | Vec n -> Mat(List.length exprs, n)
                | Scalar -> Vec(List.length exprs)
                | _ -> failwith "TStack: requires vectors or scalars"
        | TConcat exprs ->
            let total = exprs |> List.sumBy (fun e -> (shape e).ElementCount)
            Vec total
        | TReshape(s, _) -> s

    /// Validate that shapes are compatible for binary element-wise operations.
    let validateBinaryShapes (a: TensorExpr) (b: TensorExpr) : unit =
        let sa = shape a
        let sb = shape b

        if sa <> sb then
            failwith $"Shape mismatch: {sa} vs {sb}"

    /// Validate matmul inner dimension compatibility.
    let validateMatMul (a: TensorExpr) (b: TensorExpr) : unit =
        match shape a, shape b with
        | Mat(_, k1), Mat(k2, _) when k1 = k2 -> ()
        | Mat(_, k1), Vec k2 when k1 = k2 -> ()
        | Vec k1, Vec k2 when k1 = k2 -> () // dot product as matmul
        | sa, sb -> failwith $"TMatMul: inner dimensions mismatch: {sa} vs {sb}"

    /// Validate reshape preserves total element count.
    let validateReshape (target: Shape) (source: TensorExpr) : unit =
        let srcCount = (shape source).ElementCount
        let tgtCount = target.ElementCount

        if srcCount <> tgtCount then
            failwith $"TReshape: element count mismatch: {srcCount} vs {tgtCount}"

/// Convenience functions for building TensorExpr trees.
module Tensor =
    let zeros s = TZeros s
    let identity n = TIdentity n

    let ofScalar (v: float32) = TConst(Scalar, [| v |])

    let ofArray (data: float32[]) = TConst(Vec data.Length, data)

    let ofArray2D (data: float32[,]) =
        let m = Array2D.length1 data
        let n = Array2D.length2 data
        let flat = Array.init (m * n) (fun i -> data.[i / n, i % n])
        TConst(Mat(m, n), flat)

    let ofList (data: float32 list) = TConst(Vec data.Length, Array.ofList data)

    let matmul a b = TMatMul(a, b)
    let dot a b = TDot(a, b)
    let transpose a = TTranspose a
    let scale (s: Expr) t = TScale(s, t)
    let map f t = TMap(f, t)
    let neg t = TNeg t
    let exp t = TExp t
    let log t = TLog t
    let sqrt t = TSqrt t
    let abs t = TAbs t
    let sum t = TSum t
    let rowSum t = TRowSum t
    let colSum t = TColSum t
    let element t i j = TElement(t, i, j)
    let row t i = TRow(t, i)
    let col t i = TCol(t, i)
    let diag t = TDiag t
    let fromDiag t = TFromDiag t
    let stack exprs = TStack exprs
    let concat exprs = TConcat exprs
    let reshape s t = TReshape(s, t)

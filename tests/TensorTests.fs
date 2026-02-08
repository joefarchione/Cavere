module Cavere.Tests.TensorTests

open Xunit
open Cavere.Core

// ── Shape ──────────────────────────────────────────────────────────

[<Fact>]
let ``Shape element count for Scalar is 1`` () = Assert.Equal(1, Scalar.ElementCount)

[<Fact>]
let ``Shape element count for Vec`` () = Assert.Equal(5, (Vec 5).ElementCount)

[<Fact>]
let ``Shape element count for Mat`` () = Assert.Equal(12, (Mat(3, 4)).ElementCount)

// ── TensorExpr shape inference: leaf nodes ─────────────────────────

[<Fact>]
let ``TConst shape matches declared shape`` () =
    let t = TConst(Vec 3, [| 1.0f; 2.0f; 3.0f |])
    Assert.Equal(Vec 3, TensorExpr.shape t)

[<Fact>]
let ``TZeros shape matches declared shape`` () =
    let t = TZeros(Mat(2, 3))
    Assert.Equal(Mat(2, 3), TensorExpr.shape t)

[<Fact>]
let ``TIdentity shape is n by n`` () =
    let t = TIdentity 4
    Assert.Equal(Mat(4, 4), TensorExpr.shape t)

[<Fact>]
let ``TInput shape matches declared shape`` () =
    let t = TInput(0, Vec 5)
    Assert.Equal(Vec 5, TensorExpr.shape t)

// ── TensorExpr shape inference: element-wise ───────────────────────

[<Fact>]
let ``Element-wise add preserves shape`` () =
    let a = TConst(Mat(2, 3), Array.zeroCreate 6)
    let b = TConst(Mat(2, 3), Array.zeroCreate 6)
    Assert.Equal(Mat(2, 3), TensorExpr.shape (TAdd(a, b)))

[<Fact>]
let ``TNeg preserves shape`` () =
    let a = TConst(Vec 4, Array.zeroCreate 4)
    Assert.Equal(Vec 4, TensorExpr.shape (TNeg a))

[<Fact>]
let ``TScale preserves tensor shape`` () =
    let s = Const 2.0f
    let t = TConst(Mat(3, 3), Array.zeroCreate 9)
    Assert.Equal(Mat(3, 3), TensorExpr.shape (TScale(s, t)))

[<Fact>]
let ``TMap preserves shape`` () =
    let t = TConst(Vec 3, [| 1.0f; 2.0f; 3.0f |])
    Assert.Equal(Vec 3, TensorExpr.shape (TMap(Expr.exp, t)))

[<Fact>]
let ``TExp preserves shape`` () =
    let t = TZeros(Mat(2, 2))
    Assert.Equal(Mat(2, 2), TensorExpr.shape (TExp t))

// ── TensorExpr shape inference: linear algebra ─────────────────────

[<Fact>]
let ``MatMul of Mat(2,3) and Mat(3,4) gives Mat(2,4)`` () =
    let a = TZeros(Mat(2, 3))
    let b = TZeros(Mat(3, 4))
    Assert.Equal(Mat(2, 4), TensorExpr.shape (TMatMul(a, b)))

[<Fact>]
let ``MatMul of Mat(3,3) and Vec(3) gives Vec(3)`` () =
    let a = TZeros(Mat(3, 3))
    let b = TZeros(Vec 3)
    Assert.Equal(Vec 3, TensorExpr.shape (TMatMul(a, b)))

[<Fact>]
let ``MatMul of Vec and Vec gives Scalar (dot product)`` () =
    let a = TZeros(Vec 3)
    let b = TZeros(Vec 3)
    Assert.Equal(Scalar, TensorExpr.shape (TMatMul(a, b)))

[<Fact>]
let ``Transpose of Mat(2,3) gives Mat(3,2)`` () =
    let a = TZeros(Mat(2, 3))
    Assert.Equal(Mat(3, 2), TensorExpr.shape (TTranspose a))

[<Fact>]
let ``Transpose of Vec(3) gives Mat(1,3)`` () =
    let a = TZeros(Vec 3)
    Assert.Equal(Mat(1, 3), TensorExpr.shape (TTranspose a))

[<Fact>]
let ``Dot product gives Scalar`` () =
    let a = TZeros(Vec 5)
    let b = TZeros(Vec 5)
    Assert.Equal(Scalar, TensorExpr.shape (TDot(a, b)))

// ── TensorExpr shape inference: reductions ─────────────────────────

[<Fact>]
let ``TSum gives Scalar`` () =
    let t = TZeros(Mat(3, 4))
    Assert.Equal(Scalar, TensorExpr.shape (TSum t))

[<Fact>]
let ``TRowSum of Mat(3,4) gives Vec(3)`` () =
    let t = TZeros(Mat(3, 4))
    Assert.Equal(Vec 3, TensorExpr.shape (TRowSum t))

[<Fact>]
let ``TColSum of Mat(3,4) gives Vec(4)`` () =
    let t = TZeros(Mat(3, 4))
    Assert.Equal(Vec 4, TensorExpr.shape (TColSum t))

// ── TensorExpr shape inference: access ─────────────────────────────

[<Fact>]
let ``TElement gives Scalar`` () =
    let t = TZeros(Mat(3, 3))
    Assert.Equal(Scalar, TensorExpr.shape (TElement(t, 0, 0)))

[<Fact>]
let ``TRow of Mat(3,4) gives Vec(4)`` () =
    let t = TZeros(Mat(3, 4))
    Assert.Equal(Vec 4, TensorExpr.shape (TRow(t, 1)))

[<Fact>]
let ``TCol of Mat(3,4) gives Vec(3)`` () =
    let t = TZeros(Mat(3, 4))
    Assert.Equal(Vec 3, TensorExpr.shape (TCol(t, 2)))

[<Fact>]
let ``TDiag of Mat(3,4) gives Vec(3)`` () =
    let t = TZeros(Mat(3, 4))
    Assert.Equal(Vec 3, TensorExpr.shape (TDiag t))

[<Fact>]
let ``TFromDiag of Vec(3) gives Mat(3,3)`` () =
    let t = TZeros(Vec 3)
    Assert.Equal(Mat(3, 3), TensorExpr.shape (TFromDiag t))

// ── TensorExpr shape inference: construction ───────────────────────

[<Fact>]
let ``TStack of 3 Vec(4) gives Mat(3,4)`` () =
    let rows = [ TZeros(Vec 4); TZeros(Vec 4); TZeros(Vec 4) ]
    Assert.Equal(Mat(3, 4), TensorExpr.shape (TStack rows))

[<Fact>]
let ``TStack of scalars gives Vec`` () =
    let scalars = [ Tensor.ofScalar 1.0f; Tensor.ofScalar 2.0f ]
    Assert.Equal(Vec 2, TensorExpr.shape (TStack scalars))

[<Fact>]
let ``TConcat of Vec(2) and Vec(3) gives Vec(5)`` () =
    let parts = [ TZeros(Vec 2); TZeros(Vec 3) ]
    Assert.Equal(Vec 5, TensorExpr.shape (TConcat parts))

[<Fact>]
let ``TReshape from Mat(2,3) to Vec(6)`` () =
    let t = TZeros(Mat(2, 3))
    Assert.Equal(Vec 6, TensorExpr.shape (TReshape(Vec 6, t)))

// ── Operator overloads ─────────────────────────────────────────────

[<Fact>]
let ``Operator + builds TAdd`` () =
    let a = TZeros(Vec 3)
    let b = TZeros(Vec 3)
    let result = a + b

    match result with
    | TAdd(TZeros(Vec 3), TZeros(Vec 3)) -> ()
    | _ -> Assert.Fail "Expected TAdd"

[<Fact>]
let ``Operator * builds TMul`` () =
    let a = TZeros(Vec 3)
    let b = TZeros(Vec 3)
    let result = a * b

    match result with
    | TMul(TZeros(Vec 3), TZeros(Vec 3)) -> ()
    | _ -> Assert.Fail "Expected TMul"

[<Fact>]
let ``Unary negation builds TNeg`` () =
    let a = TZeros(Vec 3)
    let result = -a

    match result with
    | TNeg(TZeros(Vec 3)) -> ()
    | _ -> Assert.Fail "Expected TNeg"

// ── Tensor module convenience functions ────────────────────────────

[<Fact>]
let ``Tensor.ofArray creates Vec with correct data`` () =
    let t = Tensor.ofArray [| 1.0f; 2.0f; 3.0f |]

    match t with
    | TConst(Vec 3, data) ->
        Assert.Equal(1.0f, data.[0])
        Assert.Equal(3.0f, data.[2])
    | _ -> Assert.Fail "Expected TConst Vec"

[<Fact>]
let ``Tensor.ofArray2D creates Mat with row-major data`` () =
    let data = array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
    let t = Tensor.ofArray2D data

    match t with
    | TConst(Mat(2, 2), flat) ->
        Assert.Equal(1.0f, flat.[0])
        Assert.Equal(2.0f, flat.[1])
        Assert.Equal(3.0f, flat.[2])
        Assert.Equal(4.0f, flat.[3])
    | _ -> Assert.Fail "Expected TConst Mat(2,2)"

[<Fact>]
let ``Tensor.ofScalar creates Scalar`` () =
    let t = Tensor.ofScalar 42.0f

    match t with
    | TConst(Scalar, data) ->
        Assert.Equal(1, data.Length)
        Assert.Equal(42.0f, data.[0])
    | _ -> Assert.Fail "Expected TConst Scalar"

// ── Validation helpers ─────────────────────────────────────────────

[<Fact>]
let ``validateBinaryShapes passes for matching shapes`` () =
    let a = TZeros(Mat(2, 3))
    let b = TZeros(Mat(2, 3))
    TensorExpr.validateBinaryShapes a b // should not throw

[<Fact>]
let ``validateBinaryShapes throws for mismatched shapes`` () =
    let a = TZeros(Mat(2, 3))
    let b = TZeros(Mat(3, 2))

    Assert.Throws<System.Exception>(fun () -> TensorExpr.validateBinaryShapes a b |> ignore)
    |> ignore

[<Fact>]
let ``validateMatMul passes for compatible dims`` () =
    let a = TZeros(Mat(2, 3))
    let b = TZeros(Mat(3, 4))
    TensorExpr.validateMatMul a b // should not throw

[<Fact>]
let ``validateMatMul throws for incompatible dims`` () =
    let a = TZeros(Mat(2, 3))
    let b = TZeros(Mat(2, 4))

    Assert.Throws<System.Exception>(fun () -> TensorExpr.validateMatMul a b |> ignore)
    |> ignore

[<Fact>]
let ``validateReshape passes for matching element count`` () =
    let t = TZeros(Mat(2, 3))
    TensorExpr.validateReshape (Vec 6) t // should not throw

[<Fact>]
let ``validateReshape throws for mismatched element count`` () =
    let t = TZeros(Mat(2, 3))

    Assert.Throws<System.Exception>(fun () -> TensorExpr.validateReshape (Vec 5) t |> ignore)
    |> ignore

// ── Composition: chained operations ────────────────────────────────

[<Fact>]
let ``Chained matmul shape propagation`` () =
    let a = TZeros(Mat(2, 3))
    let b = TZeros(Mat(3, 4))
    let c = TZeros(Mat(4, 5))
    let result = Tensor.matmul (Tensor.matmul a b) c
    Assert.Equal(Mat(2, 5), TensorExpr.shape result)

[<Fact>]
let ``Complex expression: matmul + scale + add`` () =
    let a = TZeros(Mat(2, 2))
    let b = TZeros(Mat(2, 2))
    let product = Tensor.matmul a b
    let scaled = Tensor.scale 0.5f.C product
    let result = scaled + TIdentity 2
    Assert.Equal(Mat(2, 2), TensorExpr.shape result)

[<Fact>]
let ``Extract row from matmul result`` () =
    let a = TZeros(Mat(3, 4))
    let b = TZeros(Mat(4, 2))
    let product = Tensor.matmul a b // Mat(3,2)
    let row0 = Tensor.row product 0 // Vec 2
    Assert.Equal(Vec 2, TensorExpr.shape row0)

// ── ComputeGraph: compute { } CE ──────────────────────────────────

[<Fact>]
let ``compute CE builds graph with single output`` () =
    let g =
        compute {
            let! w = tensorInput "weights" (Vec 3)
            let! r = tensorInput "returns" (Vec 3)
            return TDot(w, r)
        }

    Assert.Equal(1, g.Outputs.Length)
    Assert.Equal(2, g.Inputs.Count)

[<Fact>]
let ``compute CE tracks inputs by name and shape`` () =
    let g =
        compute {
            let! a = tensorInput "A" (Mat(2, 3))
            let! b = tensorInput "B" (Mat(3, 4))
            return Tensor.matmul a b
        }

    let inputShapes = g.Inputs |> Map.values |> Seq.map snd |> Seq.toList
    Assert.Contains(Mat(2, 3), inputShapes)
    Assert.Contains(Mat(3, 4), inputShapes)

[<Fact>]
let ``compute CE output node has correct shape`` () =
    let g =
        compute {
            let! a = tensorInput "A" (Mat(2, 3))
            let! b = tensorInput "B" (Mat(3, 4))
            return Tensor.matmul a b
        }

    let outId = g.Outputs.[0]
    let outNode = g.Nodes.[outId]
    Assert.Equal(Mat(2, 4), outNode.Shape)

[<Fact>]
let ``compute CE topological order respects dependencies`` () =
    let g =
        compute {
            let! w = tensorInput "weights" (Vec 4)
            let! r = tensorInput "returns" (Vec 4)
            return TDot(w, r)
        }

    // Topo order should list inputs before outputs
    let inputIds = g.Inputs |> Map.keys |> Set.ofSeq
    let outId = g.Outputs.[0]
    let topoIdx id = g.TopoOrder |> List.findIndex (fun x -> x = id)

    for inputId in inputIds do
        Assert.True(topoIdx inputId < topoIdx outId, $"Input {inputId} should come before output {outId} in topo order")

[<Fact>]
let ``compute CE with intermediate tensorNode`` () =
    let g =
        compute {
            let! a = tensorInput "A" (Mat(3, 3))
            let! b = tensorInput "B" (Mat(3, 3))
            let! sum = tensorNode (a + b)
            let! scaled = tensorNode (Tensor.scale 2.0f.C sum)
            return scaled + TIdentity 3
        }

    // Should have: 2 inputs + 2 intermediate + 1 output = 5 nodes
    Assert.Equal(5, g.Nodes.Count)
    Assert.Equal(2, g.Inputs.Count)

[<Fact>]
let ``compute CE with element-wise chain`` () =
    let g =
        compute {
            let! v = tensorInput "v" (Vec 5)
            return Tensor.exp (Tensor.neg v)
        }

    let outId = g.Outputs.[0]
    let outNode = g.Nodes.[outId]
    Assert.Equal(Vec 5, outNode.Shape)

[<Fact>]
let ``compute CE portfolio example`` () =
    let g =
        compute {
            let! weights = tensorInput "weights" (Vec 10)
            let! returns = tensorInput "returns" (Vec 10)
            return TDot(weights, returns)
        }

    let outId = g.Outputs.[0]
    let outNode = g.Nodes.[outId]
    Assert.Equal(Scalar, outNode.Shape)
    Assert.Equal(2, g.Inputs.Count)

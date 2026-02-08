namespace Cavere.Core

type ComputeNode = {
    Id: int
    Expr: TensorExpr
    Shape: Shape
}

type ComputeGraph = {
    Nodes: Map<int, ComputeNode>
    Inputs: Map<int, string * Shape>
    Outputs: int list
    TopoOrder: int list
}

type ComputeCtx = {
    mutable NextId: int
    mutable Nodes: Map<int, ComputeNode>
    mutable Inputs: Map<int, string * Shape>
    mutable OutputIds: int list
}

module ComputeGraph =
    /// Collect direct TensorExpr dependencies (node IDs referenced via TInput).
    let rec private collectInputIds (t: TensorExpr) : Set<int> =
        match t with
        | TConst _
        | TZeros _
        | TIdentity _ -> Set.empty
        | TInput(id, _) -> Set.singleton id
        | TMap(_, a)
        | TNeg a
        | TExp a
        | TLog a
        | TSqrt a
        | TAbs a -> collectInputIds a
        | TAdd(a, b)
        | TSub(a, b)
        | TMul(a, b)
        | TDiv(a, b)
        | TMatMul(a, b)
        | TDot(a, b) -> Set.union (collectInputIds a) (collectInputIds b)
        | TScale(_, a) -> collectInputIds a
        | TTranspose a
        | TSum a
        | TRowSum a
        | TColSum a -> collectInputIds a
        | TElement(a, _, _)
        | TRow(a, _)
        | TCol(a, _)
        | TDiag a
        | TFromDiag a -> collectInputIds a
        | TStack exprs
        | TConcat exprs -> exprs |> List.map collectInputIds |> Set.unionMany
        | TReshape(_, a) -> collectInputIds a

    /// Build a mapping from node ID to the set of node IDs it depends on.
    let private buildDeps (nodes: Map<int, ComputeNode>) : Map<int, Set<int>> =
        let allIds = nodes |> Map.keys |> Set.ofSeq

        nodes
        |> Map.map (fun id node ->
            let refs = collectInputIds node.Expr
            Set.intersect refs allIds |> Set.remove id)

    /// Kahn's algorithm topological sort.
    let topoSort (nodes: Map<int, ComputeNode>) : int list =
        let deps = buildDeps nodes

        let mutable inDeg = nodes |> Map.map (fun id _ -> deps.[id].Count)

        let mutable queue = inDeg |> Map.toList |> List.filter (fun (_, d) -> d = 0) |> List.map fst

        let mutable result = []

        while not (List.isEmpty queue) do
            let current = List.head queue
            queue <- List.tail queue
            result <- current :: result

            // Reduce in-degree of dependents
            for kvp in nodes do
                let nid = kvp.Key

                if deps.[nid].Contains current then
                    let newDeg = inDeg.[nid] - 1
                    inDeg <- inDeg |> Map.add nid newDeg

                    if newDeg = 0 then
                        queue <- nid :: queue

        List.rev result

type ComputeBuilder() =
    member _.Bind(f: ComputeCtx -> TensorExpr, g: TensorExpr -> ComputeCtx -> 'R) : ComputeCtx -> 'R =
        fun ctx -> g (f ctx) ctx

    member _.Return(t: TensorExpr) : ComputeCtx -> TensorExpr = fun _ -> t

    member _.ReturnFrom(f: ComputeCtx -> TensorExpr) : ComputeCtx -> TensorExpr = f

    member _.Zero() : ComputeCtx -> unit = fun _ -> ()

    member _.Combine(a: ComputeCtx -> unit, b: ComputeCtx -> 'T) : ComputeCtx -> 'T =
        fun ctx ->
            a ctx
            b ctx

    member _.Delay(f: unit -> ComputeCtx -> 'T) : ComputeCtx -> 'T = fun ctx -> f () ctx

    member _.Run(f: ComputeCtx -> TensorExpr) : ComputeGraph =
        let ctx = {
            NextId = 0
            Nodes = Map.empty
            Inputs = Map.empty
            OutputIds = []
        }

        let result = f ctx
        let resultShape = TensorExpr.shape result
        let outId = ctx.NextId
        ctx.NextId <- outId + 1

        ctx.Nodes <-
            ctx.Nodes
            |> Map.add outId {
                Id = outId
                Expr = result
                Shape = resultShape
            }

        ctx.OutputIds <- [ outId ]

        let topo = ComputeGraph.topoSort ctx.Nodes

        {
            Nodes = ctx.Nodes
            Inputs = ctx.Inputs
            Outputs = ctx.OutputIds
            TopoOrder = topo
        }

[<AutoOpen>]
module ComputeDsl =

    let compute = ComputeBuilder()

    /// Declare a named input tensor with a given shape.
    let tensorInput (name: string) (shape: Shape) : ComputeCtx -> TensorExpr =
        fun ctx ->
            let id = ctx.NextId
            ctx.NextId <- id + 1
            ctx.Inputs <- ctx.Inputs |> Map.add id (name, shape)

            ctx.Nodes <-
                ctx.Nodes
                |> Map.add id {
                    Id = id
                    Expr = TInput(id, shape)
                    Shape = shape
                }

            TInput(id, shape)

    /// Register an intermediate node in the compute graph.
    let tensorNode (t: TensorExpr) : ComputeCtx -> TensorExpr =
        fun ctx ->
            let s = TensorExpr.shape t
            let id = ctx.NextId
            ctx.NextId <- id + 1
            ctx.Nodes <- ctx.Nodes |> Map.add id { Id = id; Expr = t; Shape = s }
            t

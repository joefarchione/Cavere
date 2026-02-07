namespace Cavere.Core

open System.IO
open Parquet
open Parquet.Schema
open Parquet.Data

type OutputFormat =
    | Csv
    | Parquet

module Output =

    // ── CSV helpers ──────────────────────────────────────────────────

    module Csv =

        let writeFold (writer: TextWriter) (values: float32[]) : unit =
            writer.WriteLine("scenario,value")
            for i in 0 .. values.Length - 1 do
                writer.WriteLine($"{i},{values.[i]}")

        let writeScan (writer: TextWriter) (data: float32[,]) : unit =
            let steps = Array2D.length1 data
            let sims = Array2D.length2 data
            let header = "step," + (Array.init sims (fun s -> $"sim_{s}") |> String.concat ",")
            writer.WriteLine(header)
            for t in 0 .. steps - 1 do
                let row = Array.init sims (fun s -> $"{data.[t, s]}") |> String.concat ","
                writer.WriteLine($"{t},{row}")

        let writeWatch (writer: TextWriter) (finals: float32[]) (watch: WatchResult) : unit =
            writer.WriteLine("scenario,final_value")
            for i in 0 .. finals.Length - 1 do
                writer.WriteLine($"{i},{finals.[i]}")
            writer.WriteLine()
            for obs in watch.Observers do
                writer.WriteLine($"observer:{obs.Name}")
                let header = "obs," + (Array.init watch.NumPaths (fun p -> $"path_{p}") |> String.concat ",")
                writer.WriteLine(header)
                let vals = Watcher.values obs.Name watch
                for o in 0 .. watch.NumObs - 1 do
                    let row = Array.init watch.NumPaths (fun p -> $"{vals.[o, p]}") |> String.concat ","
                    writer.WriteLine($"{o},{row}")

    // ── Parquet helpers ──────────────────────────────────────────────

    module Parquet =

        let writeFold (path: string) (values: float32[]) : unit =
            let schema =
                ParquetSchema(
                    DataField<int>("scenario"),
                    DataField<float32>("value"))
            let scenarios = Array.init values.Length id
            use stream = File.Create(path)
            use writer = ParquetWriter.CreateAsync(schema, stream) |> Async.AwaitTask |> Async.RunSynchronously
            use rg = writer.CreateRowGroup()
            rg.WriteColumnAsync(DataColumn(schema.DataFields.[0], scenarios)) |> Async.AwaitTask |> Async.RunSynchronously
            rg.WriteColumnAsync(DataColumn(schema.DataFields.[1], values)) |> Async.AwaitTask |> Async.RunSynchronously

        let writeScan (path: string) (data: float32[,]) : unit =
            let steps = Array2D.length1 data
            let sims = Array2D.length2 data
            let fields : Field[] =
                [| yield DataField<int>("step") :> Field
                   for s in 0 .. sims - 1 do
                       yield DataField<float32>($"sim_{s}") :> Field |]
            let schema = ParquetSchema(fields)
            let stepCol = Array.init steps id
            use stream = File.Create(path)
            use writer = ParquetWriter.CreateAsync(schema, stream) |> Async.AwaitTask |> Async.RunSynchronously
            use rg = writer.CreateRowGroup()
            rg.WriteColumnAsync(DataColumn(schema.DataFields.[0], stepCol)) |> Async.AwaitTask |> Async.RunSynchronously
            for s in 0 .. sims - 1 do
                let col = Array.init steps (fun t -> data.[t, s])
                rg.WriteColumnAsync(DataColumn(schema.DataFields.[s + 1], col)) |> Async.AwaitTask |> Async.RunSynchronously

    // ── Unified write functions ──────────────────────────────────────

    let writeFold (path: string) (format: OutputFormat) (values: float32[]) : unit =
        match format with
        | Csv ->
            use writer = new StreamWriter(path)
            Csv.writeFold writer values
        | OutputFormat.Parquet ->
            Parquet.writeFold path values

    let writeScan (path: string) (format: OutputFormat) (data: float32[,]) : unit =
        match format with
        | Csv ->
            use writer = new StreamWriter(path)
            Csv.writeScan writer data
        | OutputFormat.Parquet ->
            Parquet.writeScan path data

    let writeWatch (path: string) (format: OutputFormat) (finals: float32[]) (watch: WatchResult) : unit =
        match format with
        | Csv ->
            use writer = new StreamWriter(path)
            Csv.writeWatch writer finals watch
        | OutputFormat.Parquet ->
            Parquet.writeFold path finals

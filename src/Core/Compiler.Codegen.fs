namespace Cavere.Core

open System

/// Code generation helpers: C# templates and dynamic emitters.
[<RequireQualifiedAccess>]
module CompilerCodegen =

    // ══════════════════════════════════════════════════════════════════
    // String Helpers
    // ══════════════════════════════════════════════════════════════════

    /// Remove common leading whitespace from all lines (like Python's textwrap.dedent)
    let dedent (s: string) =
        let lines = s.Split('\n')
        let nonEmptyLines = lines |> Array.filter (fun l -> l.Trim().Length > 0)

        if nonEmptyLines.Length = 0 then
            s
        else
            let minIndent =
                nonEmptyLines
                |> Array.map (fun l -> l.Length - l.TrimStart().Length)
                |> Array.min

            lines
            |> Array.map (fun l -> if l.Length >= minIndent then l.Substring(minIndent) else l)
            |> String.concat "\n"

    // ══════════════════════════════════════════════════════════════════
    // Static C# Templates
    // ══════════════════════════════════════════════════════════════════

    let csPreamble =
        dedent
            """
        using System;
        using ILGPU;
        using ILGPU.Runtime;

        public static class GeneratedKernel
        {"""

    let csHash =
        dedent
            """
            static uint Hash(int k)
            {
                uint x = (uint)k;
                x ^= x >> 16;
                x *= 0x85ebca6bu;
                x ^= x >> 13;
                x *= 0xc2b2ae35u;
                x ^= x >> 16;
                return x;
            }"""

    let csToFloat =
        dedent
            """
            static float ToFloat(uint u) => u * 2.3283064e-10f;"""

    let csBoxMuller =
        dedent
            """
            static float BoxMuller(float u1, float u2)
            {
                float r = MathF.Sqrt(-2.0f * MathF.Log(MathF.Max(u1, 1e-10f)));
                float theta = 2.0f * MathF.PI * u2;
                return r * MathF.Cos(theta);
            }"""

    let csFindBin =
        dedent
            """
            static float FindBin(ArrayView1D<float, Stride1D.Dense> surfaces, int axisOffset, int count, float value)
            {
                int lo = 0;
                int hi = count - 2;
                while (lo < hi)
                {
                    int mid = (lo + hi + 1) / 2;
                    if (value >= surfaces[axisOffset + mid])
                        lo = mid;
                    else
                        hi = mid - 1;
                }
                return (float)lo;
            }"""

    // ══════════════════════════════════════════════════════════════════
    // Dynamic Emitters
    // ══════════════════════════════════════════════════════════════════

    let emitHelpers (_layout: SurfaceLayout) = [ csHash; csToFloat; csBoxMuller; csFindBin ] |> String.concat "\n"

    let emitAccumDecls (sb: Text.StringBuilder) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        let condCounter = ref 0

        for (id, def) in sortedAccums do
            let preamble = Text.StringBuilder()
            let initExpr = CompilerCommon.emitExprPreamble layout preamble condCounter def.Init
            sb.Append(preamble) |> ignore
            sb.AppendLine(sprintf "        float accum_%d = %s;" id initExpr) |> ignore

    let emitNormals (sb: Text.StringBuilder) (model: Model) (seedVar: string) =
        for n in 0 .. model.NormalCount - 1 do
            sb.AppendLine(sprintf "            float z_%d = BoxMuller(" n) |> ignore

            sb.AppendLine(
                sprintf "                ToFloat(Hash(%s * 6364136 + t * 2 * normalCount + %d))," seedVar (n * 2)
            )
            |> ignore

            sb.AppendLine(
                sprintf "                ToFloat(Hash(%s * 6364136 + t * 2 * normalCount + %d)));" seedVar (n * 2 + 1)
            )
            |> ignore

    let emitUniforms (sb: Text.StringBuilder) (model: Model) (seedVar: string) =
        for n in 0 .. model.UniformCount - 1 do
            sb.AppendLine(
                sprintf
                    "            float u_%d = ToFloat(Hash(%s * 7919 + t * uniformCount + %d + 1000000));"
                    n
                    seedVar
                    n
            )
            |> ignore

    let emitBernoullis (sb: Text.StringBuilder) (model: Model) (seedVar: string) =
        for n in 0 .. model.BernoulliCount - 1 do
            sb.AppendLine(
                sprintf
                    "            float b_%d = (ToFloat(Hash(%s * 5399 + t * bernoulliCount + %d + 2000000)) < 0.5f) ? 1.0f : 0.0f;"
                    n
                    seedVar
                    n
            )
            |> ignore

    let emitAccumUpdates (sb: Text.StringBuilder) (layout: SurfaceLayout) (sortedAccums: (int * AccumDef) list) =
        // Simultaneous update: compute all new values from old, then assign.
        // This ensures AccumRef cross-references see pre-update values,
        // which is critical for AD derivative accumulators.
        let condCounter = ref 0

        for (id, def) in sortedAccums do
            let preamble = Text.StringBuilder()
            let bodyExpr = CompilerCommon.emitExprPreamble layout preamble condCounter def.Body
            sb.Append(preamble) |> ignore

            sb.AppendLine(sprintf "            float new_accum_%d = %s;" id bodyExpr)
            |> ignore

        for (id, _) in sortedAccums do
            sb.AppendLine(sprintf "            accum_%d = new_accum_%d;" id id) |> ignore

    let emitObserverRecording
        (sb: Text.StringBuilder)
        (model: Model)
        (layout: SurfaceLayout)
        (bufferExpr: string)
        (totalExpr: string)
        =
        if not model.Observers.IsEmpty then
            sb.AppendLine("            if ((t + 1) % interval == 0 || t == steps - 1)")
            |> ignore

            sb.AppendLine("            {") |> ignore
            sb.AppendLine("                int slot = (t + 1) / interval - 1;") |> ignore

            sb.AppendLine("                if ((t + 1) % interval != 0) slot = numObs - 1;")
            |> ignore

            sb.AppendLine("                else slot = Math.Min(slot, numObs - 1);")
            |> ignore

            let condCounter = ref 0

            for obs in model.Observers do
                let preamble = Text.StringBuilder()
                let obsExpr = CompilerCommon.emitExprPreamble layout preamble condCounter obs.Expr
                sb.Append(preamble) |> ignore

                sb.AppendLine(
                    sprintf
                        "                %s[(%d * numObs + slot) * %s + idx] = %s;"
                        bufferExpr
                        obs.SlotIndex
                        totalExpr
                        obsExpr
                )
                |> ignore

            sb.AppendLine("            }") |> ignore

    /// Emit a result or arbitrary expression with Cond preamble support.
    let emitWithPreamble (sb: Text.StringBuilder) (layout: SurfaceLayout) (expr: Expr) (template: string -> string) =
        let preamble = Text.StringBuilder()
        let condCounter = ref 0
        let exprStr = CompilerCommon.emitExprPreamble layout preamble condCounter expr
        sb.Append(preamble) |> ignore
        sb.AppendLine(template exprStr) |> ignore

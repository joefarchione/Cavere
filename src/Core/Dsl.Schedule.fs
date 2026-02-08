namespace Cavere.Core

[<Struct>]
type Schedule = {
    Dt: float32[]
    T: float32[]
    Steps: int
} with

    member s.Duration = s.T.[s.Steps]

module Schedule =

    let constant (dt: float32) (steps: int) : Schedule =
        let dtArr = Array.create steps dt
        let yfArr = Array.init (steps + 1) (fun i -> float32 i * dt)
        { Dt = dtArr; T = yfArr; Steps = steps }

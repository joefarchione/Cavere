"""Python Expr DSL for building custom Cavere models via expression trees."""

from __future__ import annotations

from cavere._generated import simulation_pb2 as pb


def _ensure_expr(x: Expr | float | int) -> Expr:
    if isinstance(x, Expr):
        return x
    return Const(float(x))


class Expr:
    """Base class wrapping an ExprNode protobuf message."""

    def to_proto(self) -> pb.ExprNode:
        raise NotImplementedError

    def __add__(self, other: Expr | float | int) -> Expr:
        return _BinaryExpr("add", self, _ensure_expr(other))

    def __radd__(self, other: float | int) -> Expr:
        return _BinaryExpr("add", _ensure_expr(other), self)

    def __sub__(self, other: Expr | float | int) -> Expr:
        return _BinaryExpr("sub", self, _ensure_expr(other))

    def __rsub__(self, other: float | int) -> Expr:
        return _BinaryExpr("sub", _ensure_expr(other), self)

    def __mul__(self, other: Expr | float | int) -> Expr:
        return _BinaryExpr("mul", self, _ensure_expr(other))

    def __rmul__(self, other: float | int) -> Expr:
        return _BinaryExpr("mul", _ensure_expr(other), self)

    def __truediv__(self, other: Expr | float | int) -> Expr:
        return _BinaryExpr("div", self, _ensure_expr(other))

    def __rtruediv__(self, other: float | int) -> Expr:
        return _BinaryExpr("div", _ensure_expr(other), self)

    def __neg__(self) -> Expr:
        return _UnaryExpr("neg", self)


class Const(Expr):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(const_val=self.value)


class Normal(Expr):
    def __init__(self, id: int) -> None:
        self.id = id

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(normal_id=self.id)


class Uniform(Expr):
    def __init__(self, id: int) -> None:
        self.id = id

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(uniform_id=self.id)


class AccumRef(Expr):
    def __init__(self, id: int) -> None:
        self.id = id

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(accum_ref_id=self.id)


class TimeIndex(Expr):
    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(time_index=True)


class Lookup1D(Expr):
    def __init__(self, surface_id: int) -> None:
        self.surface_id = surface_id

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(lookup_1d_id=self.surface_id)


class BatchRefExpr(Expr):
    def __init__(self, surface_id: int) -> None:
        self.surface_id = surface_id

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(batch_ref_id=self.surface_id)


class Dual(Expr):
    """1st-order differentiable parameter. Mark parameters for automatic differentiation."""

    def __init__(self, index: int, value: float, name: str | None = None) -> None:
        self.index = index
        self.value = float(value)
        self.name = name if name is not None else str(index)

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(dual=pb.DualOp(index=self.index, value=self.value, name=self.name))


class HyperDual(Expr):
    """2nd-order differentiable parameter. Mark parameters for automatic differentiation."""

    def __init__(self, index: int, value: float, name: str | None = None) -> None:
        self.index = index
        self.value = float(value)
        self.name = name if name is not None else str(index)

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(
            hyper_dual=pb.HyperDualOp(index=self.index, value=self.value, name=self.name)
        )


class _BinaryExpr(Expr):
    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def to_proto(self) -> pb.ExprNode:
        binop = pb.BinaryOp(left=self.left.to_proto(), right=self.right.to_proto())
        return pb.ExprNode(**{self.op: binop})


class _UnaryExpr(Expr):
    def __init__(self, op: str, operand: Expr) -> None:
        self.op = op
        self.operand = operand

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(**{self.op: self.operand.to_proto()})


class _SelectExpr(Expr):
    def __init__(self, cond: Expr, if_true: Expr, if_false: Expr) -> None:
        self.cond = cond
        self.if_true = if_true
        self.if_false = if_false

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(
            select=pb.SelectOp(
                cond=self.cond.to_proto(),
                if_true=self.if_true.to_proto(),
                if_false=self.if_false.to_proto(),
            )
        )


class _SurfaceAtExpr(Expr):
    def __init__(self, surface_id: int, index: Expr) -> None:
        self.surface_id = surface_id
        self.index = index

    def to_proto(self) -> pb.ExprNode:
        return pb.ExprNode(
            surface_at=pb.SurfaceAtOp(surface_id=self.surface_id, index=self.index.to_proto())
        )


# ── Functions ──────────────────────────────────────────────────────────


def exp(x: Expr | float | int) -> Expr:
    return _UnaryExpr("exp", _ensure_expr(x))


def log(x: Expr | float | int) -> Expr:
    return _UnaryExpr("log", _ensure_expr(x))


def sqrt(x: Expr | float | int) -> Expr:
    return _UnaryExpr("sqrt", _ensure_expr(x))


def abs_(x: Expr | float | int) -> Expr:
    return _UnaryExpr("abs", _ensure_expr(x))


def floor(x: Expr | float | int) -> Expr:
    return _UnaryExpr("floor", _ensure_expr(x))


def max_(a: Expr | float | int, b: Expr | float | int) -> Expr:
    return _BinaryExpr("max", _ensure_expr(a), _ensure_expr(b))


def min_(a: Expr | float | int, b: Expr | float | int) -> Expr:
    return _BinaryExpr("min", _ensure_expr(a), _ensure_expr(b))


def gt(a: Expr | float | int, b: Expr | float | int) -> Expr:
    return _BinaryExpr("gt", _ensure_expr(a), _ensure_expr(b))


def gte(a: Expr | float | int, b: Expr | float | int) -> Expr:
    return _BinaryExpr("gte", _ensure_expr(a), _ensure_expr(b))


def lt(a: Expr | float | int, b: Expr | float | int) -> Expr:
    return _BinaryExpr("lt", _ensure_expr(a), _ensure_expr(b))


def lte(a: Expr | float | int, b: Expr | float | int) -> Expr:
    return _BinaryExpr("lte", _ensure_expr(a), _ensure_expr(b))


def select(cond: Expr, if_true: Expr | float | int, if_false: Expr | float | int) -> Expr:
    return _SelectExpr(cond, _ensure_expr(if_true), _ensure_expr(if_false))


def surface_at(surface_id: int, index: Expr | float | int) -> Expr:
    return _SurfaceAtExpr(surface_id, _ensure_expr(index))


# ── ModelBuilder ───────────────────────────────────────────────────────


class ModelBuilder:
    """Builds a CustomModel proto from Python Expr trees."""

    def __init__(self) -> None:
        self._accums: list[pb.AccumDef] = []
        self._surfaces: list[pb.SurfaceDef] = []
        self._next_accum_id = 0
        self._next_surface_id = 0

    def add_accum(self, init: Expr | float | int, body_fn: object) -> AccumRef:
        """Add an accumulator. body_fn receives an AccumRef and returns the body Expr."""
        accum_id = self._next_accum_id
        self._next_accum_id += 1
        ref = AccumRef(accum_id)
        body_expr = body_fn(ref)
        self._accums.append(
            pb.AccumDef(id=accum_id, init=_ensure_expr(init).to_proto(), body=body_expr.to_proto())
        )
        return ref

    def add_surface_1d(self, values: list[float], steps: int) -> int:
        sid = self._next_surface_id
        self._next_surface_id += 1
        self._surfaces.append(
            pb.SurfaceDef(id=sid, curve_1d=pb.Curve1DData(values=values, steps=steps))
        )
        return sid

    def add_surface_2d(
        self, times: list[float], spots: list[float], vols: list[float], steps: int
    ) -> int:
        sid = self._next_surface_id
        self._next_surface_id += 1
        self._surfaces.append(
            pb.SurfaceDef(
                id=sid,
                grid_2d=pb.Grid2DData(values=vols, time_axis=times, spot_axis=spots, steps=steps),
            )
        )
        return sid

    def build(
        self, result_expr: Expr, normal_count: int, uniform_count: int, steps: int
    ) -> pb.ModelSpec:
        custom = pb.CustomModel(
            result=result_expr.to_proto(),
            accums=self._accums,
            surfaces=self._surfaces,
            normal_count=normal_count,
            uniform_count=uniform_count,
            steps=steps,
        )
        return pb.ModelSpec(custom=custom)

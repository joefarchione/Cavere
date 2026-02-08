"""High-level Python client for the Cavere simulation gRPC service."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import grpc
import numpy as np
from numpy.typing import NDArray

from cavere._generated import simulation_pb2 as pb
from cavere._generated import simulation_pb2_grpc as pb_grpc

# ── Model spec builders ────────────────────────────────────────────────


def _device_enum(device: str) -> int:
    d = device.upper()
    if d == "GPU":
        return pb.GPU
    elif d == "EMULATED":
        return pb.EMULATED
    else:
        return pb.CPU


def _freq_enum(frequency: str) -> int:
    mapping = {
        "TERMINAL": pb.TERMINAL,
        "DAILY": pb.DAILY,
        "WEEKLY": pb.WEEKLY,
        "MONTHLY": pb.MONTHLY,
        "QUARTERLY": pb.QUARTERLY,
        "ANNUALLY": pb.ANNUALLY,
    }
    return mapping.get(frequency.upper(), pb.TERMINAL)


def _diff_mode_enum(mode: str) -> int:
    mapping = {
        "DUAL": pb.DIFF_DUAL,
        "HYPERDUAL_DIAG": pb.DIFF_HYPERDUAL_DIAG,
        "HYPERDUAL_FULL": pb.DIFF_HYPERDUAL_FULL,
        "ADJOINT": pb.DIFF_ADJOINT,
    }
    return mapping.get(mode.upper(), pb.DIFF_DUAL)


def call_payoff(strike: float, discounted: bool = True) -> pb.PayoffSpec:
    return pb.PayoffSpec(type=pb.PAYOFF_CALL, strike=strike, discounted=discounted)


def put_payoff(strike: float, discounted: bool = True) -> pb.PayoffSpec:
    return pb.PayoffSpec(type=pb.PAYOFF_PUT, strike=strike, discounted=discounted)


def vol_surface(time_axis: list[float], spot_axis: list[float], values: list[float]) -> pb.VolSurface:
    return pb.VolSurface(time_axis=time_axis, spot_axis=spot_axis, values=values)


def gbm(
    spot: float,
    rate: float,
    vol: float,
    steps: int,
    payoff: pb.PayoffSpec | None = None,
    observers: list[str] | None = None,
) -> pb.ModelSpec:
    m = pb.GBMModel(spot=spot, rate=rate, vol=vol, steps=steps)
    if payoff is not None:
        m.payoff.CopyFrom(payoff)
    spec = pb.ModelSpec(gbm=m)
    if observers:
        spec.observers.extend(observers)
    return spec


def gbm_local_vol(
    spot: float,
    rate: float,
    vol_surf: pb.VolSurface,
    steps: int,
    payoff: pb.PayoffSpec | None = None,
    observers: list[str] | None = None,
) -> pb.ModelSpec:
    m = pb.GBMLocalVolModel(spot=spot, rate=rate, vol_surface=vol_surf, steps=steps)
    if payoff is not None:
        m.payoff.CopyFrom(payoff)
    spec = pb.ModelSpec(gbm_local_vol=m)
    if observers:
        spec.observers.extend(observers)
    return spec


def heston(
    spot: float,
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    steps: int,
    payoff: pb.PayoffSpec | None = None,
    observers: list[str] | None = None,
) -> pb.ModelSpec:
    m = pb.HestonModel(spot=spot, rate=rate, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, steps=steps)
    if payoff is not None:
        m.payoff.CopyFrom(payoff)
    spec = pb.ModelSpec(heston=m)
    if observers:
        spec.observers.extend(observers)
    return spec


def multi_asset_heston(
    rate: float,
    assets: list[dict[str, float]],
    stock_corr: list[float],
    vol_corr: list[float],
    steps: int,
    payoffs: list[pb.PayoffSpec] | None = None,
    observers: list[str] | None = None,
) -> pb.ModelSpec:
    asset_specs = [
        pb.HestonAssetSpec(spot=a["spot"], v0=a["v0"], kappa=a["kappa"], theta=a["theta"], xi=a["xi"], rho=a["rho"])
        for a in assets
    ]
    m = pb.MultiAssetHestonModel(
        rate=rate, assets=asset_specs, stock_correlation=stock_corr, vol_correlation=vol_corr, steps=steps
    )
    if payoffs:
        m.payoffs.extend(payoffs)
    spec = pb.ModelSpec(multi_asset_heston=m)
    if observers:
        spec.observers.extend(observers)
    return spec


def vasicek(
    kappa: float, theta: float, sigma: float, r0: float, steps: int, observers: list[str] | None = None
) -> pb.ModelSpec:
    m = pb.VasicekModel(kappa=kappa, theta=theta, sigma=sigma, r0=r0, steps=steps)
    spec = pb.ModelSpec(vasicek=m)
    if observers:
        spec.observers.extend(observers)
    return spec


def cir(
    kappa: float, theta: float, sigma: float, r0: float, steps: int, observers: list[str] | None = None
) -> pb.ModelSpec:
    m = pb.CIRModel(kappa=kappa, theta=theta, sigma=sigma, r0=r0, steps=steps)
    spec = pb.ModelSpec(cir=m)
    if observers:
        spec.observers.extend(observers)
    return spec


def cirpp(
    kappa: float,
    theta: float,
    sigma: float,
    x0: float,
    forward_tenors: list[float],
    forward_rates: list[float],
    steps: int,
    observers: list[str] | None = None,
) -> pb.ModelSpec:
    m = pb.CIRPPModel(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        x0=x0,
        forward_tenors=forward_tenors,
        forward_rates=forward_rates,
        steps=steps,
    )
    spec = pb.ModelSpec(cirpp=m)
    if observers:
        spec.observers.extend(observers)
    return spec


# ── Client ─────────────────────────────────────────────────────────────


class CavereClient:
    """Thin wrapper around the Cavere gRPC simulation service."""

    def __init__(self, host: str = "localhost:5000") -> None:
        self.channel = grpc.insecure_channel(host)
        self.stub = pb_grpc.SimulationServiceStub(self.channel)

    def close(self) -> None:
        self.channel.close()

    def __enter__(self) -> CavereClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Simple simulation ──────────────────────────────────────────

    def fold(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        device: str = "CPU",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> NDArray[np.float32]:
        req = pb.SimulationRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.Fold(req)
        return np.array(resp.values, dtype=np.float32)

    def fold_watch(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        device: str = "CPU",
        frequency: str = "MONTHLY",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        req = pb.WatchRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            frequency=_freq_enum(frequency),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.FoldWatch(req)
        finals = np.array(resp.finals, dtype=np.float32)
        observers: dict[str, NDArray[np.float32]] = {}
        for od in resp.observers:
            observers[od.name] = np.array(od.values, dtype=np.float32).reshape(od.num_obs, od.num_paths)
        return finals, observers

    def scan(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        device: str = "CPU",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> NDArray[np.float32]:
        req = pb.SimulationRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.Scan(req)
        return np.array(resp.values, dtype=np.float32).reshape(resp.steps, resp.num_scenarios)

    def stream_scan(
        self, model_spec: pb.ModelSpec, num_scenarios: int, device: str = "CPU"
    ) -> Iterator[tuple[int, NDArray[np.float32]]]:
        req = pb.SimulationRequest(model=model_spec, num_scenarios=num_scenarios, device=_device_enum(device))
        for chunk in self.stub.StreamScan(req):
            arr = np.array(chunk.values, dtype=np.float32).reshape(chunk.step_count, chunk.num_scenarios)
            yield chunk.start_step, arr

    # ── Batch simulation ───────────────────────────────────────────

    def batch_fold(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        batch_values: list[float],
        device: str = "CPU",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> NDArray[np.float32]:
        req = pb.BatchRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            batch_values=batch_values,
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.BatchFold(req)
        num_batch = len(batch_values)
        return np.array(resp.values, dtype=np.float32).reshape(num_batch, num_scenarios)

    def batch_fold_watch(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        batch_values: list[float],
        device: str = "CPU",
        frequency: str = "MONTHLY",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        req = pb.BatchWatchRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            batch_values=batch_values,
            frequency=_freq_enum(frequency),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.BatchFoldWatch(req)
        finals = np.array(resp.finals, dtype=np.float32)
        observers: dict[str, NDArray[np.float32]] = {}
        for od in resp.observers:
            observers[od.name] = np.array(od.values, dtype=np.float32).reshape(od.num_obs, od.num_paths)
        return finals, observers

    def batch_fold_means(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        batch_values: list[float],
        device: str = "CPU",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> NDArray[np.float32]:
        req = pb.BatchRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            batch_values=batch_values,
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.BatchFoldMeans(req)
        return np.array(resp.values, dtype=np.float32)

    def stream_batch_fold(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        batch_values: list[float],
        device: str = "CPU",
    ) -> Iterator[tuple[int, NDArray[np.float32]]]:
        req = pb.BatchRequest(
            model=model_spec, num_scenarios=num_scenarios, device=_device_enum(device), batch_values=batch_values
        )
        for chunk in self.stub.StreamBatchFold(req):
            arr = np.array(chunk.values, dtype=np.float32)
            yield chunk.start_index, arr

    # ── Automatic differentiation ─────────────────────────────────

    def fold_diff(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        device: str = "CPU",
        diff_mode: str = "DUAL",
        frequency: str = "TERMINAL",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        """Forward-mode AD (Dual or HyperDual). Returns (values, derivative_observers)."""
        req = pb.DiffRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            diff_mode=_diff_mode_enum(diff_mode),
            frequency=_freq_enum(frequency),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.FoldDiff(req)
        finals = np.array(resp.finals, dtype=np.float32)
        observers: dict[str, NDArray[np.float32]] = {}
        for od in resp.observers:
            observers[od.name] = np.array(od.values, dtype=np.float32).reshape(od.num_obs, od.num_paths)
        return finals, observers

    def fold_adjoint(
        self,
        model_spec: pb.ModelSpec,
        num_scenarios: int,
        device: str = "CPU",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], list[int]]:
        """Reverse-mode AD. Returns (values, adjoints[scenarios, diff_vars], diff_var_indices)."""
        req = pb.AdjointRequest(
            model=model_spec,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.FoldAdjoint(req)
        values = np.array(resp.values, dtype=np.float32)
        adjoints = np.array(resp.adjoints, dtype=np.float32).reshape(resp.num_scenarios, resp.num_diff_vars)
        diff_var_indices = list(resp.diff_var_indices)
        return values, adjoints, diff_var_indices

    def recommend(self, model_spec: pb.ModelSpec) -> tuple[str | None, bool, str]:
        """Get AD mode recommendation. Returns (mode_name, has_diff_vars, description)."""
        resp = self.stub.Recommend(model_spec)
        mode_names = {
            pb.DIFF_DUAL: "DUAL",
            pb.DIFF_HYPERDUAL_DIAG: "HYPERDUAL_DIAG",
            pb.DIFF_HYPERDUAL_FULL: "HYPERDUAL_FULL",
            pb.DIFF_ADJOINT: "ADJOINT",
        }
        mode = mode_names.get(resp.recommended_mode) if resp.has_diff_vars else None
        return mode, resp.has_diff_vars, resp.description

    # ── Kernel management ──────────────────────────────────────────

    def compile_kernel(self, model_spec: pb.ModelSpec, batch: bool = False) -> tuple[str, str]:
        """Compile a model to a reusable kernel. Returns (kernel_id, csharp_source)."""
        req = pb.CompileKernelRequest(model=model_spec, batch=batch)
        resp = self.stub.CompileKernel(req)
        return resp.kernel_id, resp.csharp_source

    def fold_kernel(
        self,
        kernel_id: str,
        num_scenarios: int,
        device: str = "CPU",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> NDArray[np.float32]:
        """Run a pre-compiled kernel. Returns scenario values."""
        req = pb.KernelRunRequest(
            kernel_id=kernel_id,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.FoldKernel(req)
        return np.array(resp.values, dtype=np.float32)

    def fold_watch_kernel(
        self,
        kernel_id: str,
        num_scenarios: int,
        device: str = "CPU",
        frequency: str = "MONTHLY",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        """Run a pre-compiled kernel with observer recording."""
        req = pb.KernelWatchRequest(
            kernel_id=kernel_id,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            frequency=_freq_enum(frequency),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.FoldWatchKernel(req)
        finals = np.array(resp.finals, dtype=np.float32)
        observers: dict[str, NDArray[np.float32]] = {}
        for od in resp.observers:
            observers[od.name] = np.array(od.values, dtype=np.float32).reshape(od.num_obs, od.num_paths)
        return finals, observers

    def scan_kernel(
        self,
        kernel_id: str,
        num_scenarios: int,
        device: str = "CPU",
        device_count: int = 0,
        use_pinned: bool = False,
    ) -> NDArray[np.float32]:
        """Run a pre-compiled kernel in scan mode. Returns [steps, scenarios] array."""
        req = pb.KernelRunRequest(
            kernel_id=kernel_id,
            num_scenarios=num_scenarios,
            device=_device_enum(device),
            device_count=device_count,
            use_pinned=use_pinned,
        )
        resp = self.stub.ScanKernel(req)
        return np.array(resp.values, dtype=np.float32).reshape(resp.steps, resp.num_scenarios)

    def destroy_kernel(self, kernel_id: str) -> None:
        """Destroy a pre-compiled kernel to free server resources."""
        self.stub.DestroyKernel(pb.KernelId(id=kernel_id))

    # ── Session management ─────────────────────────────────────────

    def create_session(self, device: str = "CPU", num_scenarios: int = 10000, steps: int = 252) -> str:
        req = pb.CreateSessionRequest(device=_device_enum(device), num_scenarios=num_scenarios, steps=steps)
        resp = self.stub.CreateSession(req)
        return resp.id

    def destroy_session(self, session_id: str) -> None:
        self.stub.DestroySession(pb.SessionId(id=session_id))

    # ── Utility ────────────────────────────────────────────────────

    def source(self, model_spec: pb.ModelSpec) -> str:
        resp = self.stub.GetSource(model_spec)
        return resp.csharp_source

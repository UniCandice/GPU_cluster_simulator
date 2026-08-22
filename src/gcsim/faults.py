"""Fault and workload injections.

GOVERNING RULE, enforced by the import list below and by
`tests/test_scenarios.py::test_faults_cannot_touch_telemetry`: this module may
import `topology` and `workload` and nothing else from the simulator. It has no
path to `telemetry`, `samplers` or `metrics`.

An injection perturbs *physical state* -- a port goes down, a CRAC loses
capacity, a GPU loses SM throughput -- or the *workload's own parameters*. It
never writes a telemetry value. Everything the dashboard shows is downstream of
these mutations, which is what makes the telemetry causal rather than staged.

Handlers are registered by name so adding a scenario is a config change plus one
function, with no edits to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from gcsim.config import Injection, ScenarioConfig
from gcsim.topology import Cluster
from gcsim.workload import WorkloadState

Handler = Callable[["InjectionContext"], dict[str, Any]]

_HANDLERS: dict[str, Handler] = {}


def handler(name: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        _HANDLERS[name] = fn
        return fn
    return register


@dataclass
class InjectionContext:
    cluster: Cluster
    workload: WorkloadState
    params: dict[str, Any]
    iteration: int
    #: progress through a ramped injection, 0.0 -> 1.0
    progress: float = 1.0


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _target_gpu(ctx: InjectionContext):
    t = ctx.params["target"]
    return ctx.cluster.gpus[f"r{t['rack']}n{t['node']}g{t['gpu']}"]


@handler("gpu_throughput_derate")
def gpu_throughput_derate(ctx: InjectionContext) -> dict[str, Any]:
    """A co-resident process steals SM time.

    Chosen precisely because it leaves no fingerprint on any device counter: the
    GPU still boosts to its nominal clock, still reports ~100% utilisation, and
    is simply slower. Only the *job's* timing reveals it.
    """
    gpu = _target_gpu(ctx)
    gpu.throughput_derate = float(ctx.params["factor"])
    return {"gpu_id": gpu.gpu_id, "throughput_derate": gpu.throughput_derate}


@handler("gpu_reliability_degrade")
def gpu_reliability_degrade(ctx: InjectionContext) -> dict[str, Any]:
    """Row remapping and error handling eat memory bandwidth; RAS caps the clock.

    Nearly identical to a straggler in job timing. The separators are the
    reported clock cap (throttle_reason = RELIABILITY) and the occupancy drop on
    the victim, because stalled SMs are occupied but not retiring work.
    """
    gpu = _target_gpu(ctx)
    gpu.memory_bandwidth_factor = float(ctx.params["memory_bandwidth_factor"])
    gpu.reliability_clock_cap = float(ctx.params["clock_cap"])
    return {
        "gpu_id": gpu.gpu_id,
        "memory_bandwidth_factor": gpu.memory_bandwidth_factor,
        "reliability_clock_cap": gpu.reliability_clock_cap,
    }


@handler("leaf_uplink_failure")
def leaf_uplink_failure(ctx: InjectionContext) -> dict[str, Any]:
    """An uplink bundle fails, leaving the rack on a fraction of its capacity.

    The surviving members are given an elevated error rate: the marginal optics
    that took the bundle down in the first place are still carrying traffic.
    Errors cost goodput, which raises utilisation, which fills the queue, which
    drops frames, which cost more goodput -- the degradation compounds, as it
    does on a real flapping link.
    """
    rack_id = f"r{ctx.params['target']['rack']}"
    keep = int(ctx.params["uplinks_active"])
    error_rate = float(ctx.params.get("error_rate", 5e-4))
    leaf = ctx.cluster.leaf_of(rack_id)

    downed = []
    for i, port_id in enumerate(leaf.uplink_ids):
        port = ctx.cluster.ports[port_id]
        if i >= keep:
            port.up = False
            downed.append(port_id)
        else:
            port.error_rate = error_rate
    return {"rack_id": rack_id, "uplinks_active": keep, "downed_ports": downed,
            "survivor_error_rate": error_rate}


@handler("cooling_degrade")
def cooling_degrade(ctx: InjectionContext) -> dict[str, Any]:
    """A CRAC loses capacity, over `ramp_iterations` timesteps.

    Cooling efficiency is a *rack* property, so this moves all 32 GPUs in the
    rack together. Nothing sets a temperature: inlet temperature is recomputed
    from efficiency and the rack's own dissipation, and die temperature follows
    from that through the thermal lag.
    """
    rack = ctx.cluster.racks[f"r{ctx.params['target']['rack']}"]
    target = float(ctx.params["efficiency"])
    nominal = ctx.cluster.cfg.cooling.nominal_efficiency
    rack.cooling_efficiency = nominal + (target - nominal) * ctx.progress
    return {"rack_id": rack.rack_id, "cooling_efficiency": rack.cooling_efficiency,
            "progress": ctx.progress}


@handler("workload_output_change")
def workload_output_change(ctx: InjectionContext) -> dict[str, Any]:
    """NOT A FAULT. The job starts asking for more output.

    Mutates the workload, never the hardware. Every rank is affected identically
    and no device counter moves, which is exactly what separates it from the
    faults above.
    """
    ctx.workload.output_interval = int(ctx.params["output_interval"])
    ctx.workload.output_bytes_scale = float(ctx.params["output_bytes_scale"])
    return {"output_interval": ctx.workload.output_interval,
            "output_bytes_scale": ctx.workload.output_bytes_scale}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class _Active:
    injection: Injection
    ramp: int


@dataclass
class FaultInjector:
    """Applies a scenario's injections at the right timesteps.

    `tick` is called once per timestep before phase costs are computed, so a
    perturbation applied at timestep N is felt from timestep N onward.
    """
    cluster: Cluster
    workload: WorkloadState
    scenario: ScenarioConfig
    _pending: list[Injection] = field(default_factory=list)
    _ramping: list[_Active] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._pending = sorted(self.scenario.injections, key=lambda i: i.at_iteration)

    def tick(self, iteration: int) -> list[dict[str, Any]]:
        """Fire anything due at `iteration`; advance anything still ramping."""
        fired: list[dict[str, Any]] = []

        while self._pending and self._pending[0].at_iteration <= iteration:
            inj = self._pending.pop(0)
            ramp = int(inj.params.get("ramp_iterations", 0))
            if ramp > 0:
                self._ramping.append(_Active(injection=inj, ramp=ramp))
            payload = self._apply(inj, iteration, progress=0.0 if ramp else 1.0)
            fired.append({"type": inj.type, "at_iteration": inj.at_iteration,
                          "ramping": bool(ramp), **payload})

        still: list[_Active] = []
        for act in self._ramping:
            elapsed = iteration - act.injection.at_iteration
            progress = min(1.0, elapsed / act.ramp)
            self._apply(act.injection, iteration, progress=progress)
            if progress < 1.0:
                still.append(act)
        self._ramping = still
        return fired

    def _apply(self, inj: Injection, iteration: int, progress: float) -> dict[str, Any]:
        try:
            fn = _HANDLERS[inj.type]
        except KeyError:
            raise KeyError(
                f"no handler for injection type {inj.type!r}; "
                f"registered: {sorted(_HANDLERS)}"
            ) from None
        ctx = InjectionContext(cluster=self.cluster, workload=self.workload,
                               params=inj.params, iteration=iteration, progress=progress)
        return fn(ctx)


def registered_handlers() -> list[str]:
    return sorted(_HANDLERS)

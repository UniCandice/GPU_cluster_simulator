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

from gcsim.config import Injection, ScenarioConfig, derive_rng
from gcsim.topology import Cluster
from gcsim.workload import WorkloadState

Handler = Callable[["InjectionContext"], dict[str, Any]]

_HANDLERS: dict[str, Handler] = {}

#: Handlers that must be re-entered every timestep rather than applied once.
#: A one-shot injection sets a state and leaves it set; a persistent one owns a
#: state that changes over the run -- an episode starting, and later ending --
#: so it needs the tick to keep arriving.
_PERSISTENT: set[str] = set()


def handler(name: str, persistent: bool = False) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        _HANDLERS[name] = fn
        if persistent:
            _PERSISTENT.add(name)
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
    #: run-level facts a handler may need to plan ahead. Not telemetry: the seed
    #: is what makes a stochastic injection reproducible, and `iterations` is how
    #: far ahead there is to plan.
    seed: int = 0
    iterations: int = 0
    #: scratch owned by the injector, one dict per injection, persisted across
    #: ticks. A persistent handler memoises its plan here.
    state: dict[str, Any] = field(default_factory=dict)


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


def _plan_episodes(ctx: InjectionContext) -> list[dict[str, Any]]:
    """Draw the whole episode schedule up front, from the run's own seed.

    Planned once rather than sampled tick by tick, for two reasons. The schedule
    becomes a value that can be written into the event payload as ground truth,
    which is what lets a test check the effect against the cause. And it is
    reproducible by construction: the same (seed, params) always yields the same
    episodes, with no dependence on how many times the handler happened to run.

    Episodes are stratified rather than uniform -- the span is cut into one slot
    per episode and each episode is jittered inside its own slot. Uniform draws
    clump, and a run whose episodes all landed in the first half would leave the
    comparison window clean and the fault undetectable for reasons that have
    nothing to do with the fault. Stratification also makes episodes
    non-overlapping by construction, so at most one rank is ever derated and
    "restored to baseline" is a property that can actually be asserted.

    Victims come from a small COHORT drawn once, not redrawn per episode. A few
    hosts with a stale process on them is the situation this models, and it is
    also the difference between a fault that can be localised and one that
    cannot: spread thin enough across 128 GPUs, no rank accumulates enough
    derated time to stand clear of ordinary silicon variation, and attribution
    lands on whichever rank is naturally slowest instead.
    """
    p = ctx.params
    rng = derive_rng(ctx.seed, f"straggler_episodes:{p.get('stream', 'default')}")

    start = ctx.iteration
    end = int(p.get("until_iteration") or ctx.iterations)
    n = max(1, int(p["episodes"]))
    d_min, d_max = (int(v) for v in p["duration_iterations"])
    f_min, f_max = (float(v) for v in p["factor"])
    gpu_ids = [g.gpu_id for g in ctx.cluster.gpu_list]

    #  Drawn before the episode loop so the cohort is a property of the seed
    #  alone, not of how many episodes were requested.
    n_victims = max(1, min(int(p.get("victims", 3)), len(gpu_ids)))
    cohort = [gpu_ids[int(i)] for i in
              rng.choice(len(gpu_ids), size=n_victims, replace=False)]

    span = max(end - start, 1)
    slot = span / n

    #  Non-overlap is what makes "at most one rank derated, everything else
    #  exactly baseline" assertable, and it holds only while an episode fits
    #  inside its own slot. Raise `episodes` or `duration_iterations` far enough
    #  and an episode runs past its slot boundary; the next episode's start tick
    #  then passes while the previous one is still active, the handler's
    #  `start == iteration` test never matches again, and that episode is
    #  silently dropped -- a quieter fault than the one being injected. Fail
    #  loudly at plan time instead.
    if d_max > slot:
        raise ValueError(
            f"straggler episodes cannot fit: duration up to {d_max} timesteps in "
            f"slots of {slot:.1f} ({span} timesteps / {n} episodes). Lower "
            f"`episodes` or `duration_iterations`, or widen the span.")
    plan: list[dict[str, Any]] = []
    for k in range(n):
        duration = int(rng.integers(d_min, d_max + 1))
        room = max(slot - duration, 0.0)
        begin = int(start + k * slot + rng.uniform(0.0, room))
        plan.append({
            "gpu_id": cohort[int(rng.integers(len(cohort)))],
            "start": begin,
            "end": begin + duration,
            "factor": round(float(rng.uniform(f_min, f_max)), 4),
        })
    return plan


@handler("gpu_throughput_episodes", persistent=True)
def gpu_throughput_episodes(ctx: InjectionContext) -> dict[str, Any]:
    """Intermittent co-resident interference, roaming across the fleet.

    Same physical lever as `gpu_throughput_derate` -- SM time stolen by a
    process the job does not own -- but transient and recurring rather than a
    step change, which is how this failure mode usually presents: a rank is slow
    for a few timesteps, recovers completely, and one of its neighbours is slow
    later.

    A handful of GPUs carry the stale processes, and episodes land on those --
    so the fault is genuinely localised, just never continuously. Each episode
    derates exactly one of them and then restores it to 1.0, so between episodes
    the cluster is bit-for-bit healthy. Nothing else is touched: clock, HBM
    bandwidth, temperature and the reliability governor stay at baseline, so the
    device reports itself perfectly well throughout and this stays separable
    from thermal throttling and from RAS-driven degradation.
    """
    first = "schedule" not in ctx.state
    if first:
        ctx.state["schedule"] = _plan_episodes(ctx)
        ctx.state["active"] = None

    #  End first, then start. An episode's last timestep is `end - 1`, so a slot
    #  boundary can retire one episode and open the next on the same tick
    #  without the restore clobbering the new derate.
    active = ctx.state["active"]
    if active is not None and ctx.iteration >= active["end"]:
        ctx.cluster.gpus[active["gpu_id"]].throughput_derate = 1.0
        ctx.state["active"] = active = None

    if active is None:
        for episode in ctx.state["schedule"]:
            if episode["start"] == ctx.iteration:
                ctx.cluster.gpus[episode["gpu_id"]].throughput_derate = episode["factor"]
                ctx.state["active"] = episode
                break

    if not first:
        #  Silence on ordinary ticks. Every payload becomes a labelled mark on
        #  the dashboard timeline, and one mark per episode would bury the
        #  charts under vertical lines.
        return {}
    schedule = ctx.state["schedule"]
    return {
        "n_episodes": len(schedule),
        "gpus_affected": sorted({e["gpu_id"] for e in schedule}),
        "episodes": schedule,
    }


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
class _Live:
    """A persistent injection and the scratch it keeps across ticks."""
    injection: Injection
    started: int
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultInjector:
    """Applies a scenario's injections at the right timesteps.

    `tick` is called once per timestep before phase costs are computed, so a
    perturbation applied at timestep N is felt from timestep N onward.

    `seed` and `iterations` are carried so a stochastic injection can plan a
    reproducible schedule from the run's own seed. Neither is telemetry, and the
    import restriction at the top of this module still holds.
    """
    cluster: Cluster
    workload: WorkloadState
    scenario: ScenarioConfig
    seed: int = 0
    iterations: int = 0
    _pending: list[Injection] = field(default_factory=list)
    _ramping: list[_Active] = field(default_factory=list)
    _live: list[_Live] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._pending = sorted(self.scenario.injections, key=lambda i: i.at_iteration)

    def tick(self, iteration: int) -> list[dict[str, Any]]:
        """Fire anything due at `iteration`; advance ramps and live injections."""
        fired: list[dict[str, Any]] = []

        while self._pending and self._pending[0].at_iteration <= iteration:
            inj = self._pending.pop(0)
            ramp = int(inj.params.get("ramp_iterations", 0))
            if ramp > 0:
                self._ramping.append(_Active(injection=inj, ramp=ramp))
            live = None
            if inj.type in _PERSISTENT:
                live = _Live(injection=inj, started=iteration)
                self._live.append(live)
            payload = self._apply(inj, iteration, progress=0.0 if ramp else 1.0,
                                  state=live.state if live else None)
            fired.append({"type": inj.type, "at_iteration": inj.at_iteration,
                          "ramping": bool(ramp), **payload})

        still: list[_Active] = []
        for act in self._ramping:
            elapsed = iteration - act.injection.at_iteration
            progress = min(1.0, elapsed / act.ramp)
            #  Skip the tick it fired on. The loop above already applied it at
            #  progress 0, and elapsed is 0 here, so re-applying would set the
            #  identical value a second time -- harmless, but it makes the trace
            #  read as though the injection happened twice.
            if elapsed > 0:
                self._apply(act.injection, iteration, progress=progress)
            if progress < 1.0:
                still.append(act)
        self._ramping = still

        for live in self._live:
            if live.started == iteration:
                continue                      # already applied on the firing tick
            payload = self._apply(live.injection, iteration, progress=1.0,
                                  state=live.state)
            if payload:
                fired.append({"type": live.injection.type,
                              "at_iteration": live.injection.at_iteration, **payload})
        return fired

    def _apply(self, inj: Injection, iteration: int, progress: float,
               state: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            fn = _HANDLERS[inj.type]
        except KeyError:
            raise KeyError(
                f"no handler for injection type {inj.type!r}; "
                f"registered: {sorted(_HANDLERS)}"
            ) from None
        ctx = InjectionContext(cluster=self.cluster, workload=self.workload,
                               params=inj.params, iteration=iteration, progress=progress,
                               seed=self.seed, iterations=self.iterations,
                               state=state if state is not None else {})
        return fn(ctx)


def registered_handlers() -> list[str]:
    return sorted(_HANDLERS)

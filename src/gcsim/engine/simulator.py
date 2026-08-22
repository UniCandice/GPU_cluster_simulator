"""The main loop.

One outer timestep, in full:

    injector.tick(i)                    physical/workload state may change here
    compute_s   = f(cells, clock, memory bandwidth, SM share)
    halo_s      = fabric.solve(...)     flow contention over the current topology
    arrival     = halo_s + compute_s
    iter_time   = max(arrival) + allreduce + output          <- THE BARRIER
    wait_s      = max(arrival) - arrival                     <- barrier slack

`wait_s` is the load-bearing quantity in the whole simulator. A rank that is
slow has ``wait ~ 0`` while its 127 peers accumulate wait; the culprit is the
only one genuinely busy and every victim looks idle. That inversion is what the
diagnosis view is built on.

Sample ticks are scheduled on the event queue alongside phase boundaries, so a
tick landing mid-timestep sees a partial timestep and the emitted telemetry is a
true time-average -- coarse, exactly as a real exporter is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gcsim.config import SimConfig, derive_rng
from gcsim.engine.events import Event, EventQueue, EventTrace, EventType
from gcsim.faults import FaultInjector
from gcsim.mesh import partition
from gcsim.models.compute import achieved_occupancy, compute_time_s, ideal_compute_time_s
from gcsim.models.network import Fabric, build_halo_flows
from gcsim.models.storage import StorageModel
from gcsim.models.thermal import DeviceGovernor
from gcsim.placement import place
from gcsim.routing import Router
from gcsim.samplers import ActivityWindow, SamplerSet
from gcsim.topology import build_cluster
from gcsim.workload import Phase, WorkloadState


@dataclass
class SimulationOutput:
    config: SimConfig
    frames: dict[str, pd.DataFrame]
    summary: dict[str, Any]


class Simulator:
    def __init__(self, cfg: SimConfig, record_ticks: bool = False):
        self.cfg = cfg
        cc = cfg.cluster

        self.cluster = build_cluster(cc)
        self.router = Router(self.cluster)
        self.decomposition = partition(cfg.mesh, self.cluster.n_gpus,
                                       preferred_first_extent=cc.gpus_per_node)
        self.placement = place(self.cluster, self.decomposition, self.router,
                               strategy=cfg.workload.placement)
        self.fabric = Fabric(self.cluster, self.router)
        self.halo_flows = build_halo_flows(self.fabric, self.decomposition, self.placement)

        self.n_ranks = self.decomposition.n_ranks
        self.iterations = cfg.workload.iterations

        self.workload = WorkloadState(
            total_cells=cfg.mesh.total_cells,
            output_interval=cfg.workload.output_interval,
            output_bytes_per_cell=cfg.mesh.output_bytes_per_cell,
            dataload_bytes_per_cell=cfg.mesh.dataload_bytes_per_cell,
            allreduce_values=cfg.workload.allreduce_values,
        )
        self.injector = FaultInjector(cluster=self.cluster, workload=self.workload,
                                      scenario=cfg.scenario, seed=cfg.seed,
                                      iterations=self.iterations)
        self.storage = StorageModel(spec=cc.storage)

        self._draw_silicon()
        self.governor = DeviceGovernor(cc, self.cluster.n_gpus, self._leakage)

        self.memory_per_rank_gb = (self.decomposition.cells * cfg.mesh.bytes_per_cell) / 1e9
        self.samplers = SamplerSet(
            cluster=self.cluster, governor=self.governor, scenario=cfg.scenario.name,
            seed=cfg.seed, rank_to_gpu=self.placement.rank_to_gpu,
            memory_per_rank_gb=self.memory_per_rank_gb,
        )

        self.queue = EventQueue()
        self.trace = EventTrace(cfg.scenario.name, cfg.seed, record_ticks=record_ticks)
        self.window = ActivityWindow(self.n_ranks)

        self.sample_interval = cc.telemetry.sample_interval_s
        self.straggler_threshold = cc.telemetry.straggler_rel_threshold

        self.t = 0.0
        self._last_credit_t = 0.0
        self._next_tick = self.sample_interval
        self._current: dict[str, Any] = {}
        self._prev_stragglers: tuple[int, ...] = ()
        self._prev_congested: frozenset[str] = frozenset()

        self._alloc_performance_arrays()

    # -- setup -------------------------------------------------------------

    def _draw_silicon(self) -> None:
        """Fixed per-GPU manufacturing variation.

        The ONLY stochastic input in the model, drawn once per GPU from that
        GPU's own stable stream. Everything else in telemetry is deterministic
        given this and the workload, which is why a healthy run has a tight but
        non-zero rank spread rather than an artificial dead flat line.
        """
        n = self.cluster.n_gpus
        clock = np.empty(n)
        leak = np.empty(n)
        noise = np.empty((n, self.iterations))
        for i, gpu in enumerate(self.cluster.gpu_list):
            rng = derive_rng(self.cfg.seed, f"gpu:{gpu.gpu_id}")
            clock[i] = np.clip(rng.normal(1.0, 0.010), 0.96, 1.04)
            leak[i] = np.clip(rng.normal(1.0, 0.040), 0.90, 1.12)
            #  Per-timestep efficiency jitter: cache behaviour, ECC scrubs, DVFS
            #  residency. Small, but it is what stops the healthy baseline from
            #  being suspiciously periodic.
            noise[i] = np.clip(rng.normal(1.0, 0.005, self.iterations), 0.97, 1.03)
            gpu.silicon_clock_offset = float(clock[i])
            gpu.silicon_leakage_offset = float(leak[i])
        self._clock_offset = clock
        self._leakage = leak
        self._noise = noise

    def _alloc_performance_arrays(self) -> None:
        n = self.iterations * self.n_ranks
        self._rp = {
            "iteration": np.empty(n, dtype=np.int32),
            "rank_id": np.empty(n, dtype=np.int16),
            "compute_time_s": np.empty(n),
            "halo_wait_s": np.empty(n),
            "allreduce_wait_s": np.empty(n),
            "checkpoint_time_s": np.empty(n),
            "total_time_s": np.empty(n),
            "is_straggler": np.empty(n, dtype=bool),
        }
        self._jp: list[dict[str, Any]] = []

    # -- time --------------------------------------------------------------

    def _schedule_ticks_through(self, t_end: float) -> None:
        while self._next_tick <= t_end + 1e-12:
            self.queue.push(Event(self._next_tick, EventType.SAMPLE_TICK))
            self._next_tick += self.sample_interval

    def _credit_to(self, t: float) -> None:
        """Credit activity accumulated between the last credit point and `t`."""
        cur = self._current
        span = cur["iter_time"]
        if span <= 0:
            return
        frac = (t - self._last_credit_t) / span
        if frac <= 0:
            return
        self.window.credit(frac, cur["compute_s"], cur["occupancy"], span, cur["output_s"])
        self._last_credit_t = t

    def _drain(self, t_end: float) -> None:
        """Pop every event up to `t_end`, sampling where the ticks fall."""
        self._schedule_ticks_through(t_end)
        while self.queue and (self.queue.peek_time() or 0.0) <= t_end + 1e-12:
            ev = self.queue.pop()
            if ev.event_type is EventType.SAMPLE_TICK:
                self._credit_to(ev.time_s)
                self._fire_tick(ev.time_s)
            self.trace.record(ev)
        self._credit_to(t_end)

    def _fire_tick(self, t: float) -> None:
        span = max(self.window.span_s, 1e-9)
        output_fraction = min(1.0, self.window.output_s / span)
        transitions = self.samplers.tick(t, self.window, self.storage, output_fraction)
        for gpu_id, kind, reason in transitions:
            self.trace.record(Event(
                t,
                EventType.THROTTLE_ENGAGED if kind == "engaged" else EventType.THROTTLE_RELEASED,
                gpu_id=gpu_id, payload={"reason": reason},
            ))
        self._check_congestion(t)
        self.window.reset()

    def _check_congestion(self, t: float) -> None:
        congested = frozenset(
            row["switch_id"] for row in self.samplers.switch_rows[-len(self.cluster.switches):]
            if row["congested"]
        )
        for sw in sorted(congested - self._prev_congested):
            self.trace.record(Event(t, EventType.CONGESTION_ONSET, payload={"switch_id": sw}))
        for sw in sorted(self._prev_congested - congested):
            self.trace.record(Event(t, EventType.CONGESTION_CLEARED, payload={"switch_id": sw}))
        self._prev_congested = congested

    # -- the run -----------------------------------------------------------

    def run(self) -> SimulationOutput:
        cfg = self.cfg
        self.trace.record(Event(0.0, EventType.SIM_START, payload={
            "mesh": cfg.mesh.name, "dims": list(cfg.mesh.dims),
            "grid": list(self.decomposition.grid), "ranks": self.n_ranks,
            "placement": self.placement.strategy,
        }))

        self._data_load()
        for iteration in range(1, self.iterations + 1):
            self._timestep(iteration)

        self.trace.record(Event(self.t, EventType.SIM_END,
                                payload={"iterations": self.iterations}))
        return SimulationOutput(config=cfg, frames=self._frames(), summary=self._summary())

    def _data_load(self) -> None:
        """One-off mesh read and decomposition."""
        latency, transfer = self.storage.read(self.workload.dataload_bytes())
        duration = latency + transfer
        self.trace.record(Event(self.t, EventType.DATA_LOAD_START, payload={
            "bytes": self.workload.dataload_bytes(), "phase": Phase.DATA_LOAD.value,
        }))
        self._current = {
            "iter_time": duration,
            "compute_s": np.zeros(self.n_ranks),
            "occupancy": np.zeros(self.n_ranks),
            "output_s": duration,       # host-side staging, not GPU work
        }
        self._last_credit_t = self.t
        end = self.t + duration
        self.queue.push(Event(end, EventType.DATA_LOAD_END,
                              payload={"duration_s": duration, "latency_s": latency}))
        self._drain(end)
        self.t = end

    def _timestep(self, iteration: int) -> None:
        cfg = self.cfg
        cc = cfg.cluster
        t0 = self.t

        # --- 1. state changes -------------------------------------------
        for payload in self.injector.tick(iteration):
            self.trace.record(Event(t0, EventType.INJECTION_APPLIED,
                                    payload={"iteration": iteration, **payload}))

        # --- 2. per-rank phase costs -------------------------------------
        r2g = self.placement.rank_to_gpu
        clock_factor = self.governor.clock_factor[r2g] * self._clock_offset[r2g]
        mem_bw = np.array([g.memory_bandwidth_factor for g in self.cluster.gpu_list])[r2g]
        sm_share = np.array([g.throughput_derate for g in self.cluster.gpu_list])[r2g]
        noise = self._noise[r2g, iteration - 1]

        compute_s = compute_time_s(
            self.decomposition.cells, cc.gpu, cfg.mesh.inner_iterations,
            clock_factor=clock_factor, memory_bandwidth_factor=mem_bw,
            throughput_derate=sm_share, efficiency_noise=noise,
        )
        occupancy = achieved_occupancy(self.decomposition.cells, cc.gpu, mem_bw)

        solution = self.fabric.solve(self.halo_flows, self.n_ranks)
        self.fabric.accumulate(self.halo_flows, solution)
        halo_s = solution.rank_time_s

        # --- 3. the barrier ----------------------------------------------
        busy = halo_s + compute_s
        arrival = busy.max()
        allreduce_s = self.fabric.allreduce_time_s(
            self.n_ranks, self.workload.allreduce_bytes(), solution)
        wait_s = arrival - busy

        is_output = self.workload.is_output_iteration(iteration)
        if is_output:
            out_bytes = self.workload.output_bytes()
            out_latency, out_transfer = self.storage.write(out_bytes)
            output_s = out_latency + out_transfer
        else:
            out_bytes = 0.0
            output_s = 0.0

        iter_time = arrival + allreduce_s + output_s
        if not is_output:
            #  Give the filesystem the timestep's worth of wall clock so any
            #  writeback backlog drains -- this is what makes the *baseline*
            #  between outputs recover (or not) after a heavy output campaign.
            self.storage.advance(iter_time)

        # --- 4. record performance ---------------------------------------
        median_busy = float(np.median(busy))
        stragglers = busy > median_busy * (1.0 + self.straggler_threshold)
        self._record_performance(iteration, t0, compute_s, halo_s, wait_s,
                                 allreduce_s, output_s, iter_time, busy, stragglers)

        # --- 5. advance the clock, sampling as we go ----------------------
        self._current = {"iter_time": iter_time, "compute_s": compute_s,
                         "occupancy": occupancy, "output_s": output_s}
        self._last_credit_t = t0

        t_halo = t0 + float(halo_s.max())
        t_compute = t0 + arrival
        t_allreduce = t_compute + allreduce_s
        t_end = t0 + iter_time

        self.queue.push(Event(t0, EventType.ITERATION_START,
                              payload={"iteration": iteration}))
        self.queue.push(Event(t_halo, EventType.HALO_EXCHANGE_END, payload={
            "iteration": iteration, "phase": Phase.HALO_EXCHANGE.value,
            "max_s": float(halo_s.max()), "min_s": float(halo_s.min()),
            "bytes": float(self.halo_flows.nbytes.sum()),
        }))
        self.queue.push(Event(t_compute, EventType.COMPUTE_END, payload={
            "iteration": iteration, "phase": Phase.COMPUTE.value,
            "max_s": float(compute_s.max()), "min_s": float(compute_s.min()),
            "slowest_rank": int(np.argmax(busy)),
        }))
        self.queue.push(Event(t_allreduce, EventType.ALLREDUCE_END, payload={
            "iteration": iteration, "phase": Phase.ALLREDUCE.value,
            "duration_s": allreduce_s,
        }))
        if is_output:
            self.queue.push(Event(t_allreduce, EventType.OUTPUT_START, payload={
                "iteration": iteration, "phase": Phase.OUTPUT.value, "bytes": out_bytes,
            }))
            self.queue.push(Event(t_end, EventType.OUTPUT_END, payload={
                "iteration": iteration, "duration_s": output_s,
                "dirty_backlog_gb": self.storage.dirty_bytes / 1e9,
            }))
        self.queue.push(Event(t_end, EventType.ITERATION_END, payload={
            "iteration": iteration, "iteration_time_s": iter_time,
        }))

        current = tuple(int(i) for i in np.nonzero(stragglers)[0])
        if current != self._prev_stragglers:
            self.queue.push(Event(t_end, EventType.STRAGGLER_DETECTED, payload={
                "iteration": iteration, "ranks": list(current),
                "excess_s": float(busy.max() - median_busy),
            }))
            self._prev_stragglers = current

        self._drain(t_end)
        self.t = t_end

    def _record_performance(self, iteration, t0, compute_s, halo_s, wait_s,
                            allreduce_s, output_s, iter_time, busy, stragglers) -> None:
        lo = (iteration - 1) * self.n_ranks
        hi = lo + self.n_ranks
        rp = self._rp
        rp["iteration"][lo:hi] = iteration
        rp["rank_id"][lo:hi] = np.arange(self.n_ranks)
        rp["compute_time_s"][lo:hi] = compute_s
        rp["halo_wait_s"][lo:hi] = halo_s
        #  Barrier slack plus the rank's share of the collective. Chosen so that
        #  the four phase columns sum EXACTLY to total_time_s for every rank --
        #  an invariant the tests assert.
        rp["allreduce_wait_s"][lo:hi] = wait_s + allreduce_s
        rp["checkpoint_time_s"][lo:hi] = output_s
        rp["total_time_s"][lo:hi] = iter_time
        rp["is_straggler"][lo:hi] = stragglers

        self._jp.append({
            "scenario": self.cfg.scenario.name,
            "seed": self.cfg.seed,
            "iteration": iteration,
            "timestamp": t0,
            "iteration_time_s": iter_time,
            "compute_max_s": float(compute_s.max()),
            "compute_mean_s": float(compute_s.mean()),
            "halo_max_s": float(halo_s.max()),
            "halo_mean_s": float(halo_s.mean()),
            "allreduce_s": allreduce_s,
            "checkpoint_s": output_s,
            "slowest_rank_id": int(np.argmax(busy)),
            "fastest_rank_id": int(np.argmin(busy)),
            "rank_spread_s": float(busy.max() - busy.min()),
            "sync_overhead_s": float(wait_s.mean() + allreduce_s),
            "wait_total_s": float(wait_s.sum()),
            "straggler_count": int(stragglers.sum()),
            "throughput_iters_per_s": 1.0 / iter_time if iter_time > 0 else 0.0,
            "cumulative_runtime_s": t0 + iter_time,
        })

    # -- outputs -----------------------------------------------------------

    def _frames(self) -> dict[str, pd.DataFrame]:
        gpu_ids = np.array([self.cluster.gpu_list[int(g)].gpu_id
                            for g in self.placement.rank_to_gpu], dtype=object)
        rank = pd.DataFrame(self._rp)
        rank.insert(0, "scenario", self.cfg.scenario.name)
        rank.insert(1, "seed", self.cfg.seed)
        rank["gpu_id"] = np.tile(gpu_ids, self.iterations)

        s = self.samplers
        return {
            "telemetry_gpu": pd.DataFrame(s.gpu_rows),
            "telemetry_node": pd.DataFrame(s.node_rows),
            "telemetry_nic": pd.DataFrame(s.nic_rows),
            "telemetry_switch_port": pd.DataFrame(s.port_rows),
            "telemetry_switch_aggregate": pd.DataFrame(s.switch_rows),
            "telemetry_storage": pd.DataFrame(s.storage_rows),
            "rank_performance": rank,
            "job_performance": pd.DataFrame(self._jp),
            "events": self.trace.to_frame(),
        }

    def _summary(self) -> dict[str, Any]:
        job = pd.DataFrame(self._jp)
        cfg = self.cfg
        ideal = ideal_compute_time_s(cfg.mesh.total_cells, self.n_ranks, cfg.cluster.gpu)
        mean_iter = float(job["iteration_time_s"].mean())
        comm = float((job["halo_mean_s"] + job["allreduce_s"]).mean())
        gpu = pd.DataFrame(self.samplers.gpu_rows)
        return {
            "run_id": cfg.run_id,
            "scenario": cfg.scenario.name,
            "scenario_label": cfg.scenario.label,
            "is_fault": cfg.scenario.fault,
            "tier": cfg.scenario.tier,
            "mesh": cfg.mesh.name,
            "seed": cfg.seed,
            "iterations": self.iterations,
            "n_ranks": self.n_ranks,
            "runtime_s": float(self.t),
            "mean_iteration_time_s": mean_iter,
            "median_iteration_time_s": float(job["iteration_time_s"].median()),
            "throughput_iters_per_s": float(job["throughput_iters_per_s"].mean()),
            "mean_rank_spread_s": float(job["rank_spread_s"].mean()),
            "mean_sync_overhead_s": float(job["sync_overhead_s"].mean()),
            "comm_fraction": comm / mean_iter if mean_iter else 0.0,
            "mean_halo_s": float(job["halo_mean_s"].mean()),
            "mean_compute_s": float(job["compute_mean_s"].mean()),
            #  Halo cost measured against IDEAL compute rather than achieved.
            #  `comm_fraction` is confounded by the occupancy penalty -- a coarse
            #  mesh inflates its own denominator -- so this is the metric that
            #  isolates the surface-to-volume effect the mesh study is about.
            "halo_per_ideal_compute": float(job["halo_mean_s"].mean()) / ideal if ideal else 0.0,
            "ideal_iteration_time_s": ideal,
            "parallel_efficiency": ideal / mean_iter if mean_iter else 0.0,
            "mean_utilization_pct": float(gpu["utilization_pct"].mean()),
            "mean_sm_occupancy_pct": float(gpu["sm_occupancy_pct"].mean()),
            "mean_temperature_c": float(gpu["temperature_c"].mean()),
            "max_temperature_c": float(gpu["temperature_c"].max()),
            "mean_power_w": float(gpu["power_w"].mean()),
            "throttled_sample_fraction": float(gpu["throttled"].mean()),
            "throttle_reasons": sorted(set(gpu.loc[gpu["throttled"], "throttle_reason"])),
            "events": len(self.trace),
            **{f"mesh_{k}": v for k, v in self.decomposition.summary().items()},
        }

"""Telemetry samplers.

Samplers are the *only* place telemetry is produced, and they only ever read
simulator state. Nothing here can be written by a scenario.

They run on a fixed interval that is deliberately independent of event
resolution. Real fleets poll DCGM at about 1 Hz while a timestep here lasts tens
to hundreds of milliseconds, so a sampler sees a time-average over several
timesteps and cannot resolve the phase structure inside them. That is a real
limitation of real monitoring, and inheriting it honestly is more useful than
pretending the exporter sees everything.

Counters that are cumulative on real hardware are cumulative here. Rates are
computed by differencing against the previous sample, which is exactly what a
collector does, and is what makes the monotonicity and conservation invariants
meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gcsim.config import ClusterConfig
from gcsim.models import power as power_model
from gcsim.models.thermal import DeviceGovernor, inlet_temperature_c
from gcsim.topology import Cluster

_BYTES_TO_GBIT = 8.0 / 1e9

#: A leaf is called congested when its uplink path is under real pressure.
#: Both thresholds sit well above what a healthy bulk-synchronous halo exchange
#: produces, so `congested` means something when it moves.
UPLINK_CONGESTION_PCT = 60.0
UPLINK_QUEUE_THRESHOLD = 10.0


@dataclass
class ActivityWindow:
    """Per-rank work accumulated since the last sample tick.

    The simulator credits this as timesteps complete; the GPU sampler drains it.
    A timestep straddling a tick boundary is apportioned pro rata -- the sampler
    genuinely cannot see inside a timestep, so there is nothing finer to model.
    """
    n_ranks: int
    compute_s: np.ndarray = field(init=False)
    occupancy_weighted: np.ndarray = field(init=False)
    output_s: float = 0.0
    span_s: float = 0.0

    def __post_init__(self) -> None:
        self.compute_s = np.zeros(self.n_ranks)
        self.occupancy_weighted = np.zeros(self.n_ranks)

    def credit(self, fraction: float, compute_s: np.ndarray, occupancy: np.ndarray,
               iteration_time_s: float, output_s: float) -> None:
        self.compute_s += compute_s * fraction
        self.occupancy_weighted += occupancy * compute_s * fraction
        self.output_s += output_s * fraction
        self.span_s += iteration_time_s * fraction

    def reset(self) -> None:
        self.compute_s[:] = 0.0
        self.occupancy_weighted[:] = 0.0
        self.output_s = 0.0
        self.span_s = 0.0


class SamplerSet:
    """Owns every telemetry stream and the previous-counter snapshots."""

    def __init__(self, cluster: Cluster, governor: DeviceGovernor, scenario: str,
                 seed: int, rank_to_gpu: np.ndarray, memory_per_rank_gb: np.ndarray):
        self.cluster = cluster
        self.cfg: ClusterConfig = cluster.cfg
        self.governor = governor
        self.scenario = scenario
        self.seed = seed
        self.rank_to_gpu = rank_to_gpu
        #  Sized by the cluster with -1 for idle GPUs: a subset job leaves
        #  most GPUs unallocated, and sizing this by n_ranks scatter-indexed
        #  by GPU index corrupts it the moment the two differ.
        self.gpu_to_rank = np.full(cluster.n_gpus, -1, dtype=np.int64)
        self.gpu_to_rank[rank_to_gpu] = np.arange(rank_to_gpu.size)
        self.memory_per_rank_gb = memory_per_rank_gb

        self.gpu_rows: list[dict] = []
        self.node_rows: list[dict] = []
        self.nic_rows: list[dict] = []
        self.port_rows: list[dict] = []
        self.switch_rows: list[dict] = []
        self.storage_rows: list[dict] = []

        self._prev_nic: dict[str, tuple[float, ...]] = {}
        self._prev_port: dict[str, tuple[float, ...]] = {}
        self._prev_throttled = np.zeros(cluster.n_gpus, dtype=bool)

        #  Static per-GPU identity columns, precomputed once.
        self._gpu_meta = [
            {"gpu_id": g.gpu_id, "node_id": g.node_id, "rack_id": g.rack_id}
            for g in cluster.gpu_list
        ]

    # -- the tick ----------------------------------------------------------

    def tick(self, t_s: float, window: ActivityWindow, storage, workload_output_fraction: float
             ) -> list[tuple[str, str, str]]:
        """Sample every stream and advance the thermal loop by one interval.

        Returns any throttle transitions, so the engine can put them in the
        event trace.
        """
        span = max(window.span_s, 1e-9)

        # --- rank-space activity -> GPU-space -----------------------------
        compute_fraction_rank = np.clip(window.compute_s / span, 0.0, 1.0)
        occupancy_rank = np.divide(window.occupancy_weighted,
                                   np.maximum(window.compute_s, 1e-12))
        occupancy_rank = np.clip(occupancy_rank, 0.0, 1.0)

        compute_fraction = np.zeros(self.cluster.n_gpus)
        occupancy = np.zeros(self.cluster.n_gpus)
        mem_used = np.zeros(self.cluster.n_gpus)
        compute_fraction[self.rank_to_gpu] = compute_fraction_rank
        occupancy[self.rank_to_gpu] = occupancy_rank
        mem_used[self.rank_to_gpu] = self.memory_per_rank_gb

        gspec = self.cfg.gpu
        occ_active = power_model.occupancy_active(compute_fraction, occupancy, gspec)
        utilisation = power_model.reported_utilisation(compute_fraction, gspec)

        #  Idle GPUs are idle, not spinning at a barrier. The spin floor in
        #  both models is the signature of a rank parked in a collective, and
        #  an unallocated GPU has no rank to park -- it must read ~0, not ~99%.
        #  Masked HERE rather than at compute_fraction == 0 inside the power
        #  model, because allocated ranks also hit zero compute in a window
        #  (the mesh load), and those genuinely are resident-and-spinning.
        #  The guard keeps the full-allocation path untouched to the bit.
        idle = self.gpu_to_rank < 0
        if idle.any():
            occ_active[idle] = 0.0
            utilisation[idle] = 0.0

        # --- rack inlet temperature, from cooling health and rack load ----
        rack_power_kw = self._rack_power_kw()
        cooling = np.array([r.cooling_efficiency for r in self.cluster.rack_list])
        inlet = inlet_temperature_c(rack_power_kw, cooling, self.cfg)
        rack_of_gpu = np.array([g.rack_index for g in self.cluster.gpu_list])
        inlet_per_gpu = inlet[rack_of_gpu]

        reliability_cap = np.array([g.reliability_clock_cap for g in self.cluster.gpu_list])

        # --- advance the closed loop --------------------------------------
        #  A window containing the mesh load or a field output is not
        #  representative of steady-state operation, so it must not be used to
        #  seed the thermal model.
        self.governor.step(span, occ_active, inlet_per_gpu, reliability_cap,
                           window_is_representative=workload_output_fraction < 1e-3)

        transitions = self._throttle_transitions()

        # --- write state back onto the entities (for anything that reads it)
        for i, g in enumerate(self.cluster.gpu_list):
            g.temperature_c = float(self.governor.temperature_c[i])
            g.power_w = float(self.governor.power_w[i])
            g.clock_mhz = float(self.governor.clock_mhz[i])
            g.throttled = bool(self.governor.throttled[i])
            g.throttle_reason = str(self.governor.reason[i])
            g.occupancy = float(occ_active[i])
            g.utilisation = float(utilisation[i])
            g.memory_used_gb = float(mem_used[i])
        for r, rack in enumerate(self.cluster.rack_list):
            rack.inlet_temp_c = float(inlet[r])
            rack.power_kw = float(rack_power_kw[r])

        # --- emit ----------------------------------------------------------
        self._sample_gpu(t_s, utilisation, occ_active, mem_used)
        self._sample_node(t_s, window, storage, workload_output_fraction)
        self._sample_nic(t_s, span)
        self._sample_switch(t_s, span)
        self._sample_storage(t_s, storage)
        return transitions

    # -- individual streams -------------------------------------------------

    def _sample_gpu(self, t_s: float, utilisation: np.ndarray, occ_active: np.ndarray,
                    mem_used: np.ndarray) -> None:
        """Emit one row per GPU.

        `sm_occupancy_pct` is the TIME-WEIGHTED occupancy over the sample window,
        which is what a profiler actually reports -- not occupancy conditional on
        a kernel doing useful work. The distinction is the entire point: a rank
        stalled at a barrier keeps a spin kernel resident, so its utilisation
        stays pinned near 100 while this number falls. Reporting the conditional
        value instead would make occupancy constant and destroy the signal.
        """
        gv = self.governor
        total = self.cfg.gpu.memory_gb
        for i, meta in enumerate(self._gpu_meta):
            self.gpu_rows.append({
                **meta,
                "timestamp": t_s,
                "scenario": self.scenario,
                "seed": self.seed,
                "utilization_pct": float(utilisation[i] * 100.0),
                "sm_occupancy_pct": float(occ_active[i] * 100.0),
                "memory_used_gb": float(mem_used[i]),
                "memory_total_gb": total,
                "power_w": float(gv.power_w[i]),
                "temperature_c": float(gv.temperature_c[i]),
                "clock_mhz": float(gv.clock_mhz[i]),
                "throttled": bool(gv.throttled[i]),
                "throttle_reason": str(gv.reason[i]),
            })

    def _sample_node(self, t_s: float, window: ActivityWindow, storage,
                     output_fraction: float) -> None:
        """Host-side pressure.

        `cpu_pressure` and `io_pressure` are driven by output staging, which is
        why they move during a legitimate output campaign while every GPU
        channel stays flat. `memory_pressure` follows the writeback backlog: if
        outputs arrive faster than the filesystem drains them, dirty pages
        accumulate in host memory and the pressure keeps climbing.
        """
        n_nodes = len(self.cluster.node_list)
        dirty_per_node_gb = storage.dirty_bytes / 1e9 / n_nodes
        bg = storage.background_utilisation()

        base_cpu = 0.08                      # MPI progress threads, always on
        cpu = min(1.0, base_cpu + 0.80 * output_fraction)
        io = min(1.0, output_fraction + bg)
        mem = min(1.0, 0.12 + dirty_per_node_gb / (self.cfg.host.memory_gb * 0.5))

        for node in self.cluster.node_list:
            node.cpu_pressure, node.io_pressure, node.memory_pressure = cpu, io, mem
            self.node_rows.append({
                "node_id": node.node_id,
                "rack_id": node.rack_id,
                "timestamp": t_s,
                "scenario": self.scenario,
                "seed": self.seed,
                "cpu_pressure": cpu,
                "memory_pressure": mem,
                "io_pressure": io,
            })

    def _sample_nic(self, t_s: float, span: float) -> None:
        for nic in self.cluster.nic_list:
            cur = (nic.tx_bytes, nic.rx_bytes, nic.tx_errors, nic.rx_errors,
                   nic.tx_drops, nic.rx_drops)
            prev = self._prev_nic.get(nic.nic_id, (0.0,) * 6)
            self._prev_nic[nic.nic_id] = cur
            self.nic_rows.append({
                "node_id": nic.node_id,
                "rack_id": nic.rack_id,
                "nic_id": nic.nic_id,
                "timestamp": t_s,
                "scenario": self.scenario,
                "seed": self.seed,
                "capacity_gbps": nic.capacity_gbps,
                "tx_gbps": (cur[0] - prev[0]) * _BYTES_TO_GBIT / span,
                "rx_gbps": (cur[1] - prev[1]) * _BYTES_TO_GBIT / span,
                "tx_bytes": cur[0], "rx_bytes": cur[1],
                "tx_errors": cur[2], "rx_errors": cur[3],
                "tx_drops": cur[4], "rx_drops": cur[5],
            })

    def _sample_switch(self, t_s: float, span: float) -> None:
        cl = self.cluster
        per_switch: dict[str, dict[str, float]] = {}

        for port in cl.port_list:
            #  Spine-side ports mirror their leaf-side partner: it is the same
            #  cable, so accounting it twice would break conservation.
            src = cl.ports[port.mirror_of] if port.mirror_of else port
            tx_bytes, rx_bytes = (src.rx_bytes, src.tx_bytes) if port.mirror_of else \
                                 (src.tx_bytes, src.rx_bytes)
            tx_err, rx_err = (src.rx_errors, src.tx_errors) if port.mirror_of else \
                             (src.tx_errors, src.rx_errors)
            tx_drop, rx_drop = (src.rx_drops, src.tx_drops) if port.mirror_of else \
                               (src.tx_drops, src.rx_drops)

            cur = (tx_bytes, rx_bytes, tx_err, rx_err, tx_drop, rx_drop)
            prev = self._prev_port.get(port.port_id, (0.0,) * 6)
            self._prev_port[port.port_id] = cur

            tx_gbps = (cur[0] - prev[0]) * _BYTES_TO_GBIT / span
            rx_gbps = (cur[1] - prev[1]) * _BYTES_TO_GBIT / span
            cap = port.capacity_gbps if port.up else 0.0
            util = 0.0 if cap <= 0 else min(100.0, max(tx_gbps, rx_gbps) / cap * 100.0)

            self.port_rows.append({
                "timestamp": t_s, "scenario": self.scenario, "seed": self.seed,
                "switch_id": port.switch_id, "switch_tier": port.switch_tier,
                "domain_id": port.domain_id, "port_id": port.port_id,
                "port_role": port.role, "peer_id": port.peer_id,
                "capacity_gbps": cap, "link_up": port.up,
                "tx_gbps": tx_gbps, "rx_gbps": rx_gbps,
                "tx_bytes": cur[0], "rx_bytes": cur[1],
                "tx_errors": cur[2], "rx_errors": cur[3],
                "tx_drops": cur[4], "rx_drops": cur[5],
                "utilisation_pct": util,
                #  Utilisation is a window average, but queue depth is the
                #  high-water mark reached DURING the exchange. The two look
                #  inconsistent on purpose: a bulk-synchronous job moves its
                #  entire halo in a burst that occupies a few percent of the
                #  sample window, so a link can average 4% and still queue hard.
                #  Averaging the queue away would hide exactly that.
                "queue_depth": src.queue_depth,
            })

            agg = per_switch.setdefault(port.switch_id, {
                "tx": 0.0, "rx": 0.0, "up_tx": 0.0, "up_rx": 0.0,
                "up_cap": 0.0, "down_cap": 0.0, "queue": 0.0,
            })
            agg["tx"] += tx_gbps
            agg["rx"] += rx_gbps
            if port.role == "uplink":
                agg["up_tx"] += tx_gbps
                agg["up_rx"] += rx_gbps
                agg["up_cap"] += cap
                #  Only uplink queueing counts toward congestion. A downlink
                #  port queues deeply during every halo burst because the host
                #  NIC is the intended bottleneck of a bulk-synchronous
                #  exchange -- that is the job working correctly, not a fault.
                agg["queue"] = max(agg["queue"], src.queue_depth)
            else:
                agg["down_cap"] += cap

        for switch in cl.switches.values():
            a = per_switch.get(switch.switch_id)
            if a is None:
                continue
            up_cap = a["up_cap"]
            up_util = 0.0 if up_cap <= 0 else min(100.0, max(a["up_tx"], a["up_rx"]) / up_cap * 100.0)
            #  Oversubscription is itself an observable: when uplinks fail, the
            #  ratio jumps, which localises the fault to a domain without any
            #  reference to ground truth.
            #  A spine has no uplinks, so it is not oversubscribed relative to
            #  anything and the ratio is undefined -- NaN, not inf. Reserve inf
            #  for a leaf that has uplink ports but has lost every one of them,
            #  where "infinitely oversubscribed" is the honest reading. An inf
            #  parked in a telemetry table for a structural reason is a trap for
            #  anything that later aggregates the column.
            if not switch.uplink_ids:
                oversub = float("nan")
            else:
                oversub = (a["down_cap"] / up_cap) if up_cap > 0 else float("inf")
            self.switch_rows.append({
                "timestamp": t_s, "scenario": self.scenario, "seed": self.seed,
                "switch_id": switch.switch_id, "switch_tier": switch.tier,
                "domain_id": switch.domain_id,
                "aggregate_tx_gbps": a["tx"], "aggregate_rx_gbps": a["rx"],
                "uplink_utilisation_pct": up_util,
                "oversubscription_ratio": oversub,
                "max_queue_depth": a["queue"],
                "congested": bool(up_util > UPLINK_CONGESTION_PCT
                                  or a["queue"] > UPLINK_QUEUE_THRESHOLD),
            })

    def _sample_storage(self, t_s: float, storage) -> None:
        self.storage_rows.append({
            "timestamp": t_s, "scenario": self.scenario, "seed": self.seed,
            "backend_id": "fs0", **storage.sample(),
        })

    # -- helpers ------------------------------------------------------------

    def _rack_power_kw(self) -> np.ndarray:
        """Total dissipation per rack: every GPU plus the hosts."""
        gv = self.governor
        rack_of_gpu = np.array([g.rack_index for g in self.cluster.gpu_list])
        gpu_kw = np.bincount(rack_of_gpu, weights=gv.power_w,
                             minlength=self.cfg.racks) / 1000.0
        host_kw = self.cfg.nodes_per_rack * self.cfg.host.idle_power_w / 1000.0
        return gpu_kw + host_kw

    def _throttle_transitions(self) -> list[tuple[str, str, str]]:
        now = self.governor.throttled
        engaged = np.nonzero(now & ~self._prev_throttled)[0]
        released = np.nonzero(~now & self._prev_throttled)[0]
        self._prev_throttled = now.copy()
        out = [(self.cluster.gpu_list[i].gpu_id, "engaged", str(self.governor.reason[i]))
               for i in engaged]
        out += [(self.cluster.gpu_list[i].gpu_id, "released", "NONE") for i in released]
        return out

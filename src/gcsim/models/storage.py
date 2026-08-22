"""The shared parallel filesystem behind every node.

Field output is a *legitimate phase*, not a fault, and it is the main reason
storage telemetry moves at all. Two mechanisms produce the observable behaviour:

1.  **Concurrent writers queue.** All 16 nodes dump at once, so the filesystem
    saturates and latency follows the M/M/1 form ``base / (1 - rho)``. That is
    why a checkpoint costs far more than ``bytes / bandwidth`` would suggest,
    and why the healthy baseline still shows a latency spike every 100
    timesteps.

2.  **Writeback outlives the write.** A dump leaves a backlog -- replication,
    metadata, page-cache flush -- that drains in the background at a bounded
    rate. If outputs arrive faster than the backlog drains, the residual load
    never reaches zero, so the *baseline* between dumps rises too.

Mechanism 2 is what makes the `phase_change` scenario a sustained rise rather
than a train of isolated spikes. Nothing about it is a hardware fault: the
filesystem is behaving exactly as specified, the job is simply asking for more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gcsim.config import StorageSpec

_GBPS_TO_BPS = 1e9 / 8.0


@dataclass
class StorageWindow:
    """Accumulators the sampler drains once per telemetry interval."""
    elapsed_s: float = 0.0
    busy_s: float = 0.0
    read_bytes: float = 0.0
    write_bytes: float = 0.0
    #: time-weighted sums, so the reported value is a true window average --
    #: which is how a real exporter reports and why short stalls get smeared
    read_latency_s_weighted: float = 0.0
    write_latency_s_weighted: float = 0.0
    utilisation_weighted: float = 0.0
    queue_weighted: float = 0.0

    def reset(self) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, 0.0)


@dataclass
class StorageModel:
    spec: StorageSpec
    dirty_bytes: float = 0.0
    #: bound on the writeback backlog. Past this, writes stop being absorbed and
    #: the job blocks on the filesystem directly.
    dirty_capacity_bytes: float = 512e9
    window: StorageWindow = field(default_factory=StorageWindow)

    @property
    def capacity_bps(self) -> float:
        return self.spec.capacity_gbps * _GBPS_TO_BPS

    @property
    def drain_bps(self) -> float:
        return self.spec.dirty_drain_gbps * _GBPS_TO_BPS

    # -- background --------------------------------------------------------

    def background_utilisation(self) -> float:
        """Filesystem load from writeback alone, with no job I/O in flight."""
        if self.dirty_bytes <= 0.0:
            return 0.0
        return min(self.drain_bps / self.capacity_bps, self.spec.max_utilisation)

    def advance(self, dt_s: float) -> None:
        """Let `dt_s` of wall time pass with no foreground I/O."""
        rho = self.background_utilisation()
        if self.dirty_bytes > 0.0:
            drained = min(self.dirty_bytes, self.drain_bps * dt_s)
            self.dirty_bytes -= drained
            self.window.read_bytes += 0.0
        lat_r = self.spec.base_read_latency_ms * 1e-3 / max(1.0 - rho, 1e-3)
        lat_w = self.spec.base_write_latency_ms * 1e-3 / max(1.0 - rho, 1e-3)
        self._observe(dt_s, busy_s=0.0, rho=rho, lat_r=lat_r, lat_w=lat_w)

    # -- foreground --------------------------------------------------------

    def write(self, total_bytes: float) -> tuple[float, float]:
        """A job-wide collective write. Returns (latency_s, transfer_s).

        Foreground traffic competes with whatever writeback is still draining,
        so a backlog left by the previous output directly slows this one.
        """
        rho_bg = self.background_utilisation()
        effective = max(self.capacity_bps * (1.0 - rho_bg), self.capacity_bps * 0.05)
        transfer_s = total_bytes / effective

        rho = self.spec.max_utilisation
        latency_s = self.spec.base_write_latency_ms * 1e-3 / max(1.0 - rho, 1e-3)

        #  The write leaves a backlog behind it. Once the backlog is at capacity
        #  nothing more is absorbed and the job simply waits for the disk.
        headroom = max(self.dirty_capacity_bytes - self.dirty_bytes, 0.0)
        self.dirty_bytes += min(total_bytes, headroom)

        duration = latency_s + transfer_s
        self.window.write_bytes += total_bytes
        self._observe(duration, busy_s=duration, rho=rho,
                      lat_r=latency_s, lat_w=latency_s)
        return latency_s, transfer_s

    def read(self, total_bytes: float) -> tuple[float, float]:
        """A job-wide collective read, used by the one-off mesh load phase."""
        rho_bg = self.background_utilisation()
        effective = max(self.capacity_bps * (1.0 - rho_bg), self.capacity_bps * 0.05)
        transfer_s = total_bytes / effective
        rho = self.spec.max_utilisation
        latency_s = self.spec.base_read_latency_ms * 1e-3 / max(1.0 - rho, 1e-3)
        duration = latency_s + transfer_s
        self.window.read_bytes += total_bytes
        self._observe(duration, busy_s=duration, rho=rho, lat_r=latency_s, lat_w=latency_s)
        return latency_s, transfer_s

    # -- observation -------------------------------------------------------

    def _observe(self, dt_s: float, busy_s: float, rho: float,
                 lat_r: float, lat_w: float) -> None:
        w = self.window
        w.elapsed_s += dt_s
        w.busy_s += busy_s
        w.read_latency_s_weighted += lat_r * dt_s
        w.write_latency_s_weighted += lat_w * dt_s
        w.utilisation_weighted += rho * dt_s
        w.queue_weighted += (rho ** 2 / max(1.0 - rho, 1e-3)) * dt_s

    def sample(self) -> dict[str, float]:
        """Drain the window into a telemetry row and reset it."""
        w = self.window
        span = max(w.elapsed_s, 1e-9)
        row = {
            "read_latency_ms": w.read_latency_s_weighted / span * 1e3,
            "write_latency_ms": w.write_latency_s_weighted / span * 1e3,
            "throughput_gbps": (w.read_bytes + w.write_bytes) * 8.0 / 1e9 / span,
            "queue_depth": w.queue_weighted / span,
            "utilisation_pct": w.utilisation_weighted / span * 100.0,
            "dirty_backlog_gb": self.dirty_bytes / 1e9,
        }
        w.reset()
        return row

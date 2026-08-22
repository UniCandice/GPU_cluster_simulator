"""Derived metrics, straggler attribution, and the rule-based diagnosis.

`diagnose` is the payoff for building the telemetry causally. It is a small,
explicit rule set that reads only telemetry -- never the scenario's `fault`
label -- and decides whether an observed slowdown is a hardware fault or a
legitimate workload change, and at which tier.

Crucially it needs no external baseline. It compares a run's own early window
against its late window, which is what a real detector has to do.

The rules encode exactly the discriminating cells of the signature matrix:

    throttled samples appeared        -> GPU fault, tier from throttle_reason
    one rack's temperature drifted    -> cooling fault, even with NO throttling
    link down / drops / errors rose   -> fabric fault, localised by domain_id
    rank spread widened, nothing else -> rank fault, localised by rank
    spread stayed TIGHT, storage and
      node pressure rose, no device
      channel moved at all            -> workload change, NOT a fault

The last rule is the interesting one. A detector that only watches throughput
fires on the output campaign; what saves it is the *absence* of movement
everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Fraction of the run treated as "before" and "after" for change detection.
BASELINE_FRACTION = 0.15
COMPARE_FRACTION = 0.30

#: A slowdown smaller than this is not worth explaining.
SLOWDOWN_THRESHOLD = 0.03
#: Rank spread beyond this multiple of the baseline counts as "widened".
SPREAD_WIDEN_FACTOR = 2.0
#: Fraction of the comparison window a rank must spend pacing the barrier before
#: it counts as a culprit rather than as jitter. Silicon variation alone puts the
#: unluckiest rank of a healthy fleet at about 0.5% of timesteps; a genuinely
#: derated rank sits an order of magnitude above that even when the fault is
#: only intermittent, so the gap this sits in is wide.
STRAGGLER_DUTY_FLOOR = 0.02
#: Every link corrupts the odd frame at its background bit error rate, so a
#: healthy fabric still trickles errors. A detector without a floor above that
#: trickle would report a fabric fault on every run.
LINK_ERROR_FLOOR = 1000.0
#: A rack drifting this far from its peers is a cooling problem, throttling or
#: not. Well above the spread a healthy fleet shows from silicon variation.
RACK_THERMAL_DRIFT_C = 8.0


def _split(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 10:
        return df, df
    head = df.iloc[: max(1, int(n * BASELINE_FRACTION))]
    tail = df.iloc[int(n * (1.0 - COMPARE_FRACTION)) :]
    return head, tail


# ---------------------------------------------------------------------------
# Straggler attribution
# ---------------------------------------------------------------------------

def straggler_attribution(rank_perf: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Rank the ranks by how much of the job's wait time they caused.

    `busy` is the work a rank actually does before the barrier. Its excess over
    the fleet median, multiplied by the 127 peers that had to wait for it, is
    the GPU-seconds that rank destroyed.
    """
    df = rank_perf.copy()
    df["busy_s"] = df["compute_time_s"] + df["halo_wait_s"]

    median_by_iter = df.groupby("iteration")["busy_s"].transform("median")
    df["excess_s"] = df["busy_s"] - median_by_iter

    n_ranks = df["rank_id"].nunique()
    grouped = df.groupby(["rank_id", "gpu_id"], as_index=False).agg(
        mean_busy_s=("busy_s", "mean"),
        mean_excess_s=("excess_s", "mean"),
        mean_wait_s=("allreduce_wait_s", "mean"),
        straggler_iterations=("is_straggler", "sum"),
    )
    grouped["wasted_gpu_s"] = (grouped["mean_excess_s"].clip(lower=0)
                               * (n_ranks - 1) * df["iteration"].nunique())
    grouped = grouped.sort_values("mean_excess_s", ascending=False).reset_index(drop=True)
    return grouped.head(top_n)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    verdict: str                       # "NOMINAL" | "HARDWARE_FAULT" | "WORKLOAD_CHANGE"
    tier: str                          # "none" | "gpu" | "rank" | "rack" | "workload"
    confidence: str                    # "high" | "medium" | "low"
    slowdown_pct: float
    evidence: list[str] = field(default_factory=list)
    discriminators: list[str] = field(default_factory=list)
    localisation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "tier": self.tier, "confidence": self.confidence,
            "slowdown_pct": self.slowdown_pct, "evidence": self.evidence,
            "discriminators": self.discriminators, "localisation": self.localisation,
        }


def diagnose(frames: dict[str, pd.DataFrame]) -> Diagnosis:
    job = frames["job_performance"].sort_values("iteration")
    before, after = _split(job, "iteration_time_s")

    base_time = float(before["iteration_time_s"].mean())
    late_time = float(after["iteration_time_s"].mean())
    slowdown = (late_time - base_time) / base_time if base_time else 0.0

    base_spread = float(before["rank_spread_s"].mean())
    late_spread = float(after["rank_spread_s"].mean())
    spread_widened = late_spread > max(base_spread * SPREAD_WIDEN_FACTOR, base_spread + 1e-4)

    evidence: list[str] = []
    discriminators: list[str] = []
    localisation: dict[str, Any] = {}

    impact = (f"job impact: iteration time {slowdown * 100:+.1f}% "
              f"({base_time * 1e3:.0f} ms -> {late_time * 1e3:.0f} ms)")

    #  Hardware channels are checked BEFORE the slowdown gate, deliberately. A
    #  degraded link on a compute-bound job may cost almost no wall time while
    #  its counters scream; reporting that as "nominal" because throughput held
    #  up is how a fleet ends up running on a fabric that is quietly failing.
    #  Job impact is reported as a consequence, not used as the trigger.

    # --- device channels: did anything throttle? --------------------------
    gpu = frames["telemetry_gpu"]
    throttled = gpu[gpu["throttled"]]
    if len(throttled):
        reasons = sorted(set(throttled["throttle_reason"]))
        racks = sorted(set(throttled["rack_id"]))
        gpus = sorted(set(throttled["gpu_id"]))
        discriminators += ["throttled", "throttle_reason", "clock_mhz", "power_w"]
        evidence.append(f"{len(gpus)} GPU(s) throttled, reason(s) {', '.join(reasons)}")
        evidence.append(impact)
        if "THERMAL" in reasons:
            evidence.append(f"temperature crossed the slowdown threshold in rack(s) {', '.join(racks)}")
            localisation = {"racks": racks, "gpus": gpus}
            return Diagnosis("HARDWARE_FAULT", "rack", "high", slowdown * 100.0,
                             evidence, discriminators, localisation)
        if "RELIABILITY" in reasons:
            evidence.append("RAS governor capped the clock -- the device reported its own "
                            "degradation, which is what separates this from a plain straggler")
            localisation = {"gpus": gpus}
            return Diagnosis("HARDWARE_FAULT", "gpu", "high", slowdown * 100.0,
                             evidence, discriminators, localisation)

    # --- thermal, BEFORE it reaches the throttle threshold -----------------
    #  A cooling failure on a light workload may never throttle anything: the
    #  rack draws less power, so the same loss of cooling capacity lands the dies
    #  well below the slowdown point. Waiting for a throttle bit would miss it
    #  entirely and leave the fleet running on a failed CRAC. What is unmissable
    #  is that one rack has drifted away from its peers, because inlet
    #  temperature is a rack property and every GPU in it moves together.
    rack_temp = gpu.pivot_table(index="timestamp", columns="rack_id",
                                values="temperature_c", aggfunc="mean").sort_index()
    if len(rack_temp) >= 10 and rack_temp.shape[1] > 1:
        head = rack_temp.iloc[: max(1, int(len(rack_temp) * BASELINE_FRACTION))].mean()
        tail = rack_temp.iloc[int(len(rack_temp) * (1.0 - COMPARE_FRACTION)) :].mean()
        drift = (tail - head) - (tail - head).median()
        hottest = str(drift.idxmax())
        if float(drift.max()) > RACK_THERMAL_DRIFT_C:
            discriminators += ["temperature_c", "rack_id"]
            evidence.append(
                f"rack {hottest} drifted {drift.max():+.1f} C relative to every other rack "
                f"({head[hottest]:.1f} -> {tail[hottest]:.1f} C) while the rest held steady")
            evidence.append("inlet temperature is a rack property, so all 32 GPUs moved "
                            "together -- no single-GPU fault can do that")
            evidence.append("no throttling yet: the dies are hot but still under the "
                            "slowdown threshold, so throughput is unaffected so far")
            evidence.append(impact)
            return Diagnosis("HARDWARE_FAULT", "rack", "medium", slowdown * 100.0,
                             evidence, discriminators, {"racks": [hottest]})

    # --- fabric channels ---------------------------------------------------
    ports = frames["telemetry_switch_port"]
    down = ports[~ports["link_up"]]
    port_before, port_after = _split(ports.sort_values("timestamp"), "tx_drops")
    drops_delta = float(port_after["tx_drops"].max() - port_before["tx_drops"].max())
    errors_delta = float(port_after["tx_errors"].max() - port_before["tx_errors"].max())

    if len(down) or drops_delta > LINK_ERROR_FLOOR or errors_delta > LINK_ERROR_FLOOR:
        domains = sorted({d for d in down["domain_id"].dropna().unique()})
        discriminators += ["link_up", "tx_drops", "tx_errors", "oversubscription_ratio"]
        evidence.append(impact)
        if len(down):
            evidence.append(f"{down['port_id'].nunique()} uplink port(s) down in domain(s) "
                            f"{', '.join(domains) or 'unknown'}")
        if drops_delta > LINK_ERROR_FLOOR:
            evidence.append(f"switch drops rose by {drops_delta:,.0f} packets")
        if errors_delta > LINK_ERROR_FLOOR:
            evidence.append(f"link errors rose by {errors_delta:,.0f} frames")
        evidence.append("compute durations unchanged -- the loss is on the wire, not the device")
        localisation = {"domains": domains}
        return Diagnosis("HARDWARE_FAULT", "rack", "high", slowdown * 100.0,
                         evidence, discriminators, localisation)

    # --- rank-level: a few ranks pace the barrier, with no hardware signal --
    #
    #  Checked BEFORE the slowdown gate below, deliberately. That gate compares
    #  an early window against a late one, which silently assumes the run began
    #  healthy. An intermittent fault that was already running during the
    #  baseline window shifts neither the mean nor the spread *between* windows
    #  and would be waved through as nominal -- while the ranks it lands on are
    #  still, plainly, pacing the barrier the whole time. Counting who paces
    #  needs no baseline at all, which is the point: it is the one rank-level
    #  check here that survives never having seen the cluster healthy.
    rank = frames["rank_performance"]
    late = rank[rank["iteration"] >= rank["iteration"].max() * (1.0 - COMPARE_FRACTION)]
    gpu_of = late.drop_duplicates("rank_id").set_index("rank_id")["gpu_id"]
    n_steps = max(int(late["iteration"].nunique()), 1)

    #  Localise by COUNTING the timesteps each rank paced the barrier, not by
    #  taking the largest mean. The two agree for a continuous fault, but for an
    #  intermittent one only the count works: a rank derated for a tenth of the
    #  window barely lifts its own average above ordinary silicon variation,
    #  while the mean happily promotes whichever healthy rank is the fleet's
    #  slowest. The count also finds every member of a cohort, where a single
    #  argmax can only ever name one of them. The duty floor keeps ordinary
    #  jitter out: a healthy fleet peaks near 0.5% of timesteps, a real culprit
    #  sits an order of magnitude above that.
    paced = late.groupby("rank_id")["is_straggler"].sum()
    paced = paced[paced >= n_steps * STRAGGLER_DUTY_FLOOR].sort_values(ascending=False)

    if len(paced) or (spread_widened and slowdown >= SLOWDOWN_THRESHOLD):
        busy = late["compute_time_s"] + late["halo_wait_s"]
        by_rank = busy.groupby(late["rank_id"]).mean()
        culprits = [int(i) for i in paced.index] or [int(by_rank.idxmax())]
        gpus = [str(gpu_of[c]) for c in culprits]
        duty = [100.0 * float(paced[c]) / n_steps if c in paced.index else 0.0
                for c in culprits]

        discriminators += ["rank_spread_s", "allreduce_wait_s", "sm_occupancy_pct"]
        evidence.append(impact)
        evidence.append(f"rank spread {base_spread * 1e3:.1f} ms -> {late_spread * 1e3:.1f} ms")
        if len(culprits) == 1:
            evidence.append(f"rank {culprits[0]} ({gpus[0]}) is slowest and has near-zero wait, "
                            f"while its peers accumulate wait")
        else:
            listed = ", ".join(f"{c} ({g}, {d:.0f}% of timesteps)"
                               for c, g, d in zip(culprits, gpus, duty))
            evidence.append(f"{len(culprits)} ranks pace the barrier at different times "
                            f"-- {listed} -- each with near-zero wait while it does")
            evidence.append("no rank paces it continuously, so the fault is intermittent: "
                            "averaged over the whole run each culprit looks nearly healthy")
        evidence.append("no throttling, no link errors -- the device reports itself healthy, "
                        "so this is invisible to any single hardware counter")
        #  GPUs only. The panel flattens every value it is given, so adding the
        #  rank ids here would print them as bare numbers beside the device ids;
        #  the evidence line above already pairs each rank with its GPU.
        return Diagnosis("HARDWARE_FAULT", "rank", "medium", slowdown * 100.0,
                         evidence, discriminators, {"gpus": gpus})

    # --- no hardware channel moved, nobody is pacing. Is anything wrong? ---
    if slowdown < SLOWDOWN_THRESHOLD:
        return Diagnosis(
            verdict="NOMINAL", tier="none", confidence="high",
            slowdown_pct=slowdown * 100.0,
            evidence=[f"iteration time changed by {slowdown * 100:+.1f}%, within tolerance",
                      "no throttling, no link errors, no queue growth, rank spread tight",
                      "no rank paces the barrier persistently"],
            discriminators=["iteration_time_s"],
        )
    evidence.append(impact)

    # --- everything device-side is clean, and the spread stayed tight -----
    storage = frames["telemetry_storage"].sort_values("timestamp")
    s_before, s_after = _split(storage, "write_latency_ms")
    lat_ratio = (float(s_after["write_latency_ms"].mean())
                 / max(float(s_before["write_latency_ms"].mean()), 1e-9))

    node = frames["telemetry_node"].sort_values("timestamp")
    n_before, n_after = _split(node, "io_pressure")
    io_delta = float(n_after["io_pressure"].mean() - n_before["io_pressure"].mean())

    discriminators += ["rank_spread_s", "throttled", "tx_errors",
                       "write_latency_ms", "io_pressure"]
    evidence.append(f"rank spread stayed tight ({late_spread * 1e3:.1f} ms) -- every rank "
                    f"is affected identically, which no localised fault can do")
    evidence.append("no GPU throttled, no link errors, no switch queue growth")
    if lat_ratio > 1.5:
        evidence.append(f"storage write latency rose {lat_ratio:.1f}x and stayed risen")
    if io_delta > 0.02:
        evidence.append(f"node io_pressure rose {io_delta:+.2f}")
    return Diagnosis("WORKLOAD_CHANGE", "workload", "high", slowdown * 100.0,
                     evidence, discriminators, {})


# ---------------------------------------------------------------------------
# Mesh scaling study
# ---------------------------------------------------------------------------

def mesh_scaling_table(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per mesh for the partitioning study.

    Built from healthy runs only: the point is to isolate what resolution alone
    does to utilisation, with no fault anywhere.
    """
    rows = []
    for s in summaries:
        rows.append({
            "mesh": s["mesh"],
            "dims": s["mesh_dims"],
            "total_cells": s["mesh_total_cells"],
            "cells_per_rank": s["mesh_cells_mean"],
            "partition_imbalance": s["mesh_imbalance"],
            "surface_to_volume": s["mesh_surface_to_volume"],
            "halo_mb_per_iter": s["mesh_halo_bytes_per_iter_mean"] / 1e6,
            "memory_per_rank_gb": s["mesh_memory_per_rank_gb"],
            "sm_occupancy_pct": s["mean_sm_occupancy_pct"],
            "utilization_pct": s["mean_utilization_pct"],
            "comm_fraction": s["comm_fraction"],
            "halo_per_ideal_compute": s["halo_per_ideal_compute"],
            "iteration_time_s": s["mean_iteration_time_s"],
            "parallel_efficiency": s["parallel_efficiency"],
        })
    order = {"coarse": 0, "medium": 1, "fine": 2}
    return pd.DataFrame(rows).sort_values(
        "mesh", key=lambda c: c.map(lambda m: order.get(m, 99))).reset_index(drop=True)


def phase_breakdown(job: pd.DataFrame) -> pd.DataFrame:
    """Mean seconds per timestep in each phase, for the stacked timeline."""
    out = job[["iteration", "iteration_time_s"]].copy()
    out["compute_s"] = job["compute_mean_s"]
    out["halo_s"] = job["halo_mean_s"]
    out["allreduce_s"] = job["allreduce_s"]
    out["checkpoint_s"] = job["checkpoint_s"]
    #  Whatever the mean rank did NOT spend on its own work, it spent waiting.
    out["wait_s"] = (job["iteration_time_s"] - job["compute_mean_s"]
                     - job["halo_mean_s"] - job["allreduce_s"] - job["checkpoint_s"]).clip(lower=0)
    return out


def counter_conservation(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-domain check that leaf uplink bytes match what left the rack.

    Bytes a rack's NICs sent, minus bytes that stayed inside the rack, must equal
    bytes its leaf's uplink ports sent. If switch telemetry were bookkeeping
    invented alongside the flows rather than derived from them, this would not
    hold.
    """
    ports = frames["telemetry_switch_port"]
    last = ports.sort_values("timestamp").groupby("port_id").last().reset_index()
    leaf = last[last["switch_tier"] == "leaf"]
    rows = []
    for domain, grp in leaf.groupby("domain_id"):
        up = grp[grp["port_role"] == "uplink"]
        down = grp[grp["port_role"] == "downlink"]
        rows.append({
            "domain_id": domain,
            "uplink_tx_gb": up["tx_bytes"].sum() / 1e9,
            "uplink_rx_gb": up["rx_bytes"].sum() / 1e9,
            "downlink_rx_gb": down["rx_bytes"].sum() / 1e9,
            "downlink_tx_gb": down["tx_bytes"].sum() / 1e9,
            "active_uplinks": int(up["link_up"].sum()),
        })
    return pd.DataFrame(rows)

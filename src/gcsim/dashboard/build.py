"""Turn `runs/` into one self-contained HTML file.

Design notes
------------
The page carries no external dependency -- no CDN, no charting library, no
server. Charts are hand-drawn SVG driven entirely by CSS custom properties, so
light and dark are the same code path and the whole file is a few hundred kB
instead of the 5 MB a bundled charting library would cost.

Everything is decimated before it is embedded. Full-resolution Parquet stays on
disk for real analysis; the page only needs enough resolution to make each
signature legible:

    job series      all 1000 timesteps (only a handful of numbers per step)
    rank heatmaps   128 ranks x 96 time bins
    telemetry       96 time bins, aggregated per rack

Values are rounded hard on the way out. A temperature travels as tenths of a
degree in an integer, not a float64 -- across 18 runs that is the difference
between a 3 MB page and a 12 MB one.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gcsim.config import load_config
from gcsim.mesh import DIRECTION_NAMES, partition
from gcsim.metrics import (STRAGGLER_DUTY_FLOOR, counter_conservation,
                           mesh_scaling_table, straggler_attribution)
from gcsim.placement import KIND_CODES, KIND_NAMES, place
from gcsim.routing import CROSS_DOMAIN, Router
from gcsim.telemetry import read_run
from gcsim.topology import build_cluster

TIME_BINS = 96
TEMPLATE = Path(__file__).parent / "template.html"
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "dashboard" / "index.html"

#: Fraction of a run treated as "before" / "after" when deriving the signature
#: matrix. Matches metrics.diagnose, so the table and the verdict never disagree.
HEAD, TAIL = 0.15, 0.30


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _r(values, places: int = 3):
    """Round for transport. `places=0` yields ints."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if places == 0:
        return [int(v) for v in np.rint(arr)]
    return [round(float(v), places) for v in arr]


def _bin_series(t: np.ndarray, v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of `v` in each time bin, filling empty bins from their neighbours.

    An empty bin means the sampler produced nothing in that window, which reads
    as "unchanged" -- not as a drop to zero. Leading empties are back-filled from
    the first real sample rather than carried forward from nothing: seeding the
    carry at zero puts a 0 degree C, 0% occupancy column at the start of every
    short run, which is not a measurement and badly distorts any colour scale
    derived from the data range.
    """
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
    total = np.bincount(idx, weights=v, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    out = np.divide(total, np.maximum(count, 1))

    filled = np.nonzero(count > 0)[0]
    if filled.size == 0:
        return out
    empty = count == 0
    if empty.any():
        last = out[filled[0]]                 # back-fill, never zero
        for i in range(len(out)):
            if empty[i]:
                out[i] = last
            else:
                last = out[i]
    return out


def _edges(runtime: float, n_bins: int = TIME_BINS) -> np.ndarray:
    return np.linspace(0.0, max(runtime, 1e-6), n_bins + 1)


def _bin_count(n_samples: int) -> int:
    """How many time bins this run can actually support.

    Never more than there are samples. A 39 s run sampled at 1 Hz has 39 real
    observations; spreading them over 96 bins would fill three-fifths of the
    columns with duplicated neighbours and invent blocky structure that is an
    artefact of the binning, not of the simulation.
    """
    return int(max(8, min(TIME_BINS, n_samples)))


def _window(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    if n < 6:
        m = float(np.mean(values))
        return m, m
    head = values[: max(1, int(n * HEAD))]
    tail = values[int(n * (1.0 - TAIL)):]
    return float(np.mean(head)), float(np.mean(tail))


def _direction(before: float, after: float, rel: float = 0.05, floor: float = 0.0) -> str:
    if abs(after - before) <= max(abs(before) * rel, floor):
        return "flat"
    return "up" if after > before else "down"


# ---------------------------------------------------------------------------
# per-run extraction
# ---------------------------------------------------------------------------

def _job_block(job: pd.DataFrame) -> dict[str, Any]:
    """Phase decomposition per timestep, in milliseconds.

    `wait_ms` is what the mean rank spent blocked at the barrier: the timestep
    minus its own work. Stacked with the other four it accounts for the whole
    timestep, so the stack height IS the timestep and nothing is hidden.
    """
    wait = (job["iteration_time_s"] - job["compute_mean_s"] - job["halo_mean_s"]
            - job["allreduce_s"] - job["checkpoint_s"]).clip(lower=0)
    return {
        "iteration": [int(v) for v in job["iteration"]],
        "t": _r(job["timestamp"], 2),
        "iter_ms": _r(job["iteration_time_s"] * 1e3, 2),
        "compute_ms": _r(job["compute_mean_s"] * 1e3, 2),
        "halo_ms": _r(job["halo_mean_s"] * 1e3, 2),
        "allreduce_ms": _r(job["allreduce_s"] * 1e3, 3),
        "output_ms": _r(job["checkpoint_s"] * 1e3, 2),
        "wait_ms": _r(wait * 1e3, 2),
        "spread_ms": _r(job["rank_spread_s"] * 1e3, 2),
    }


def _rank_block(rank: pd.DataFrame, gpu: pd.DataFrame, runtime: float,
                n_ranks: int, n_bins: int) -> dict[str, Any]:
    """Rank x time heatmaps.

    Three channels, chosen because between them they separate every scenario:
    barrier wait (who is waiting for whom), occupancy (who is actually doing
    work), and temperature (which physical region is affected).
    """
    edges = _edges(runtime, n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0

    #  Wait comes from rank_performance, which is indexed by timestep. Map it
    #  onto the same wall-clock bins as the telemetry.
    iters = rank["iteration"].to_numpy()
    n_iters = int(iters.max())
    bins = np.clip(((iters - 1) / n_iters * n_bins).astype(int), 0, n_bins - 1)
    ranks = rank["rank_id"].to_numpy()
    wait = np.zeros((n_ranks, n_bins))
    count = np.zeros((n_ranks, n_bins))
    np.add.at(wait, (ranks, bins), rank["allreduce_wait_s"].to_numpy() * 1e3)
    np.add.at(count, (ranks, bins), 1.0)
    wait /= np.maximum(count, 1.0)

    #  Occupancy and temperature come from GPU telemetry, already on wall time.
    #  Row order is the run's own rank -> GPU map, read back from the frame,
    #  NOT every GPU in the telemetry: a subset job leaves idle GPUs in
    #  telemetry_gpu with no row in a rank-indexed heatmap to land in, and
    #  sorting ids only agreed with rank order while the job was the whole
    #  cluster under packed placement anyway.
    first = rank[rank["iteration"] == rank["iteration"].min()]
    gpu_ids = first.sort_values("rank_id")["gpu_id"].tolist()
    order = {g: i for i, g in enumerate(gpu_ids)}
    occ = np.zeros((n_ranks, n_bins))
    temp = np.zeros((n_ranks, n_bins))
    for gid, grp in gpu.groupby("gpu_id"):
        i = order.get(gid)
        if i is None:                     # idle GPU: no rank, no row
            continue
        t = grp["timestamp"].to_numpy()
        occ[i] = _bin_series(t, grp["sm_occupancy_pct"].to_numpy(), edges)
        temp[i] = _bin_series(t, grp["temperature_c"].to_numpy(), edges)

    busy = rank["compute_time_s"] + rank["halo_wait_s"]
    late = rank["iteration"] > n_iters * (1.0 - TAIL)
    by_rank = busy[late].groupby(rank.loc[late, "rank_id"]).mean()
    wait_by_rank = rank.loc[late].groupby("rank_id")["allreduce_wait_s"].mean()

    return {
        "t": _r(centres, 1),
        "gpu_ids": gpu_ids,
        #  Tenths of a unit as integers: same information, a third of the bytes.
        "wait": [_r(row, 0) for row in wait],
        "occupancy": [_r(row * 10, 0) for row in occ],
        "temperature": [_r(row * 10, 0) for row in temp],
        "mean_busy_ms": _r(by_rank.reindex(range(n_ranks)).fillna(0.0) * 1e3, 2),
        "mean_wait_ms": _r(wait_by_rank.reindex(range(n_ranks)).fillna(0.0) * 1e3, 2),
    }


def _telemetry_block(frames: dict[str, pd.DataFrame], runtime: float,
                     n_bins: int) -> dict[str, Any]:
    gpu = frames["telemetry_gpu"]
    edges = _edges(runtime, n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0

    racks = sorted(gpu["rack_id"].unique())
    temperature, clock, power = {}, {}, {}
    for rack, grp in gpu.groupby("rack_id"):
        agg = grp.groupby("timestamp")[["temperature_c", "clock_mhz", "power_w"]].mean()
        t = agg.index.to_numpy()
        temperature[rack] = _r(_bin_series(t, agg["temperature_c"].to_numpy(), edges), 2)
        clock[rack] = _r(_bin_series(t, agg["clock_mhz"].to_numpy(), edges), 1)
        power[rack] = _r(_bin_series(t, agg["power_w"].to_numpy(), edges), 1)

    #  Fleet means cover the GPUs the job runs on. Averaging 96 idle GPUs
    #  into a 32-rank job's occupancy line would flatten every signal the
    #  page exists to show. At full allocation the filter passes everything
    #  and the numbers are unchanged.
    allocated = set(frames["rank_performance"]["gpu_id"].unique())
    fleet = (gpu[gpu["gpu_id"].isin(allocated)]
             .groupby("timestamp")[["utilization_pct", "sm_occupancy_pct"]].mean())
    ft = fleet.index.to_numpy()
    throttled = gpu.groupby("timestamp")["throttled"].sum()

    storage = frames["telemetry_storage"].sort_values("timestamp")
    st = storage["timestamp"].to_numpy()
    node = frames["telemetry_node"].groupby("timestamp")[
        ["cpu_pressure", "memory_pressure", "io_pressure"]].mean()
    nt = node.index.to_numpy()

    return {
        "t": _r(centres, 1),
        "racks": racks,
        "rack_temperature": temperature,
        "rack_clock": clock,
        "rack_power": power,
        "utilisation": _r(_bin_series(ft, fleet["utilization_pct"].to_numpy(), edges), 2),
        "occupancy": _r(_bin_series(ft, fleet["sm_occupancy_pct"].to_numpy(), edges), 2),
        "throttled_gpus": _r(_bin_series(throttled.index.to_numpy(),
                                         throttled.to_numpy().astype(float), edges), 0),
        "storage_write_ms": _r(_bin_series(st, storage["write_latency_ms"].to_numpy(), edges), 3),
        "storage_queue": _r(_bin_series(st, storage["queue_depth"].to_numpy(), edges), 2),
        "storage_dirty_gb": _r(_bin_series(st, storage["dirty_backlog_gb"].to_numpy(), edges), 2),
        "node_cpu": _r(_bin_series(nt, node["cpu_pressure"].to_numpy(), edges), 3),
        "node_mem": _r(_bin_series(nt, node["memory_pressure"].to_numpy(), edges), 3),
        "node_io": _r(_bin_series(nt, node["io_pressure"].to_numpy(), edges), 3),
    }


def _fabric_block(frames: dict[str, pd.DataFrame], runtime: float,
                  n_bins: int) -> dict[str, Any]:
    ports = frames["telemetry_switch_port"]
    agg = frames["telemetry_switch_aggregate"]
    edges = _edges(runtime, n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0

    leaves = sorted(agg.loc[agg["switch_tier"] == "leaf", "domain_id"].dropna().unique())
    uplink_util, oversub = {}, {}
    for domain, grp in agg[agg["switch_tier"] == "leaf"].groupby("domain_id"):
        grp = grp.sort_values("timestamp")
        t = grp["timestamp"].to_numpy()
        uplink_util[domain] = _r(_bin_series(t, grp["uplink_utilisation_pct"].to_numpy(), edges), 2)
        oversub[domain] = _r(_bin_series(t, grp["oversubscription_ratio"].to_numpy(), edges), 2)

    up = ports[(ports["switch_tier"] == "leaf") & (ports["port_role"] == "uplink")]
    drops, errors, active = {}, {}, {}
    for domain, grp in up.groupby("domain_id"):
        per_t = grp.groupby("timestamp").agg(
            drops=("tx_drops", "sum"), errors=("tx_errors", "sum"),
            active=("link_up", "sum"))
        t = per_t.index.to_numpy()
        drops[domain] = _r(_bin_series(t, per_t["drops"].to_numpy(), edges) / 1e6, 3)
        errors[domain] = _r(_bin_series(t, per_t["errors"].to_numpy(), edges) / 1e6, 3)
        active[domain] = _r(_bin_series(t, per_t["active"].to_numpy().astype(float), edges), 1)

    return {
        "t": _r(centres, 1),
        "leaves": leaves,
        "uplink_util": uplink_util,
        "oversubscription": oversub,
        "drops_m": drops,
        "errors_m": errors,
        "active_uplinks": active,
        "conservation": counter_conservation(frames).round(3).to_dict("records"),
    }


def _signature_row(frames: dict[str, pd.DataFrame], job: pd.DataFrame) -> dict[str, str]:
    """Derive one column of the signature matrix from telemetry alone.

    Nothing here consults the ground-truth label, so a reader can check the
    verdict against the evidence rather than taking it on trust.
    """
    gpu = frames["telemetry_gpu"]
    ports = frames["telemetry_switch_port"]
    storage = frames["telemetry_storage"].sort_values("timestamp")
    node = frames["telemetry_node"].sort_values("timestamp")

    def dirn(series, rel=0.05, floor=0.0):
        return _direction(*_window(np.asarray(series, dtype=float)), rel, floor)

    fleet = gpu.groupby("timestamp")[
        ["utilization_pct", "sm_occupancy_pct", "power_w", "temperature_c"]].mean()

    throttled = gpu[gpu["throttled"]]
    reasons = sorted(set(throttled["throttle_reason"])) if len(throttled) else []
    n_throttled = int(throttled["gpu_id"].nunique()) if len(throttled) else 0

    up = ports[(ports["switch_tier"] == "leaf") & (ports["port_role"] == "uplink")]
    err_total = up.groupby("timestamp")["tx_errors"].sum()
    down = ports[~ports["link_up"]]

    #  Which racks moved thermally, relative to the fleet? Inlet temperature is a
    #  rack property, so a cooling failure shows as one rack drifting from the
    #  rest -- visible here even when it never reaches the throttle threshold.
    rack_t = gpu.pivot_table(index="timestamp", columns="rack_id",
                             values="temperature_c", aggfunc="mean").sort_index()
    drift = "flat"
    hot_racks: list[str] = []
    if len(rack_t) >= 6 and rack_t.shape[1] > 1:
        head = rack_t.iloc[: max(1, int(len(rack_t) * HEAD))].mean()
        tail = rack_t.iloc[int(len(rack_t) * (1.0 - TAIL)):].mean()
        delta = (tail - head) - (tail - head).median()
        hot_racks = sorted(delta[delta > 8.0].index.tolist())
        drift = "up" if hot_racks else "flat"

    spread_before, spread_after = _window(job["rank_spread_s"].to_numpy())

    #  How many ranks paced the barrier for a meaningful share of the late
    #  window. Every other row in this matrix is an early-vs-late comparison,
    #  which is structurally blind to a fault that was already running during
    #  the early window: both windows move together and every channel reports
    #  "flat". This row needs no baseline, so it is the one that still sees an
    #  intermittent straggler -- and it is the same signal `metrics.diagnose`
    #  localises on, sharing its threshold rather than keeping a second copy.
    rank = frames["rank_performance"]
    late_rank = rank[rank["iteration"] >= rank["iteration"].max() * (1.0 - TAIL)]
    n_late = max(int(late_rank["iteration"].nunique()), 1)
    pacers = late_rank.groupby("rank_id")["is_straggler"].sum()
    n_pacers = int((pacers >= n_late * STRAGGLER_DUTY_FLOOR).sum())

    return {
        "iteration_time": dirn(job["iteration_time_s"], 0.03),
        "rank_spread": _direction(spread_before, spread_after, 1.0),
        "barrier_pacers": "up" if n_pacers else "flat",
        "barrier_pacer_count": str(n_pacers),
        "compute": dirn(job["compute_mean_s"], 0.03),
        "halo": dirn(job["halo_mean_s"], 0.10),
        "utilisation": dirn(fleet["utilization_pct"], 0.01),
        "occupancy": dirn(fleet["sm_occupancy_pct"], 0.03),
        "power": dirn(fleet["power_w"], 0.03),
        "rack_thermal_drift": drift,
        "hot_racks": ",".join(hot_racks),
        "throttled": "up" if reasons else "flat",
        "throttle_reason": ",".join(reasons) if reasons else "NONE",
        "throttled_gpus": str(n_throttled),
        "link_errors": "up" if float(err_total.max()) > 1000 else "flat",
        "link_down": "up" if len(down) else "flat",
        "down_domains": ",".join(sorted(d for d in down["domain_id"].dropna().unique())),
        "storage_latency": dirn(storage["write_latency_ms"], 0.30),
        "node_io": dirn(node["io_pressure"], 0.10, 0.01),
    }


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------

def seed_run_counts(runs_dir: Path | str) -> dict[int, int]:
    """How many run directories each seed has on disk, keyed by seed.

    The count matters because seeds are no longer guaranteed to be full sweeps:
    a quick `--scenarios straggler --meshes coarse` run leaves a seed with one
    directory, and the pages need to say so rather than looking identical to an
    eighteen-run sweep.
    """
    runs_dir = Path(runs_dir)
    counts: dict[int, int] = {}
    for d in sorted(runs_dir.iterdir()):
        summary = d / "summary.json"
        if d.is_dir() and summary.exists():
            with summary.open("r", encoding="utf-8") as fh:
                s = int(json.load(fh)["seed"])
            counts[s] = counts.get(s, 0) + 1
    return counts


def seeds_under(runs_dir: Path | str) -> list[int]:
    """Every seed with runs on disk, ascending."""
    return sorted(seed_run_counts(runs_dir))


def build_payload(runs_dir: Path | str, seed: int | None = None) -> dict[str, Any]:
    """Payload for ONE seed.

    Runs are keyed by `scenario__mesh`, which carries no seed, so loading two
    seeds at once means the second silently overwrites the first and the page
    shows a mixture of both while claiming to show all of them. A payload is
    therefore one seed's worth, and `seed` says which.

    Passing None is only a convenience for the single-seed case: it resolves to
    the one seed present and refuses to guess when there is more than one.
    """
    runs_dir = Path(runs_dir)
    bundle = load_config()

    available = seeds_under(runs_dir)
    if not available:
        raise FileNotFoundError(
            f"no runs found under {runs_dir}; run scripts/run_all.py first")
    if seed is None:
        if len(available) > 1:
            raise ValueError(
                f"runs under {runs_dir} span seeds {available}; pass seed= to say "
                f"which one to build, or use build_dashboard() to build all of them")
        seed = available[0]
    elif seed not in available:
        raise FileNotFoundError(f"no runs for seed {seed} under {runs_dir}; have {available}")

    run_dirs = []
    for p in sorted(runs_dir.iterdir()):
        if not (p.is_dir() and (p / "summary.json").exists()):
            continue
        with (p / "summary.json").open("r", encoding="utf-8") as fh:
            if int(json.load(fh)["seed"]) == seed:
                run_dirs.append(p)

    runs: dict[str, Any] = {}
    healthy_summaries: list[dict] = []

    for d in run_dirs:
        frames, summary = read_run(d)
        key = f"{summary['scenario']}__{summary['mesh']}"
        job = frames["job_performance"].sort_values("iteration")
        runtime = float(summary["runtime_s"])
        n_ranks = int(summary["n_ranks"])
        n_bins = _bin_count(frames["telemetry_gpu"]["timestamp"].nunique())

        events = frames["events"]
        marks = []
        for _, row in events[events["event_type"] == "INJECTION_APPLIED"].iterrows():
            body = json.loads(row["payload"])
            marks.append({"t": round(float(row["timestamp"]), 2),
                          "iteration": int(body.get("iteration", 0)),
                          "label": body.get("type", "injection")})

        runs[key] = {
            "summary": summary,
            "job": _job_block(job),
            "marks": marks,
            "ranks": _rank_block(frames["rank_performance"], frames["telemetry_gpu"],
                                 runtime, n_ranks, n_bins),
            "telemetry": _telemetry_block(frames, runtime, n_bins),
            "fabric": _fabric_block(frames, runtime, n_bins),
            "signature": _signature_row(frames, job),
            "attribution": straggler_attribution(
                frames["rank_performance"], top_n=6).round(5).to_dict("records"),
        }
        if summary["scenario"] == "healthy":
            healthy_summaries.append(summary)

    #  Partition geometry, so the process-grid map can show raggedness directly
    #  rather than asserting it in prose.
    #
    #  Link class per halo face comes from `placement`, which already computed it
    #  when it mapped ranks onto GPUs -- deriving it again here by comparing rank
    #  indices would let the picture drift away from the model it claims to show.
    cc = bundle.cluster
    cluster = build_cluster(cc)
    router = Router(cluster)
    ic = cc.interconnect
    latency_us = {
        "intranode": ic.intranode.latency_us,
        "intra_domain": 2.0 * ic.nic.latency_us,
        "cross_domain": 2.0 * ic.nic.latency_us + 2.0 * ic.leaf_uplink.latency_us,
    }

    #  Geometry follows the JOB, not the cluster. With an allocation configured,
    #  the decomposition, the faces, and which racks exchange with which are all
    #  properties of the subset actually running -- describing the 128-rank
    #  layout under a 32-rank run's charts would be the config asserting a
    #  topology nobody is using. Without an allocation this is exactly the old
    #  full-cluster computation.
    alloc = bundle.workload.allocation
    job_ranks = alloc.n_ranks if alloc else cc.n_gpus

    partitions = {}
    for name, mesh in bundle.meshes.items():
        d = partition(mesh, job_ranks, preferred_first_extent=cc.gpus_per_node)
        placement = place(cluster, d, router, strategy=bundle.workload.placement,
                          allocation=alloc)
        rack_of = np.array([cluster.gpu(int(g)).rack_index
                            for g in placement.rank_to_gpu])

        #  Averaged over ranks, not read off rank 0: on a ragged mesh the faces
        #  differ between ranks and quoting one of them would misreport the rest.
        #  Link class per direction: uniform across ranks for the shipped full
        #  layout, but a subset placement can put the same face on NVLink for
        #  one rank and across a rack for another -- reported as "mixed" rather
        #  than whichever rank 0 happened to get.
        face_cells = d.face_cells.mean(axis=0)
        total_face = float(face_cells.sum()) or 1.0
        faces = []
        for i, direction in enumerate(DIRECTION_NAMES):
            kinds = {int(k) for k in placement.neighbour_kind[:, i]}
            kind = KIND_NAMES[kinds.pop()] if len(kinds) == 1 else "mixed"
            faces.append({
                "dir": direction,
                "cells": round(float(face_cells[i]), 1),
                "kind": kind,
                "latency_us": round(latency_us[kind], 1) if kind in latency_us else None,
                "byte_share": round(100.0 * float(face_cells[i]) / total_face, 2),
            })

        #  Which rack pairs this job's halo actually joins, and which racks host
        #  ranks at all. For the full cluster the pairs close into the familiar
        #  ring; a subset spanning two racks yields one link and two idle racks,
        #  and the drawing should say so instead of implying a cycle.
        cross = KIND_CODES[CROSS_DOMAIN]
        links: set[tuple[int, int]] = set()
        for r in range(d.n_ranks):
            for dd in range(placement.neighbour_kind.shape[1]):
                if placement.neighbour_kind[r, dd] == cross:
                    a, b = int(rack_of[r]), int(rack_of[d.neighbours[r, dd]])
                    if a != b:
                        links.add((min(a, b), max(a, b)))

        partitions[name] = {
            "grid": list(d.grid),
            "dims": list(mesh.dims),
            "cells": [int(c) for c in d.cells],
            "coords": [[int(v) for v in row] for row in d.coords],
            "imbalance": round(d.imbalance, 5),
            "surface_to_volume": round(d.surface_to_volume, 5),
            "extents": [int(v) for v in d.extents[0]],
            "faces": faces,
            "n_racks": cc.racks,
            "active_racks": sorted({int(x) for x in rack_of}),
            "rack_links": [list(p) for p in sorted(links)],
            "halo_mb_per_iter": round(
                float(d.halo_bytes_per_iteration().sum(axis=1).mean()) / 1e6, 2),
            "label": mesh.label,
            "note": mesh.note,
        }

    return {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "seed": int(seed),
            "cluster": f"{cc.racks} racks x {cc.nodes_per_rack} nodes x "
                       f"{cc.gpus_per_node} GPUs = {cc.n_gpus} ranks",
            "gpu_model": cc.gpu.model,
            "workload": f"{bundle.workload.iterations} timesteps, field output every "
                        f"{bundle.workload.output_interval}"
                        + (f", {bundle.workload.allocation.n_ranks} of {cc.n_gpus} GPUs "
                           f"({bundle.workload.placement})"
                           if bundle.workload.allocation else ""),
            "sample_interval_s": cc.telemetry.sample_interval_s,
            "slowdown_c": cc.gpu.thermal_slowdown_c,
            "n_ranks": cc.n_gpus,
            #  What workload.yaml configures RIGHT NOW, as opposed to what any
            #  run on disk was made with. Runs persist across config edits, so
            #  the two can disagree -- and the page must say so rather than
            #  quietly showing a 128-rank run under a 32-rank configuration.
            "configured_gpus": (bundle.workload.allocation.n_ranks
                                if bundle.workload.allocation else cc.n_gpus),
            "gpus_per_node": cc.gpus_per_node,
            "gpus_per_rack": cc.gpus_per_rack,
            "racks": cc.racks,
        },
        "meshes": [m for m in ("coarse", "medium", "fine") if m in bundle.meshes],
        "default_mesh": bundle.default_mesh,
        "scenarios": [
            {"name": s.name, "label": s.label, "fault": s.fault,
             "tier": s.tier, "description": s.description}
            for s in (bundle.scenarios[n] for n in bundle.scenario_order)
        ],
        "mesh_study": (mesh_scaling_table(healthy_summaries).round(5).to_dict("records")
                       if healthy_summaries else []),
        "partitions": partitions,
        "runs": runs,
    }


def open_in_browser(path: Path) -> bool:
    """Show a built dashboard, returning whether a browser actually took it.

    Never fatal. A build that cannot reach a browser -- CI, a container, a
    machine with no display -- is still a successful build, so a failure here
    is reported to the caller and swallowed rather than raised.

    Goes through `as_uri()` rather than passing the path as a string: this
    project lives under paths with spaces in them, and a bare Windows path is
    not a URL a browser will accept.
    """
    import webbrowser
    try:
        return webbrowser.open(Path(path).resolve().as_uri())
    except Exception:
        return False


def build_dashboard(runs_dir: Path | str,
                    out_path: Path | str | None = None) -> tuple[list[Path], str]:
    """Render one dashboard per seed. Returns every path and the `Generated` stamp.

    One file per seed rather than one file for everything, because runs are
    keyed by `scenario__mesh` -- no seed in the key -- so a combined page would
    show whichever seed happened to load last while its header claimed to show
    both.

    With a single seed on disk the output is `dashboard/index.html`, exactly as
    before. With several, each gets `index_seed{N}.html` and the plain
    `index.html` is written again for the highest seed, so existing links and
    bookmarks keep resolving to something real rather than to a stale mixture.

    The returned list always ends with that plain `index.html`, which is the
    canonical entry point: the per-seed pages are reachable from its header.
    Callers wanting to show one page should open the last.

    The stamp is returned rather than discarded so a caller can print it: the
    output paths never change between builds, so it is the only thing that tells
    you at a glance whether the page you are looking at is this build.
    """
    runs_dir = Path(runs_dir)
    counts = seed_run_counts(runs_dir)
    seeds = sorted(counts)
    if not seeds:
        raise FileNotFoundError(
            f"no runs found under {runs_dir}; run scripts/run_all.py first")

    out = Path(out_path) if out_path else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    html = TEMPLATE.read_text(encoding="utf-8")
    marker = "/*__GCSIM_DATA__*/null"
    if marker not in html:
        raise ValueError(f"template {TEMPLATE} is missing the data marker")

    def path_for(seed: int) -> Path:
        return out if len(seeds) == 1 else out.with_name(f"{out.stem}_seed{seed}{out.suffix}")

    #  Every page lists every seed, so the header can offer one-click switching
    #  without the reader having to guess a filename. The run count rides along
    #  because seeds are not all alike any more: "view 8 (1)" tells the reader
    #  that page holds a single run before they click through to it.
    others = [{"seed": s, "href": path_for(s).name, "n_runs": counts[s]} for s in seeds]

    written: list[Path] = []
    stamp = ""
    for seed in seeds:
        payload = build_payload(runs_dir, seed=seed)
        payload["meta"]["seed_links"] = others
        stamp = payload["meta"]["generated"]
        blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        page = html.replace(marker, blob)

        target = path_for(seed)
        target.write_text(page, encoding="utf-8")
        written.append(target)

        #  Highest seed also lands on the bare filename, so a link to
        #  dashboard/index.html never points at nothing.
        if len(seeds) > 1 and seed == seeds[-1]:
            out.write_text(page, encoding="utf-8")
            written.append(out)

    return written, stamp

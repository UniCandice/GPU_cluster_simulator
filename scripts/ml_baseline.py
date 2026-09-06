#!/usr/bin/env python
"""A learned fault classifier on the simulator's own telemetry.

    python scripts/ml_baseline.py all --runs-dir runs_ml --seeds 42 1 2 3 4

Three stages, each usable on its own:

  generate   simulate every scenario x mesh for each seed into --runs-dir,
             with the fault's target and onset re-drawn per seed so that
             neither entity identity nor iteration number is a label proxy
  features   cut every run into iteration windows and compute dimensionless,
             fleet-relative features per window, per (window, GPU) and per
             (window, rack); labels come from the INJECTION_APPLIED payload
  train      fit gradient-boosted trees (+ a linear and an unsupervised
             baseline), evaluate on held-out seeds and held-out meshes,
             compare with metrics.diagnose on the same runs, and write
             metrics.json / figures.json for the slide deck

Nothing in the simulator is modified. Randomisation is done in memory by
rebuilding the frozen config dataclasses; the shipped runs/ stay untouched.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gcsim.config import ConfigBundle, derive_rng, load_config  # noqa: E402
from gcsim.telemetry import read_run  # noqa: E402

CLASSES = ["healthy", "straggler", "network_domain", "thermal", "gpu_degradation", "phase_change"]
FAULT_CLASSES = {"straggler", "network_domain", "thermal", "gpu_degradation"}
MESHES = ["coarse", "medium", "fine"]
DEFAULT_SEEDS = [42, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def variant_bundle(bundle: ConfigBundle, scenario: str, seed: int,
                   randomise: bool) -> tuple[ConfigBundle, dict[str, Any]]:
    """Re-draw the fault's target and onset for this seed, in memory.

    The yaml fixes thermal on rack 1 at iteration 250, the network fault on rack
    2 at 300, the bad GPU at r3n1g2 at 350 and the output campaign at 500. A
    model trained on those runs could learn "rack 1 means thermal" or "iteration
    300 means network" and score perfectly while learning nothing about the
    physics. Re-drawing both per seed removes the shortcut; the true target is
    read back from the injection payload, never from here.
    """
    sc = bundle.scenarios[scenario]
    choice: dict[str, Any] = {}
    if not randomise or not sc.injections:
        return bundle, choice
    rng = derive_rng(seed, f"ml:{scenario}:variant")
    new_inj = []
    for inj in sc.injections:
        params = dict(inj.params)
        at = inj.at_iteration
        if inj.type == "cooling_degrade":
            params["target"] = {"rack": int(rng.integers(4))}
            at = int(rng.integers(200, 451))
        elif inj.type == "leaf_uplink_failure":
            params["target"] = {"rack": int(rng.integers(4))}
            at = int(rng.integers(200, 601))
        elif inj.type == "gpu_reliability_degrade":
            params["target"] = {"rack": int(rng.integers(4)), "node": int(rng.integers(4)),
                                "gpu": int(rng.integers(8))}
            at = int(rng.integers(200, 601))
        elif inj.type == "workload_output_change":
            at = int(rng.integers(200, 601))
        # gpu_throughput_episodes: cohort and schedule are already seed-keyed
        new_inj.append(dataclasses.replace(inj, at_iteration=at, params=params))
        choice = {"type": inj.type, "at_iteration": at, "target": params.get("target")}
    new_sc = dataclasses.replace(sc, injections=tuple(new_inj))
    return dataclasses.replace(bundle, scenarios={**bundle.scenarios, scenario: new_sc}), choice


def generate(args: argparse.Namespace) -> None:
    from gcsim.scenarios import run_scenario
    runs_dir: Path = args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    base = load_config()
    todo = [(sc, m, s) for s in args.seeds for sc in base.scenario_order for m in args.meshes
            if sc in args.scenarios]
    t_all = time.perf_counter()
    for n, (scenario, mesh, seed) in enumerate(todo, 1):
        run_id = f"{scenario}__{mesh}__seed{seed}"
        if (runs_dir / run_id / "summary.json").exists() and not args.force:
            print(f"[{n:3d}/{len(todo)}] {run_id:38s} cached")
            continue
        bundle, choice = variant_bundle(base, scenario, seed, not args.no_randomise)
        print(f"[{n:3d}/{len(todo)}] {run_id:38s} ...", end="", flush=True)
        res = run_scenario(scenario, mesh=mesh, seed=seed, bundle=bundle, out_dir=runs_dir)
        manifest[run_id] = {"scenario": scenario, "mesh": mesh, "seed": seed,
                            "randomised": not args.no_randomise, **choice}
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        print(f" {res.wall_seconds:5.1f}s  rules={res.summary['diagnosis']['verdict']}"
              f"/{res.summary['diagnosis']['tier']}")
    print(f"generate: {len(todo)} runs, {time.perf_counter() - t_all:.0f}s")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
#
# Every feature is either dimensionless (a ratio, a fraction, a coefficient of
# variation) or a deviation from the fleet / rack median, so the same model
# covers coarse, medium and fine meshes whose iteration times differ 20x.
# Windows are defined in iterations, not seconds, for the same reason: 100
# iterations is ~4 telemetry samples on coarse and ~80 on fine.
#
# Deliberately NOT used as features: scenario / seed (labels), the events
# table (the label source), is_straggler / straggler_count / slowest_rank_id
# (detector outputs and identities baked into the simulator), every entity id
# (identity), iteration / timestamp (position in the run), and absolute
# per-iteration times or memory sizes (mesh identifiers). See the deck.

def _onset_and_targets(frames: dict[str, pd.DataFrame], scenario: str) -> dict[str, Any]:
    """Ground truth read back from the injector's own event, never from yaml."""
    ev = frames["events"]
    fired = ev[ev["event_type"] == "INJECTION_APPLIED"]
    out: dict[str, Any] = {"onset": None, "gpus": set(), "rack": None, "episodes": []}
    if fired.empty:
        return out
    payload = json.loads(fired.iloc[0]["payload"])
    out["onset"] = int(payload.get("at_iteration", fired.iloc[0]["timestamp"]))
    if scenario == "straggler":
        out["episodes"] = payload["episodes"]
        out["onset"] = None  # labelled per window from the episode schedule
    elif scenario == "gpu_degradation":
        out["gpus"] = {payload["gpu_id"]}
        out["rack"] = payload["gpu_id"][: payload["gpu_id"].index("n")]
    elif scenario in ("thermal", "network_domain"):
        out["rack"] = payload["rack_id"]
    return out


def _window_label(scenario: str, s: int, e: int, truth: dict[str, Any]) -> tuple[str, float]:
    """Class of window [s, e) and the fraction of it the fault covers.

    Pre-onset windows of a faulted run are labelled healthy -- the injector has
    not fired, so the hardware is healthy. Windows straddling the onset are
    marked 'straddle' and left out of training and scoring; the time-to-detect
    sweep still runs on them.
    """
    if scenario == "healthy":
        return "healthy", 0.0
    if scenario == "straggler":
        covered = np.zeros(e - s, dtype=bool)
        for ep in truth["episodes"]:
            a, b = max(ep["start"], s), min(ep["end"], e)
            if b > a:
                covered[a - s:b - s] = True
        frac = float(covered.mean())
        return ("straggler" if frac > 0 else "healthy"), frac
    onset = truth["onset"]
    if e <= onset:
        return "healthy", 0.0
    if s >= onset:
        return scenario, 1.0
    return "straddle", float((e - onset) / (e - s))


def _cv(x: np.ndarray, axis=None) -> float | np.ndarray:
    m = np.mean(x, axis=axis)
    return np.std(x, axis=axis) / np.where(m == 0, np.nan, m)


def _slope_rel(y: np.ndarray) -> float:
    if len(y) < 3 or np.mean(y) == 0:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0] * len(y) / np.mean(y))


def _delta_ratio(df: pd.DataFrame, num_cols: list[str], den_cols: list[str], by: str) -> pd.Series:
    """(last - first) of cumulative counters within the window, per entity, as a ratio."""
    g = df.groupby(by, sort=False)
    num = sum(g[c].last() - g[c].first() for c in num_cols)
    den = sum(g[c].last() - g[c].first() for c in den_cols)
    return (num / den.where(den > 0, np.nan)) * 1e9  # per GB


class RunTables:
    """One run's frames pre-pivoted once, so a window is a row slice."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        job = frames["job_performance"].sort_values("iteration").reset_index(drop=True)
        self.job = job
        self.iters = job["iteration"].to_numpy()
        self.stamp = job["timestamp"].to_numpy()
        self.iter_time = job["iteration_time_s"].to_numpy()

        rank = frames["rank_performance"]
        piv = lambda c: rank.pivot(index="iteration", columns="rank_id", values=c)  # noqa: E731
        self.compute = piv("compute_time_s").to_numpy()
        self.halo = piv("halo_wait_s").to_numpy()
        self.wait = piv("allreduce_wait_s").to_numpy()
        self.busy = self.compute + self.halo
        med = np.median(self.busy, axis=1, keepdims=True)
        self.excess = self.busy / med - 1.0
        rank_gpu = rank.drop_duplicates("rank_id").set_index("rank_id")["gpu_id"]
        self.rank_gpu = rank_gpu.reindex(piv("compute_time_s").columns).to_numpy()

        gpu = frames["telemetry_gpu"]
        self.gpu_t = np.sort(gpu["timestamp"].unique())
        gp = lambda c: gpu.pivot(index="timestamp", columns="gpu_id", values=c)  # noqa: E731
        occ = gp("sm_occupancy_pct")
        self.gpu_ids = np.array(occ.columns)
        self.gpu_rack = np.array([g[: g.index("n")] for g in self.gpu_ids])
        self.gpu_node = np.array([g[: g.index("g")] for g in self.gpu_ids])
        self.occ = occ.to_numpy(dtype=float)
        self.util = gp("utilization_pct").to_numpy(dtype=float)
        self.temp = gp("temperature_c").to_numpy(dtype=float)
        self.power = gp("power_w").to_numpy(dtype=float)
        self.clock = gp("clock_mhz").to_numpy(dtype=float)
        self.throttled = gp("throttled").to_numpy(dtype=float)
        reason = gp("throttle_reason")
        self.thr_thermal = (reason == "THERMAL").to_numpy(dtype=float)
        self.thr_reliab = (reason == "RELIABILITY").to_numpy(dtype=float)
        self.racks = np.array(sorted(set(self.gpu_rack)))
        self.rack_mask = np.stack([self.gpu_rack == r for r in self.racks])  # (4, 128)

        self.nic = frames["telemetry_nic"].sort_values(["nic_id", "timestamp"])
        ports = frames["telemetry_switch_port"]
        self.uplinks = ports[(ports["switch_tier"] == "leaf") & (ports["port_role"] == "uplink")] \
            .sort_values(["port_id", "timestamp"])
        agg = frames["telemetry_switch_aggregate"]
        self.agg = agg[agg["switch_tier"] == "leaf"].copy()
        self.agg["oversubscription_ratio"] = self.agg["oversubscription_ratio"].replace(np.inf, 16.0)
        self.storage = frames["telemetry_storage"].sort_values("timestamp")
        self.node = frames["telemetry_node"]

    # -- window helpers ----------------------------------------------------
    def wall_span(self, s: int, e: int) -> tuple[float, float]:
        i0 = np.searchsorted(self.iters, s)
        i1 = np.searchsorted(self.iters, e)
        t0 = float(self.stamp[i0])
        t1 = float(self.stamp[i1]) if i1 < len(self.stamp) else float(self.stamp[-1] + self.iter_time[-1])
        return t0, t1

    def tel_rows(self, t0: float, t1: float) -> slice:
        return slice(int(np.searchsorted(self.gpu_t, t0)), int(np.searchsorted(self.gpu_t, t1)))

    def frame_in(self, df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
        ts = df["timestamp"].to_numpy()
        return df[(ts >= t0) & (ts < t1)]

    # -- window-level features --------------------------------------------
    def window_features(self, s: int, e: int) -> dict[str, float]:
        f: dict[str, float] = {}
        i0, i1 = np.searchsorted(self.iters, s), np.searchsorted(self.iters, e)
        job = self.job.iloc[i0:i1]
        it = job["iteration_time_s"].to_numpy()
        mean_it = float(it.mean())
        f["job_iter_cv"] = float(_cv(it))
        f["job_iter_slope_rel"] = _slope_rel(it)
        f["job_spread_rel"] = float(job["rank_spread_s"].mean() / mean_it)
        f["job_sync_rel"] = float(job["sync_overhead_s"].mean() / mean_it)
        f["job_halo_over_compute"] = float(job["halo_mean_s"].mean() / job["compute_mean_s"].mean())
        f["job_halo_max_rel"] = float((job["halo_max_s"] / job["halo_mean_s"]).mean() - 1)
        f["job_compute_max_rel"] = float((job["compute_max_s"] / job["compute_mean_s"]).mean() - 1)
        f["job_ckpt_rel"] = float(job["checkpoint_s"].mean() / mean_it)
        f["job_output_duty"] = float((job["checkpoint_s"] > 0).mean())

        busy, ex = self.busy[i0:i1], self.excess[i0:i1]
        pacer = np.argmax(busy, axis=1)
        counts = np.bincount(pacer, minlength=busy.shape[1]) / len(pacer)
        top = np.sort(counts)[::-1]
        f["rank_pace_top1"] = float(top[0])
        f["rank_pace_top3"] = float(top[:3].sum())
        nz = counts[counts > 0]
        f["rank_pace_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(busy.shape[1]))
        f["rank_excess_max_mean"] = float(ex.mean(axis=0).max())
        f["rank_excess_p99"] = float(np.percentile(ex, 99))
        f["rank_compute_disp"] = float(np.mean(_cv(self.compute[i0:i1], axis=1)))
        f["rank_halo_disp"] = float(np.mean(_cv(self.halo[i0:i1], axis=1)))
        f["rank_wait_min_rel"] = float(self.wait[i0:i1].mean(axis=0).min() / mean_it)

        t0, t1 = self.wall_span(s, e)
        r = self.tel_rows(t0, t1)
        if r.stop - r.start >= 1:
            occ, util = self.occ[r], self.util[r]
            f["gpu_occ_mean"] = float(occ.mean())
            f["gpu_util_occ_gap"] = float((util - occ).mean())
            occ_g = occ.mean(axis=0)
            f["gpu_occ_min_rel"] = float(occ_g.min() / np.median(occ_g))
            clk_g = self.clock[r].mean(axis=0)
            f["gpu_clock_min_rel"] = float(clk_g.min() / np.median(clk_g) - 1)
            f["gpu_power_cv"] = float(_cv(self.power[r].mean(axis=0)))
            temp = self.temp[r]
            rack_mean = np.stack([temp[:, m].mean(axis=1) for m in self.rack_mask])  # (4, T)
            drift = rack_mean - np.median(rack_mean, axis=0, keepdims=True)
            f["gpu_rack_temp_drift"] = float(drift.max(axis=0).mean())
            hot = int(np.argmax(rack_mean[:, -1]))
            f["gpu_rack_temp_rise"] = float(rack_mean[hot, -1] - rack_mean[hot, 0])
            f["gpu_temp_max_minus_med"] = float((temp.max(axis=1) - np.median(temp, axis=1)).mean())
            f["gpu_throttled_frac"] = float(self.throttled[r].mean())
            f["gpu_throttle_thermal_frac"] = float(self.thr_thermal[r].mean())
            f["gpu_throttle_reliab_frac"] = float(self.thr_reliab[r].mean())

        nic = self.frame_in(self.nic, t0, t1)
        if len(nic) >= 2 * nic["nic_id"].nunique():
            err = _delta_ratio(nic, ["tx_errors", "rx_errors"], ["tx_bytes", "rx_bytes"], "nic_id")
            drop = _delta_ratio(nic, ["tx_drops", "rx_drops"], ["tx_bytes", "rx_bytes"], "nic_id")
            f["nic_err_rate"] = float(err.mean())
            f["nic_err_max"] = float(err.max())
            f["nic_drop_rate"] = float(drop.mean())
            f["nic_tx_cv"] = float(_cv(nic.groupby("nic_id")["tx_gbps"].mean().to_numpy()))

        up = self.frame_in(self.uplinks, t0, t1)
        if len(up) >= 2 * up["port_id"].nunique():
            f["port_down_frac"] = float(1.0 - up["link_up"].astype(float).mean())
            f["port_down_max_domain"] = float((1.0 - up.groupby("domain_id")["link_up"]
                                                .mean().astype(float)).max())
            err = _delta_ratio(up, ["tx_errors", "rx_errors"], ["tx_bytes", "rx_bytes"], "port_id")
            drop = _delta_ratio(up, ["tx_drops", "rx_drops"], ["tx_bytes", "rx_bytes"], "port_id")
            dom = up.drop_duplicates("port_id").set_index("port_id")["domain_id"]
            f["port_err_rate_max_domain"] = float(err.groupby(dom).mean().max())
            f["port_drop_rate_max_domain"] = float(drop.groupby(dom).mean().max())
            f["port_util_max"] = float(up["utilisation_pct"].max())
            f["port_queue_max"] = float(up["queue_depth"].max())

        agg = self.frame_in(self.agg, t0, t1)
        if len(agg):
            f["agg_oversub_max"] = float(agg["oversubscription_ratio"].max())
            f["agg_uplink_util_max"] = float(agg["uplink_utilisation_pct"].max())
            f["agg_queue_max"] = float(agg["max_queue_depth"].max())
            f["agg_congested_frac"] = float(agg["congested"].astype(float).mean())

        st = self.frame_in(self.storage, t0, t1)
        if len(st):
            f["storage_write_lat_mean"] = float(st["write_latency_ms"].mean())
            f["storage_write_lat_max"] = float(st["write_latency_ms"].max())
            f["storage_dirty_backlog_mean"] = float(st["dirty_backlog_gb"].mean())
            f["storage_util_mean"] = float(st["utilisation_pct"].mean())
            f["storage_active_frac"] = float((st["throughput_gbps"] > 0).mean())

        nd = self.frame_in(self.node, t0, t1)
        if len(nd):
            f["node_io_pressure_mean"] = float(nd["io_pressure"].mean())
            f["node_io_pressure_max"] = float(nd["io_pressure"].max())
            f["node_cpu_pressure_mean"] = float(nd["cpu_pressure"].mean())
            f["node_mem_pressure_mean"] = float(nd["memory_pressure"].mean())

        # absolute features: mesh identifiers, kept out of the primary set
        f["abs_iter_time_s"] = mean_it
        f["abs_compute_mean_s"] = float(job["compute_mean_s"].mean())
        return f

    # -- entity-level features ---------------------------------------------
    def gpu_features(self, s: int, e: int) -> pd.DataFrame:
        i0, i1 = np.searchsorted(self.iters, s), np.searchsorted(self.iters, e)
        busy, ex = self.busy[i0:i1], self.excess[i0:i1]
        mean_it = float(self.iter_time[i0:i1].mean())
        pacer = np.argmax(busy, axis=1)
        duty = np.bincount(pacer, minlength=busy.shape[1]) / len(pacer)
        by_rank = pd.DataFrame({
            "gpu_id": self.rank_gpu,
            "e_pace_duty": duty,
            "e_excess_mean": ex.mean(axis=0),
            "e_excess_max": ex.max(axis=0),
            "e_wait_rel": self.wait[i0:i1].mean(axis=0) / mean_it,
            "e_compute_cv": _cv(self.compute[i0:i1], axis=0),
        })
        t0, t1 = self.wall_span(s, e)
        r = self.tel_rows(t0, t1)
        cols = {"gpu_id": self.gpu_ids, "rack_id": self.gpu_rack, "node_id": self.gpu_node}
        if r.stop - r.start >= 1:
            occ = self.occ[r].mean(axis=0)
            temp = self.temp[r].mean(axis=0)
            gap = (self.util[r] - self.occ[r]).mean(axis=0)
            power, clock = self.power[r].mean(axis=0), self.clock[r].mean(axis=0)
            rack_med_occ = np.zeros_like(occ)
            rack_med_temp = np.zeros_like(temp)
            for m in self.rack_mask:
                rack_med_occ[m] = np.median(occ[m])
                rack_med_temp[m] = np.median(temp[m])
            cols.update({
                "e_occ_minus_fleet_med": occ - np.median(occ),
                "e_occ_minus_rack_med": occ - rack_med_occ,
                "e_temp_minus_rack_med": temp - rack_med_temp,
                "e_temp_minus_fleet_med": temp - np.median(temp),
                "e_rack_med_temp_minus_fleet_med": rack_med_temp - np.median(temp),
                "e_power_rel_fleet_med": power / np.median(power) - 1,
                "e_clock_rel_fleet_med": clock / np.median(clock) - 1,
                "e_util_occ_gap_minus_fleet_med": gap - np.median(gap),
                "e_throttled_frac": self.throttled[r].mean(axis=0),
            })
        tel = pd.DataFrame(cols)
        out = tel.merge(by_rank, on="gpu_id", how="left")
        nic = self.frame_in(self.nic, t0, t1)
        if len(nic) >= 2 * nic["nic_id"].nunique():
            err = _delta_ratio(nic, ["tx_errors", "rx_errors"], ["tx_bytes", "rx_bytes"], "nic_id")
            node_of = nic.drop_duplicates("nic_id").set_index("nic_id")["node_id"]
            node_err = err.groupby(node_of).mean()
            tx = nic.groupby("node_id")["tx_gbps"].mean()
            out["e_node_nic_err_rate"] = out["node_id"].map(node_err).to_numpy()
            out["e_node_tx_rel_med"] = (out["node_id"].map(tx) / tx.median() - 1).to_numpy()
        return out

    def rack_features(self, s: int, e: int) -> pd.DataFrame:
        i0, i1 = np.searchsorted(self.iters, s), np.searchsorted(self.iters, e)
        busy = self.busy[i0:i1]
        pacer_gpu = self.rank_gpu[np.argmax(busy, axis=1)]
        pacer_rack = np.array([g[: g.index("n")] for g in pacer_gpu])
        halo_rank = self.halo[i0:i1].mean(axis=0)
        rank_rack = np.array([g[: g.index("n")] for g in self.rank_gpu])
        t0, t1 = self.wall_span(s, e)
        r = self.tel_rows(t0, t1)
        rows = []
        up_all = self.frame_in(self.uplinks, t0, t1)
        agg_all = self.frame_in(self.agg, t0, t1)
        nic_all = self.frame_in(self.nic, t0, t1)
        temp = self.temp[r] if r.stop > r.start else None
        rack_means = None
        if temp is not None:
            rack_means = np.stack([temp[:, m].mean(axis=1) for m in self.rack_mask])
        for k, rack in enumerate(self.racks):
            row: dict[str, Any] = {"rack_id": rack}
            if rack_means is not None:
                others = np.delete(rack_means, k, axis=0)
                row["k_temp_minus_others_med"] = float((rack_means[k] - np.median(others, axis=0)).mean())
                row["k_temp_rise"] = float(rack_means[k, -1] - rack_means[k, 0])
                row["k_throttled_frac"] = float(self.throttled[r][:, self.rack_mask[k]].mean())
            up = up_all[up_all["domain_id"] == rack]
            if len(up) >= 2 * up["port_id"].nunique() and len(up):
                row["k_leaf_down_frac"] = float(1.0 - up["link_up"].astype(float).mean())
                row["k_leaf_err_rate"] = float(_delta_ratio(up, ["tx_errors", "rx_errors"],
                                                            ["tx_bytes", "rx_bytes"], "port_id").mean())
                row["k_leaf_drop_rate"] = float(_delta_ratio(up, ["tx_drops", "rx_drops"],
                                                             ["tx_bytes", "rx_bytes"], "port_id").mean())
                row["k_leaf_queue_max"] = float(up["queue_depth"].max())
                row["k_leaf_util_max"] = float(up["utilisation_pct"].max())
            ag = agg_all[agg_all["domain_id"] == rack]
            if len(ag):
                row["k_leaf_oversub"] = float(ag["oversubscription_ratio"].max())
            nic = nic_all[nic_all["rack_id"] == rack]
            if len(nic) >= 2 * nic["nic_id"].nunique() and len(nic):
                row["k_nic_err_rate"] = float(_delta_ratio(nic, ["tx_errors", "rx_errors"],
                                                           ["tx_bytes", "rx_bytes"], "nic_id").mean())
            row["k_pace_share"] = float((pacer_rack == rack).mean())
            row["k_halo_rel"] = float(halo_rank[rank_rack == rack].mean() / halo_rank.mean())
            rows.append(row)
        return pd.DataFrame(rows)


def build_features(args: argparse.Namespace) -> None:
    runs_dir: Path = args.runs_dir
    W, stride = args.window, args.ttd_stride
    run_dirs = sorted(d for d in runs_dir.iterdir() if (d / "summary.json").exists())
    win_rows, gpu_rows, rack_rows = [], [], []
    t_all = time.perf_counter()
    for n, d in enumerate(run_dirs, 1):
        frames, summary = read_run(d)
        scenario, mesh, seed = summary["scenario"], summary["mesh"], int(summary["seed"])
        truth = _onset_and_targets(frames, scenario)
        rt = RunTables(frames)
        last = int(rt.iters[-1])
        starts = list(range(1, last - W + 2, stride))
        for s in starts:
            e = s + W
            label, frac = _window_label(scenario, s, e, truth)
            base = {"run_id": d.name, "scenario": scenario, "mesh": mesh, "seed": seed,
                    "start": s, "end": e, "label": label, "fault_frac": frac,
                    "train_stride": ((s - 1) % args.stride == 0)}
            t0, t1 = rt.wall_span(s, e)
            base["t0"], base["t1"] = t0, t1
            win_rows.append({**base, **rt.window_features(s, e)})
            # entity tables only on the training stride, and only for windows
            # that carry a clean label
            if base["train_stride"] and label != "straddle":
                g = rt.gpu_features(s, e)
                pos = np.zeros(len(g), dtype=bool)
                if label == "straggler":
                    active = {ep["gpu_id"] for ep in truth["episodes"]
                              if min(ep["end"], e) > max(ep["start"], s)}
                    pos = g["gpu_id"].isin(active).to_numpy()
                elif label == "gpu_degradation":
                    pos = g["gpu_id"].isin(truth["gpus"]).to_numpy()
                elif label == "thermal":
                    pos = (g["rack_id"] == truth["rack"]).to_numpy()
                g.insert(0, "positive", pos)
                for k, v in base.items():
                    g[k] = v
                gpu_rows.append(g)
                k_ = rt.rack_features(s, e)
                k_.insert(0, "positive", (k_["rack_id"] == truth["rack"]).to_numpy()
                          if label in ("thermal", "network_domain") else False)
                for k, v in base.items():
                    k_[k] = v
                rack_rows.append(k_)
        print(f"[{n:3d}/{len(run_dirs)}] {d.name:38s} {len(starts)} windows  "
              f"{time.perf_counter() - t_all:5.0f}s", flush=True)
    out = runs_dir / "ml"
    out.mkdir(exist_ok=True)
    pd.DataFrame(win_rows).to_parquet(out / "features_window.parquet", index=False)
    pd.concat(gpu_rows, ignore_index=True).to_parquet(out / "features_gpu.parquet", index=False)
    pd.concat(rack_rows, ignore_index=True).to_parquet(out / "features_rack.parquet", index=False)
    print(f"features: {len(win_rows)} windows, {sum(len(x) for x in gpu_rows)} gpu rows, "
          f"{time.perf_counter() - t_all:.0f}s")

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

META = {"run_id", "scenario", "mesh", "seed", "start", "end", "label", "fault_frac",
        "train_stride", "t0", "t1", "positive", "gpu_id", "rack_id", "node_id"}
TABLE_PREFIXES = ["job", "rank", "gpu", "nic", "port", "agg", "storage", "node"]
#  Checkpoint time is a wall-clock cost (bytes over a shared filesystem) divided
#  by an iteration time that differs 20x between meshes, so the ratio identifies
#  the mesh rather than the workload phase. `job_output_duty` -- the fraction of
#  iterations that wrote output -- carries the same information and transfers.
WALLTIME_RATIOS = {"job_ckpt_rel"}
ABLATIONS = {
    "full": (),
    "-throttle": ("gpu_throttle", "gpu_throttled"),
    "-rank_performance": ("rank_",),
    "-fabric": ("nic_", "port_", "agg_"),
    "-gpu_telemetry": ("gpu_",),
}


def _cols(df: pd.DataFrame, drop: tuple[str, ...] = (), absolute: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in META and c not in WALLTIME_RATIOS
            and (absolute or not c.startswith("abs_"))]
    return [c for c in cols if not any(c.startswith(p) for p in drop)]


def _hgb(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=seed)


def _logreg():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))


def _scores(y_true, y_pred) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix, f1_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).astype(float)
    row = cm.sum(axis=1, keepdims=True)
    is_fault_t = np.isin(y_true, list(FAULT_CLASSES))
    is_fault_p = np.isin(y_pred, list(FAULT_CLASSES))
    tp = (is_fault_t & is_fault_p).sum()
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": {c: float(v) for c, v in zip(CLASSES, per_class)},
        "confusion": (cm / np.where(row == 0, 1, row)).round(4).tolist(),
        "confusion_counts": cm.astype(int).tolist(),
        "fault_precision": float(tp / max(is_fault_p.sum(), 1)),
        "fault_recall": float(tp / max(is_fault_t.sum(), 1)),
        "false_alarm_rate": float((is_fault_p & ~is_fault_t).sum() / max((~is_fault_t).sum(), 1)),
        "n": int(len(y_true)),
    }


def _fit_eval(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], model=None) -> tuple[dict, Any]:
    model = model or _hgb()
    model.fit(tr[cols], tr["label"])
    return _scores(te["label"], model.predict(te[cols])), model


def _rules_class(diag: dict[str, Any]) -> str:
    v, tier = diag.get("verdict"), diag.get("tier")
    if v == "NOMINAL":
        return "healthy"
    if v == "WORKLOAD_CHANGE":
        return "phase_change"
    if tier == "gpu":
        return "gpu_degradation"
    if tier == "rank":
        return "straggler"
    if tier == "rack":
        return "network_domain" if "domains" in diag.get("localisation", {}) else "thermal"
    return "unknown"


def _rollup(pred: pd.Series, min_windows: int = 3) -> str:
    counts = pred[pred != "healthy"].value_counts()
    if len(counts) and counts.iloc[0] >= min_windows:
        return str(counts.index[0])
    return "healthy"


def _run_truth(runs_dir: Path, run_id: str) -> dict[str, Any]:
    frames, summary = read_run(runs_dir / run_id)
    truth = _onset_and_targets(frames, summary["scenario"])
    truth["cohort"] = {ep["gpu_id"] for ep in truth["episodes"]}
    truth["diagnosis"] = summary["diagnosis"]
    return truth


def _rules_localised(diag: dict[str, Any], scenario: str, truth: dict[str, Any]) -> bool | None:
    loc = diag.get("localisation", {})
    if scenario == "thermal":
        return truth["rack"] in loc.get("racks", [])
    if scenario == "network_domain":
        return truth["rack"] in loc.get("domains", [])
    if scenario == "gpu_degradation":
        return bool(truth["gpus"] & set(loc.get("gpus", [])))
    if scenario == "straggler":
        return bool(truth["cohort"] & set(loc.get("gpus", [])))
    return None


def _permutation_importance(model, te: pd.DataFrame, cols: list[str], n_repeats: int = 5,
                            seed: int = 0) -> dict[str, Any]:
    """Macro-F1 drop and per-class F1 drop when one feature is shuffled."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    X = te[cols].to_numpy(dtype=float)
    y = te["label"].to_numpy()
    base_all = f1_score(y, model.predict(te[cols]), labels=CLASSES, average=None, zero_division=0)
    out = {}
    for j, c in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            f = f1_score(y, model.predict(pd.DataFrame(Xp, columns=cols)), labels=CLASSES,
                         average=None, zero_division=0)
            drops.append(base_all - f)
        d = np.mean(drops, axis=0)
        out[c] = {"macro": float(d.mean()), "per_class": {k: float(v) for k, v in zip(CLASSES, d)}}
    by_table = {}
    for p in TABLE_PREFIXES:
        vals = [v["macro"] for c, v in out.items() if c.startswith(p + "_")]
        by_table[p] = float(sum(vals)) if vals else 0.0
    top_per_class = {k: sorted(((v["per_class"][k], c) for c, v in out.items()), reverse=True)[:4]
                     for k in CLASSES}
    return {"per_feature": out, "by_table": by_table,
            "top_per_class": {k: [[c, round(v, 4)] for v, c in lst] for k, lst in top_per_class.items()}}


def _ranking_metrics(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    """Per-window ranking quality of an entity score, averaged per scenario."""
    from sklearn.metrics import roc_auc_score
    rows = {}
    for scenario, sub in df.groupby("label"):
        if scenario not in FAULT_CLASSES:
            continue
        p1, pk, auc = [], [], []
        for _, w in sub.groupby(["run_id", "start"]):
            pos = w["positive"].to_numpy()
            if pos.sum() == 0 or pos.all():
                continue
            order = np.argsort(-w[score_col].to_numpy())
            k = int(pos.sum())
            p1.append(float(pos[order[0]]))
            pk.append(float(pos[order[:k]].mean()))
            auc.append(float(roc_auc_score(pos, w[score_col])))
        if p1:
            rows[scenario] = {"precision_at_1": float(np.mean(p1)), "precision_at_k": float(np.mean(pk)),
                              "auroc": float(np.mean(auc)), "n_windows": len(p1)}
    return rows


def train(args: argparse.Namespace) -> None:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    runs_dir: Path = args.runs_dir
    out_dir: Path = args.out or (runs_dir / "ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name: str) -> None:
        timing[name] = round(time.perf_counter() - t0 - sum(timing.values()), 1)

    win = pd.read_parquet(out_dir / "features_window.parquet")
    labelled = win[win["train_stride"] & (win["label"] != "straddle")].reset_index(drop=True)
    test_seeds = set(args.test_seeds)
    is_test = labelled["seed"].isin(test_seeds)
    tr, te = labelled[~is_test], labelled[is_test]
    cols = _cols(labelled)
    metrics: dict[str, Any] = {
        "dataset": {
            "runs": int(win["run_id"].nunique()),
            "seeds": sorted(int(s) for s in win["seed"].unique()),
            "test_seeds": sorted(test_seeds),
            "window_iterations": int(win["end"].iloc[0] - win["start"].iloc[0]),
            "windows_labelled": int(len(labelled)),
            "windows_train": int(len(tr)), "windows_test": int(len(te)),
            "windows_ttd": int(len(win)),
            "class_counts": {c: int((labelled["label"] == c).sum()) for c in CLASSES},
            "class_by_mesh": {m: {c: int(((labelled["label"] == c) & (labelled["mesh"] == m)).sum())
                                  for c in CLASSES} for m in MESHES},
            "n_features": len(cols), "features": cols,
        }
    }
    print(f"windows: {len(labelled)} labelled ({len(tr)} train / {len(te)} test), {len(cols)} features")

    # -- 1. seed holdout -----------------------------------------------------
    seed_hold, hgb = _fit_eval(tr, te, cols)
    metrics["seed_holdout"] = seed_hold
    metrics["seed_holdout_logreg"], _ = _fit_eval(tr, te, cols, _logreg())
    print(f"seed holdout  HGB macro-F1 {seed_hold['macro_f1']:.3f}   "
          f"logreg {metrics['seed_holdout_logreg']['macro_f1']:.3f}")

    # -- 2. cross-validation by seed ------------------------------------------
    f1s = []
    for tri, tei in GroupKFold(n_splits=labelled["seed"].nunique()).split(labelled, groups=labelled["seed"]):
        s, _ = _fit_eval(labelled.iloc[tri], labelled.iloc[tei], cols)
        f1s.append(s["macro_f1"])
    metrics["cv_by_seed"] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                             "folds": [float(x) for x in f1s]}
    lap("classifier")

    # -- 3. mesh holdout ------------------------------------------------------
    mesh_hold = {}
    for name, train_m, test_m in [("coarse+fine->medium", ["coarse", "fine"], "medium"),
                                  ("coarse+medium->fine", ["coarse", "medium"], "fine"),
                                  ("medium+fine->coarse", ["medium", "fine"], "coarse")]:
        a, b = labelled[labelled["mesh"].isin(train_m)], labelled[labelled["mesh"] == test_m]
        s_rel, _ = _fit_eval(a, b, cols)
        s_abs, _ = _fit_eval(a, b, _cols(labelled, absolute=True))
        s_lr, _ = _fit_eval(a, b, cols, _logreg())
        mesh_hold[name] = {"macro_f1": s_rel["macro_f1"], "per_class_f1": s_rel["per_class_f1"],
                           "macro_f1_with_absolute": s_abs["macro_f1"],
                           "macro_f1_logreg": s_lr["macro_f1"], "per_class_f1_logreg": s_lr["per_class_f1"]}
        print(f"mesh holdout {name:22s} GBT {s_rel['macro_f1']:.3f}  (+absolute {s_abs['macro_f1']:.3f})"
              f"  logreg {s_lr['macro_f1']:.3f}")
    metrics["mesh_holdout"] = mesh_hold

    # -- 4. ablations ---------------------------------------------------------
    if not args.no_ablations:
        abl = {}
        for name, drop in ABLATIONS.items():
            s, _ = _fit_eval(tr, te, _cols(labelled, drop=drop))
            abl[name] = {"macro_f1": s["macro_f1"], "per_class_f1": s["per_class_f1"]}
        s, _ = _fit_eval(tr, te, _cols(labelled, absolute=True))
        abl["+absolute"] = {"macro_f1": s["macro_f1"], "per_class_f1": s["per_class_f1"]}
        s, _ = _fit_eval(tr, te, cols + sorted(WALLTIME_RATIOS))
        abl["+ckpt_rel"] = {"macro_f1": s["macro_f1"], "per_class_f1": s["per_class_f1"]}
        perm = tr.copy()
        perm["label"] = np.random.default_rng(0).permutation(perm["label"].to_numpy())
        s, _ = _fit_eval(perm, te, cols)
        abl["label_permutation"] = {"macro_f1": s["macro_f1"], "per_class_f1": s["per_class_f1"]}
        metrics["ablations"] = abl
        print("ablations: " + "  ".join(f"{k} {v['macro_f1']:.3f}" for k, v in abl.items()))
    lap("ablations")

    # -- 5. streaming predictions on every stride-10 window of the test seeds --
    stream = win[win["seed"].isin(test_seeds)].copy()
    stream["pred"] = hgb.predict(stream[cols])
    proba = hgb.predict_proba(stream[cols])
    fault_idx = [list(hgb.classes_).index(c) for c in FAULT_CLASSES if c in hgb.classes_]
    stream["p_fault"] = proba[:, fault_idx].sum(axis=1)

    # -- 6. run level: ML roll-up vs the rule set on the same held-out runs ----
    truths = {rid: _run_truth(runs_dir, rid) for rid in stream["run_id"].unique()}
    run_rows = []
    for rid, g in stream.groupby("run_id"):
        scenario, mesh = g["scenario"].iloc[0], g["mesh"].iloc[0]
        truth = truths[rid]
        ml = _rollup(g["pred"])
        rules = _rules_class(truth["diagnosis"])
        run_rows.append({"run_id": rid, "scenario": scenario, "mesh": mesh, "seed": int(g["seed"].iloc[0]),
                         "ml": ml, "rules": rules, "ml_correct": ml == scenario,
                         "rules_correct": rules == scenario,
                         "rules_localised": _rules_localised(truth["diagnosis"], scenario, truth),
                         "onset": truth["onset"], "target_rack": truth["rack"],
                         "target_gpus": sorted(truth["gpus"] | truth["cohort"])})
    runs_df = pd.DataFrame(run_rows)
    per_scn = {}
    for scenario, g in runs_df.groupby("scenario"):
        per_scn[scenario] = {"n": int(len(g)), "ml_correct": int(g["ml_correct"].sum()),
                             "rules_correct": int(g["rules_correct"].sum()),
                             "rules_localised": int(g["rules_localised"].fillna(False).astype(bool).sum())
                             if scenario in FAULT_CLASSES else None}
    sensitivity = {}
    for k in (2, 3, 5):
        sensitivity[k] = float(np.mean([_rollup(g["pred"], k) == g["scenario"].iloc[0]
                                        for _, g in stream.groupby("run_id")]))
    metrics["run_level"] = {"per_scenario": per_scn, "ml_accuracy": float(runs_df["ml_correct"].mean()),
                            "rules_accuracy": float(runs_df["rules_correct"].mean()),
                            "rollup_min_windows": 3, "rollup_sensitivity": sensitivity,
                            "runs": run_rows}
    print(f"run level: ML {runs_df['ml_correct'].sum()}/{len(runs_df)}  "
          f"rules {runs_df['rules_correct'].sum()}/{len(runs_df)}")

    # -- 7. time to detect ----------------------------------------------------
    ttd_rows = []
    for rid, g in stream.groupby("run_id"):
        g = g.sort_values("end")
        scenario = g["scenario"].iloc[0]
        truth = truths[rid]
        if scenario == "healthy":
            continue
        onset = truth["onset"] if truth["onset"] is not None else 1
        cand = g[g["end"] >= onset]
        hit = (cand["pred"] == scenario).to_numpy()
        det = None
        for i in range(len(hit) - 1):
            if hit[i] and hit[i + 1]:
                det = cand.iloc[i]
                break
        onset_t = float(np.interp(onset, g["start"], g["t0"]))
        ttd_rows.append({"run_id": rid, "scenario": scenario, "mesh": g["mesh"].iloc[0],
                         "seed": int(g["seed"].iloc[0]), "onset": int(onset),
                         "detected": det is not None,
                         "latency_iterations": int(det["end"] - onset) if det is not None else None,
                         "latency_s": float(det["t1"] - onset_t) if det is not None else None,
                         "run_length_s": float(g["t1"].max())})
    ttd_df = pd.DataFrame(ttd_rows)
    ttd_summary = {}
    for (scenario, mesh), g in ttd_df.groupby(["scenario", "mesh"]):
        d = g[g["detected"]]
        ttd_summary[f"{scenario}/{mesh}"] = {
            "detected": int(g["detected"].sum()), "n": int(len(g)),
            "median_iterations": float(d["latency_iterations"].median()) if len(d) else None,
            "median_s": float(d["latency_s"].median()) if len(d) else None,
            "max_s": float(d["latency_s"].max()) if len(d) else None,
        }
    fa = {}
    for scenario in ("healthy", "phase_change"):
        g = stream[stream["scenario"] == scenario]
        fa[scenario] = float(g["pred"].isin(FAULT_CLASSES).mean())
    metrics["time_to_detect"] = {"summary": ttd_summary, "runs": ttd_rows,
                                 "false_alarm_window_rate": fa, "rule": "2 consecutive windows"}
    lap("run_level_ttd")

    # -- 8. unsupervised, window level ---------------------------------------
    healthy_tr = tr[tr["label"] == "healthy"]
    imp = SimpleImputer(strategy="median").fit(healthy_tr[cols])
    Xh, Xt = imp.transform(healthy_tr[cols]), imp.transform(te[cols])
    iso = IsolationForest(n_estimators=300, random_state=0).fit(Xh)
    score_te = -iso.score_samples(Xt)
    thr = np.percentile(-iso.score_samples(Xh), 95)
    scaler = StandardScaler().fit(Xh)
    pca = PCA(n_components=0.95, random_state=0).fit(scaler.transform(Xh))
    Zt = scaler.transform(Xt)
    resid = np.sqrt(((Zt - pca.inverse_transform(pca.transform(Zt))) ** 2).mean(axis=1))
    y_fault = te["label"].isin(FAULT_CLASSES).to_numpy()
    unsup = {"isolation_forest": {"auroc_is_fault": float(roc_auc_score(y_fault, score_te)),
                                  "flag_rate_by_class": {c: float((score_te[(te["label"] == c).to_numpy()] > thr).mean())
                                                         for c in CLASSES}},
             "pca_residual": {"auroc_is_fault": float(roc_auc_score(y_fault, resid)),
                              "n_components": int(pca.n_components_)},
             "threshold": "95th percentile of healthy training windows"}
    metrics["unsupervised_window"] = unsup
    print("unsupervised AUROC(is_fault): iforest %.3f  pca %.3f  | phase_change flagged %.0f%%" % (
        unsup["isolation_forest"]["auroc_is_fault"], unsup["pca_residual"]["auroc_is_fault"],
        100 * unsup["isolation_forest"]["flag_rate_by_class"]["phase_change"]))
    lap("unsupervised")

    # -- 9. localisation ------------------------------------------------------
    gpu = pd.read_parquet(out_dir / "features_gpu.parquet")
    gcols = _cols(gpu)
    gtr, gte = gpu[~gpu["seed"].isin(test_seeds)], gpu[gpu["seed"].isin(test_seeds)].copy()
    ghgb = _hgb().fit(gtr[gcols], gtr["positive"].astype(int))
    gte["s_hgb"] = ghgb.predict_proba(gte[gcols])[:, 1]
    gh = gtr[gtr["label"] == "healthy"]
    gh = gh.sample(min(len(gh), 60000), random_state=0)
    gimp = SimpleImputer(strategy="median").fit(gh[gcols])
    Gh, Gt = gimp.transform(gh[gcols]), gimp.transform(gte[gcols])
    giso = IsolationForest(n_estimators=200, random_state=0).fit(Gh)
    gte["s_iforest"] = -giso.score_samples(Gt)
    gsc = StandardScaler().fit(Gh)
    gpca = PCA(n_components=0.95, random_state=0).fit(gsc.transform(Gh))
    Zg = gsc.transform(Gt)
    gte["s_pca"] = np.sqrt(((Zg - gpca.inverse_transform(gpca.transform(Zg))) ** 2).mean(axis=1))
    loc_gpu = {m: _ranking_metrics(gte, f"s_{m}") for m in ("hgb", "iforest", "pca")}
    # run-level localisation: average the supervised score over post-onset windows
    ml_loc: dict[str, bool] = {}
    for rid, g in gte[gte["label"].isin(FAULT_CLASSES)].groupby("run_id"):
        scenario = g["label"].iloc[0]
        truth = truths.get(rid) or _run_truth(runs_dir, rid)
        mean_s = g.groupby("gpu_id")["s_hgb"].mean().sort_values(ascending=False)
        if scenario == "thermal":
            ml_loc[rid] = bool(g.groupby("rack_id")["s_hgb"].mean().idxmax() == truth["rack"])
        elif scenario == "gpu_degradation":
            ml_loc[rid] = bool(mean_s.index[0] in truth["gpus"])
        elif scenario == "straggler":
            ml_loc[rid] = bool(mean_s.index[0] in truth["cohort"])
    rack = pd.read_parquet(out_dir / "features_rack.parquet")
    kcols = _cols(rack)
    ktr, kte = rack[~rack["seed"].isin(test_seeds)], rack[rack["seed"].isin(test_seeds)].copy()
    khgb = _hgb().fit(ktr[kcols], ktr["positive"].astype(int))
    kte["s_hgb"] = khgb.predict_proba(kte[kcols])[:, 1]
    kh = ktr[ktr["label"] == "healthy"]
    kimp = SimpleImputer(strategy="median").fit(kh[kcols])
    Kh, Kt = kimp.transform(kh[kcols]), kimp.transform(kte[kcols])
    kiso = IsolationForest(n_estimators=200, random_state=0).fit(Kh)
    kte["s_iforest"] = -kiso.score_samples(Kt)
    ksc = StandardScaler().fit(Kh)
    kpca = PCA(n_components=0.95, random_state=0).fit(ksc.transform(Kh))
    Zk = ksc.transform(Kt)
    kte["s_pca"] = np.sqrt(((Zk - kpca.inverse_transform(kpca.transform(Zk))) ** 2).mean(axis=1))
    rack_top1: dict[str, Any] = {}
    for m in ("hgb", "iforest", "pca"):
        rack_top1[m] = {}
        for scenario in ("thermal", "network_domain"):
            sub = kte[kte["label"] == scenario]
            hits = [bool(w.loc[w[f"s_{m}"].idxmax(), "positive"]) for _, w in sub.groupby(["run_id", "start"])]
            rack_top1[m][scenario] = {"top1_accuracy": float(np.mean(hits)) if hits else None,
                                      "n_windows": len(hits)}
    for rid, g in kte[kte["label"] == "network_domain"].groupby("run_id"):
        ml_loc[rid] = bool(g.groupby("rack_id")["s_hgb"].mean().idxmax() == truths[rid]["rack"])
    loc_run = {}
    for scenario in sorted(FAULT_CLASSES):
        rids = [r for r in ml_loc if r.startswith(scenario + "__")]
        loc_run[scenario] = {"ml_localised": int(sum(ml_loc[r] for r in rids)), "n": len(rids),
                             "rules_localised": per_scn[scenario]["rules_localised"]}
    metrics["localisation"] = {"gpu": loc_gpu, "rack": rack_top1, "run_level": loc_run,
                               "gpu_features": gcols, "rack_features": kcols,
                               "gpu_rows_train": int(len(gtr)), "gpu_rows_test": int(len(gte))}
    print("localisation p@1 (hgb): " + "  ".join(f"{k} {v['precision_at_1']:.2f}" for k, v in loc_gpu["hgb"].items()))
    lap("localisation")

    # -- 10. importance -------------------------------------------------------
    metrics["importance"] = _permutation_importance(hgb, te, cols)
    lap("importance")

    # -- 11. figures: one held-out medium run per scenario --------------------
    traces = {}
    keys = ["gpu_rack_temp_drift", "rank_halo_disp", "rank_compute_disp", "rank_pace_top1",
            "rank_pace_entropy", "storage_write_lat_mean", "gpu_clock_min_rel", "port_down_max_domain",
            "gpu_throttle_reliab_frac", "job_spread_rel", "p_fault"]
    for scenario in CLASSES:
        g = stream[(stream["scenario"] == scenario) & (stream["mesh"] == "medium")]
        if g.empty:
            continue
        rid = sorted(g["run_id"].unique())[0]
        g = g[g["run_id"] == rid].sort_values("start")
        traces[scenario] = {"run_id": rid, "onset": truths[rid]["onset"],
                            "start": g["start"].astype(int).tolist(), "end": g["end"].astype(int).tolist(),
                            "label": g["label"].tolist(), "pred": g["pred"].tolist(),
                            **{k: [None if pd.isna(v) else round(float(v), 5) for v in g[k]] for k in keys}}
    example = None
    sg = gte[gte["label"] == "straggler"]
    if len(sg):
        rid = sorted(sg["run_id"].unique())[0]
        w = sg[sg["run_id"] == rid]
        w = w[w["start"] == w["start"].iloc[len(w) // 2]].sort_values("s_hgb", ascending=False)
        example = {"run_id": rid, "start": int(w["start"].iloc[0]),
                   "ranked": [[g_, round(float(s_), 4), bool(p_)] for g_, s_, p_ in
                              zip(w["gpu_id"].head(10), w["s_hgb"].head(10), w["positive"].head(10))],
                   "n_positive": int(w["positive"].sum())}
    figures = {"traces": traces, "straggler_ranking_example": example}

    timing["total"] = round(time.perf_counter() - t0, 1)
    metrics["timing_s"] = timing
    try:
        import subprocess
        metrics["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                        cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        metrics["git_commit"] = None
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True, default=str))
    (out_dir / "figures.json").write_text(json.dumps(figures, indent=1, default=str))
    print(f"wrote {out_dir / 'metrics.json'}  ({timing['total']:.0f}s)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs_ml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="simulate the training matrix")
    g.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    g.add_argument("--meshes", nargs="+", default=MESHES)
    g.add_argument("--scenarios", nargs="+", default=CLASSES)
    g.add_argument("--no-randomise", action="store_true",
                   help="keep the yaml's fixed targets and onsets")
    g.add_argument("--force", action="store_true", help="re-simulate cached runs")

    f = sub.add_parser("features", help="window the runs into feature tables")
    f.add_argument("--window", type=int, default=100, help="iterations per window")
    f.add_argument("--stride", type=int, default=25)
    f.add_argument("--ttd-stride", type=int, default=10, help="stride for time-to-detect")

    t = sub.add_parser("train", help="fit, evaluate, write metrics.json")
    t.add_argument("--test-seeds", type=int, nargs="+", default=[3, 4])
    t.add_argument("--out", type=Path, default=None, help="default <runs-dir>/ml")
    t.add_argument("--no-ablations", action="store_true")

    a = sub.add_parser("all", help="generate -> features -> train")
    for p in (a,):
        p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
        p.add_argument("--meshes", nargs="+", default=MESHES)
        p.add_argument("--scenarios", nargs="+", default=CLASSES)
        p.add_argument("--no-randomise", action="store_true")
        p.add_argument("--force", action="store_true")
        p.add_argument("--window", type=int, default=100)
        p.add_argument("--stride", type=int, default=25)
        p.add_argument("--ttd-stride", type=int, default=10)
        p.add_argument("--test-seeds", type=int, nargs="+", default=[3, 4])
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--no-ablations", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in ("generate", "all"):
        generate(args)
    if args.cmd in ("features", "all"):
        build_features(args)
    if args.cmd in ("train", "all"):
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Telemetry schemas and persistence.

Nine tables per run, written as Parquet. The schema lives here rather than being
implied by whatever the samplers happened to emit, so a missing or renamed
column fails loudly instead of silently disappearing from the dashboard.

Column-order and dtype normalisation also make the reproducibility test
meaningful: two runs at the same seed must serialise byte-identically, which
they cannot do if column order depends on dict insertion order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

#: table -> ordered columns. Also the contract TELEMETRY.md documents.
SCHEMAS: dict[str, list[str]] = {
    "telemetry_gpu": [
        "scenario", "seed", "timestamp", "gpu_id", "node_id", "rack_id",
        "utilization_pct", "sm_occupancy_pct", "memory_used_gb", "memory_total_gb",
        "power_w", "temperature_c", "clock_mhz", "throttled", "throttle_reason",
    ],
    "telemetry_node": [
        "scenario", "seed", "timestamp", "node_id", "rack_id",
        "cpu_pressure", "memory_pressure", "io_pressure",
    ],
    "telemetry_nic": [
        "scenario", "seed", "timestamp", "node_id", "rack_id", "nic_id",
        "capacity_gbps", "tx_gbps", "rx_gbps", "tx_bytes", "rx_bytes",
        "tx_errors", "rx_errors", "tx_drops", "rx_drops",
    ],
    "telemetry_switch_port": [
        "scenario", "seed", "timestamp", "switch_id", "switch_tier", "domain_id",
        "port_id", "port_role", "peer_id", "capacity_gbps", "link_up",
        "tx_gbps", "rx_gbps", "tx_bytes", "rx_bytes",
        "tx_errors", "rx_errors", "tx_drops", "rx_drops",
        "utilisation_pct", "queue_depth",
    ],
    "telemetry_switch_aggregate": [
        "scenario", "seed", "timestamp", "switch_id", "switch_tier", "domain_id",
        "aggregate_tx_gbps", "aggregate_rx_gbps", "uplink_utilisation_pct",
        "oversubscription_ratio", "max_queue_depth", "congested",
    ],
    "telemetry_storage": [
        "scenario", "seed", "timestamp", "backend_id",
        "read_latency_ms", "write_latency_ms", "throughput_gbps",
        "queue_depth", "utilisation_pct", "dirty_backlog_gb",
    ],
    "rank_performance": [
        "scenario", "seed", "iteration", "rank_id", "gpu_id",
        "compute_time_s", "halo_wait_s", "allreduce_wait_s", "checkpoint_time_s",
        "total_time_s", "is_straggler",
    ],
    "job_performance": [
        "scenario", "seed", "iteration", "timestamp", "iteration_time_s",
        "compute_max_s", "compute_mean_s", "halo_max_s", "halo_mean_s",
        "allreduce_s", "checkpoint_s", "slowest_rank_id", "fastest_rank_id",
        "rank_spread_s", "sync_overhead_s", "wait_total_s", "straggler_count",
        "throughput_iters_per_s", "cumulative_runtime_s",
    ],
    "events": [
        "scenario", "seed", "timestamp", "event_type", "rank_id", "gpu_id", "payload",
    ],
}

#: Columns that are legitimately null. A spine port belongs to no single network
#: domain, and job-wide events belong to no single rank. Everything else must be
#: populated.
NULLABLE_COLUMNS: dict[str, set[str]] = {
    "telemetry_switch_port": {"domain_id"},
    "telemetry_switch_aggregate": {"domain_id"},
    "events": {"rank_id", "gpu_id"},
}

#: Counters that must never decrease. Asserted in tests.
CUMULATIVE_COLUMNS: dict[str, list[str]] = {
    "telemetry_nic": ["tx_bytes", "rx_bytes", "tx_errors", "rx_errors", "tx_drops", "rx_drops"],
    "telemetry_switch_port": ["tx_bytes", "rx_bytes", "tx_errors", "rx_errors",
                              "tx_drops", "rx_drops"],
}

#: Gauges and their physical bounds. Asserted in tests.
BOUNDED_COLUMNS: dict[str, dict[str, tuple[float, float]]] = {
    "telemetry_gpu": {
        "utilization_pct": (0.0, 100.0),
        "sm_occupancy_pct": (0.0, 100.0),
        "temperature_c": (0.0, 120.0),
    },
    "telemetry_node": {
        "cpu_pressure": (0.0, 1.0),
        "memory_pressure": (0.0, 1.0),
        "io_pressure": (0.0, 1.0),
    },
    "telemetry_storage": {"utilisation_pct": (0.0, 100.0)},
}


def conform(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reorder to the declared schema and fail loudly on a mismatch."""
    expected = SCHEMAS[name]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}")
    extra = [c for c in df.columns if c not in expected]
    if extra:
        raise KeyError(f"{name}: unexpected columns {extra}")
    return df.loc[:, expected].reset_index(drop=True)


def write_run(frames: dict[str, pd.DataFrame], summary: dict[str, Any],
              out_dir: Path | str) -> Path:
    """Persist one run. Returns the directory written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        conform(name, df).to_parquet(out / f"{name}.parquet", index=False)
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return out


def read_run(run_dir: Path | str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    d = Path(run_dir)
    frames = {name: pd.read_parquet(d / f"{name}.parquet")
              for name in SCHEMAS if (d / f"{name}.parquet").exists()}
    with (d / "summary.json").open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    return frames, summary


def frame_digest(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Stable content hash per table, for the reproducibility test."""
    import hashlib
    out = {}
    for name in sorted(frames):
        df = conform(name, frames[name])
        payload = pd.util.hash_pandas_object(df, index=False).values.tobytes()
        out[name] = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return out

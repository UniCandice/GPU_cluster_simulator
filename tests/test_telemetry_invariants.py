"""Invariants every telemetry stream must satisfy.

These are the tests that would catch telemetry drifting away from the simulation
it is supposed to be describing. The most important one is counter conservation:
it is the difference between switch counters *derived from routed traffic* and
switch counters invented alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gcsim.models.network import build_halo_flows
from gcsim.routing import CROSS_DOMAIN
from gcsim.telemetry import (BOUNDED_COLUMNS, CUMULATIVE_COLUMNS, NULLABLE_COLUMNS,
                             SCHEMAS, conform)

#: A healthy fabric still corrupts the odd frame at its background BER, so
#: "no errors" means "nothing above the physical noise floor".
ERROR_FLOOR = 1000.0


def _iterations_covered(frames) -> int:
    """Timesteps that had completed by the final telemetry sample.

    Sampling stops on its own 1 Hz clock, which does not line up with the end of
    the run, so the last sample misses whatever ran after it. Any conservation
    check has to compare like with like.
    """
    #  Counters are credited when a timestep is computed, so a timestep that had
    #  STARTED by the last sample is already reflected in it -- even if it had
    #  not finished.
    last_sample = frames["telemetry_nic"]["timestamp"].max()
    job = frames["job_performance"]
    return int((job["timestamp"] <= last_sample).sum())


@pytest.fixture(scope="module")
def frames(healthy):
    return healthy.frames


def test_every_declared_table_is_present_and_conforms(frames):
    assert set(frames) == set(SCHEMAS)
    for name, df in frames.items():
        assert len(df) > 0, name
        conform(name, df)


def test_no_missing_values_outside_the_declared_nullable_columns(frames):
    for name, df in frames.items():
        nullable = NULLABLE_COLUMNS.get(name, set())
        checked = df.drop(columns=list(nullable))
        assert not checked.isna().any().any(), f"{name} has unexpected nulls"

    #  ...and the nullable ones are null only where they are meant to be.
    ports = frames["telemetry_switch_port"]
    assert ports.loc[ports["switch_tier"] == "leaf", "domain_id"].notna().all()
    assert ports.loc[ports["switch_tier"] == "spine", "domain_id"].isna().all()


def test_gauges_stay_inside_physical_bounds(frames, bundle):
    for name, cols in BOUNDED_COLUMNS.items():
        df = frames[name]
        for col, (lo, hi) in cols.items():
            assert df[col].min() >= lo - 1e-9, f"{name}.{col} below {lo}"
            assert df[col].max() <= hi + 1e-9, f"{name}.{col} above {hi}"

    gpu = frames["telemetry_gpu"]
    g = bundle.cluster.gpu
    assert (gpu["memory_used_gb"] <= gpu["memory_total_gb"]).all()
    assert gpu["power_w"].max() <= g.board_power_cap_w + 1e-9
    assert gpu["clock_mhz"].between(g.min_clock_mhz, g.base_clock_mhz).all()
    assert set(gpu["throttle_reason"]) <= {"NONE", "THERMAL", "POWER_CAP", "RELIABILITY"}
    #  A throttle reason and the throttle flag must always agree.
    assert ((gpu["throttle_reason"] != "NONE") == gpu["throttled"]).all()


def test_cumulative_counters_never_decrease(frames):
    for name, cols in CUMULATIVE_COLUMNS.items():
        df = frames[name]
        key = "nic_id" if name == "telemetry_nic" else "port_id"
        for _, grp in df.sort_values("timestamp").groupby(key):
            for col in cols:
                assert (grp[col].diff().dropna() >= -1e-6).all(), f"{name}.{col} went backwards"


def test_phase_times_sum_exactly_to_the_timestep(frames):
    """The barrier decomposition must be exhaustive, per rank, every timestep.

    If these four columns did not sum to the timestep, `wait` would be a free
    parameter rather than the barrier slack it is supposed to be -- and the
    whole straggler-attribution story would rest on nothing.
    """
    r = frames["rank_performance"]
    total = (r["compute_time_s"] + r["halo_wait_s"]
             + r["allreduce_wait_s"] + r["checkpoint_time_s"])
    assert np.allclose(total, r["total_time_s"], rtol=1e-12, atol=1e-12)


def test_every_rank_leaves_the_barrier_together(frames):
    """A synchronised job: total_time_s is identical across ranks each timestep."""
    r = frames["rank_performance"]
    spread = r.groupby("iteration")["total_time_s"].agg(lambda s: s.max() - s.min())
    assert spread.max() < 1e-12

    job = frames["job_performance"].set_index("iteration")["iteration_time_s"]
    per_iter = r.groupby("iteration")["total_time_s"].first()
    assert np.allclose(per_iter, job.loc[per_iter.index])


def test_phase_times_are_non_negative(frames):
    r = frames["rank_performance"]
    for col in ("compute_time_s", "halo_wait_s", "allreduce_wait_s", "checkpoint_time_s"):
        assert (r[col] >= 0).all(), col


def test_event_trace_is_monotonic_and_covers_the_run(frames):
    ev = frames["events"]
    assert (ev["timestamp"].diff().dropna() >= -1e-9).all()
    assert ev["event_type"].iloc[0] == "SIM_START"
    assert ev["event_type"].iloc[-1] == "SIM_END"
    counts = ev["event_type"].value_counts()
    assert counts["ITERATION_END"] == 1000
    assert counts["DATA_LOAD_END"] == 1
    #  Output every 100 timesteps, and the trace has to show all ten.
    assert counts["OUTPUT_END"] == 10


def test_telemetry_timestamps_fall_on_the_sample_interval(frames, bundle):
    """Sampling runs on its own clock, independent of when phases happen."""
    interval = bundle.cluster.telemetry.sample_interval_s
    ts = np.sort(frames["telemetry_gpu"]["timestamp"].unique())
    assert np.allclose(np.diff(ts), interval)
    assert ts[0] == pytest.approx(interval)

    #  ...and it is genuinely coarser than the timestep, so a sample averages
    #  over several timesteps and cannot resolve the phase structure inside them.
    mean_iter = frames["job_performance"]["iteration_time_s"].mean()
    assert interval > 3 * mean_iter


def test_one_row_per_entity_per_sample(frames, cluster):
    expectations = {
        "telemetry_gpu": len(cluster.gpus),
        "telemetry_node": len(cluster.nodes),
        "telemetry_nic": len(cluster.nics),
        "telemetry_switch_port": len(cluster.ports),
        "telemetry_switch_aggregate": len(cluster.switches),
        "telemetry_storage": 1,
    }
    for name, expected in expectations.items():
        df = frames[name]
        assert (df.groupby("timestamp").size() == expected).all(), name


# ---------------------------------------------------------------------------
# Counter conservation
# ---------------------------------------------------------------------------

def test_uplink_bytes_equal_the_traffic_that_actually_left_the_rack(
        healthy, cluster, router, decomposition, placement, bundle):
    """The proof that switch telemetry is derived, not invented.

    Bytes recorded on a leaf's uplink ports must equal the halo traffic whose
    route genuinely left that rack -- recomputed here from the flow table rather
    than read back from the same accounting path.
    """
    from gcsim.models.network import Fabric

    fabric = Fabric(cluster, router)
    flows = build_halo_flows(fabric, decomposition, placement)
    iterations = _iterations_covered(healthy.frames)
    assert iterations > 900

    expected: dict[str, float] = {}
    for i in range(flows.n_flows):
        if flows.kind[i] != CROSS_DOMAIN:
            continue
        rack = cluster.gpu_list[int(placement.rank_to_gpu[flows.src_rank[i]])].rack_id
        expected[rack] = expected.get(rack, 0.0) + float(flows.nbytes[i]) * iterations

    ports = healthy.frames["telemetry_switch_port"]
    last = ports.sort_values("timestamp").groupby("port_id").last().reset_index()
    uplinks = last[(last["switch_tier"] == "leaf") & (last["port_role"] == "uplink")]
    observed = uplinks.groupby("domain_id")["tx_bytes"].sum()

    assert set(observed.index) == set(expected)
    for rack, want in expected.items():
        assert observed[rack] == pytest.approx(want, rel=1e-9)


def test_nothing_crosses_a_rack_boundary_without_being_counted(healthy):
    """Global conservation: every byte sent cross-rack is received cross-rack."""
    ports = healthy.frames["telemetry_switch_port"]
    last = ports.sort_values("timestamp").groupby("port_id").last().reset_index()
    uplinks = last[(last["switch_tier"] == "leaf") & (last["port_role"] == "uplink")]
    assert uplinks["tx_bytes"].sum() == pytest.approx(uplinks["rx_bytes"].sum(), rel=1e-9)


def test_spine_counters_mirror_their_leaf_partners(healthy):
    """Same cable, counted once. tx on the leaf side is rx on the spine side."""
    ports = healthy.frames["telemetry_switch_port"]
    last = ports.sort_values("timestamp").groupby("port_id").last()
    leaf_tx = last[last["switch_tier"] == "leaf"]
    spine = last[last["switch_tier"] == "spine"]
    assert spine["rx_bytes"].sum() == pytest.approx(
        leaf_tx[leaf_tx["port_role"] == "uplink"]["tx_bytes"].sum(), rel=1e-9)


def test_intra_node_traffic_never_reaches_a_switch(
        healthy, cluster, router, decomposition, placement, bundle):
    """NVLink exchanges must not appear in NIC or switch counters.

    The +/-x faces are the largest in the decomposition -- twice the area of the
    others -- so if they leaked into the fabric counters every fabric metric
    would be inflated by half.
    """
    from gcsim.models.network import Fabric
    from gcsim.routing import INTRANODE

    fabric = Fabric(cluster, router)
    flows = build_halo_flows(fabric, decomposition, placement)
    iterations = _iterations_covered(healthy.frames)

    off_node = float(flows.nbytes[flows.kind != INTRANODE].sum()) * iterations
    on_node = float(flows.nbytes[flows.kind == INTRANODE].sum()) * iterations
    assert on_node > 0

    nic = healthy.frames["telemetry_nic"]
    last = nic.sort_values("timestamp").groupby("nic_id").last()
    assert last["tx_bytes"].sum() == pytest.approx(off_node, rel=1e-9)
    #  Full duplex: everything one NIC sent, another received.
    assert last["rx_bytes"].sum() == pytest.approx(off_node, rel=1e-9)

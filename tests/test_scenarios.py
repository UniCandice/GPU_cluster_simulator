"""Scenario behaviour: does each degradation produce the signature it should?

Every test here reads only telemetry. None of them consults the scenario's
`fault` label except to check that the rule-based diagnosis agrees with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
LATE = 700          # timestep after which every injection has taken effect
EARLY = 150         # timestep before which nothing has
#: Every link corrupts the odd frame at its background BER, so "clean" means
#: "nothing above the physical noise floor", not literally zero.
ERROR_FLOOR = 1000.0


def _window(df, column="iteration"):
    return df[df[column] < EARLY], df[df[column] > LATE]


def _gpu_window(gpu, job):
    """Split GPU telemetry on wall time rather than timestep index."""
    t_early = job[job["iteration"] == EARLY]["timestamp"].iloc[0]
    t_late = job[job["iteration"] == LATE]["timestamp"].iloc[0]
    return gpu[gpu["timestamp"] < t_early], gpu[gpu["timestamp"] > t_late]


# ---------------------------------------------------------------------------
# The architectural rule
# ---------------------------------------------------------------------------

def test_faults_cannot_reach_telemetry():
    """`faults.py` must have no import path to any telemetry-producing module.

    This is the governing rule of the whole design made mechanical: an injection
    perturbs physical state and nothing else. If this test fails, a scenario has
    gained the ability to write its own signature, and every result the
    simulator produces stops meaning anything.
    """
    forbidden = {"gcsim.telemetry", "gcsim.samplers", "gcsim.metrics", "gcsim.scenarios"}
    seen: set[str] = set()

    def imports_of(module: str) -> set[str]:
        path = SRC / (module.replace(".", "/") + ".py")
        if not path.exists():
            return set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
            elif isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
        return {m for m in out if m.startswith("gcsim")}

    def walk(module: str) -> None:
        if module in seen:
            return
        seen.add(module)
        for dep in imports_of(module):
            assert dep not in forbidden, f"gcsim.faults reaches {dep} via {module}"
            walk(dep)

    walk("gcsim.faults")
    assert "gcsim.topology" in seen and "gcsim.workload" in seen


def test_diagnosis_agrees_with_ground_truth_on_every_scenario(runs):
    for name, run in runs.items():
        truth = run.summary["is_fault"]
        verdict = run.summary["diagnosis"]["verdict"]
        assert (verdict == "HARDWARE_FAULT") == truth, f"{name}: got {verdict}"


# ---------------------------------------------------------------------------
# healthy
# ---------------------------------------------------------------------------

def test_healthy_is_quiet_on_every_hardware_channel(healthy):
    gpu = healthy.frames["telemetry_gpu"]
    ports = healthy.frames["telemetry_switch_port"]
    assert not healthy.frames["telemetry_switch_aggregate"]["congested"].any()

    assert not gpu["throttled"].any()
    assert gpu["clock_mhz"].nunique() == 1                     # never left boost
    assert ports["tx_drops"].max() < ERROR_FLOOR
    assert ports["rx_drops"].max() < ERROR_FLOOR
    assert ports["link_up"].all()
    assert healthy.summary["diagnosis"]["verdict"] == "NOMINAL"


def test_healthy_still_shows_periodic_storage_spikes(healthy):
    """Field output is a legitimate phase. The baseline is not flat, and should
    not be: a detector calibrated on a dead-flat baseline would fire on it."""
    job = healthy.frames["job_performance"]
    output_steps = job[job["checkpoint_s"] > 0]
    assert len(output_steps) == 10
    assert output_steps["iteration_time_s"].min() > job["iteration_time_s"].median() * 1.3


def test_healthy_rank_spread_is_tight_but_not_zero(healthy):
    """Silicon variation is the only stochastic input, and it is small."""
    job = healthy.frames["job_performance"]
    ratio = job["rank_spread_s"].mean() / job["iteration_time_s"].mean()
    assert 0.001 < ratio < 0.10


# ---------------------------------------------------------------------------
# straggler -- and the barrier-stall fingerprint
# ---------------------------------------------------------------------------

def test_straggler_widens_the_spread_and_slows_the_job(runs):
    job = runs["straggler"].frames["job_performance"]
    early, late = _window(job)
    assert late["iteration_time_s"].mean() > early["iteration_time_s"].mean() * 1.2
    assert late["rank_spread_s"].mean() > early["rank_spread_s"].mean() * 4


def test_the_culprit_is_the_one_rank_that_never_waits(runs):
    """The inversion the whole diagnosis rests on.

    A slow rank has ~zero barrier slack while all 127 peers accumulate wait. The
    culprit looks busy; every victim looks idle. Reading the wait column
    naively would blame the wrong 127 GPUs.
    """
    rank = runs["straggler"].frames["rank_performance"]
    late = rank[rank["iteration"] > LATE]
    busy = late["compute_time_s"] + late["halo_wait_s"]
    culprit = int(busy.groupby(late["rank_id"]).mean().idxmax())

    wait = late.groupby("rank_id")["allreduce_wait_s"].mean()
    peers = wait.drop(index=culprit)
    assert wait[culprit] < peers.min() * 0.1
    assert peers.min() > 0.03          # every peer really is stalled


def test_straggler_leaves_no_fingerprint_on_any_device_counter(runs):
    """A stolen-SM straggler is invisible to the hardware.

    The GPU still boosts, still reports full utilisation and never throttles.
    Only the job's own timing gives it away, which is exactly why this failure
    mode survives in real fleets.
    """
    run = runs["straggler"]
    gpu = run.frames["telemetry_gpu"]
    assert not gpu["throttled"].any()
    assert gpu["clock_mhz"].nunique() == 1
    assert run.frames["telemetry_switch_port"]["tx_errors"].max() < ERROR_FLOOR

    #  The victim's occupancy is HIGHER than its peers', not lower: it is the
    #  only rank still doing a full timestep of work while everyone else waits.
    #  Nothing about the device is degraded, so there is nothing to see.
    _, late = _gpu_window(gpu, run.frames["job_performance"])
    victim = late[late["gpu_id"] == "r1n2g5"]["sm_occupancy_pct"].mean()
    peers = late[late["gpu_id"] != "r1n2g5"]["sm_occupancy_pct"].mean()
    assert victim > peers


def test_barrier_stall_shows_as_high_utilisation_with_falling_occupancy(runs):
    """The two-channel signature of a synchronisation stall.

    Peers of the straggler keep a spin kernel resident, so utilisation stays
    pinned near 100 while occupancy and power collapse. A model that treated
    utilisation and occupancy as the same signal could not show this.
    """
    run = runs["straggler"]
    gpu, job = run.frames["telemetry_gpu"], run.frames["job_performance"]
    early, late = _gpu_window(gpu, job)

    peers_early = early[early["gpu_id"] != "r1n2g5"]
    peers_late = late[late["gpu_id"] != "r1n2g5"]

    #  Utilisation barely moves: the spin kernel is still resident.
    assert peers_early["utilization_pct"].mean() > 98.0
    assert peers_late["utilization_pct"].mean() > 98.0

    #  Everything that tracks real work collapses.
    assert peers_late["sm_occupancy_pct"].mean() < peers_early["sm_occupancy_pct"].mean()
    assert peers_late["power_w"].mean() < peers_early["power_w"].mean() * 0.92
    assert peers_late["temperature_c"].mean() < peers_early["temperature_c"].mean()

    #  ...while the culprit, the only rank still doing a full timestep of work,
    #  is the one GPU in the cluster that gets HOTTER. Exactly backwards from
    #  where a temperature-led search would look.
    victim_early = early[early["gpu_id"] == "r1n2g5"]["temperature_c"].mean()
    victim_late = late[late["gpu_id"] == "r1n2g5"]["temperature_c"].mean()
    assert victim_late > victim_early
    assert victim_late > peers_late["temperature_c"].mean()


def test_straggler_amplification_is_set_by_the_barrier_not_the_derate(runs, healthy):
    """The job grows by the victim's excess over the PREVIOUS pacer.

    A synchronised job runs at the speed of its slowest rank, so slowing one
    rank costs only what it adds beyond whoever was already slowest -- not its
    own full slowdown. Silicon variation means another rank was already setting
    the pace, so the job loses less than the victim does. Getting this wrong
    overstates the cost of every straggler in the fleet.
    """
    def busy(frames):
        r = frames["rank_performance"]
        r = r[r["iteration"] > LATE]
        return (r["compute_time_s"] + r["halo_wait_s"]).groupby(r["rank_id"]).mean()

    def job_time(frames):
        j = frames["job_performance"]
        return j[j["iteration"] > LATE]["iteration_time_s"].mean()

    healthy_busy = busy(healthy.frames)
    slow_busy = busy(runs["straggler"].frames)
    delta_job = job_time(runs["straggler"].frames) - job_time(healthy.frames)

    #  The barrier simply moved from the old pacer to the victim.
    assert delta_job == pytest.approx(slow_busy.max() - healthy_busy.max(), rel=0.02)

    #  ...and that is strictly LESS than the victim's own slowdown.
    victim = int(slow_busy.idxmax())
    own_slowdown = slow_busy[victim] - healthy_busy[victim]
    assert 0 < delta_job < own_slowdown

    #  The victim really is running at 75% SM throughput on its variable work.
    assert own_slowdown > 0.15 * healthy_busy[victim]


# ---------------------------------------------------------------------------
# network_domain
# ---------------------------------------------------------------------------

def test_network_fault_slows_communication_and_leaves_compute_alone(runs):
    """The separator between a fabric fault and a compute fault."""
    job = runs["network_domain"].frames["job_performance"]
    early, late = _window(job)

    assert late["halo_mean_s"].mean() > early["halo_mean_s"].mean() * 3
    assert late["compute_mean_s"].mean() == pytest.approx(
        early["compute_mean_s"].mean(), rel=0.02)


def test_network_fault_is_confined_to_one_domain(runs):
    """Errors and downed links localise the fault without any ground truth."""
    ports = runs["network_domain"].frames["telemetry_switch_port"]
    last = ports.sort_values("timestamp").groupby("port_id").last().reset_index()
    leaf = last[last["switch_tier"] == "leaf"]

    down = leaf[~leaf["link_up"]]
    assert set(down["domain_id"]) == {"r2"}
    assert len(down) == 7                                  # 8 uplinks, 1 survives

    #  Background BER trickles a handful of errors onto every link, so the test
    #  is that r2 sits orders of magnitude above that floor and nothing else does.
    errored = leaf[leaf["tx_errors"] > ERROR_FLOOR]
    assert set(errored["domain_id"]) == {"r2"}
    assert errored["tx_errors"].min() > 1e5
    assert leaf[leaf["domain_id"] != "r2"]["tx_drops"].max() < ERROR_FLOOR


def test_oversubscription_ratio_exposes_the_failure_directly(runs):
    agg = runs["network_domain"].frames["telemetry_switch_aggregate"]
    r2 = agg[agg["domain_id"] == "r2"].sort_values("timestamp")
    r0 = agg[agg["domain_id"] == "r0"].sort_values("timestamp")
    assert r2["oversubscription_ratio"].iloc[-1] == pytest.approx(
        r2["oversubscription_ratio"].iloc[0] * 8)
    assert r0["oversubscription_ratio"].nunique() == 1      # untouched


def test_no_gpu_throttles_during_a_fabric_fault(runs):
    """A fabric fault must not be mistakable for a device fault."""
    gpu = runs["network_domain"].frames["telemetry_gpu"]
    assert not gpu["throttled"].any()
    assert gpu["clock_mhz"].nunique() == 1


# ---------------------------------------------------------------------------
# thermal
# ---------------------------------------------------------------------------

def test_thermal_fault_moves_the_whole_rack_together(runs):
    """Inlet temperature is a rack property, so all 32 GPUs heat as a block."""
    run = runs["thermal"]
    gpu, job = run.frames["telemetry_gpu"], run.frames["job_performance"]
    _, late = _gpu_window(gpu, job)

    hot = late[late["rack_id"] == "r1"]
    cool = late[late["rack_id"] != "r1"]
    assert hot["temperature_c"].mean() > cool["temperature_c"].mean() + 15
    assert set(gpu[gpu["throttled"]]["rack_id"]) == {"r1"}
    assert gpu[gpu["throttled"]]["gpu_id"].nunique() == 32


def test_thermal_signature_is_clock_and_power_falling_together(runs):
    """What distinguishes a throttled GPU from an idle one.

    An idle GPU sheds power with its clock at boost. A throttled GPU sheds both,
    while utilisation stays pinned because the work has not gone away.
    """
    run = runs["thermal"]
    gpu, job = run.frames["telemetry_gpu"], run.frames["job_performance"]
    early, late = _gpu_window(gpu, job)

    a = early[early["rack_id"] == "r1"]
    b = late[late["rack_id"] == "r1"]

    assert b["temperature_c"].mean() > a["temperature_c"].mean() + 20
    assert b["clock_mhz"].mean() < a["clock_mhz"].mean()
    assert b["power_w"].mean() < a["power_w"].mean()
    assert b["utilization_pct"].mean() > 95.0
    assert set(gpu[gpu["throttled"]]["throttle_reason"]) == {"THERMAL"}


def test_throttle_events_appear_in_the_trace(runs):
    ev = runs["thermal"].frames["events"]
    engaged = ev[ev["event_type"] == "THROTTLE_ENGAGED"]
    assert len(engaged) >= 32
    assert all(g.startswith("r1n") for g in engaged["gpu_id"])


# ---------------------------------------------------------------------------
# gpu_degradation -- the hard discrimination
# ---------------------------------------------------------------------------

def test_degradation_and_straggler_are_indistinguishable_on_job_timing(runs):
    """Two different faults, the same throughput and spread signature.

    If the simulator only produced timing data, these two would be one scenario.
    """
    a = runs["straggler"].frames["job_performance"]
    b = runs["gpu_degradation"].frames["job_performance"]
    a, b = a[a["iteration"] > LATE], b[b["iteration"] > LATE]

    assert b["iteration_time_s"].mean() == pytest.approx(
        a["iteration_time_s"].mean(), rel=0.05)
    assert b["rank_spread_s"].mean() == pytest.approx(a["rank_spread_s"].mean(), rel=0.10)
    assert b["straggler_count"].mean() == pytest.approx(a["straggler_count"].mean(), abs=1)


def test_only_throttle_reason_and_occupancy_separate_them(runs):
    """...and here is what does separate them.

    The degraded device reports its own condition: the RAS governor caps the
    clock, and the victim's occupancy drops because its SMs are stalled on
    memory. The straggler's device reports nothing at all.
    """
    deg = runs["gpu_degradation"].frames["telemetry_gpu"]
    strag = runs["straggler"].frames["telemetry_gpu"]

    assert set(deg[deg["throttled"]]["throttle_reason"]) == {"RELIABILITY"}
    assert deg[deg["throttled"]]["gpu_id"].nunique() == 1
    assert not strag["throttled"].any()

    #  Compare the two PACING ranks against each other. Both run at ~100% duty
    #  while their peers wait, so duty cycle cancels out and what is left is the
    #  device itself: the degraded GPU's SMs are stalled on memory, so it retires
    #  less per resident cycle than the straggler's perfectly healthy SMs do.
    def pacing_occupancy(gpu_frame, job_frame, gpu_id):
        t_late = job_frame[job_frame["iteration"] == LATE]["timestamp"].iloc[0]
        late = gpu_frame[gpu_frame["timestamp"] > t_late]
        return late[late["gpu_id"] == gpu_id]["sm_occupancy_pct"].mean()

    deg_victim = pacing_occupancy(deg, runs["gpu_degradation"].frames["job_performance"],
                                  "r3n1g2")
    strag_victim = pacing_occupancy(strag, runs["straggler"].frames["job_performance"],
                                    "r1n2g5")
    assert deg_victim < strag_victim * 0.95
    assert deg["clock_mhz"].min() < strag["clock_mhz"].min()

    assert runs["gpu_degradation"].summary["diagnosis"]["tier"] == "gpu"
    assert runs["straggler"].summary["diagnosis"]["tier"] == "rank"


# ---------------------------------------------------------------------------
# phase_change -- the legitimate one
# ---------------------------------------------------------------------------

def test_workload_change_slows_the_job_with_every_device_channel_clean(runs):
    """The assertion that encodes "this is not a hardware fault".

    Throughput drops by ~10%, which is enough to fire any threshold detector.
    Nothing else moves.
    """
    run = runs["phase_change"]
    gpu = run.frames["telemetry_gpu"]
    ports = run.frames["telemetry_switch_port"]
    job = run.frames["job_performance"]
    early, late = _window(job)

    assert late["iteration_time_s"].mean() > early["iteration_time_s"].mean() * 1.05

    assert not gpu["throttled"].any()
    assert gpu["clock_mhz"].nunique() == 1
    assert ports["tx_errors"].max() < ERROR_FLOOR
    assert ports["tx_drops"].max() < ERROR_FLOOR
    assert ports["link_up"].all()
    #  No leaf ever reports uplink congestion, in this scenario or the baseline.
    agg = run.frames["telemetry_switch_aggregate"]
    assert not agg["congested"].any()


def test_workload_change_affects_every_rank_identically(runs):
    """No localised fault can produce a uniform slowdown across 128 ranks."""
    job = runs["phase_change"].frames["job_performance"]
    early, late = _window(job)
    assert late["rank_spread_s"].mean() == pytest.approx(
        early["rank_spread_s"].mean(), rel=0.05)
    assert late["straggler_count"].mean() == pytest.approx(
        early["straggler_count"].mean(), abs=1)


def test_workload_change_shows_up_where_it_should(runs):
    """Storage and host pressure, not device counters."""
    run = runs["phase_change"]
    storage = run.frames["telemetry_storage"].sort_values("timestamp")
    node = run.frames["telemetry_node"].sort_values("timestamp")
    n = len(storage)

    early = storage.iloc[: n // 5]
    late = storage.iloc[-n // 3 :]
    assert late["write_latency_ms"].mean() > early["write_latency_ms"].mean() * 2
    assert late["dirty_backlog_gb"].max() > early["dirty_backlog_gb"].max()

    m = len(node)
    assert (node.iloc[-m // 3 :]["io_pressure"].mean()
            > node.iloc[: m // 5]["io_pressure"].mean() + 0.02)


def test_output_volume_is_the_cause_and_it_is_in_the_trace(runs):
    ev = runs["phase_change"].frames["events"]
    injections = ev[ev["event_type"] == "INJECTION_APPLIED"]
    assert len(injections) == 1
    assert "workload_output_change" in injections["payload"].iloc[0]

    job = runs["phase_change"].frames["job_performance"]
    #  Output every 100 timesteps for the first 500, then every 20.
    assert (job[job["iteration"] <= 500]["checkpoint_s"] > 0).sum() == 5
    assert (job[job["iteration"] > 500]["checkpoint_s"] > 0).sum() == 25


# ---------------------------------------------------------------------------
# The same fault, a different workload
# ---------------------------------------------------------------------------

def test_cooling_failure_is_caught_before_it_ever_throttles(bundle):
    """The same CRAC failure has a different consequence on a lighter workload.

    A coarse-mesh job is communication-bound and draws far less power, so the
    rack it sits in settles ~30 C cooler than the same rack running the medium
    mesh. The identical cooling failure never reaches the slowdown threshold and
    costs no throughput at all -- but the rack has still failed, and a detector
    that waits for a throttle bit would leave the fleet running on it.

    What gives it away is that one rack drifted away from its peers. Inlet
    temperature is a rack property, so all 32 GPUs move together; nothing at the
    GPU or node tier can produce that pattern.
    """
    from gcsim.scenarios import run_scenario

    run = run_scenario("thermal", mesh="coarse", seed=42, bundle=bundle, out_dir=None)
    gpu = run.frames["telemetry_gpu"]
    diagnosis = run.summary["diagnosis"]

    #  Nothing throttled and throughput is untouched.
    assert not gpu["throttled"].any()
    assert gpu["temperature_c"].max() < bundle.cluster.gpu.thermal_slowdown_c
    assert abs(diagnosis["slowdown_pct"]) < 3.0

    #  The affected rack is still unambiguous, and it is the whole rack. Compare
    #  the late window: averaging over the whole run dilutes the drift with the
    #  250 healthy timesteps that preceded the injection.
    _, late = _gpu_window(gpu, run.frames["job_performance"])
    by_rack = late.groupby("rack_id")["temperature_c"].mean()
    assert by_rack["r1"] > by_rack.drop(index="r1").max() + 8.0
    assert late[late["rack_id"] == "r1"]["gpu_id"].nunique() == 32

    assert diagnosis["verdict"] == "HARDWARE_FAULT"
    assert diagnosis["tier"] == "rack"
    assert diagnosis["localisation"]["racks"] == ["r1"]
    #  Lower confidence than the throttling case, which is the honest call.
    assert diagnosis["confidence"] == "medium"


def test_thermal_impact_scales_with_how_hard_the_workload_pushes_the_rack(bundle):
    """Fault severity is not a property of the fault alone.

    Identical cooling degradation, three workloads: the throughput cost rises
    with the power the rack is dissipating, from nothing on the coarse mesh to
    double digits on the fine one.
    """
    from gcsim.scenarios import run_scenario

    impact = {}
    for mesh in ("coarse", "medium", "fine"):
        run = run_scenario("thermal", mesh=mesh, seed=42, bundle=bundle, out_dir=None)
        impact[mesh] = (run.summary["diagnosis"]["slowdown_pct"],
                        run.frames["telemetry_gpu"]["temperature_c"].max())

    assert impact["coarse"][0] < impact["medium"][0] < impact["fine"][0]
    assert impact["coarse"][1] < impact["medium"][1] < impact["fine"][1]
    assert impact["coarse"][0] < 3.0 and impact["fine"][0] > 10.0

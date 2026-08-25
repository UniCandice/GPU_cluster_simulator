"""Scenario behaviour: does each degradation produce the signature it should?

Every test here reads only telemetry. None of them consults the scenario's
`fault` label except to check that the rule-based diagnosis agrees with it.
"""

from __future__ import annotations

import ast
import json
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
# Straggler episodes: reading the injected schedule back as ground truth
# ---------------------------------------------------------------------------
#
# The episodic straggler plans its episodes up front and writes them into the
# INJECTION_APPLIED payload. Tests read that schedule rather than inferring
# episode windows from timings, so cause and effect stay separable: if a test
# had to find the episodes by looking for slow timesteps, it could no longer
# tell whether the slowdown landed where the injection said it would.

def _episodes(run):
    """The episode schedule, as the injector planned it."""
    events = run.frames["events"]
    fired = events[events["event_type"] == "INJECTION_APPLIED"]
    return json.loads(fired.iloc[0]["payload"])["episodes"]


def _episode_spans(run):
    """Episodes as (gpu_id, t_start, t_end) in wall-clock seconds."""
    stamp = run.frames["job_performance"].set_index("iteration")["timestamp"]
    last = int(stamp.index.max())
    return [(e["gpu_id"], float(stamp.loc[e["start"]]), float(stamp.loc[min(e["end"], last)]))
            for e in _episodes(run)]


def _episode_of(run, timestamps):
    """Index of the episode covering each timestamp, or -1 outside every one."""
    stamps = np.asarray(timestamps, dtype=float)
    out = np.full(len(stamps), -1, dtype=int)
    for k, (_, start, end) in enumerate(_episode_spans(run)):
        out[(stamps >= start) & (stamps < end)] = k
    return out


def _split_by_episode(run, gpu):
    """GPU telemetry split into (victim rows, peer rows in-episode, rows outside).

    The victim changes from episode to episode, so "peer" is only meaningful
    relative to whichever rank is being derated at that moment.
    """
    where = _episode_of(run, gpu["timestamp"].to_numpy())
    spans = _episode_spans(run)
    victim_id = np.array([spans[i][0] if i >= 0 else "" for i in where])
    is_victim = (gpu["gpu_id"].to_numpy() == victim_id) & (where >= 0)
    return gpu[is_victim], gpu[(where >= 0) & ~is_victim], gpu[where < 0]


def _quiet_iterations(run):
    """Timesteps after the first episode that no episode covers."""
    job = run.frames["job_performance"]
    iteration = job["iteration"].to_numpy()
    covered = np.zeros(len(job), dtype=bool)
    episodes = _episodes(run)
    for e in episodes:
        covered |= (iteration >= e["start"]) & (iteration < e["end"])
    started = iteration >= min(e["start"] for e in episodes)
    return job[~covered & started]


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

def test_the_early_late_comparison_almost_misses_the_episodic_straggler(runs):
    """The scenario's whole point, asserted as a negative.

    Episodes run from the first timestep, so the early window is as contaminated
    as the late one and the before/after comparison that every other scenario
    leans on reports almost nothing: a couple of percent on iteration time and a
    rank spread that barely moves. A detector built only on window deltas calls
    this cluster healthy. The fault is real -- the next test measures it -- but
    it is invisible to that particular question, which is why `diagnose` does
    not ask it first.
    """
    job = runs["straggler"].frames["job_performance"]
    early, late = _window(job)
    assert late["iteration_time_s"].mean() < early["iteration_time_s"].mean() * 1.05
    assert late["rank_spread_s"].mean() < early["rank_spread_s"].mean() * 1.5


def test_but_the_cost_is_real_when_measured_against_the_episodes(runs, healthy):
    """...and here is everything the window comparison hid.

    Split the same run by whether an episode was actually running and the fault
    is unmistakable. The information was never missing; it was averaged away.
    """
    run = runs["straggler"]
    job = run.frames["job_performance"]
    iteration = job["iteration"].to_numpy()
    covered = np.zeros(len(job), dtype=bool)
    for e in _episodes(run):
        covered |= (iteration >= e["start"]) & (iteration < e["end"])

    inside, outside = job[covered], job[~covered]
    assert inside["iteration_time_s"].mean() > outside["iteration_time_s"].mean() * 1.20
    assert inside["rank_spread_s"].mean() > outside["rank_spread_s"].mean() * 3.0

    #  ...and the job as a whole really did pay for it.
    healthy_job = healthy.frames["job_performance"]
    assert (job["iteration_time_s"].sum()
            > healthy_job["iteration_time_s"].sum() * 1.05)


def test_between_episodes_the_cluster_returns_to_baseline(runs, healthy):
    """Every episode releases the GPU completely -- no residue, no drift.

    This is the property that separates an intermittent fault from a degrading
    one. A quiet timestep in the straggler run is not merely *close* to healthy,
    it is the same number: the derate goes back to exactly 1.0, and nothing else
    was ever touched, so the whole downstream pipeline recomputes identically.
    An assertion this tight is only honest because the simulator is
    deterministic under a fixed seed -- and it would fail loudly if an episode
    ever restored to 0.999 instead of 1.0.
    """
    quiet = _quiet_iterations(runs["straggler"])
    healthy_job = healthy.frames["job_performance"]
    same = healthy_job[healthy_job["iteration"].isin(quiet["iteration"])]

    assert len(quiet) > 100            # there really is a quiet majority to check
    assert quiet["iteration_time_s"].mean() == pytest.approx(
        same["iteration_time_s"].mean(), rel=1e-9)
    assert quiet["rank_spread_s"].mean() == pytest.approx(
        same["rank_spread_s"].mean(), rel=1e-9)


def test_the_culprit_is_the_rank_the_injector_actually_derated(runs):
    """The inversion the whole diagnosis rests on, checked against ground truth.

    A slow rank has ~zero barrier slack while all 127 peers accumulate wait. The
    culprit looks busy; every victim looks idle. Reading the wait column naively
    would blame the wrong 127 GPUs.

    Stronger than the old form of this test: because the schedule is known, the
    busiest rank inside an episode can be checked to be *the rank the injector
    chose*, not merely some rank that happened to be slow.
    """
    run = runs["straggler"]
    episode = max(_episodes(run), key=lambda e: e["end"] - e["start"])

    rank = run.frames["rank_performance"]
    window = rank[(rank["iteration"] >= episode["start"])
                  & (rank["iteration"] < episode["end"])]
    busy = window["compute_time_s"] + window["halo_wait_s"]
    culprit = int(busy.groupby(window["rank_id"]).mean().idxmax())

    assert window.loc[window["rank_id"] == culprit, "gpu_id"].iloc[0] == episode["gpu_id"]

    wait = window.groupby("rank_id")["allreduce_wait_s"].mean()
    peers = wait.drop(index=culprit)
    assert wait[culprit] < peers.min() * 0.1
    assert peers.min() > 0.0           # every peer really is stalled


def test_the_pacer_changes_hands_across_episodes(runs):
    """A small cohort carries the fault, and the barrier passes between them.

    This is what makes the episodic straggler a different object from
    `gpu_degradation`, which is pinned to one device for the whole run: the
    fault is localised, but never to only one rank and never continuously.
    """
    run = runs["straggler"]
    victims = {e["gpu_id"] for e in _episodes(run)}
    assert 1 < len(victims) <= 8          # a cohort, not one GPU and not the fleet

    #  ...and the fleet-level consequence: over the late window the identity of
    #  the slowest rank is not constant.
    rank = run.frames["rank_performance"]
    late = rank[rank["iteration"] > LATE].copy()
    late["busy"] = late["compute_time_s"] + late["halo_wait_s"]
    pacer = late.loc[late.groupby("iteration")["busy"].idxmax(), "rank_id"]
    assert pacer.nunique() > 1


def test_the_diagnosis_recovers_the_whole_cohort_from_telemetry(runs):
    """The classifier finds every injected victim, and only those.

    The strongest statement this suite makes about the episodic straggler: the
    set of GPUs named in the diagnosis is *exactly* the set the injector chose,
    recovered from timing telemetry with no access to the schedule. It works
    because localisation counts the timesteps each rank paced the barrier rather
    than ranking mean busy time -- a mean over the window is dominated by
    ordinary silicon variation once a fault is only present a tenth of the time.
    """
    run = runs["straggler"]
    injected = sorted({e["gpu_id"] for e in _episodes(run)})
    diagnosed = sorted(run.summary["diagnosis"]["localisation"]["gpus"])
    assert diagnosed == injected


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
    victim, peers_in, _ = _split_by_episode(run, gpu)
    assert victim["sm_occupancy_pct"].mean() > peers_in["sm_occupancy_pct"].mean()


def test_barrier_stall_shows_as_high_utilisation_with_falling_occupancy(runs):
    """The two-channel signature of a synchronisation stall.

    Peers of the straggler keep a spin kernel resident, so utilisation stays
    pinned near 100 while occupancy and power collapse. A model that treated
    utilisation and occupancy as the same signal could not show this.
    """
    run = runs["straggler"]
    gpu = run.frames["telemetry_gpu"]
    victim, peers_in, outside = _split_by_episode(run, gpu)

    #  Utilisation barely moves: the spin kernel is still resident.
    assert outside["utilization_pct"].mean() > 98.0
    assert peers_in["utilization_pct"].mean() > 98.0

    #  Everything that tracks real work collapses while an episode is running.
    assert peers_in["sm_occupancy_pct"].mean() < outside["sm_occupancy_pct"].mean() * 0.9
    assert peers_in["power_w"].mean() < outside["power_w"].mean() * 0.95

    #  Temperature is deliberately NOT compared in-episode against out-of-episode.
    #  A die has thermal mass: it integrates over a window considerably longer
    #  than one episode, so the two populations come out within a few tenths of
    #  a degree of each other and the sign of the difference is not meaningful.
    #  That is a real property of the channel, not a gap in the model -- it is
    #  why an intermittent fault is close to invisible to thermal monitoring.
    #
    #  The comparison that does survive is spatial rather than temporal: at the
    #  same instant, the culprit is the only rank still doing a full timestep of
    #  work, so it is HOTTER than the ranks waiting on it. Exactly backwards
    #  from where a temperature-led search would look.
    assert victim["temperature_c"].mean() > peers_in["temperature_c"].mean()
    assert victim["sm_occupancy_pct"].mean() > peers_in["sm_occupancy_pct"].mean()


def test_straggler_amplification_is_set_by_the_barrier_not_the_derate(runs, healthy):
    """The job grows by the victim's excess over the PREVIOUS pacer.

    A synchronised job runs at the speed of its slowest rank, so slowing one
    rank costs only what it adds beyond whoever was already slowest -- not its
    own full slowdown. Silicon variation means another rank was already setting
    the pace, so the job loses less than the victim does. Getting this wrong
    overstates the cost of every straggler in the fleet.
    """
    #  Measured inside one episode rather than over the late window: averaging
    #  across quiet timesteps would mix a derated victim with a healthy one and
    #  understate both sides of the comparison.
    run = runs["straggler"]
    episode = max(_episodes(run), key=lambda e: e["end"] - e["start"])
    lo, hi = episode["start"], episode["end"]

    def busy(frames):
        r = frames["rank_performance"]
        r = r[(r["iteration"] >= lo) & (r["iteration"] < hi)]
        return (r["compute_time_s"] + r["halo_wait_s"]).groupby(r["rank_id"]).mean()

    def job_time(frames):
        j = frames["job_performance"]
        return j[(j["iteration"] >= lo) & (j["iteration"] < hi)]["iteration_time_s"].mean()

    healthy_busy = busy(healthy.frames)
    slow_busy = busy(run.frames)
    delta_job = job_time(run.frames) - job_time(healthy.frames)

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

def test_degradation_is_persistent_where_the_straggler_is_transient(runs):
    """The discriminator that intermittency adds.

    While the straggler was a permanent derate pinned to one GPU, these two
    scenarios produced near-identical throughput and spread and only device
    counters could separate them. An episodic straggler adds a second,
    purely temporal discriminator that needs no device counter at all:
    `gpu_degradation` owns the barrier in every single timestep, because the
    fault never goes away, while the episodic straggler keeps handing it back.

    Note what this does NOT say. Neither run throttles on demand, and both look
    like "a slow rank" to any detector that only reports a fleet average -- the
    separation lives in *when*, not in *how much*.
    """
    def pacer(frames):
        r = frames["rank_performance"]
        r = r[r["iteration"] > LATE].copy()
        r["busy"] = r["compute_time_s"] + r["halo_wait_s"]
        return r.loc[r.groupby("iteration")["busy"].idxmax(), "rank_id"]

    degraded, episodic = pacer(runs["gpu_degradation"].frames), pacer(runs["straggler"].frames)

    #  One device owns the barrier for essentially the whole late window...
    assert degraded.nunique() == 1
    #  ...where the episodic straggler's pacer keeps changing.
    assert episodic.nunique() > degraded.nunique()
    assert episodic.value_counts(normalize=True).iloc[0] < 1.0


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
    #  The straggler's pacer has to be taken per-episode now, since outside an
    #  episode it is not pacing anything.
    t_late = runs["gpu_degradation"].frames["job_performance"]
    t_late = t_late[t_late["iteration"] == LATE]["timestamp"].iloc[0]
    deg_late = deg[deg["timestamp"] > t_late]
    deg_victim = deg_late[deg_late["gpu_id"] == "r3n1g2"]["sm_occupancy_pct"].mean()

    strag_victim = _split_by_episode(runs["straggler"], strag)[0]["sm_occupancy_pct"].mean()
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


# ---------------------------------------------------------------------------
# the signature matrix itself
# ---------------------------------------------------------------------------

def test_every_fault_differs_from_healthy_somewhere_in_the_signature_matrix(runs):
    """No faulted scenario may be cell-identical to `healthy`.

    The matrix's own claim is that you can read across a row to find the channel
    that separates two scenarios. If a fault matches `healthy` in every cell,
    that claim is simply false for it, and the page contradicts its own
    diagnosis panel.

    This is a regression guard with a specific history. Almost every row is an
    early-vs-late comparison, which is structurally blind to a fault already
    running during the early window; when the straggler's episodes were moved to
    start at timestep 1, its entire column went flat and nothing caught it. The
    `barrier_pacers` row exists because it needs no baseline.
    """
    from gcsim.dashboard.build import _signature_row

    signatures = {name: _signature_row(run.frames, run.frames["job_performance"])
                  for name, run in runs.items()}
    healthy_sig = signatures["healthy"]
    #  Counts carry detail for the tooltip, not a verdict; compare the cells the
    #  matrix actually renders.
    cells = [k for k in healthy_sig if not k.endswith(("_count", "_gpus", "_racks", "_domains"))]

    for name, sig in signatures.items():
        if name in ("healthy", "phase_change"):
            continue
        differing = [k for k in cells if sig[k] != healthy_sig[k]]
        assert differing, f"{name} is indistinguishable from healthy in every signature cell"


def test_the_pacing_row_does_not_fire_on_the_two_non_faults(runs):
    """...and the row added to catch the straggler must not cry wolf.

    `healthy` and `phase_change` are the two runs with no hardware fault in them.
    A detector that flagged either would have moved the problem rather than
    solved it: `phase_change` in particular slows the job by ~10% and is exactly
    the case a naive detector fires on.
    """
    from gcsim.dashboard.build import _signature_row

    for name in ("healthy", "phase_change", "network_domain"):
        sig = _signature_row(runs[name].frames, runs[name].frames["job_performance"])
        assert sig["barrier_pacers"] == "flat", name
        assert sig["barrier_pacer_count"] == "0", name

    #  ...while the three faults that do localise to hardware all register.
    for name in ("straggler", "thermal", "gpu_degradation"):
        sig = _signature_row(runs[name].frames, runs[name].frames["job_performance"])
        assert sig["barrier_pacers"] == "up", name
        assert int(sig["barrier_pacer_count"]) >= 1, name


# ---------------------------------------------------------------------------
# the connectivity card
# ---------------------------------------------------------------------------

def test_halo_faces_ride_the_links_the_card_claims(bundle, cluster, router):
    """+/-x on NVLink, +/-y inside the rack, +/-z across racks -- every mesh.

    The connectivity card states this arrangement as fact and takes its link
    classes from `placement.neighbour_kind`, so the picture cannot disagree with
    the model by construction. What it *can* do is quietly stop being true if the
    process grid or the placement strategy changes shape, at which point the card
    would keep drawing a topology nobody is running. Hence asserting the
    arrangement itself rather than the drawing of it.
    """
    from gcsim.mesh import DIRECTION_NAMES, partition
    from gcsim.placement import KIND_NAMES, place

    expected = {"-x": "intranode", "+x": "intranode",
                "-y": "intra_domain", "+y": "intra_domain",
                "-z": "cross_domain", "+z": "cross_domain"}

    for name, mesh in bundle.meshes.items():
        d = partition(mesh, cluster.n_gpus,
                      preferred_first_extent=bundle.cluster.gpus_per_node)
        placement = place(cluster, d, router, strategy=bundle.workload.placement)
        for rank in (0, 37, 99, cluster.n_gpus - 1):
            for i, direction in enumerate(DIRECTION_NAMES):
                got = KIND_NAMES[int(placement.neighbour_kind[rank, i])]
                assert got == expected[direction], f"{name} rank {rank} {direction}: {got}"


def test_the_racks_form_one_ring_through_z(bundle, cluster):
    """The +/-z neighbours close into a single cycle over all four racks.

    This is the property the card's ring drawing depends on, and the reason a
    fault on one rack surfaces on a rack that is not adjacent to it in id order.
    Two 2-cycles or a self-loop would still render as "a ring" while meaning
    something completely different about which racks a fault can reach.
    """
    from gcsim.mesh import partition

    per_rack = bundle.cluster.gpus_per_rack
    n_racks = cluster.n_gpus // per_rack
    assert n_racks > 2                     # a 2-rack cluster rings trivially

    for name, mesh in bundle.meshes.items():
        d = partition(mesh, cluster.n_gpus,
                      preferred_first_extent=bundle.cluster.gpus_per_node)
        ring = [int(d.neighbours[k * per_rack][5]) // per_rack for k in range(n_racks)]

        #  Walk +z from rack 0: it must reach every rack exactly once and return.
        visited, current = [], 0
        for _ in range(n_racks):
            current = ring[current]
            visited.append(current)
        assert sorted(visited) == list(range(n_racks)), f"{name}: {ring} is not one cycle"
        assert visited[-1] == 0, f"{name}: ring does not close, {visited}"
        assert all(ring[k] != k for k in range(n_racks)), f"{name}: self-loop in {ring}"


# ---------------------------------------------------------------------------
# partial run sets
# ---------------------------------------------------------------------------

def test_sparse_run_sets_build_per_seed_pages(bundle, tmp_path):
    """A seed holding one run must still produce a working page.

    Nothing guarantees a seed is a full sweep: `--seed 8 --scenarios straggler
    --meshes coarse` leaves exactly one directory, and that case shipped broken
    -- the payload builder assumed healthy runs exist (mesh_study) and the page
    assumed the default scenario x mesh combination exists. Every prior test
    built full sweeps, so the gap was reachable from the README and covered by
    nothing.
    """
    from gcsim.dashboard.build import build_dashboard, build_payload
    from gcsim.scenarios import run_scenario

    runs = tmp_path / "runs"
    run_scenario("straggler", mesh="coarse", seed=8, bundle=bundle, out_dir=runs)
    run_scenario("phase_change", mesh="coarse", seed=7, bundle=bundle, out_dir=runs)

    payload = build_payload(runs, seed=8)
    assert list(payload["runs"]) == ["straggler__coarse"]
    assert payload["meta"]["seed"] == 8
    #  No healthy runs -> no mesh study. The page must degrade, not throw.
    assert payload["mesh_study"] == []
    #  Geometry is config-derived, so it is complete even on a sparse seed.
    assert sorted(payload["partitions"]) == sorted(bundle.meshes)

    #  Seeds are COMPLETELY separate (a deliberate layout change): every seed
    #  gets its own standalone index_seed{N}.html, there is no chooser or
    #  shared index.html, and pages carry no links to each other. A stale
    #  index.html from the earlier layout must be removed, not orphaned at the
    #  best-known filename.
    dash = tmp_path / "dash"
    dash.mkdir()
    (dash / "index.html").write_text("stale chooser from an older build")
    paths, _ = build_dashboard(runs, out_path=dash / "index.html")
    assert [q.name for q in paths] == ["index_seed7.html", "index_seed8.html"]
    assert not (dash / "index.html").exists()
    page = paths[1].read_text(encoding="utf-8")
    assert "seed_links" not in page
    assert '"seed":8' in page


def test_wait_heatmap_step_aligns_with_injection_marker(bundle, tmp_path):
    """The wait channel must sit on the same wall-clock axis as the marker.

    The wait rows used to be placed by iteration NUMBER scaled onto the axis,
    which assumes every timestep costs the same wall time. A fault breaks that
    assumption by definition: gpu_degradation lengthens every post-injection
    timestep, so iteration 350 sits at 30% of the wall clock but 35% of the
    count, and the wait step rendered several bins to the right of the
    injection marker -- a phantom delay, while occupancy and temperature
    (binned by real timestamps) moved at the true moment.
    """
    import numpy as np
    from gcsim.dashboard.build import build_payload
    from gcsim.scenarios import run_scenario

    runs = tmp_path / "runs"
    run_scenario("gpu_degradation", mesh="medium", seed=42, bundle=bundle, out_dir=runs)
    run = build_payload(runs, seed=42)["runs"]["gpu_degradation__medium"]

    inj_t = float(run["marks"][0]["t"])
    centres = np.asarray(run["ranks"]["t"], dtype=float)
    width = float(centres[1] - centres[0])
    fleet_wait = np.asarray(run["ranks"]["wait"], dtype=float).mean(axis=0)

    #  Baseline from bins that end comfortably before the injection.
    pre = fleet_wait[centres < inj_t - width]
    assert pre.size >= 3
    stepped = np.flatnonzero(fleet_wait > max(3.0 * pre.mean(), pre.mean() + 5.0))
    assert stepped.size, "no wait step found at all"

    #  The step is physically instantaneous (the victim slows in the injection
    #  iteration itself), so the first elevated bin must be the one holding the
    #  injection time -- one bin of slack for the boundary bin's blend.
    assert abs(centres[stepped[0]] - inj_t) <= 1.5 * width, (
        f"wait step at t={centres[stepped[0]]:.1f}s but injection at "
        f"t={inj_t:.1f}s (bin width {width:.2f}s)")

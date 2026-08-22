"""Reproducibility.

The seeding scheme keys every stochastic stream by the entity's *identity*
rather than its index. The payoff is tested here: two runs at the same seed are
bit-identical, and -- more usefully -- a healthy run and a faulted run share
every stream they have in common, so any difference between them is a
*consequence of the injection* rather than a different roll of the dice.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gcsim.config import derive_rng, load_config
from gcsim.engine.simulator import Simulator
from gcsim.scenarios import run_scenario
from gcsim.telemetry import frame_digest

FAST_MESH = "coarse"      # ~0.5 s per run


def test_same_seed_reproduces_every_stream_exactly(bundle):
    a = run_scenario("thermal", mesh=FAST_MESH, seed=7, bundle=bundle, out_dir=None)
    b = run_scenario("thermal", mesh=FAST_MESH, seed=7, bundle=bundle, out_dir=None)
    assert frame_digest(a.frames) == frame_digest(b.frames)


def test_different_seeds_differ_in_detail_but_not_in_character(bundle):
    a = run_scenario("healthy", mesh=FAST_MESH, seed=7, bundle=bundle, out_dir=None)
    b = run_scenario("healthy", mesh=FAST_MESH, seed=8, bundle=bundle, out_dir=None)

    assert frame_digest(a.frames) != frame_digest(b.frames)
    #  Same physics, so the summary statistics must agree closely.
    for key in ("mean_iteration_time_s", "parallel_efficiency",
                "mean_sm_occupancy_pct", "mean_temperature_c"):
        assert b.summary[key] == pytest.approx(a.summary[key], rel=0.02), key
    assert a.summary["diagnosis"]["verdict"] == b.summary["diagnosis"]["verdict"]


def test_keys_are_stable_strings_not_positions():
    """The property that makes healthy and faulted runs diffable.

    Streams are keyed by entity identity, so adding a rack, reordering a loop or
    introducing a new scenario cannot perturb an unrelated GPU's noise.
    """
    a = derive_rng(42, "gpu:r1n2g5").normal(size=5)
    b = derive_rng(42, "gpu:r1n2g5").normal(size=5)
    c = derive_rng(42, "gpu:r1n2g6").normal(size=5)
    d = derive_rng(43, "gpu:r1n2g5").normal(size=5)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_silicon_variation_is_identical_across_scenarios(bundle):
    """Every GPU's manufacturing draw is the same in every scenario.

    So when a faulted run diverges from the healthy one, the divergence is the
    fault -- not a GPU that happened to be drawn slower this time.
    """
    sims = [Simulator(bundle.build(name, mesh=FAST_MESH, seed=42))
            for name in ("healthy", "straggler", "thermal", "phase_change")]
    reference = sims[0]
    for sim in sims[1:]:
        assert np.array_equal(sim._clock_offset, reference._clock_offset)
        assert np.array_equal(sim._leakage, reference._leakage)
        assert np.array_equal(sim._noise, reference._noise)


def test_healthy_and_faulted_runs_agree_before_the_injection(bundle):
    """Nothing diverges until the injection actually fires.

    Every timestep before the injection must be identical to the healthy run, to
    the last bit.

    Uses `gpu_degradation` rather than `straggler`, because the straggler's
    episodes now begin at timestep 1 precisely so that it has no clean baseline
    -- which makes it the one scenario that cannot answer this question. Not
    `thermal` either: on the coarse mesh used here the dies never reach the
    slowdown threshold, so a cooling failure costs no wall time at all and the
    divergence half of this test would be vacuous. `gpu_degradation` moves
    compute time directly, on any mesh.

    The injection timestep is read from config so this cannot silently drift out
    of date again when a scenario is retimed.
    """
    cfg = bundle.build("gpu_degradation", mesh=FAST_MESH, seed=42)
    fires_at = min(i.at_iteration for i in cfg.scenario.injections)

    healthy = run_scenario("healthy", mesh=FAST_MESH, seed=42, bundle=bundle, out_dir=None)
    faulted = run_scenario("gpu_degradation", mesh=FAST_MESH, seed=42, bundle=bundle, out_dir=None)

    h = healthy.frames["job_performance"]
    f = faulted.frames["job_performance"]
    before = h["iteration"] < fires_at

    assert before.sum() > 10
    assert np.array_equal(h.loc[before, "iteration_time_s"].to_numpy(),
                          f.loc[before, "iteration_time_s"].to_numpy())
    #  ...and they diverge afterwards, so the test is not vacuous. Checked with
    #  room to spare rather than on the very next timestep, so it stays true if
    #  the injection is ever given a ramp.
    after = h["iteration"] > fires_at + 150
    assert not np.array_equal(h.loc[after, "iteration_time_s"].to_numpy(),
                              f.loc[after, "iteration_time_s"].to_numpy())


def test_config_loading_is_pure(bundle):
    """Loading config twice gives equal values, and a run never mutates it."""
    fresh = load_config()
    assert fresh.cluster == bundle.cluster
    assert fresh.scenario_order == bundle.scenario_order

    run_scenario("network_domain", mesh=FAST_MESH, seed=42, bundle=bundle, out_dir=None)
    assert load_config().cluster == bundle.cluster


def test_written_run_round_trips(bundle, tmp_path):
    from gcsim.telemetry import read_run

    result = run_scenario("healthy", mesh=FAST_MESH, seed=42, bundle=bundle,
                          out_dir=tmp_path)
    assert result.path is not None and result.path.exists()

    frames, summary = read_run(result.path)
    assert frame_digest(frames) == frame_digest(result.frames)
    assert summary["run_id"] == result.run_id
    assert summary["diagnosis"]["verdict"] == result.summary["diagnosis"]["verdict"]


def test_a_whole_run_directory_is_byte_identical(bundle, tmp_path):
    """Two identical invocations must produce identical files on disk.

    Not just identical dataframes -- identical bytes, summary.json included.
    Wall-clock timing is deliberately kept out of the persisted summary for
    exactly this reason: it measures the machine, not the simulation.
    """
    import filecmp

    a = run_scenario("thermal", mesh=FAST_MESH, seed=42, bundle=bundle, out_dir=tmp_path / "a")
    b = run_scenario("thermal", mesh=FAST_MESH, seed=42, bundle=bundle, out_dir=tmp_path / "b")

    cmp = filecmp.dircmp(a.path, b.path)
    assert not cmp.left_only and not cmp.right_only
    matched, mismatched, errors = filecmp.cmpfiles(
        a.path, b.path, cmp.common_files, shallow=False)
    assert not mismatched and not errors, f"differed: {mismatched} {errors}"
    assert len(matched) >= 10          # nine tables plus summary.json

    assert "wall_seconds" not in json.loads((a.path / "summary.json").read_text())

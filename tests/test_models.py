"""The physical models, tested in isolation from the engine."""

from __future__ import annotations

import numpy as np
import pytest

from gcsim.models.compute import (achieved_occupancy, compute_time_s,
                                  fixed_overhead_s, ideal_compute_time_s)
from gcsim.models.power import (board_power_w, clock_factor_for_power_cap,
                                occupancy_active, reported_utilisation)
from gcsim.models.storage import StorageModel
from gcsim.models.thermal import DeviceGovernor, inlet_temperature_c


# --- compute ---------------------------------------------------------------

def test_occupancy_saturates_in_subdomain_size(bundle):
    g = bundle.cluster.gpu
    cells = np.array([1e3, 1e5, g.occupancy_half_cells, 1e7, 1e9])
    occ = achieved_occupancy(cells, g)
    assert (np.diff(occ) > 0).all()
    assert occ[2] == pytest.approx(0.5)
    assert occ[-1] < 1.0 and occ[0] < 0.05


def test_losing_memory_bandwidth_drags_occupancy_down(bundle):
    """Stalled SMs are occupied but not retiring work.

    This coupling is what makes `gpu_degradation` visible in the occupancy
    channel while a plain straggler is not.
    """
    g = bundle.cluster.gpu
    cells = np.array([1.7e6])
    assert achieved_occupancy(cells, g, 0.85) < achieved_occupancy(cells, g, 1.0)


def test_compute_time_responds_to_each_health_term(bundle):
    """Each health term scales the variable part; the fixed overhead does not.

    Halving the clock does NOT halve the timestep, because 300 kernel launches
    cost the same either way. That is the correct behaviour and it is why a
    throttled GPU loses less throughput than its clock drop suggests.
    """
    g = bundle.cluster.gpu
    cells = np.array([1.7e6])
    fixed = fixed_overhead_s(g, 20)

    def variable(**kw):
        return compute_time_s(cells, g, 20, **kw)[0] - fixed

    base = variable()
    assert variable(clock_factor=0.5) == pytest.approx(2 * base)
    assert variable(throughput_derate=0.75) == pytest.approx(base / 0.75)
    #  Memory bandwidth acts through ONE mechanism -- stalled SMs -- so it costs
    #  throughput proportionally. Modelling it separately from occupancy as well
    #  would double-count. The point is that this single mechanism produces TWO
    #  observables: the rank slows AND its occupancy visibly drops. A stolen-SM
    #  straggler slows by the same amount with occupancy untouched, and that is
    #  the only thing separating the two scenarios.
    assert variable(memory_bandwidth_factor=0.85) == pytest.approx(base / 0.85)
    assert achieved_occupancy(cells, g, 0.85)[0] < achieved_occupancy(cells, g, 1.0)[0]
    assert compute_time_s(cells, g, 20, throughput_derate=0.85)[0] == pytest.approx(
        compute_time_s(cells, g, 20, memory_bandwidth_factor=0.85)[0])

    #  ...and the total is strictly less than a naive clock-proportional guess.
    assert compute_time_s(cells, g, 20, clock_factor=0.5)[0] < 2 * (fixed + base)


def test_cost_asymptotes_as_the_subdomain_vanishes(bundle):
    """Strong scaling stops paying: below some size you pay for the device anyway.

    Time goes as cells/occupancy and occupancy as cells/(cells+half), so the cost
    tends to `occupancy_half_cells * seconds_per_cell_update` plus the fixed
    per-timestep overhead. Halving a tiny subdomain buys almost nothing.
    """
    g = bundle.cluster.gpu
    fixed = fixed_overhead_s(g, 20)
    asymptote = fixed + g.occupancy_half_cells * g.seconds_per_cell_update

    assert compute_time_s(np.array([1.0]), g, 20)[0] == pytest.approx(asymptote, rel=1e-4)
    tiny = compute_time_s(np.array([2000.0]), g, 20)[0]
    half_of_tiny = compute_time_s(np.array([1000.0]), g, 20)[0]
    assert half_of_tiny > 0.98 * tiny          # halving the work saves ~nothing

    #  A well-sized subdomain is nowhere near the asymptote.
    assert compute_time_s(np.array([1.7e6]), g, 20)[0] > 5 * asymptote


def test_ideal_time_is_unattainable(bundle):
    """The efficiency denominator assumes perfect balance and full occupancy."""
    g = bundle.cluster.gpu
    ideal = ideal_compute_time_s(216_000_000, 128, g)
    actual = compute_time_s(np.array([216_000_000 / 128]), g, 20)[0]
    assert ideal < actual


# --- power -----------------------------------------------------------------

def test_utilisation_stays_high_while_occupancy_collapses(bundle):
    """The barrier-stall fingerprint, at the level of the model.

    A rank doing 5% of a window's work still reports ~100% utilisation because
    the collective's spin kernel remains resident. Any model that derived power
    from utilisation would miss this entirely.
    """
    g = bundle.cluster.gpu
    busy = np.array([1.0])
    idle = np.array([0.05])
    occ = np.array([0.87])

    assert reported_utilisation(idle, g)[0] > 0.98
    assert reported_utilisation(busy, g)[0] == pytest.approx(1.0)
    assert occupancy_active(idle, occ, g)[0] < 0.2 * occupancy_active(busy, occ, g)[0]


def test_power_falls_with_occupancy_and_with_clock(bundle):
    g = bundle.cluster.gpu
    leak = np.array([1.0])
    full = board_power_w(np.array([0.87]), np.array([1.0]), leak, g)[0]
    slow = board_power_w(np.array([0.87]), np.array([0.8]), leak, g)[0]
    spinning = board_power_w(np.array([0.05]), np.array([1.0]), leak, g)[0]

    assert spinning < slow < full <= g.board_power_cap_w
    assert spinning == pytest.approx(g.idle_power_w + (g.max_power_w - g.idle_power_w) * 0.05)
    #  Power sheds faster than clock: exponent 2.2 on the frequency term.
    assert slow / full < 0.8 / 1.0


def test_leaky_parts_hit_the_power_cap_first(bundle):
    g = bundle.cluster.gpu
    occ = np.array([1.0, 1.0])
    leak = np.array([0.90, 1.12])
    factors = clock_factor_for_power_cap(occ, leak, g)
    assert factors[1] < factors[0] <= 1.0


# --- thermal ---------------------------------------------------------------

def test_inlet_temperature_is_a_rack_property(bundle):
    cc = bundle.cluster
    load = np.array([20.0, 20.0])
    healthy = inlet_temperature_c(load, np.array([1.0, 1.0]), cc)
    degraded = inlet_temperature_c(load, np.array([0.30, 1.0]), cc)

    assert healthy[0] == pytest.approx(cc.cooling.base_inlet_temp_c)
    assert degraded[0] > degraded[1] == pytest.approx(cc.cooling.base_inlet_temp_c)


def _settle(gov, inlet, occ=0.87, steps=400, dt=1.0):
    n = gov.n
    for _ in range(steps):
        gov.step(dt, np.full(n, occ), np.full(n, inlet), np.ones(n))
    return gov


def test_healthy_load_never_throttles(bundle):
    gov = DeviceGovernor(bundle.cluster, 4, np.ones(4))
    _settle(gov, bundle.cluster.cooling.base_inlet_temp_c)
    assert not gov.throttled.any()
    assert gov.temperature_c.max() < bundle.cluster.gpu.thermal_slowdown_c
    assert (gov.reason == "NONE").all()


def test_the_loop_runs_temperature_up_then_clock_down_then_power_down(bundle):
    """The one place an output feeds back into an input.

    Order matters: raising the inlet must move temperature first, and only then
    the clock, and power must follow the clock rather than lead it.
    """
    g = bundle.cluster.gpu
    gov = DeviceGovernor(bundle.cluster, 4, np.ones(4))
    _settle(gov, bundle.cluster.cooling.base_inlet_temp_c)
    cool_t, cool_clk, cool_p = (gov.temperature_c.mean(), gov.clock_factor.mean(),
                                gov.power_w.mean())

    _settle(gov, 60.0)
    assert gov.temperature_c.mean() > cool_t
    assert gov.throttled.all() and (gov.reason == "THERMAL").all()
    assert gov.clock_factor.mean() < cool_clk
    assert gov.power_w.mean() < cool_p
    assert gov.temperature_c.max() >= g.thermal_slowdown_c


def test_clock_moves_in_hardware_steps(bundle):
    g = bundle.cluster.gpu
    gov = DeviceGovernor(bundle.cluster, 4, np.ones(4))
    _settle(gov, 60.0)
    remainder = np.mod(gov.clock_mhz, g.clock_step_mhz)
    assert np.allclose(remainder, 0.0)
    assert (gov.clock_mhz >= g.min_clock_mhz).all()


def test_hysteresis_prevents_flapping(bundle):
    """Once engaged, throttling holds until the die is clearly back under."""
    g = bundle.cluster.gpu
    gov = DeviceGovernor(bundle.cluster, 2, np.ones(2))
    _settle(gov, 60.0)
    assert gov.throttled.all()

    #  Cool to just inside the hysteresis band: still throttled.
    n = gov.n
    gov.temperature_c[:] = g.thermal_slowdown_c - g.thermal_hysteresis_c + 0.5
    gov._apply_governors(np.full(n, 0.87), np.ones(n))
    assert gov.throttled.all()

    #  Clearly below: released.
    gov.temperature_c[:] = g.thermal_slowdown_c - g.thermal_hysteresis_c - 5.0
    gov._apply_governors(np.full(n, 0.87), np.ones(n))
    assert not gov.throttled.any()


def test_reliability_cap_is_attributed_separately(bundle):
    """A RAS-capped clock must not be reported as a thermal event."""
    gov = DeviceGovernor(bundle.cluster, 2, np.ones(2))
    n = gov.n
    gov.temperature_c[:] = 50.0
    gov._apply_governors(np.full(n, 0.5), np.array([1.0, 0.90]))
    assert gov.reason[0] == "NONE" and not gov.throttled[0]
    assert gov.reason[1] == "RELIABILITY" and gov.throttled[1]
    assert gov.clock_factor[1] < gov.clock_factor[0]


# --- storage ---------------------------------------------------------------

def test_concurrent_writes_queue_superlinearly(bundle):
    s = StorageModel(spec=bundle.cluster.storage)
    latency, transfer = s.write(8.64e9)
    #  Queueing dominates: latency alone is many times the base service time.
    assert latency > bundle.cluster.storage.base_write_latency_ms * 1e-3 * 10
    assert transfer == pytest.approx(8.64e9 / s.capacity_bps, rel=0.05)


def test_writeback_backlog_drains_and_slows_the_next_write(bundle):
    """Why a heavy output campaign raises the baseline, not just the peaks."""
    s = StorageModel(spec=bundle.cluster.storage)
    _, clean = s.write(8.64e9)
    assert s.dirty_bytes > 0

    _, contended = s.write(8.64e9)     # backlog still draining
    assert contended > clean

    s.advance(1000.0)
    assert s.dirty_bytes == 0
    assert s.background_utilisation() == 0.0


def test_storage_sample_is_a_window_average(bundle):
    """Reported latency is time-weighted, which is how an exporter reports.

    It is also why a short stall inside a sample window gets smeared rather than
    showing up as a spike.
    """
    s = StorageModel(spec=bundle.cluster.storage)
    s.advance(10.0)
    quiet = s.sample()
    s.write(8.64e9)
    s.advance(0.1)
    busy = s.sample()
    assert busy["write_latency_ms"] > quiet["write_latency_ms"] * 5
    assert busy["utilisation_pct"] > quiet["utilisation_pct"]
    assert 0.0 <= quiet["utilisation_pct"] <= 100.0

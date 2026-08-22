"""Temperature, and the governors that respond to it.

This module owns the simulator's central feedback loop. Everything else is a
one-way function; this is the only place where an output feeds back into an
input:

    work  ->  occupancy  ->  power  ->  temperature  ->  throttle  ->  clock
                  ^                                                     |
                  +-----------------------------------------------------+
                        (applies to the NEXT interval, not this one)

The one-interval lag is deliberate and is why the thermal scenario produces a
*ramp* followed by a clock staircase rather than an instantaneous step. It also
means a throttling GPU overshoots slightly before settling, which is what real
parts do.

Temperature uses a first-order lag driven by the rack's cold-aisle inlet:

    dT/dt = (P * R_th - (T - T_inlet)) / tau

with steady state ``T = T_inlet + P * R_th``. At 700 W and R_th = 0.06 that is
42 C above inlet, so a healthy rack at 24 C inlet settles around 66 C.

`T_inlet` is a **rack** property. Degrading one CRAC therefore moves all 32 GPUs
in that rack together, which is what turns a thermal fault into a contiguous
block in the rank heatmap rather than a scatter of hot GPUs.
"""

from __future__ import annotations

import numpy as np

from gcsim.config import ClusterConfig
from gcsim.models.power import board_power_w, clock_factor_for_power_cap

#: Ordered by precedence: the first governor that binds names the reason.
NONE = "NONE"
THERMAL = "THERMAL"
POWER_CAP = "POWER_CAP"
RELIABILITY = "RELIABILITY"


def inlet_temperature_c(rack_power_kw: np.ndarray, cooling_efficiency: np.ndarray,
                        cfg: ClusterConfig) -> np.ndarray:
    """Cold-aisle inlet temperature per rack.

    A CRAC running below capacity fails to remove the rack's own heat, so the
    inlet climbs in proportion to what the rack is dissipating:

        inlet = base + (1/efficiency - 1) * rack_power_kw * coupling

    At full efficiency the second term vanishes and every rack sits at base.
    Note the coupling to `rack_power_kw`: a rack full of throttled (and
    therefore cooler-running) GPUs heats its own inlet less, so the loop is
    self-limiting rather than divergent.
    """
    eff = np.clip(np.asarray(cooling_efficiency, np.float64), 1e-3, None)
    excess = (1.0 / eff) - 1.0
    return cfg.cooling.base_inlet_temp_c + excess * rack_power_kw * cfg.cooling.coupling_c_per_kw


class DeviceGovernor:
    """Per-GPU thermal / power / reliability state, vectorised over the fleet.

    Holds the mutable dynamic state for all GPUs as numpy arrays. `step` advances
    it by one telemetry sample interval.
    """

    def __init__(self, cfg: ClusterConfig, n_gpus: int, leakage: np.ndarray):
        self.cfg = cfg
        self.gpu = cfg.gpu
        self.n = n_gpus
        self.leakage = leakage

        self.temperature_c = np.full(n_gpus, cfg.cooling.base_inlet_temp_c)
        self.power_w = np.full(n_gpus, cfg.gpu.idle_power_w)
        self.clock_factor = np.ones(n_gpus)
        self.throttled = np.zeros(n_gpus, dtype=bool)
        self.reason = np.full(n_gpus, NONE, dtype=object)
        #  The job under study is assumed to have been running long enough to be
        #  at thermal equilibrium. Starting the dies at inlet temperature instead
        #  would spend the first ~3 tau (a minute of simulated time) climbing a
        #  cold-start transient that has nothing to do with the phenomena being
        #  studied, and would sit right on top of the injection points.
        self._warm_start = True

    # -- the loop ---------------------------------------------------------

    def step(self, dt_s: float, occ_active: np.ndarray, inlet_c: np.ndarray,
             reliability_cap: np.ndarray, window_is_representative: bool = True) -> None:
        """Advance one sample interval.

        Order matters. Power and temperature are evaluated with the clock that
        was in force *during* the interval; the governors then set the clock for
        the *next* interval. Reversing this would collapse the feedback loop
        into an algebraic identity and lose the ramp entirely.
        """
        g = self.gpu

        # 1. power drawn during the interval, at the clock that was in force
        raw = np.minimum(board_power_w(occ_active, self.clock_factor, self.leakage, g),
                         g.board_power_cap_w)

        #  Warm start. The job under study is assumed to have been running long
        #  enough to be at equilibrium, so on the first REPRESENTATIVE window --
        #  one that is all steady-state timestepping, with no mesh load or field
        #  output in it -- both lags are snapped to their settled values at once:
        #  the power sensor's EMA and the thermal RC. Snapping only the thermal
        #  one would leave the dies chasing a power reading that is itself still
        #  climbing, and either way a transient lasting ~3 tau would run straight
        #  through the injection points and contaminate every before/after
        #  comparison in the study.
        if self._warm_start and window_is_representative                 and float(np.mean(occ_active)) > 2.0 * g.spin_occupancy:
            self.power_w = raw.copy()
            self.temperature_c = inlet_c + raw * g.thermal_resistance_c_per_w
            self._warm_start = False
            self._apply_governors(occ_active, reliability_cap)
            return

        alpha = g.power_ema_alpha
        self.power_w = alpha * raw + (1.0 - alpha) * self.power_w

        # 2. temperature responds with a first-order lag
        steady = inlet_c + self.power_w * g.thermal_resistance_c_per_w
        decay = np.exp(-dt_s / g.thermal_time_constant_s)
        self.temperature_c = steady + (self.temperature_c - steady) * decay

        # 3. governors set the clock for the next interval
        self._apply_governors(occ_active, reliability_cap)

    def _apply_governors(self, occ_active: np.ndarray, reliability_cap: np.ndarray) -> None:
        g = self.gpu
        floor = g.min_clock_mhz / g.base_clock_mhz

        # --- thermal, with hysteresis so it does not flap on the threshold ---
        engage = self.temperature_c >= g.thermal_slowdown_c
        hold = self.throttled & (self.temperature_c >= g.thermal_slowdown_c - g.thermal_hysteresis_c)
        thermally_limited = engage | hold
        over = np.maximum(self.temperature_c - g.thermal_slowdown_c, 0.0)
        #  While held inside the hysteresis band the die is below the threshold,
        #  so the proportional term is zero -- but the part must NOT jump back to
        #  full boost, or it would immediately reheat and re-engage. Real parts
        #  hold at least one clock bin down until they are clearly clear, so the
        #  ceiling while limited is one step below nominal.
        one_step = g.clock_step_mhz / g.base_clock_mhz
        thermal_factor = np.where(
            thermally_limited,
            np.clip(1.0 - g.thermal_derate_per_c * over, floor, 1.0 - one_step),
            1.0,
        )

        # --- power cap: leaky parts bind first -----------------------------
        power_factor = clock_factor_for_power_cap(occ_active, self.leakage, g)
        power_factor = np.clip(power_factor, floor, 1.0)

        # --- RAS governor: set by gpu_degradation --------------------------
        rel_factor = np.clip(reliability_cap, floor, 1.0)

        stack = np.stack([thermal_factor, power_factor, rel_factor])
        factor = stack.min(axis=0)

        # Quantise to real clock steps -- throttling shows up as a staircase.
        mhz = np.floor(factor * g.base_clock_mhz / g.clock_step_mhz) * g.clock_step_mhz
        mhz = np.clip(mhz, g.min_clock_mhz, g.base_clock_mhz)
        self.clock_factor = mhz / g.base_clock_mhz

        # Attribute the limit to whichever governor actually bound. Precedence
        # matters only for exact ties, and thermal is the most consequential.
        binding = np.argmin(stack, axis=0)
        limited = factor < 1.0 - 1e-12
        names = np.array([THERMAL, POWER_CAP, RELIABILITY], dtype=object)
        self.reason = np.where(limited, names[binding], NONE)
        self.throttled = limited

    # -- readouts ---------------------------------------------------------

    @property
    def clock_mhz(self) -> np.ndarray:
        return self.clock_factor * self.gpu.base_clock_mhz

    def hw_slowdown(self) -> np.ndarray:
        """GPUs past the hardware slowdown threshold, not just the software one."""
        return self.temperature_c >= self.gpu.thermal_hw_slowdown_c

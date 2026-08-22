"""Board power.

The important modelling choice: **dynamic power tracks occupancy, not
utilisation.** A rank blocked in a collective keeps a spin kernel resident, so
`utilization_pct` stays pinned near 100 while the SMs retire almost nothing.
Power follows the work actually done, so it falls. That divergence between two
channels that naive models treat as the same signal is the single clearest
fingerprint of a synchronisation stall, and it only exists because the two are
computed from different quantities here.

    P = P_idle * leakage + (P_max - P_idle) * occupancy_active * (f/f_base)^k

The exponent ``k ~ 2.2`` comes from dynamic power going as C*V^2*f with voltage
scaling roughly linearly with frequency over the DVFS range. It is why a
throttled GPU sheds power faster than it sheds clock -- and why clock and power
falling *together* under sustained load is the throttling signature, as opposed
to power alone falling when a GPU simply runs out of work.
"""

from __future__ import annotations

import numpy as np

from gcsim.config import GpuSpec


def occupancy_active(compute_fraction: np.ndarray, occupancy: np.ndarray,
                     gpu: GpuSpec) -> np.ndarray:
    """Time-weighted occupancy over a sample window.

    `compute_fraction` is the share of the window spent doing real work; the
    remainder is spent spinning in the collective at `spin_occupancy`, which is
    small but not zero -- a spin kernel does burn some power.
    """
    compute_fraction = np.clip(np.asarray(compute_fraction, np.float64), 0.0, 1.0)
    return compute_fraction * occupancy + (1.0 - compute_fraction) * gpu.spin_occupancy


def reported_utilisation(compute_fraction: np.ndarray, gpu: GpuSpec) -> np.ndarray:
    """What `nvidia-smi utilization.gpu` would report.

    It measures the fraction of time *any* kernel is resident, and the collective
    spins rather than blocking, so a rank waiting at a barrier still reports
    almost fully busy. Deliberately near-flat -- that is the point.
    """
    compute_fraction = np.clip(np.asarray(compute_fraction, np.float64), 0.0, 1.0)
    return compute_fraction + (1.0 - compute_fraction) * gpu.spin_utilisation


def board_power_w(occ_active: np.ndarray, clock_factor: np.ndarray,
                  leakage: np.ndarray, gpu: GpuSpec) -> np.ndarray:
    """Instantaneous board power, before the cap is applied."""
    dynamic = (gpu.max_power_w - gpu.idle_power_w) * occ_active \
        * np.power(np.clip(clock_factor, 1e-6, None), gpu.power_clock_exponent)
    return gpu.idle_power_w * leakage + dynamic


def clock_factor_for_power_cap(occ_active: np.ndarray, leakage: np.ndarray,
                               gpu: GpuSpec) -> np.ndarray:
    """Largest clock factor that keeps the board inside its power cap.

    Invert the power expression for f/f_base. Leaky parts hit this first, which
    is why in a real fleet the same workload power-caps some GPUs and not
    others.
    """
    headroom = gpu.board_power_cap_w - gpu.idle_power_w * leakage
    denom = (gpu.max_power_w - gpu.idle_power_w) * np.maximum(occ_active, 1e-9)
    ratio = np.clip(headroom / denom, 1e-6, None)
    return np.clip(np.power(ratio, 1.0 / gpu.power_clock_exponent), 0.0, 1.0)

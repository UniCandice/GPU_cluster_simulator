"""Compute cost of advancing one subdomain by one outer timestep.

Three effects, all of which the mesh study depends on:

1.  **Work scales with owned cells.** `seconds_per_cell_update` is the cost at
    base clock and full SM occupancy, inclusive of every inner linear-solver
    iteration.

2.  **Small subdomains cannot fill the device.** Achieved occupancy follows a
    saturation curve ``cells / (cells + occupancy_half_cells)``. A rank holding
    120k cells reaches only about a third of peak, so its effective throughput
    is a third of peak too. This is why the coarse mesh is inefficient for a
    *modelled* reason rather than a fudge factor -- and it surfaces directly as
    the `sm_occupancy_pct` telemetry channel.

3.  **A stencil solver is memory-bandwidth bound.** Losing HBM bandwidth costs
    throughput roughly proportionally, and it drags achieved occupancy down with
    it because the SMs sit stalled on memory. That coupling is what separates
    `gpu_degradation` from `straggler` in the occupancy channel.

On top of all of that sits a fixed per-timestep overhead: 15 kernels per inner
iteration over 20 inner iterations is 300 launches and synchronisations, costing
1.5 ms whatever they are working on.

One consequence is worth stating explicitly. Because time goes as
``cells / occupancy`` and occupancy goes as ``cells / (cells + half)``, the cost
asymptotes to ``occupancy_half_cells * seconds_per_cell_update`` -- about 22 ms
here -- as the subdomain shrinks toward nothing. That is not an artefact: it is
the model saying that below some size you pay for the device whether you fill it
or not, which is precisely why strong scaling stops paying. It also means
`coarse` is slow for a *derived* reason rather than a tuned constant.
"""

from __future__ import annotations

import numpy as np

from gcsim.config import GpuSpec


def achieved_occupancy(cells: np.ndarray, gpu: GpuSpec,
                       memory_bandwidth_factor: np.ndarray | float = 1.0) -> np.ndarray:
    """SM occupancy actually achieved while the solver kernels are resident.

    Saturating in subdomain size, and scaled down by any loss of memory
    bandwidth (stalled SMs are occupied but not retired).
    """
    cells = np.asarray(cells, dtype=np.float64)
    fill = cells / (cells + gpu.occupancy_half_cells)
    return np.clip(fill * np.asarray(memory_bandwidth_factor, dtype=np.float64), 0.0, 1.0)


def compute_time_s(cells: np.ndarray, gpu: GpuSpec, inner_iterations: int,
                   clock_factor: np.ndarray | float = 1.0,
                   memory_bandwidth_factor: np.ndarray | float = 1.0,
                   throughput_derate: np.ndarray | float = 1.0,
                   efficiency_noise: np.ndarray | float = 1.0) -> np.ndarray:
    """Seconds to advance each rank's subdomain by one outer timestep.

    Parameters
    ----------
    clock_factor
        Reported clock / base clock. Set by the thermal, power-cap and
        reliability governors -- visible in telemetry.
    memory_bandwidth_factor
        Achievable HBM bandwidth fraction. Visible only indirectly, through
        occupancy.
    throughput_derate
        Fraction of SM time this rank actually gets. A co-resident process
        lowers this and it is visible in NO device counter at all -- the GPU
        still boosts, still reports 100% utilisation, and is simply slower.
    """
    cells = np.asarray(cells, dtype=np.float64)
    occ = achieved_occupancy(cells, gpu, memory_bandwidth_factor)

    ideal = cells * gpu.seconds_per_cell_update
    scale = occ * np.asarray(clock_factor, np.float64) \
        * np.asarray(throughput_derate, np.float64) * np.asarray(efficiency_noise, np.float64)
    fixed = fixed_overhead_s(gpu, inner_iterations)
    return fixed + ideal / np.maximum(scale, 1e-9)


def fixed_overhead_s(gpu: GpuSpec, inner_iterations: int) -> float:
    """Per-timestep cost that does not shrink with the subdomain.

    15 kernels per inner iteration over 20 inner iterations is 300 launches and
    300 synchronisations per timestep, whatever each one is working on.
    """
    return gpu.kernels_per_inner_iteration * inner_iterations * gpu.kernel_launch_overhead_s


def ideal_compute_time_s(total_cells: int, n_ranks: int, gpu: GpuSpec) -> float:
    """Perfectly balanced, fully occupied, zero-communication reference time.

    The denominator of parallel efficiency. Deliberately unattainable: it
    assumes every rank owns exactly ``total_cells / n_ranks`` cells and that the
    device is saturated, so it isolates *everything* the real run loses to
    partitioning, occupancy, communication and faults.
    """
    return (total_cells / n_ranks) * gpu.seconds_per_cell_update

"""The distributed CFD job: phases and the mutable workload state.

Deliberately not a physics model. One outer timestep is:

    HALO_EXCHANGE -> COMPUTE -> ALLREDUCE -> [OUTPUT]

with a one-off DATA_LOAD before the first timestep. The CFD framing exists to
motivate domain decomposition; nothing here solves anything.

`WorkloadState` is mutable because a *legitimate* workload change is one of the
scenarios. When the run switches to a full-field output campaign it is this
object that changes -- not any piece of hardware -- which is the whole reason
that scenario is separable from a fault.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    DATA_LOAD = "DATA_LOAD"
    HALO_EXCHANGE = "HALO_EXCHANGE"
    COMPUTE = "COMPUTE"
    ALLREDUCE = "ALLREDUCE"
    OUTPUT = "OUTPUT"


@dataclass
class WorkloadState:
    """Job parameters that a scenario is allowed to change mid-run."""

    total_cells: int
    output_interval: int
    output_bytes_per_cell: float
    dataload_bytes_per_cell: float
    allreduce_values: int

    #: Multiplier applied by the `workload_output_change` injection. A restart
    #: checkpoint writes conserved state; a visualisation dump writes far more.
    output_bytes_scale: float = 1.0

    #: GPU ids the job actually occupies, in global index order; empty means
    #: the whole cluster. Filled by the Simulator from the placement. This is
    #: how the allocated set reaches faults.py, which may import workload but
    #: has no path to the placement -- an injection drawing victims must draw
    #: from GPUs that are running something, or it silently does nothing.
    allocated_gpu_ids: tuple[str, ...] = ()

    def is_output_iteration(self, iteration: int) -> bool:
        #  Iterations are 1-based for this test so that timestep `output_interval`
        #  is the first one that writes, rather than timestep 0.
        return iteration > 0 and iteration % self.output_interval == 0

    def output_bytes(self) -> float:
        return self.total_cells * self.output_bytes_per_cell * self.output_bytes_scale

    def dataload_bytes(self) -> float:
        return self.total_cells * self.dataload_bytes_per_cell

    def allreduce_bytes(self) -> float:
        return float(self.allreduce_values) * 8.0

"""Mesh partitioning and the resolution study."""

from __future__ import annotations

import numpy as np
import pytest

from gcsim.mesh import block_partition, choose_process_grid, partition


def test_process_grid_minimises_surface_area(bundle):
    grid = choose_process_grid(128, (600, 600, 600), preferred_first_extent=8)
    assert grid == (8, 4, 4)
    assert np.prod(grid) == 128

    def surface(g):
        a, b, c = (600 / g[i] for i in range(3))
        return 2 * (a * b + b * c + a * c)

    for candidate in [(2, 8, 8), (16, 4, 2), (32, 2, 2), (128, 1, 1), (4, 4, 8)]:
        assert surface(grid) <= surface(candidate) + 1e-9


def test_block_partition_conserves_and_spreads_remainder():
    parts = block_partition(250, 8)
    assert parts.sum() == 250
    assert parts.max() - parts.min() == 1
    assert (parts == 32).sum() == 250 % 8          # remainder to the low ranks
    assert block_partition(600, 8).std() == 0.0    # exact division, no spread


@pytest.mark.parametrize("mesh_name", ["coarse", "medium", "fine"])
def test_every_cell_is_owned_exactly_once(bundle, mesh_name):
    mesh = bundle.meshes[mesh_name]
    d = partition(mesh, 128, preferred_first_extent=8)
    assert int(d.cells.sum()) == mesh.total_cells
    assert (d.extents.prod(axis=1) == d.cells).all()


def test_coarse_mesh_is_ragged_and_the_others_are_exact(bundle):
    """The control case: real load imbalance on a perfectly healthy cluster.

    250 divides neither 8 nor 4, so the low-index ranks own more cells. No fault
    is present anywhere; the mesh alone is responsible.
    """
    coarse = partition(bundle.meshes["coarse"], 128, preferred_first_extent=8)
    assert coarse.imbalance > 1.03
    assert coarse.cells.min() < coarse.cells.max()

    for name in ("medium", "fine"):
        d = partition(bundle.meshes[name], 128, preferred_first_extent=8)
        assert d.imbalance == 1.0
        assert d.cells.min() == d.cells.max()


def test_surface_to_volume_falls_with_resolution(bundle):
    """Compute scales with volume, communication with surface area.

    This single relationship is what the whole mesh study rests on.
    """
    ratios = [partition(bundle.meshes[n], 128, preferred_first_extent=8).surface_to_volume
              for n in ("coarse", "medium", "fine")]
    assert ratios[0] > ratios[1] > ratios[2]


def test_periodic_domain_gives_every_rank_six_neighbours(bundle):
    d = partition(bundle.meshes["medium"], 128, preferred_first_extent=8)
    assert d.neighbours.shape == (128, 6)
    for rank in range(128):
        assert len(set(d.neighbours[rank])) == 6
        assert rank not in d.neighbours[rank]

    #  Opposite directions must agree: if A is B's +x neighbour then B is A's -x.
    for rank in range(128):
        for d_idx, opposite in ((0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)):
            nb = d.neighbours[rank, d_idx]
            assert d.neighbours[nb, opposite] == rank


def test_shared_faces_have_matching_areas(bundle):
    """The face A sends to B must be the same size as the one B sends to A."""
    d = partition(bundle.meshes["coarse"], 128, preferred_first_extent=8)
    for rank in range(128):
        for d_idx, opposite in ((0, 1), (2, 3), (4, 5)):
            nb = d.neighbours[rank, d_idx]
            assert d.face_cells[rank, d_idx] == d.face_cells[nb, opposite]


# ---------------------------------------------------------------------------
# The headline result
# ---------------------------------------------------------------------------

def test_occupancy_and_efficiency_rise_with_resolution(mesh_runs):
    """Mesh resolution alone controls GPU utilisation.

    Same cluster, same code, no fault: refining the mesh gives each rank more
    volume per unit of surface, so occupancy rises and more of the timestep is
    spent on useful work.
    """
    occ = [mesh_runs[m].summary["mean_sm_occupancy_pct"] for m in ("coarse", "medium", "fine")]
    eff = [mesh_runs[m].summary["parallel_efficiency"] for m in ("coarse", "medium", "fine")]

    assert occ[0] < occ[1] < occ[2]
    assert eff[0] < eff[1] < eff[2]
    #  The coarse mesh cannot fill the SMs and spends a large share of each
    #  timestep communicating; the fine mesh does neither. Roughly 3x in
    #  occupancy and 3x in efficiency, from resolution alone.
    assert occ[0] < 35.0 and occ[2] > 85.0
    assert occ[2] > 2.5 * occ[0]
    assert eff[0] < 0.4 and eff[2] > 0.85


def test_relative_communication_cost_falls_with_resolution(mesh_runs):
    """Halo cost measured against IDEAL compute, not achieved.

    `comm_fraction` is confounded: a coarse mesh inflates its own denominator
    through the occupancy penalty. Dividing by the ideal compute time isolates
    the surface-to-volume effect, and there the separation is a clean 5x.
    """
    ratio = [mesh_runs[m].summary["halo_per_ideal_compute"]
             for m in ("coarse", "medium", "fine")]
    assert ratio[0] > ratio[1] > ratio[2]
    assert ratio[0] > 3 * ratio[2]


def test_memory_footprint_tracks_cells_per_rank(mesh_runs):
    gb = [mesh_runs[m].summary["mesh_memory_per_rank_gb"] for m in ("coarse", "medium", "fine")]
    assert gb[0] < gb[1] < gb[2]
    #  All three still fit in an 80 GB device, so memory is never the limiter here.
    assert gb[2] < 80.0

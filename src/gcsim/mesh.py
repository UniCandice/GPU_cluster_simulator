"""Mesh partitioning -- the centrepiece of the model.

A uniform Cartesian box is decomposed onto a 3D process grid, one subdomain per
rank. Two consequences drive almost everything downstream:

1.  **Compute scales with volume, communication with surface area.** Refining
    the mesh raises cells-per-rank faster than it raises halo-face cells, so the
    compute:communication ratio -- and therefore achieved GPU utilisation --
    is a direct function of resolution. That is the mesh study.

2.  **Uniform meshes do not always divide evenly.** When an axis length is not a
    multiple of its process-grid extent, the leftover cells go to the low-index
    ranks. The result is real load imbalance on a perfectly healthy cluster,
    which is the control case for every straggler scenario.

The domain is triply periodic -- a Taylor-Green vortex box, which is the
canonical case for this geometry -- so every rank has exactly six neighbours
and no rank is privileged by sitting on a wall. Any imbalance in the reported
metrics therefore comes from partitioning alone. A walled case would break
that: boundary ranks would own five faces instead of six.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcsim.config import MeshConfig

#: Halo directions in a fixed order: -x, +x, -y, +y, -z, +z.
#: `axis` is the axis the exchange crosses; the face area is the product of the
#: subdomain extents on the *other two* axes.
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, -1), (0, +1), (1, -1), (1, +1), (2, -1), (2, +1))
DIRECTION_NAMES: tuple[str, ...] = ("-x", "+x", "-y", "+y", "-z", "+z")


def choose_process_grid(n_ranks: int, dims: tuple[int, int, int],
                        preferred_first_extent: int | None = None) -> tuple[int, int, int]:
    """Pick (px, py, pz) with px*py*pz == n_ranks minimising total surface area.

    Ties are broken toward putting the *largest* extent on the x axis, and
    toward matching `preferred_first_extent` (the GPUs per node) when possible.
    That is not cosmetic: with row-major rank ordering it places the +/-x
    exchanges -- the largest faces -- entirely inside a node on the fastest
    link, and pushes +/-z across racks. See placement.py.
    """
    nx, ny, nz = dims
    best: tuple[float, int, int, tuple[int, int, int]] | None = None

    for px in range(1, n_ranks + 1):
        if n_ranks % px:
            continue
        rest = n_ranks // px
        for py in range(1, rest + 1):
            if rest % py:
                continue
            pz = rest // py
            a, b, c = nx / px, ny / py, nz / pz
            surface = 2.0 * (a * b + b * c + a * c)
            match_pref = 0 if (preferred_first_extent is not None and px == preferred_first_extent) else 1
            key = (round(surface, 6), match_pref, -px, (px, py, pz))
            if best is None or key < best:
                best = key

    assert best is not None
    return best[3]


def block_partition(n: int, parts: int) -> np.ndarray:
    """Split `n` cells over `parts` ranks, remainder to the low-index ranks.

    This is the standard block distribution. When `n % parts != 0` the first
    `n % parts` ranks own one extra cell along that axis -- the origin of
    partition raggedness.
    """
    base = n // parts
    extra = n % parts
    out = np.full(parts, base, dtype=np.int64)
    out[:extra] += 1
    return out


@dataclass(frozen=True)
class Decomposition:
    """The result of partitioning one mesh over one rank count."""

    mesh: MeshConfig
    n_ranks: int
    grid: tuple[int, int, int]
    #: (n_ranks, 3) grid coordinates of each rank
    coords: np.ndarray
    #: (n_ranks,) cells owned by each rank
    cells: np.ndarray
    #: (n_ranks, 3) subdomain extents
    extents: np.ndarray
    #: (n_ranks, 6) neighbour rank index, in DIRECTIONS order
    neighbours: np.ndarray
    #: (n_ranks, 6) boundary cells on each face
    face_cells: np.ndarray

    # -- derived geometry --------------------------------------------------

    @property
    def boundary_cells(self) -> np.ndarray:
        """(n_ranks,) total halo-face cells per rank."""
        return self.face_cells.sum(axis=1)

    @property
    def mean_cells(self) -> float:
        return float(self.cells.mean())

    @property
    def imbalance(self) -> float:
        """max/mean cells. 1.0 is a perfect partition."""
        return float(self.cells.max() / self.cells.mean())

    @property
    def surface_to_volume(self) -> float:
        """Halo-face cells per owned cell, averaged over ranks."""
        return float((self.boundary_cells / self.cells).mean())

    def halo_bytes_per_face(self) -> np.ndarray:
        """(n_ranks, 6) bytes sent on each face *per inner iteration*."""
        return self.face_cells * self.mesh.halo_bytes_per_boundary_cell

    def halo_bytes_per_iteration(self) -> np.ndarray:
        """(n_ranks, 6) bytes sent on each face per outer timestep.

        The halo is exchanged once per inner linear-solver iteration, not once
        per timestep. That factor is what makes communication matter at all for
        an implicit solver.
        """
        return self.halo_bytes_per_face() * self.mesh.inner_iterations

    def summary(self) -> dict[str, float | int | str]:
        return {
            "mesh": self.mesh.name,
            "dims": "x".join(str(d) for d in self.mesh.dims),
            "total_cells": int(self.cells.sum()),
            "n_ranks": self.n_ranks,
            "grid": "x".join(str(g) for g in self.grid),
            "cells_min": int(self.cells.min()),
            "cells_max": int(self.cells.max()),
            "cells_mean": self.mean_cells,
            "imbalance": self.imbalance,
            "surface_to_volume": self.surface_to_volume,
            "halo_bytes_per_iter_mean": float(self.halo_bytes_per_iteration().sum(axis=1).mean()),
            "memory_per_rank_gb": float(self.cells.max() * self.mesh.bytes_per_cell / 1e9),
        }


def partition(mesh: MeshConfig, n_ranks: int,
              preferred_first_extent: int | None = None) -> Decomposition:
    """Decompose `mesh` over `n_ranks` ranks."""
    grid = choose_process_grid(n_ranks, mesh.dims, preferred_first_extent)
    px, py, pz = grid

    slabs = [block_partition(mesh.dims[a], grid[a]) for a in range(3)]

    #: Rank ordering is row-major with x fastest: r = i + px*(j + py*k).
    #: Combined with the grid choice above, this is what puts +/-x on NVLink.
    r = np.arange(n_ranks, dtype=np.int64)
    coords = np.stack([r % px, (r // px) % py, r // (px * py)], axis=1)

    ex = slabs[0][coords[:, 0]]
    ey = slabs[1][coords[:, 1]]
    ez = slabs[2][coords[:, 2]]
    extents = np.stack([ex, ey, ez], axis=1)
    cells = ex * ey * ez

    def rank_of(c: np.ndarray) -> np.ndarray:
        return c[:, 0] + px * (c[:, 1] + py * c[:, 2])

    n = n_ranks
    neighbours = np.empty((n, 6), dtype=np.int64)
    face_cells = np.empty((n, 6), dtype=np.int64)
    limits = np.array(grid)

    for d, (axis, step) in enumerate(DIRECTIONS):
        nb = coords.copy()
        nb[:, axis] = (nb[:, axis] + step) % limits[axis]   # triply periodic
        neighbours[:, d] = rank_of(nb)
        other = [a for a in range(3) if a != axis]
        face_cells[:, d] = extents[:, other[0]] * extents[:, other[1]]

    return Decomposition(
        mesh=mesh, n_ranks=n_ranks, grid=grid, coords=coords, cells=cells,
        extents=extents, neighbours=neighbours, face_cells=face_cells,
    )

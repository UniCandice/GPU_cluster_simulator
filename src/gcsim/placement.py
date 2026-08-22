"""Mapping MPI ranks onto physical GPUs.

Placement is where the mesh decomposition meets the cluster topology, and it
matters a great deal. The 8 x 4 x 4 process grid has its longest axis on x, and
rank ordering puts x fastest, so under `packed` placement:

    +/-x neighbours  -> same node        (NVLink-class, ~2 us)
    +/-y neighbours  -> same rack        (through the leaf, ~10 us)
    +/-z neighbours  -> different rack   (through the spine, ~34 us)

The largest halo faces therefore ride the fastest link, and cross-rack traffic
is confined to the two smallest faces. This is what a competent scheduler does,
and it is why a rack-level fabric fault in this model hits exactly the +/-z
exchanges while leaving intra-node exchanges untouched.

`scatter` is provided as the deliberately bad counterfactual: consecutive ranks
are spread across racks, so the *largest* faces are pushed onto the *slowest*
links. It exists so the effect of placement can be demonstrated rather than
asserted -- see tests/test_topology.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcsim.mesh import Decomposition
from gcsim.routing import CROSS_DOMAIN, INTRA_DOMAIN, INTRANODE, Router
from gcsim.topology import Cluster

#: Integer codes for neighbour link class, ordered slowest-last.
KIND_CODES = {INTRANODE: 0, INTRA_DOMAIN: 1, CROSS_DOMAIN: 2}
KIND_NAMES = (INTRANODE, INTRA_DOMAIN, CROSS_DOMAIN)


@dataclass(frozen=True)
class Placement:
    strategy: str
    #: (n_ranks,) global GPU index owned by each rank
    rank_to_gpu: np.ndarray
    #: (n_gpus,) rank running on each GPU
    gpu_to_rank: np.ndarray
    #: (n_ranks, 6) link class of each halo neighbour, in mesh.DIRECTIONS order
    neighbour_kind: np.ndarray

    def kind_counts(self) -> dict[str, int]:
        return {name: int((self.neighbour_kind == code).sum())
                for name, code in KIND_CODES.items()}

    def cross_domain_fraction(self) -> float:
        return float((self.neighbour_kind == KIND_CODES[CROSS_DOMAIN]).mean())


def place(cluster: Cluster, decomposition: Decomposition, router: Router,
          strategy: str = "packed") -> Placement:
    """Assign ranks to GPUs."""
    n = decomposition.n_ranks
    if n != cluster.n_gpus:
        raise ValueError(f"{n} ranks but {cluster.n_gpus} GPUs; this model is one rank per GPU")

    if strategy == "packed":
        #  Identity. Rank r -> GPU r, so ranks 0..7 share node 0, ranks 0..31
        #  share rack 0. Combined with x-fastest rank ordering this is the
        #  topology-aware mapping described above.
        rank_to_gpu = np.arange(n, dtype=np.int64)
    elif strategy == "scatter":
        #  Round-robin over racks: rank r lands in rack r % n_racks. Adjacent
        #  ranks are now maximally far apart.
        racks = cluster.cfg.racks
        per_rack = cluster.cfg.gpus_per_rack
        r = np.arange(n, dtype=np.int64)
        rank_to_gpu = (r % racks) * per_rack + (r // racks)
    else:
        raise ValueError(f"unknown placement strategy {strategy!r}")

    gpu_to_rank = np.empty_like(rank_to_gpu)
    gpu_to_rank[rank_to_gpu] = np.arange(n, dtype=np.int64)

    neighbour_kind = np.empty(decomposition.neighbours.shape, dtype=np.int8)
    for rank in range(n):
        src_gpu = int(rank_to_gpu[rank])
        for d in range(decomposition.neighbours.shape[1]):
            dst_gpu = int(rank_to_gpu[decomposition.neighbours[rank, d]])
            neighbour_kind[rank, d] = KIND_CODES[router.kind(src_gpu, dst_gpu)]

    return Placement(strategy=strategy, rank_to_gpu=rank_to_gpu,
                     gpu_to_rank=gpu_to_rank, neighbour_kind=neighbour_kind)

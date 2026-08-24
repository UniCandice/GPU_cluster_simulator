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

from gcsim.config import AllocationConfig
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
    #: (n_gpus,) rank running on each GPU, -1 where the GPU is idle. Sized by
    #: the CLUSTER, not the job: a subset job leaves most entries at -1, and
    #: sizing this by n_ranks silently corrupts it the moment they differ.
    gpu_to_rank: np.ndarray
    #: (n_ranks, 6) link class of each halo neighbour, in mesh.DIRECTIONS order
    neighbour_kind: np.ndarray
    #: GPU ids the job occupies, in global index order (not rank order, so the
    #: set reads the same whatever the placement strategy did with it).
    allocated_gpu_ids: tuple[str, ...] = ()

    def kind_counts(self) -> dict[str, int]:
        return {name: int((self.neighbour_kind == code).sum())
                for name, code in KIND_CODES.items()}

    def cross_domain_fraction(self) -> float:
        return float((self.neighbour_kind == KIND_CODES[CROSS_DOMAIN]).mean())


def allocation_pool(cluster: Cluster, allocation: "AllocationConfig | None") -> np.ndarray:
    """Candidate GPU indices for the job, in global index order.

    The whole cluster when no allocation (or no restriction) is given; the
    GPUs of the listed racks or nodes otherwise. Order matters: `packed` takes
    a prefix of this, so global index order is what keeps packed-at-full an
    identity map -- today's behaviour, bit for bit.
    """
    if allocation is None or (allocation.racks is None and allocation.nodes is None):
        return np.arange(cluster.n_gpus, dtype=np.int64)
    if allocation.racks is not None:
        wanted = set(allocation.racks)
        pool = [g.index for g in cluster.gpu_list if g.rack_index in wanted]
        if not pool:
            raise ValueError(f"allocation racks {sorted(wanted)} match no GPUs; "
                             f"cluster has racks 0..{cluster.cfg.racks - 1}")
    else:
        wanted = set(allocation.nodes)
        pool = [g.index for g in cluster.gpu_list if g.node_id in wanted]
        if not pool:
            have = sorted({g.node_id for g in cluster.gpu_list})
            raise ValueError(f"allocation nodes {sorted(wanted)} match no GPUs; "
                             f"cluster has {have[:4]} ... {have[-1]}")
    return np.array(pool, dtype=np.int64)


def place(cluster: Cluster, decomposition: Decomposition, router: Router,
          strategy: str = "packed",
          allocation: "AllocationConfig | None" = None) -> Placement:
    """Assign ranks to GPUs, over the whole cluster or an allocated slice."""
    n = decomposition.n_ranks
    pool = allocation_pool(cluster, allocation)
    if n > pool.size:
        raise ValueError(
            f"{n} ranks cannot fit the allocation pool of {pool.size} GPUs"
            + (f" (racks {list(allocation.racks)})" if allocation and allocation.racks else "")
            + (f" (nodes {list(allocation.nodes)})" if allocation and allocation.nodes else "")
            + f"; the cluster has {cluster.n_gpus} in total")

    if strategy == "packed":
        #  A prefix of the pool. At full allocation this is the identity map:
        #  rank r -> GPU r, so ranks 0..7 share node 0 and ranks 0..31 share
        #  rack 0. Combined with x-fastest rank ordering this is the
        #  topology-aware mapping described above.
        rank_to_gpu = pool[:n].copy()
    elif strategy == "scatter":
        #  Round-robin over the racks present in the pool: one GPU from each
        #  live rack per round, so adjacent ranks are maximally far apart. For
        #  the full cluster this reproduces the historical formula
        #  (r % racks) * per_rack + (r // racks) exactly.
        by_rack: dict[int, list[int]] = {}
        for idx in pool:
            by_rack.setdefault(int(cluster.gpu(int(idx)).rack_index), []).append(int(idx))
        queues = [list(g) for _, g in sorted(by_rack.items())]
        heads = [0] * len(queues)
        order: list[int] = []
        while len(order) < n:
            took = False
            for qi, q in enumerate(queues):
                if heads[qi] < len(q) and len(order) < n:
                    order.append(q[heads[qi]])
                    heads[qi] += 1
                    took = True
            if not took:                      # unreachable: n <= pool.size
                break
        rank_to_gpu = np.array(order, dtype=np.int64)
    else:
        raise ValueError(f"unknown placement strategy {strategy!r}")

    #  Sized by the cluster and defaulted to -1, so idle GPUs answer "no rank"
    #  instead of aliasing whatever rank happened to share their index.
    gpu_to_rank = np.full(cluster.n_gpus, -1, dtype=np.int64)
    gpu_to_rank[rank_to_gpu] = np.arange(n, dtype=np.int64)

    neighbour_kind = np.empty(decomposition.neighbours.shape, dtype=np.int8)
    for rank in range(n):
        src_gpu = int(rank_to_gpu[rank])
        for d in range(decomposition.neighbours.shape[1]):
            dst_gpu = int(rank_to_gpu[decomposition.neighbours[rank, d]])
            neighbour_kind[rank, d] = KIND_CODES[router.kind(src_gpu, dst_gpu)]

    allocated = tuple(cluster.gpu(int(i)).gpu_id for i in np.sort(rank_to_gpu))
    return Placement(strategy=strategy, rank_to_gpu=rank_to_gpu,
                     gpu_to_rank=gpu_to_rank, neighbour_kind=neighbour_kind,
                     allocated_gpu_ids=allocated)

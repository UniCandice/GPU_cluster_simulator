"""Routing: which links a GPU-to-GPU flow actually traverses.

The path determines both the cost of a transfer and which counters move, which
is what makes fault *tier* recoverable from telemetry alone:

    pair            path                                  switch hops
    --------------  ------------------------------------  -----------
    same node       GPU -> GPU (NVLink-class fabric)                 0
    same domain     NIC -> leaf -> NIC                               1
    cross domain    NIC -> leaf -> spine -> leaf -> NIC              3

    latency      = sum(hop latency for hop in path)
    effective_bw = min(hop bandwidth for hop in path)     <- bottleneck, not mean

A degraded *leaf uplink* therefore touches only traffic entering or leaving that
one rack, while every intra-rack and intra-node exchange is untouched. A
degraded *spine* would touch cross-domain traffic globally. The two are
distinguishable in the port counters without knowing the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from gcsim.topology import RX, TX, Cluster

INTRANODE = "intranode"
INTRA_DOMAIN = "intra_domain"
CROSS_DOMAIN = "cross_domain"


@dataclass(frozen=True)
class Hop:
    """One directional bottleneck on a path."""
    channel_id: str
    direction: str


@dataclass(frozen=True)
class Path:
    src_gpu: str
    dst_gpu: str
    kind: str
    hops: tuple[Hop, ...]
    #  Leaf downlink ports the flow crosses, as (port_id, direction). Uplink
    #  ports are credited through the bundle channel instead, because ECMP
    #  spreads a flow across whichever members happen to be up.
    port_credits: tuple[tuple[str, str], ...]
    switch_hops: int
    latency_us: float

    @property
    def latency_s(self) -> float:
        return self.latency_us * 1e-6


class Router:
    """Computes and caches paths for one cluster.

    Paths depend only on topology, never on link health: a downed uplink reduces
    the bundle's capacity rather than changing the route. That mirrors ECMP over
    a link aggregation group and keeps the cache valid for the whole run.
    """

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self._cache: dict[tuple[int, int], Path] = {}

    def route(self, src_index: int, dst_index: int) -> Path:
        key = (src_index, dst_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._build(src_index, dst_index)
        self._cache[key] = path
        return path

    def _build(self, src_index: int, dst_index: int) -> Path:
        cl = self.cluster
        ic = cl.cfg.interconnect
        src = cl.gpu(src_index)
        dst = cl.gpu(dst_index)

        if src.node_id == dst.node_id:
            return Path(
                src_gpu=src.gpu_id, dst_gpu=dst.gpu_id, kind=INTRANODE,
                hops=(Hop(f"intranode:{src.node_id}", TX),),
                port_credits=(),
                switch_hops=0,
                latency_us=ic.intranode.latency_us,
            )

        src_leaf = cl.racks[src.rack_id].leaf_id
        dst_leaf = cl.racks[dst.rack_id].leaf_id
        src_down = f"{src_leaf}:down_{src.nic_id}"
        dst_down = f"{dst_leaf}:down_{dst.nic_id}"

        if src.rack_id == dst.rack_id:
            #  node -> leaf -> node. The leaf fabric itself is non-blocking, so
            #  the only bottlenecks are the two host NICs.
            return Path(
                src_gpu=src.gpu_id, dst_gpu=dst.gpu_id, kind=INTRA_DOMAIN,
                hops=(Hop(f"nic:{src.nic_id}", TX), Hop(f"nic:{dst.nic_id}", RX)),
                port_credits=((src_down, RX), (dst_down, TX)),
                switch_hops=1,
                latency_us=2.0 * ic.nic.latency_us,
            )

        return Path(
            src_gpu=src.gpu_id, dst_gpu=dst.gpu_id, kind=CROSS_DOMAIN,
            hops=(
                Hop(f"nic:{src.nic_id}", TX),
                Hop(f"uplink:{src.rack_id}", TX),
                Hop(f"uplink:{dst.rack_id}", RX),
                Hop(f"nic:{dst.nic_id}", RX),
            ),
            port_credits=((src_down, RX), (dst_down, TX)),
            switch_hops=3,
            latency_us=2.0 * ic.nic.latency_us + 2.0 * ic.leaf_uplink.latency_us,
        )

    # -- helpers used by tests and by placement ----------------------------

    def kind(self, src_index: int, dst_index: int) -> str:
        return self.route(src_index, dst_index).kind

    def worst_latency_us(self, indices: list[int]) -> float:
        """Worst pairwise latency across a set of ranks.

        A ring or tree spanning racks is paced by its longest hop, so this is
        the right pessimistic bound for a collective over `indices`. Not on the
        collective path today: `allreduce_time_s` inlines the cross-rack
        expression directly, and the two agree whenever the job spans more than
        one rack -- which the shipped 128-rank config always does. Kept because
        this is the general form (correct for a subset of ranks confined to one
        node or rack, where the inline expression is not); a test pins the
        agreement so the two cannot drift apart silently.
        """
        cl = self.cluster
        racks = {cl.gpu(i).rack_id for i in indices}
        nodes = {cl.gpu(i).node_id for i in indices}
        ic = cl.cfg.interconnect
        if len(racks) > 1:
            return 2.0 * ic.nic.latency_us + 2.0 * ic.leaf_uplink.latency_us
        if len(nodes) > 1:
            return 2.0 * ic.nic.latency_us
        return ic.intranode.latency_us

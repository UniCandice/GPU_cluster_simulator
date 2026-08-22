"""Cluster topology: racks, nodes, GPUs, NICs, switches, ports and channels.

Everything mutable in here is *physical state*. Fault injectors are allowed to
reach in and change these values -- a port goes down, a rack's cooling degrades,
a GPU's achievable clock drops. Nothing in this module knows what telemetry is.

Structure
---------
    Rack (= one network domain AND one thermal domain)
      +- Leaf switch
      |    +- downlink ports  (one per NIC in the rack)
      |    +- uplink ports    (`count_per_leaf`, spread over the spines)
      +- Node x nodes_per_rack
           +- NIC x nics_per_node
           +- GPU x gpus_per_node

A rack being simultaneously a network domain and a thermal domain is the reason
rack-scoped faults show up as a *contiguous block* of ranks: the same 32 GPUs
share a leaf switch and a CRAC unit.

Channels
--------
Bandwidth is accounted on `Channel` objects rather than on individual ports.
A channel is a directional bottleneck that a flow can contend for:

    intranode:<node>   the node's internal GPU fabric (NVLink-class)
    nic:<nic>          one host NIC, full duplex
    uplink:<rack>      the leaf's uplink bundle, aggregated

Aggregating the uplink bundle into one channel is a deliberate simplification:
with many concurrent flows, ECMP spreads traffic across member links evenly
enough that the bundle behaves like one fat pipe. Per-port telemetry is
recovered by dividing the channel's bytes across its *active* member ports,
which is also what makes the counter-conservation invariant hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gcsim.config import ClusterConfig

TX = "tx"
RX = "rx"


# ---------------------------------------------------------------------------
# Leaf entities
# ---------------------------------------------------------------------------

@dataclass
class Gpu:
    gpu_id: str
    index: int                 # global index, 0 .. n_gpus-1
    node_id: str
    node_index: int
    rack_id: str
    rack_index: int
    local_index: int           # 0 .. gpus_per_node-1
    nic_id: str
    memory_gb: float

    # --- mutable physical state; only faults.py writes these ---------------
    #  Fraction of SM throughput this rank actually gets. A co-resident process
    #  time-slicing the device lowers this WITHOUT lowering the reported clock,
    #  which is precisely what makes that failure mode hard to spot.
    throughput_derate: float = 1.0
    memory_bandwidth_factor: float = 1.0  # achievable HBM bandwidth fraction
    reliability_clock_cap: float = 1.0    # governor cap imposed by the RAS subsystem

    # --- mutable dynamic state; models/ writes these ----------------------
    temperature_c: float = 0.0
    power_w: float = 0.0
    clock_mhz: float = 0.0
    throttled: bool = False
    throttle_reason: str = "NONE"
    occupancy: float = 0.0
    utilisation: float = 0.0
    memory_used_gb: float = 0.0

    # fixed per-GPU silicon variation, drawn once from the GPU's own stream
    silicon_clock_offset: float = 1.0
    silicon_leakage_offset: float = 1.0


@dataclass
class Nic:
    nic_id: str
    index: int
    node_id: str
    rack_id: str
    capacity_gbps: float
    tx_bytes: float = 0.0
    rx_bytes: float = 0.0
    tx_errors: float = 0.0
    rx_errors: float = 0.0
    tx_drops: float = 0.0
    rx_drops: float = 0.0


@dataclass
class Node:
    node_id: str
    index: int
    rack_id: str
    rack_index: int
    gpu_ids: list[str] = field(default_factory=list)
    nic_ids: list[str] = field(default_factory=list)
    cpu_cores: int = 0
    memory_gb: float = 0.0

    # mutable dynamic state
    cpu_pressure: float = 0.0
    memory_pressure: float = 0.0
    io_pressure: float = 0.0
    dirty_bytes: float = 0.0     # host page cache awaiting writeback


@dataclass
class Port:
    port_id: str
    switch_id: str
    switch_tier: str            # "leaf" | "spine"
    domain_id: str | None       # rack id for leaf ports, None for spine ports
    role: str                   # "downlink" | "uplink"
    peer_id: str
    capacity_gbps: float
    #: The port at the other end of the same cable. Spine-side counters are
    #: mirrored from their leaf-side partner rather than accounted twice, which
    #: is what keeps the counter-conservation invariant exact.
    mirror_of: str | None = None

    # mutable physical state
    up: bool = True
    #: Extra frame loss beyond the link's background BER. A marginal optic --
    #: the kind that takes a bundle down in the first place -- runs with an
    #: elevated error rate, and every corrupted frame has to be retransmitted.
    error_rate: float = 0.0

    # cumulative counters
    tx_bytes: float = 0.0
    rx_bytes: float = 0.0
    tx_errors: float = 0.0
    rx_errors: float = 0.0
    tx_drops: float = 0.0
    rx_drops: float = 0.0

    # instantaneous, refreshed each phase
    queue_depth: float = 0.0
    utilisation: float = 0.0


@dataclass
class Switch:
    switch_id: str
    tier: str                   # "leaf" | "spine"
    domain_id: str | None
    downlink_ids: list[str] = field(default_factory=list)
    uplink_ids: list[str] = field(default_factory=list)


@dataclass
class Rack:
    rack_id: str
    index: int
    node_ids: list[str] = field(default_factory=list)
    leaf_id: str = ""

    # mutable physical state: the CRAC. Inlet temperature is derived from this
    # and the rack's own power draw, so degrading it moves all 32 GPUs together.
    cooling_efficiency: float = 1.0
    inlet_temp_c: float = 0.0
    power_kw: float = 0.0


@dataclass
class Channel:
    """A directional bottleneck a flow can contend for."""
    channel_id: str
    kind: str                       # "intranode" | "nic" | "uplink"
    latency_us: float
    nominal_gbps: float
    port_ids: list[str] = field(default_factory=list)
    base_ber: float = 0.0

    # per-phase accounting, reset by the network model each exchange
    load_bytes: dict[str, float] = field(default_factory=lambda: {TX: 0.0, RX: 0.0})
    flow_count: dict[str, int] = field(default_factory=lambda: {TX: 0, RX: 0})
    queue_depth: dict[str, float] = field(default_factory=lambda: {TX: 0.0, RX: 0.0})
    loss_rate: dict[str, float] = field(default_factory=lambda: {TX: 0.0, RX: 0.0})

    def reset_load(self) -> None:
        for d in (TX, RX):
            self.load_bytes[d] = 0.0
            self.flow_count[d] = 0
            self.queue_depth[d] = 0.0
            self.loss_rate[d] = 0.0


# ---------------------------------------------------------------------------
# The cluster
# ---------------------------------------------------------------------------

class Cluster:
    """Built once per run. Holds every entity and the id -> object indexes."""

    def __init__(self, cfg: ClusterConfig):
        self.cfg = cfg
        self.racks: dict[str, Rack] = {}
        self.nodes: dict[str, Node] = {}
        self.gpus: dict[str, Gpu] = {}
        self.nics: dict[str, Nic] = {}
        self.switches: dict[str, Switch] = {}
        self.ports: dict[str, Port] = {}
        self.channels: dict[str, Channel] = {}

        # ordered views, used everywhere that vectorises over entities
        self.gpu_list: list[Gpu] = []
        self.node_list: list[Node] = []
        self.nic_list: list[Nic] = []
        self.rack_list: list[Rack] = []
        self.port_list: list[Port] = []

    # -- convenience -------------------------------------------------------
    @property
    def n_gpus(self) -> int:
        return len(self.gpu_list)

    def gpu(self, index: int) -> Gpu:
        return self.gpu_list[index]

    def leaf_of(self, rack_id: str) -> Switch:
        return self.switches[self.racks[rack_id].leaf_id]

    def uplink_channel(self, rack_id: str) -> Channel:
        return self.channels[f"uplink:{rack_id}"]

    def nic_channel(self, nic_id: str) -> Channel:
        return self.channels[f"nic:{nic_id}"]

    def intranode_channel(self, node_id: str) -> Channel:
        return self.channels[f"intranode:{node_id}"]

    def reset_channel_load(self) -> None:
        for ch in self.channels.values():
            ch.reset_load()


def build_cluster(cfg: ClusterConfig) -> Cluster:
    """Construct the full topology from the cluster config."""
    ic = cfg.interconnect
    cl = Cluster(cfg)

    # --- spines -----------------------------------------------------------
    for s in range(ic.spine_count):
        sid = f"spine{s}"
        cl.switches[sid] = Switch(switch_id=sid, tier="spine", domain_id=None)

    uplinks_per_spine = ic.leaf_uplink.count // ic.spine_count
    if uplinks_per_spine * ic.spine_count != ic.leaf_uplink.count:
        raise ValueError(
            f"leaf_uplink.count_per_leaf ({ic.leaf_uplink.count}) must divide "
            f"spine_count ({ic.spine_count})"
        )

    gpu_index = 0
    node_index = 0
    nic_index = 0

    for r in range(cfg.racks):
        rack_id = f"r{r}"
        leaf_id = f"leaf{r}"
        rack = Rack(rack_id=rack_id, index=r, leaf_id=leaf_id,
                    cooling_efficiency=cfg.cooling.nominal_efficiency,
                    inlet_temp_c=cfg.cooling.base_inlet_temp_c)
        leaf = Switch(switch_id=leaf_id, tier="leaf", domain_id=rack_id)
        cl.racks[rack_id] = rack
        cl.switches[leaf_id] = leaf

        # --- uplink ports (leaf side and mirrored spine side) -------------
        uplink_port_ids: list[str] = []
        for u in range(ic.leaf_uplink.count):
            spine_id = f"spine{u // uplinks_per_spine}"
            lp_id = f"{leaf_id}:up{u}"
            sp_id = f"{spine_id}:{rack_id}up{u}"
            cl.ports[lp_id] = Port(
                port_id=lp_id, switch_id=leaf_id, switch_tier="leaf",
                domain_id=rack_id, role="uplink", peer_id=spine_id,
                capacity_gbps=ic.leaf_uplink.bandwidth_gbps,
            )
            cl.ports[sp_id] = Port(
                port_id=sp_id, switch_id=spine_id, switch_tier="spine",
                domain_id=None, role="downlink", peer_id=leaf_id,
                capacity_gbps=ic.leaf_uplink.bandwidth_gbps,
                mirror_of=lp_id,
            )
            leaf.uplink_ids.append(lp_id)
            cl.switches[spine_id].downlink_ids.append(sp_id)
            uplink_port_ids.append(lp_id)

        cl.channels[f"uplink:{rack_id}"] = Channel(
            channel_id=f"uplink:{rack_id}", kind="uplink",
            latency_us=ic.leaf_uplink.latency_us,
            nominal_gbps=ic.leaf_uplink.total_bandwidth_gbps,
            port_ids=uplink_port_ids, base_ber=ic.leaf_uplink.ber,
        )

        # --- nodes --------------------------------------------------------
        for n in range(cfg.nodes_per_rack):
            node_id = f"{rack_id}n{n}"
            node = Node(node_id=node_id, index=node_index, rack_id=rack_id,
                        rack_index=r, cpu_cores=cfg.host.cpu_cores,
                        memory_gb=cfg.host.memory_gb)
            cl.nodes[node_id] = node
            rack.node_ids.append(node_id)

            cl.channels[f"intranode:{node_id}"] = Channel(
                channel_id=f"intranode:{node_id}", kind="intranode",
                latency_us=ic.intranode.latency_us,
                nominal_gbps=ic.intranode.bandwidth_gbps,
                base_ber=ic.intranode.ber,
            )

            # --- NICs, each on its own leaf downlink port -----------------
            for k in range(cfg.nics_per_node):
                nic_id = f"{node_id}nic{k}"
                cl.nics[nic_id] = Nic(nic_id=nic_id, index=nic_index,
                                      node_id=node_id, rack_id=rack_id,
                                      capacity_gbps=ic.nic.bandwidth_gbps)
                node.nic_ids.append(nic_id)
                nic_index += 1

                dp_id = f"{leaf_id}:down_{nic_id}"
                cl.ports[dp_id] = Port(
                    port_id=dp_id, switch_id=leaf_id, switch_tier="leaf",
                    domain_id=rack_id, role="downlink", peer_id=nic_id,
                    capacity_gbps=ic.nic.bandwidth_gbps,
                )
                leaf.downlink_ids.append(dp_id)

                cl.channels[f"nic:{nic_id}"] = Channel(
                    channel_id=f"nic:{nic_id}", kind="nic",
                    latency_us=ic.nic.latency_us,
                    nominal_gbps=ic.nic.bandwidth_gbps,
                    port_ids=[dp_id], base_ber=ic.nic.ber,
                )

            # --- GPUs -----------------------------------------------------
            for g in range(cfg.gpus_per_node):
                gpu_id = f"{node_id}g{g}"
                nic_id = node.nic_ids[g % cfg.nics_per_node]
                cl.gpus[gpu_id] = Gpu(
                    gpu_id=gpu_id, index=gpu_index, node_id=node_id,
                    node_index=node_index, rack_id=rack_id, rack_index=r,
                    local_index=g, nic_id=nic_id, memory_gb=cfg.gpu.memory_gb,
                    temperature_c=cfg.cooling.base_inlet_temp_c,
                    clock_mhz=cfg.gpu.base_clock_mhz,
                    power_w=cfg.gpu.idle_power_w,
                )
                node.gpu_ids.append(gpu_id)
                gpu_index += 1

            node_index += 1

    # ordered views
    cl.rack_list = [cl.racks[f"r{r}"] for r in range(cfg.racks)]
    cl.node_list = sorted(cl.nodes.values(), key=lambda x: x.index)
    cl.nic_list = sorted(cl.nics.values(), key=lambda x: x.index)
    cl.gpu_list = sorted(cl.gpus.values(), key=lambda x: x.index)
    cl.port_list = list(cl.ports.values())
    return cl


def channel_capacity_gbps(cluster: Cluster, channel: Channel) -> float:
    """Currently available capacity, accounting for downed member ports.

    A channel with no member ports (the intranode fabric) always runs at its
    nominal rate -- there is no port for a fault to take down.
    """
    if not channel.port_ids:
        return channel.nominal_gbps
    return sum(cluster.ports[p].capacity_gbps for p in channel.port_ids if cluster.ports[p].up)


def active_ports(cluster: Cluster, channel: Channel) -> list[Port]:
    return [cluster.ports[p] for p in channel.port_ids if cluster.ports[p].up]

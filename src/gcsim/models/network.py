"""Flow-level contention, queueing and loss on the fabric.

A halo exchange posts 768 concurrent flows (128 ranks x 6 neighbours). This
module answers: given the current health of every port, how long does each flow
take, and what do the switch and NIC counters read afterwards?

Model
-----
Flows share each channel by *fair share*, and a flow runs at the rate of its
slowest hop:

    share(channel)   = available_capacity / concurrent_flows
    bottleneck(flow) = min(share(h) for h in hops(flow))     <- min, not mean
    duration(flow)   = latency(path) + bytes / bottleneck(flow)

That is a max-min-fair approximation rather than an exact fixed point. It is
solved in two passes: the first pass establishes durations and hence channel
utilisation, and the second re-solves with the goodput that utilisation costs.

Queueing and loss are *consequences*, never inputs:

    rho    = bytes / (capacity * duration)
    queue  = rho^2 / (1 - rho)                     mean M/M/1 queue length
    loss   = overflow beyond the port buffer, plus any physical error rate
    goodput= capacity * (1 - retransmit_penalty * loss)

A degraded optic contributes physical `error_rate`; congestion contributes
overflow. Both cost goodput, which lengthens the exchange, which raises
utilisation further -- so a marginal link under load degrades non-linearly, as
real ones do.

Storage traffic is assumed to use a separate storage fabric and therefore does
not contend with halo traffic here. See README ("Limitations").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcsim.routing import Path, Router
from gcsim.topology import RX, TX, Cluster

MAX_HOPS = 4
_GBPS_TO_BPS = 1e9 / 8.0


@dataclass
class FlowSet:
    """A fixed set of concurrent transfers, precompiled against the fabric.

    Built once per run: the halo exchange repeats identically every timestep, so
    only *link health* changes between solves.
    """
    src_rank: np.ndarray          # (F,)
    dst_rank: np.ndarray          # (F,)
    nbytes: np.ndarray            # (F,)
    latency_s: np.ndarray         # (F,)
    hops: np.ndarray              # (F, MAX_HOPS) channel-direction ids, -1 padded
    port_rx: np.ndarray           # (F,) leaf downlink port id credited on rx, or -1
    port_tx: np.ndarray           # (F,) leaf downlink port id credited on tx, or -1
    kind: np.ndarray              # (F,) routing.INTRANODE / INTRA_DOMAIN / CROSS_DOMAIN codes

    @property
    def n_flows(self) -> int:
        return self.nbytes.shape[0]


@dataclass
class FabricSolution:
    """Everything one exchange produced."""
    flow_time_s: np.ndarray       # (F,)
    rank_time_s: np.ndarray       # (n_ranks,) when each rank's exchange completes
    duration_s: float             # wall time of the whole exchange
    cd_bytes: np.ndarray          # (C,) bytes carried per channel-direction
    cd_rho: np.ndarray            # (C,) utilisation
    cd_queue: np.ndarray          # (C,) mean queue depth in packets
    cd_loss: np.ndarray           # (C,) frame loss rate
    cd_drop_packets: np.ndarray   # (C,)
    cd_error_packets: np.ndarray  # (C,)
    cd_queue_delay_s: np.ndarray  # (C,)


class Fabric:
    """Contention solver bound to one cluster.

    Channel-directions are flattened to integer ids so the whole solve is
    vectorised. Solutions are cached against a fingerprint of link health, which
    makes 1000 identical timesteps cost one solve.
    """

    def __init__(self, cluster: Cluster, router: Router):
        self.cluster = cluster
        self.router = router
        ic = cluster.cfg.interconnect
        self.packet_bytes = ic.packet_bytes
        self.buffer_packets = float(ic.buffer_packets)
        self.max_util = ic.max_utilisation
        self.retransmit_penalty = ic.retransmit_penalty

        # --- flatten (channel, direction) -> id ---------------------------
        self.cd_key: dict[tuple[str, str], int] = {}
        self.cd_channel: list[str] = []
        self.cd_direction: list[str] = []
        for cid in cluster.channels:
            for d in (TX, RX):
                self.cd_key[(cid, d)] = len(self.cd_channel)
                self.cd_channel.append(cid)
                self.cd_direction.append(d)
        self.n_cd = len(self.cd_channel)
        self.cd_base_ber = np.array(
            [cluster.channels[c].base_ber for c in self.cd_channel], dtype=np.float64)

        # --- port ids -> int ----------------------------------------------
        self.port_ids: list[str] = list(cluster.ports)
        self.port_index = {p: i for i, p in enumerate(self.port_ids)}
        self.n_ports = len(self.port_ids)

        #: member ports of each channel-direction, for capacity and for
        #: spreading bundle traffic back onto individual ports
        self._cd_members = [
            np.array([self.port_index[p] for p in cluster.channels[c].port_ids], dtype=np.int64)
            for c in self.cd_channel
        ]
        self._cd_nominal = np.array(
            [cluster.channels[c].nominal_gbps for c in self.cd_channel], dtype=np.float64)

        self._cache: dict[tuple, FabricSolution] = {}

    # -- compiling flows ---------------------------------------------------

    def path_hops(self, path: Path) -> list[int]:
        return [self.cd_key[(h.channel_id, h.direction)] for h in path.hops]

    # -- health ------------------------------------------------------------

    def _port_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cl = self.cluster
        up = np.array([cl.ports[p].up for p in self.port_ids], dtype=bool)
        cap = np.array([cl.ports[p].capacity_gbps for p in self.port_ids], dtype=np.float64)
        err = np.array([cl.ports[p].error_rate for p in self.port_ids], dtype=np.float64)
        return up, cap, err

    def _capacity_and_error(self) -> tuple[np.ndarray, np.ndarray]:
        """Available capacity (bytes/s) and physical loss rate per channel-direction."""
        up, cap, err = self._port_state()
        caps = np.empty(self.n_cd)
        errs = np.empty(self.n_cd)
        for i, members in enumerate(self._cd_members):
            if members.size == 0:
                #  No member ports: the intranode fabric. Nothing can take it
                #  down, so it always runs at its nominal rate.
                caps[i] = self._cd_nominal[i]
                errs[i] = 0.0
                continue
            live = members[up[members]]
            caps[i] = float(cap[live].sum())
            errs[i] = float(err[live].mean()) if live.size else 0.0
        return caps * _GBPS_TO_BPS, errs

    def _fingerprint(self, tag: str) -> tuple:
        up, cap, err = self._port_state()
        return (tag, up.tobytes(), cap.tobytes(), err.tobytes())

    # -- the solve ---------------------------------------------------------

    def solve(self, flows: FlowSet, n_ranks: int, tag: str = "halo") -> FabricSolution:
        key = self._fingerprint(tag)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        sol = self._solve_uncached(flows, n_ranks)
        self._cache[key] = sol
        return sol

    def _solve_uncached(self, flows: FlowSet, n_ranks: int) -> FabricSolution:
        cap, phys_err = self._capacity_and_error()
        hops = flows.hops
        valid = hops >= 0
        flat = hops[valid]

        counts = np.bincount(flat, minlength=self.n_cd).astype(np.float64)
        cd_bytes = np.bincount(flat, weights=np.repeat(flows.nbytes, valid.sum(axis=1)),
                               minlength=self.n_cd)

        def flow_times(effective_cap: np.ndarray, extra_latency: np.ndarray) -> np.ndarray:
            share = effective_cap / np.maximum(counts, 1.0)
            # min over the flow's hops; padded slots must not win the minimum
            per_hop = np.where(valid, share[hops], np.inf)
            bottleneck = np.maximum(per_hop.min(axis=1), 1.0)
            lat_hop = np.where(valid, extra_latency[hops], 0.0)
            return flows.latency_s + lat_hop.sum(axis=1) + flows.nbytes / bottleneck

        # ---- pass 1: durations with no queueing feedback ------------------
        zero = np.zeros(self.n_cd)
        t1 = flow_times(cap, zero)
        dur1 = self._channel_durations(flows, t1)
        rho1 = self._utilisation(cd_bytes, cap, dur1)
        queue1, loss1 = self._queue_and_loss(rho1, phys_err)
        goodput = np.clip(1.0 - self.retransmit_penalty * loss1, 0.05, 1.0)
        queue_delay = queue1 * self.packet_bytes / np.maximum(cap, 1.0)

        # ---- pass 2: re-solve at the goodput that congestion left us ------
        t2 = flow_times(cap * goodput, queue_delay)
        dur2 = self._channel_durations(flows, t2)
        rho2 = self._utilisation(cd_bytes, cap * goodput, dur2)
        queue2, loss2 = self._queue_and_loss(rho2, phys_err)

        #  Every lost frame is a drop. Frames lost to corruption are drops AND
        #  errors; frames lost to buffer overflow are drops only. Counting them
        #  any other way would let a link lose traffic without any counter
        #  moving, which is exactly the failure this model exists to make visible.
        packets = cd_bytes / self.packet_bytes
        drops = packets * loss2
        errors = packets * phys_err + cd_bytes * 8.0 * self.cd_base_ber

        # A rank's exchange is done when all six sends AND all six receives are
        # done -- the flows are the same objects viewed from both ends.
        rank_time = np.zeros(n_ranks)
        np.maximum.at(rank_time, flows.src_rank, t2)
        np.maximum.at(rank_time, flows.dst_rank, t2)

        return FabricSolution(
            flow_time_s=t2,
            rank_time_s=rank_time,
            duration_s=float(t2.max(initial=0.0)),
            cd_bytes=cd_bytes,
            cd_rho=rho2,
            cd_queue=queue2,
            cd_loss=loss2,
            cd_drop_packets=drops,
            cd_error_packets=errors,
            cd_queue_delay_s=queue2 * self.packet_bytes / np.maximum(cap, 1.0),
        )

    # -- pieces ------------------------------------------------------------

    def _channel_durations(self, flows: FlowSet, t: np.ndarray) -> np.ndarray:
        """Wall time each channel-direction was in use."""
        dur = np.zeros(self.n_cd)
        valid = flows.hops >= 0
        for h in range(flows.hops.shape[1]):
            sel = valid[:, h]
            if sel.any():
                np.maximum.at(dur, flows.hops[sel, h], t[sel])
        return dur

    def _utilisation(self, cd_bytes: np.ndarray, cap: np.ndarray, dur: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = cd_bytes / np.maximum(cap * dur, 1e-12)
        return np.clip(np.nan_to_num(rho, nan=0.0, posinf=self.max_util), 0.0, self.max_util)

    def _queue_and_loss(self, rho: np.ndarray, phys_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean queue length and total frame loss.

        `rho^2/(1-rho)` is the M/M/1 mean number waiting. Once that exceeds the
        port buffer the excess is lost; the smooth form keeps the derivative
        finite so the two-pass solve stays stable.
        """
        queue = rho ** 2 / np.maximum(1.0 - rho, 1e-3)
        overflow = np.maximum(queue - self.buffer_packets, 0.0)
        congestion_loss = overflow / (overflow + self.buffer_packets)
        return queue, np.clip(congestion_loss + phys_err, 0.0, 0.9)

    # -- posting results back onto topology counters -----------------------

    def accumulate(self, flows: FlowSet, sol: FabricSolution) -> None:
        """Credit the solved exchange to NIC and switch-port counters.

        Bundle channels are spread evenly across their *active* member ports,
        which is what ECMP does in aggregate and what makes the
        counter-conservation invariant hold: bytes a node's NIC sends outside
        its rack equal bytes its leaf's uplink ports send.
        """
        cl = self.cluster

        # --- NIC counters, straight from the channel totals ----------------
        for i, (cid, direction) in enumerate(zip(self.cd_channel, self.cd_direction)):
            if not cid.startswith("nic:"):
                continue
            nic = cl.nics[cid.split(":", 1)[1]]
            if direction == TX:
                nic.tx_bytes += sol.cd_bytes[i]
                nic.tx_drops += sol.cd_drop_packets[i]
                nic.tx_errors += sol.cd_error_packets[i]
            else:
                nic.rx_bytes += sol.cd_bytes[i]
                nic.rx_drops += sol.cd_drop_packets[i]
                nic.rx_errors += sol.cd_error_packets[i]

        # --- queue state on the leaf downlink ports, from their NIC channel -
        for i, (cid, direction) in enumerate(zip(self.cd_channel, self.cd_direction)):
            if not cid.startswith("nic:"):
                continue
            for port_id in cl.channels[cid].port_ids:
                port = cl.ports[port_id]
                port.queue_depth = max(port.queue_depth, sol.cd_queue[i])
                port.utilisation = max(port.utilisation, sol.cd_rho[i])

        # --- leaf downlink ports, from the flows that crossed them ---------
        for arr, direction in ((flows.port_rx, RX), (flows.port_tx, TX)):
            sel = arr >= 0
            if not sel.any():
                continue
            totals = np.bincount(arr[sel], weights=flows.nbytes[sel], minlength=self.n_ports)
            for pi in np.nonzero(totals)[0]:
                port = cl.ports[self.port_ids[pi]]
                if direction == TX:
                    port.tx_bytes += totals[pi]
                else:
                    port.rx_bytes += totals[pi]

        # --- uplink bundles, spread across live members --------------------
        for i, (cid, direction) in enumerate(zip(self.cd_channel, self.cd_direction)):
            if not cid.startswith("uplink:"):
                continue
            members = [cl.ports[p] for p in cl.channels[cid].port_ids if cl.ports[p].up]
            if not members:
                continue
            share = 1.0 / len(members)
            for port in members:
                if direction == TX:
                    port.tx_bytes += sol.cd_bytes[i] * share
                    port.tx_drops += sol.cd_drop_packets[i] * share
                    port.tx_errors += sol.cd_error_packets[i] * share
                else:
                    port.rx_bytes += sol.cd_bytes[i] * share
                    port.rx_drops += sol.cd_drop_packets[i] * share
                    port.rx_errors += sol.cd_error_packets[i] * share
                port.queue_depth = sol.cd_queue[i]
                port.utilisation = sol.cd_rho[i]

    # -- collectives -------------------------------------------------------

    def allreduce_time_s(self, n_ranks: int, message_bytes: float,
                         sol: FabricSolution | None = None) -> float:
        """Cost of the job-wide reduction that closes each timestep.

        Small payloads, so this is latency-dominated and modelled as a
        double-binary tree: ``2 * ceil(log2 P)`` sequential hops at the worst
        pairwise latency in the job, plus a ring-style bandwidth term that is
        negligible at these sizes. Queueing delay on the congested uplinks is
        added, which is why a fabric fault lengthens the collective as well as
        the halo exchange.
        """
        ic = self.cluster.cfg.interconnect
        base_latency_s = (2.0 * ic.nic.latency_us + 2.0 * ic.leaf_uplink.latency_us) * 1e-6

        queue_s = 0.0
        if sol is not None:
            uplinks = [i for i, c in enumerate(self.cd_channel) if c.startswith("uplink:")]
            if uplinks:
                queue_s = float(sol.cd_queue_delay_s[uplinks].max())

        steps = 2.0 * float(np.ceil(np.log2(max(n_ranks, 2))))
        cap_bps = ic.leaf_uplink.total_bandwidth_gbps * _GBPS_TO_BPS
        bandwidth_term = 2.0 * (n_ranks - 1) / n_ranks * message_bytes / cap_bps
        return steps * (base_latency_s + queue_s) + bandwidth_term


def build_halo_flows(fabric: Fabric, decomposition, placement) -> FlowSet:
    """Compile the halo exchange into a FlowSet.

    One flow per (rank, direction): 128 x 6 = 768 concurrent transfers, each
    carrying `inner_iterations` rounds' worth of one face.
    """
    router = fabric.router
    nbytes_per_face = decomposition.halo_bytes_per_iteration()   # (n_ranks, 6)
    n_ranks, n_dirs = nbytes_per_face.shape

    src_rank, dst_rank, nbytes, latency, hop_rows = [], [], [], [], []
    port_rx, port_tx, kinds = [], [], []

    for rank in range(n_ranks):
        src_gpu = int(placement.rank_to_gpu[rank])
        for d in range(n_dirs):
            dst = int(decomposition.neighbours[rank, d])
            dst_gpu = int(placement.rank_to_gpu[dst])
            path = router.route(src_gpu, dst_gpu)

            hops = fabric.path_hops(path)
            row = hops + [-1] * (MAX_HOPS - len(hops))

            prx, ptx = -1, -1
            for port_id, direction in path.port_credits:
                if direction == RX:
                    prx = fabric.port_index[port_id]
                else:
                    ptx = fabric.port_index[port_id]

            src_rank.append(rank)
            dst_rank.append(dst)
            nbytes.append(float(nbytes_per_face[rank, d]))
            latency.append(path.latency_s)
            hop_rows.append(row)
            port_rx.append(prx)
            port_tx.append(ptx)
            kinds.append(path.kind)

    return FlowSet(
        src_rank=np.array(src_rank, dtype=np.int64),
        dst_rank=np.array(dst_rank, dtype=np.int64),
        nbytes=np.array(nbytes, dtype=np.float64),
        latency_s=np.array(latency, dtype=np.float64),
        hops=np.array(hop_rows, dtype=np.int64),
        port_rx=np.array(port_rx, dtype=np.int64),
        port_tx=np.array(port_tx, dtype=np.int64),
        kind=np.array(kinds, dtype=object),
    )

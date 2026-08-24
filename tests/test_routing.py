"""Routing: hop counts, latency ordering, and bottleneck-not-mean bandwidth."""

from __future__ import annotations

import numpy as np
import pytest

from gcsim.models.network import Fabric, build_halo_flows
from gcsim.routing import CROSS_DOMAIN, INTRA_DOMAIN, INTRANODE


def test_hop_counts_by_pair_class(cluster, router):
    #  GPUs 0..7 are node r0n0; 8..15 are r0n1; 32.. is rack r1.
    same_node = router.route(0, 3)
    same_rack = router.route(0, 9)
    cross = router.route(0, 40)

    assert (same_node.kind, same_node.switch_hops) == (INTRANODE, 0)
    assert (same_rack.kind, same_rack.switch_hops) == (INTRA_DOMAIN, 1)
    assert (cross.kind, cross.switch_hops) == (CROSS_DOMAIN, 3)


def test_latency_increases_with_distance(router):
    assert (router.route(0, 3).latency_us
            < router.route(0, 9).latency_us
            < router.route(0, 40).latency_us)


def test_cross_domain_path_traverses_both_leaf_bundles(cluster, router):
    path = router.route(0, 40)
    channels = [h.channel_id for h in path.hops]
    assert "uplink:r0" in channels and "uplink:r1" in channels
    #  ...and both host NICs, which is where the real bottleneck usually is.
    assert sum(c.startswith("nic:") for c in channels) == 2


def test_intranode_path_has_no_switch_ports(router):
    path = router.route(0, 3)
    assert path.port_credits == ()
    assert path.hops[0].channel_id.startswith("intranode:")


def test_effective_bandwidth_is_the_bottleneck_not_the_mean(cluster, router):
    """A flow runs at the rate of its slowest hop.

    Taking a mean would let a fat NVLink hop mask a starved uplink, which is
    exactly the error that makes fabric faults invisible.
    """
    fabric = Fabric(cluster, router)
    cap, _ = fabric._capacity_and_error()
    path = router.route(0, 40)
    hops = fabric.path_hops(path)
    per_hop = cap[hops]
    assert per_hop.min() < per_hop.mean()
    #  the NIC (200 Gbps) is narrower than the 8x400 Gbps uplink bundle
    assert np.isclose(per_hop.min(), 200e9 / 8)


def test_downing_uplinks_only_affects_that_domain(cluster, router):
    fabric = Fabric(cluster, router)
    before, _ = fabric._capacity_and_error()
    idx = {c: i for i, c in enumerate(fabric.cd_channel)}

    leaf = cluster.leaf_of("r2")
    for port_id in leaf.uplink_ids[1:]:
        cluster.ports[port_id].up = False

    after, _ = fabric._capacity_and_error()
    assert after[idx["uplink:r2"]] == pytest.approx(before[idx["uplink:r2"]] / 8)
    assert after[idx["uplink:r0"]] == before[idx["uplink:r0"]]
    assert after[idx["uplink:r1"]] == before[idx["uplink:r1"]]
    assert after[idx["intranode:r2n0"]] == before[idx["intranode:r2n0"]]

    for port_id in leaf.uplink_ids:
        cluster.ports[port_id].up = True


def test_route_cache_is_stable_under_link_failure(cluster, router):
    """Routes are topology, not health.

    A downed member of a bundle reduces the bundle's capacity; it does not
    reroute the flow. That mirrors ECMP over a link aggregation group and keeps
    the compiled flow table valid for the whole run.
    """
    before = router.route(0, 40)
    cluster.ports[cluster.leaf_of("r0").uplink_ids[0]].up = False
    after = router.route(0, 40)
    assert before is after
    cluster.ports[cluster.leaf_of("r0").uplink_ids[0]].up = True


def test_halo_flow_set_shape(cluster, router, decomposition, placement):
    fabric = Fabric(cluster, router)
    flows = build_halo_flows(fabric, decomposition, placement)
    assert flows.n_flows == 128 * 6
    #  Two of six directions cross a rack boundary under packed placement.
    assert (flows.kind == CROSS_DOMAIN).sum() == 128 * 2
    assert (flows.kind == INTRANODE).sum() == 128 * 2
    assert (flows.kind == INTRA_DOMAIN).sum() == 128 * 2
    #  Intra-node flows never touch a switch port.
    intranode = flows.kind == INTRANODE
    assert (flows.port_rx[intranode] == -1).all()


def test_port_queue_state_is_instantaneous_not_a_high_water_mark(
        cluster, router, decomposition, placement):
    """`queue_depth` and `utilisation` must be refreshed, never accumulated.

    Port documents these two as instantaneous and refreshed each phase. They
    were previously combined with max() against whatever the last phase left
    behind, which quietly made them a monotone high-water mark over the whole
    run: a gauge that could rise and never fall.

    Nothing in the shipped scenarios exposes it -- healthy runs are steady
    state and `leaf_uplink_failure` persists to the end, so the running peak
    happens to equal the current value. It breaks the moment a fabric fault is
    transient, which is the direction the straggler has already gone.

    Asserted by planting a stale peak rather than by degrading a link, because
    which way a fault moves this gauge is not obvious: squeezing a rack onto
    one uplink actually *lowers* the downlink queue, since rate-limiting
    upstream reduces the load the NIC channel is offered. Planting the peak
    tests the property directly and does not depend on that.
    """
    fabric = Fabric(cluster, router)
    flows = build_halo_flows(fabric, decomposition, placement)
    leaf = cluster.leaf_of("r2")
    downlinks = [cluster.ports[p] for p in leaf.downlink_ids]

    def exchange():
        sol = fabric.solve(flows, decomposition.n_ranks)
        fabric.accumulate(flows, sol)
        return max(p.queue_depth for p in downlinks)

    steady = exchange()
    assert steady > 0                      # there is something to overwrite

    #  Same input twice must give the same reading, not a doubled one.
    assert exchange() == pytest.approx(steady)

    #  A peak left by some earlier, worse phase must not survive into this one.
    for port in downlinks:
        port.queue_depth = steady * 1000.0
        port.utilisation = 1.0
    assert exchange() == pytest.approx(steady)
    assert all(p.utilisation < 1.0 for p in downlinks)

    #  And the same for the uplink branch, which used plain assignment.
    uplinks = [cluster.ports[p] for p in leaf.uplink_ids]
    up_steady = max(p.queue_depth for p in uplinks)
    for port in uplinks:
        port.queue_depth = up_steady + 1e6
    exchange()
    assert max(p.queue_depth for p in uplinks) == pytest.approx(up_steady)


def test_collective_step_latency_has_three_tiers(cluster, router):
    """worst_latency_us prices a collective by the widest span its ranks cover.

    With gpus_per_node=8 and 4 nodes per rack: one node is NVLink only, one
    rack crosses two NICs, and anything wider crosses the spine. Asserted
    against the config expressions rather than literal microseconds, so the
    test follows the config if the link speeds are ever retuned.
    """
    ic = cluster.cfg.interconnect
    per_node = cluster.cfg.gpus_per_node
    per_rack = cluster.cfg.gpus_per_rack

    same_node = router.worst_latency_us(list(range(per_node)))
    same_rack = router.worst_latency_us([0, per_node])
    cross_rack = router.worst_latency_us([0, per_rack])

    assert same_node == pytest.approx(ic.intranode.latency_us)
    assert same_rack == pytest.approx(2.0 * ic.nic.latency_us)
    assert cross_rack == pytest.approx(2.0 * ic.nic.latency_us
                                       + 2.0 * ic.leaf_uplink.latency_us)
    assert same_node < same_rack < cross_rack


def test_allreduce_cost_tracks_where_the_ranks_actually_are(cluster, router):
    """The collective is priced by placement, not by a config constant.

    The previous implementation hardcoded the cross-rack worst case, so a job
    confined to one node was charged 34 us per tree step for 2 us hops -- and
    this test fails on that code, which is the point of it. Identical rank
    count, identical bytes; only where the GPUs sit differs.
    """
    import numpy as np
    from gcsim.models.network import Fabric

    fabric = Fabric(cluster, router)
    per_node = cluster.cfg.gpus_per_node
    per_rack = cluster.cfg.gpus_per_rack

    one_node = np.arange(per_node)
    four_racks = np.arange(per_node) * per_rack // 2   # spread over racks

    cheap = fabric.allreduce_time_s(per_node, 64.0, None, gpu_indices=one_node)
    dear = fabric.allreduce_time_s(per_node, 64.0, None, gpu_indices=four_racks)
    assert cheap < dear

    #  ...and the full job still lands exactly on the cross-rack bound the old
    #  code hardcoded, which is why every shipped scenario is byte-identical.
    ic = cluster.cfg.interconnect
    all_gpus = np.arange(cluster.n_gpus)
    import math
    steps = 2.0 * math.ceil(math.log2(cluster.n_gpus))
    expected_latency = (2.0 * ic.nic.latency_us + 2.0 * ic.leaf_uplink.latency_us) * 1e-6
    got = fabric.allreduce_time_s(cluster.n_gpus, 64.0, None, gpu_indices=all_gpus)
    bandwidth = 2.0 * (cluster.n_gpus - 1) / cluster.n_gpus * 64.0         / (ic.leaf_uplink.total_bandwidth_gbps * 1e9 / 8.0)
    assert got == pytest.approx(steps * expected_latency + bandwidth)

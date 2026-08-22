"""Topology construction and placement."""

from __future__ import annotations

import numpy as np
import pytest

from gcsim.placement import KIND_CODES, place
from gcsim.routing import CROSS_DOMAIN, INTRA_DOMAIN, INTRANODE
from gcsim.topology import build_cluster


def test_cluster_shape(cluster, bundle):
    cc = bundle.cluster
    assert len(cluster.racks) == cc.racks == 4
    assert len(cluster.nodes) == cc.racks * cc.nodes_per_rack == 16
    assert len(cluster.gpus) == 128
    assert len(cluster.nics) == 16 * cc.nics_per_node == 32
    #  4 leaves x (8 downlinks + 8 uplinks) + 2 spines x 16 downlinks
    assert len(cluster.ports) == 4 * (8 + 8) + 2 * 16 == 96


def test_every_gpu_belongs_to_exactly_one_node_and_rack(cluster):
    from_nodes = [g for node in cluster.node_list for g in node.gpu_ids]
    assert sorted(from_nodes) == sorted(cluster.gpus)
    assert len(from_nodes) == len(set(from_nodes))
    for gpu in cluster.gpu_list:
        assert gpu.gpu_id in cluster.nodes[gpu.node_id].gpu_ids
        assert gpu.node_id in cluster.racks[gpu.rack_id].node_ids
        assert gpu.nic_id in cluster.nodes[gpu.node_id].nic_ids


def test_spine_ports_mirror_leaf_uplinks(cluster):
    """Each cable is accounted once, on its leaf side, and mirrored on the spine.

    Counting both ends independently would double the fabric's apparent traffic
    and break the conservation invariant.
    """
    spine_ports = [p for p in cluster.port_list if p.switch_tier == "spine"]
    assert len(spine_ports) == 32
    for port in spine_ports:
        assert port.mirror_of is not None
        partner = cluster.ports[port.mirror_of]
        assert partner.switch_tier == "leaf" and partner.role == "uplink"
    #  ...and leaf-side ports are never mirrors, so there is exactly one origin.
    assert all(p.mirror_of is None for p in cluster.port_list if p.switch_tier == "leaf")


def test_the_leaf_is_non_blocking(cluster, bundle):
    """Uplink capacity matches downlink capacity.

    Deliberate: it leaves the host NIC as the intended bottleneck and gives the
    uplink path real headroom in health. Without that headroom `congested` and
    the queue counters would be pinned high in the baseline and would mean
    nothing when a fault actually arrives.
    """
    ic = bundle.cluster.interconnect
    down = bundle.cluster.nodes_per_rack * bundle.cluster.nics_per_node * ic.nic.bandwidth_gbps
    up = ic.leaf_uplink.count * ic.leaf_uplink.bandwidth_gbps
    assert down == up == 1600.0


def test_packed_placement_puts_largest_faces_on_the_fastest_link(decomposition, placement):
    """The decomposition mapping is the whole reason topology matters here.

    With an 8x4x4 grid on 8-GPU nodes, +/-x must be intra-node, +/-y intra-rack
    and +/-z cross-rack. The +/-x faces are also the *largest*, so the biggest
    transfers ride NVLink and only the two smallest faces ever cross a spine.
    """
    kinds = placement.neighbour_kind
    assert set(np.unique(kinds[:, 0:2])) == {KIND_CODES[INTRANODE]}
    assert set(np.unique(kinds[:, 2:4])) == {KIND_CODES[INTRA_DOMAIN]}
    assert set(np.unique(kinds[:, 4:6])) == {KIND_CODES[CROSS_DOMAIN]}

    faces = decomposition.face_cells[0]
    assert faces[0] == faces[1] > faces[2]        # +/-x is the biggest face


def test_scatter_placement_is_measurably_worse(cluster, decomposition, router):
    """A topology-blind mapping pushes the largest faces onto the slowest links.

    Included so the benefit of placement is demonstrated rather than asserted.
    """
    packed = place(cluster, decomposition, router, strategy="packed")
    scatter = place(cluster, decomposition, router, strategy="scatter")

    def cross_rack_bytes(p):
        mask = p.neighbour_kind == KIND_CODES[CROSS_DOMAIN]
        return float((decomposition.halo_bytes_per_iteration() * mask).sum())

    #  Both mappings send two of six directions across a rack -- the counts are
    #  identical. What differs is WHICH two, and that is the whole point.
    assert packed.cross_domain_fraction() == scatter.cross_domain_fraction() == 2 / 6

    #  What changes is WHICH directions land where. Under packed the two LARGEST
    #  faces (+/-x) stay on NVLink and only the two smallest cross a rack. Under
    #  scatter that is exactly inverted: +/-x now crosses racks.
    assert set(np.unique(packed.neighbour_kind[:, 0:2])) == {KIND_CODES[INTRANODE]}
    assert set(np.unique(scatter.neighbour_kind[:, 0:2])) == {KIND_CODES[CROSS_DOMAIN]}

    #  The +/-x faces are twice the area of +/-z, so a topology-blind mapping
    #  doubles the bytes crossing the scarcest resource in the cluster.
    assert cross_rack_bytes(scatter) == pytest.approx(cross_rack_bytes(packed) * 2)


def test_build_is_deterministic(bundle):
    a, b = build_cluster(bundle.cluster), build_cluster(bundle.cluster)
    assert [g.gpu_id for g in a.gpu_list] == [g.gpu_id for g in b.gpu_list]
    assert list(a.ports) == list(b.ports)
    assert list(a.channels) == list(b.channels)

"""Job subset allocation on the fixed cluster.

The cluster stays 4 racks x 4 nodes x 8 GPUs; the allocation block chooses how
many of those GPUs the job occupies and from which pool. The properties pinned
here: the default path is exactly the historical whole-cluster behaviour, the
strategies distribute a subset the way their names claim, requests that cannot
fit fail loudly, idle GPUs read as idle rather than as spinning, and a
stochastic fault can never land on a GPU that is not running anything.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from gcsim.config import AllocationConfig
from gcsim.mesh import partition
from gcsim.placement import place


def _subset_cfg(bundle, scenario, mesh, seed, n_ranks, strategy="packed",
                racks=None, nodes=None):
    cfg = bundle.build(scenario, mesh=mesh, seed=seed)
    alloc = AllocationConfig(n_ranks=n_ranks, racks=racks, nodes=nodes)
    return replace(cfg, workload=replace(cfg.workload, allocation=alloc,
                                         placement=strategy))


def _placed(bundle, cluster, router, n_ranks, strategy, racks=None, nodes=None):
    d = partition(bundle.meshes["coarse"], n_ranks,
                  preferred_first_extent=bundle.cluster.gpus_per_node)
    alloc = AllocationConfig(n_ranks=n_ranks, racks=racks, nodes=nodes)
    return place(cluster, d, router, strategy=strategy, allocation=alloc)


# ---------------------------------------------------------------------------
# placement of a subset
# ---------------------------------------------------------------------------

def test_packed_subset_fills_the_first_rack(bundle, cluster, router):
    p = _placed(bundle, cluster, router, 32, "packed")
    racks = {cluster.gpu(int(g)).rack_index for g in p.rank_to_gpu}
    assert racks == {0}
    assert list(p.rank_to_gpu) == list(range(32))          # a prefix, in order


def test_scatter_subset_spreads_evenly_across_racks(bundle, cluster, router):
    p = _placed(bundle, cluster, router, 32, "scatter")
    per_rack = np.bincount([cluster.gpu(int(g)).rack_index for g in p.rank_to_gpu],
                           minlength=bundle.cluster.racks)
    assert list(per_rack) == [8, 8, 8, 8]


def test_scatter_at_full_allocation_is_the_historical_formula(bundle, cluster, router):
    """The round-robin generalisation must degenerate to exactly the old map."""
    d = partition(bundle.meshes["coarse"], cluster.n_gpus,
                  preferred_first_extent=bundle.cluster.gpus_per_node)
    p = place(cluster, d, router, strategy="scatter")      # no allocation at all
    racks, per_rack = bundle.cluster.racks, bundle.cluster.gpus_per_rack
    r = np.arange(cluster.n_gpus)
    assert np.array_equal(p.rank_to_gpu, (r % racks) * per_rack + (r // racks))


def test_rack_pool_restriction_is_honoured(bundle, cluster, router):
    p = _placed(bundle, cluster, router, 16, "scatter", racks=(1,))
    assert {cluster.gpu(int(g)).rack_index for g in p.rank_to_gpu} == {1}
    assert all(g.startswith("r1") for g in p.allocated_gpu_ids)


def test_request_larger_than_the_pool_raises(bundle, cluster, router):
    with pytest.raises(ValueError, match="cannot fit"):
        _placed(bundle, cluster, router, 40, "packed", racks=(1,))


def test_gpu_to_rank_is_minus_one_exactly_on_idle_gpus(bundle, cluster, router):
    p = _placed(bundle, cluster, router, 32, "scatter")
    assert p.gpu_to_rank.size == cluster.n_gpus
    idle = p.gpu_to_rank < 0
    assert int(idle.sum()) == cluster.n_gpus - 32
    #  ...and it round-trips on the allocated ones.
    for rank, gpu in enumerate(p.rank_to_gpu):
        assert p.gpu_to_rank[gpu] == rank
    assert not np.isin(np.nonzero(idle)[0], p.rank_to_gpu).any()


# ---------------------------------------------------------------------------
# the collective sees the allocation
# ---------------------------------------------------------------------------

def test_allreduce_is_cheaper_for_a_packed_node_than_a_scattered_eight(
        bundle, cluster, router):
    """Same rank count, same bytes; only where the GPUs sit differs.

    Builds directly on the landed placement-aware collective: an 8-rank job
    packed into one node pays intranode latency per tree step, the same job
    scattered across racks pays the spine crossing.
    """
    from gcsim.models.network import Fabric

    fabric = Fabric(cluster, router)
    packed = _placed(bundle, cluster, router, 8, "packed")
    spread = _placed(bundle, cluster, router, 8, "scatter")
    assert {cluster.gpu(int(g)).node_id for g in packed.rank_to_gpu} == {"r0n0"}

    cheap = fabric.allreduce_time_s(8, 64.0, None, gpu_indices=packed.rank_to_gpu)
    dear = fabric.allreduce_time_s(8, 64.0, None, gpu_indices=spread.rank_to_gpu)
    assert cheap < dear


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_memory_guard_rejects_a_mesh_that_cannot_fit_the_ranks(bundle):
    """4 ranks of the medium mesh is 108 GB per device against 80 GB.

    Previously this configuration ran to completion and reported the
    impossible memory use as ordinary telemetry.
    """
    from gcsim.engine.simulator import Simulator

    cfg = _subset_cfg(bundle, "healthy", "medium", 42, n_ranks=4)
    with pytest.raises(ValueError, match="medium.*4 ranks.*GB"):
        Simulator(cfg)


def test_racks_and_nodes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        AllocationConfig(n_ranks=8, racks=(0,), nodes=("r0n0",))


# ---------------------------------------------------------------------------
# one short subset run, end to end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def subset_run(bundle):
    """straggler on the coarse mesh, 32 ranks scattered over all four racks."""
    from gcsim.engine.simulator import Simulator

    cfg = _subset_cfg(bundle, "straggler", "coarse", 42, n_ranks=32,
                      strategy="scatter")
    return Simulator(cfg).run()


def test_idle_gpus_read_as_idle_not_as_spinning(subset_run):
    gpu = subset_run.frames["telemetry_gpu"]
    rank = subset_run.frames["rank_performance"]
    allocated = set(rank["gpu_id"].unique())
    assert len(allocated) == 32

    late = gpu[gpu["timestamp"] > gpu["timestamp"].max() * 0.5]
    idle = late[~late["gpu_id"].isin(allocated)]
    active = late[late["gpu_id"].isin(allocated)]
    assert idle["gpu_id"].nunique() == 96

    #  Idle is idle: no resident kernel, so no spin floor in either channel.
    assert idle["utilization_pct"].max() < 5.0
    assert idle["sm_occupancy_pct"].max() < 1.0
    #  Idle devices dissipate idle power only, so they settle well below the
    #  working GPUs and near their rack inlet.
    assert idle["temperature_c"].mean() < active["temperature_c"].mean() - 2.0
    assert not idle["throttled"].any()


def test_straggler_cohort_is_drawn_from_allocated_gpus_only(subset_run):
    events = subset_run.frames["events"]
    fired = events[events["event_type"] == "INJECTION_APPLIED"]
    episodes = json.loads(fired.iloc[0]["payload"])["episodes"]
    victims = {e["gpu_id"] for e in episodes}

    allocated = set(subset_run.frames["rank_performance"]["gpu_id"].unique())
    assert victims <= allocated
    assert victims                                  # drew someone, not nobody


def test_counter_conservation_holds_for_a_subset_job(subset_run):
    """Scattered ranks put real traffic on every leaf; the books must balance."""
    ports = subset_run.frames["telemetry_switch_port"]
    last = ports[ports["timestamp"] == ports["timestamp"].max()]
    uplinks = last[(last["switch_tier"] == "leaf") & (last["port_role"] == "uplink")]
    assert uplinks["tx_bytes"].sum() > 0            # cross-rack traffic exists
    assert uplinks["tx_bytes"].sum() == pytest.approx(
        uplinks["rx_bytes"].sum(), rel=1e-9)

    from gcsim.metrics import counter_conservation
    c = counter_conservation(subset_run.frames)
    assert np.allclose(c["uplink_tx_gb"], c["uplink_rx_gb"])


def test_summary_records_the_allocation(subset_run):
    s = subset_run.summary
    assert s["n_ranks"] == 32
    assert s["allocated_gpus"] == 32
    assert s["placement"] == "scatter"

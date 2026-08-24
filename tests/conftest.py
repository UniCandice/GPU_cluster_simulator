"""Shared fixtures.

Scenario runs are session-scoped: the full six-scenario sweep on the medium mesh
costs a few seconds, and every behavioural test reads from the same set, so a
regression shows up consistently across tests rather than in one flaky place.
"""

from __future__ import annotations

import pytest

from gcsim.config import load_config
from gcsim.mesh import partition
from gcsim.placement import place
from gcsim.routing import Router
from gcsim.scenarios import run_scenario
from gcsim.topology import build_cluster

TEST_MESH = "medium"


@pytest.fixture(scope="session")
def bundle():
    """The configuration with any allocation block stripped.

    The behavioural suite pins the FULL-cluster reference physics -- 128 ranks,
    the shipped fault targets, the documented signature matrix. The allocation
    block in workload.yaml is a user knob, and leaving it live here would let an
    experiment in a config file silently re-parameterise several dozen tests.
    Allocation behaviour is tested deliberately, in test_allocation.py, from
    configs those tests construct themselves.
    """
    from dataclasses import replace
    b = load_config()
    if b.workload.allocation is not None:
        b = replace(b, workload=replace(b.workload, allocation=None))
    return b


@pytest.fixture(scope="session")
def cluster(bundle):
    return build_cluster(bundle.cluster)


@pytest.fixture(scope="session")
def router(cluster):
    return Router(cluster)


@pytest.fixture(scope="session")
def decomposition(bundle, cluster):
    return partition(bundle.meshes[TEST_MESH], cluster.n_gpus,
                     preferred_first_extent=bundle.cluster.gpus_per_node)


@pytest.fixture(scope="session")
def placement(cluster, decomposition, router):
    return place(cluster, decomposition, router, strategy="packed")


@pytest.fixture(scope="session")
def runs(bundle):
    """Every scenario on the medium mesh, run once."""
    return {name: run_scenario(name, mesh=TEST_MESH, seed=42, bundle=bundle, out_dir=None)
            for name in bundle.scenario_order}


@pytest.fixture(scope="session")
def healthy(runs):
    return runs["healthy"]


@pytest.fixture(scope="session")
def mesh_runs(bundle):
    """Healthy runs at all three resolutions, for the mesh study tests."""
    return {name: run_scenario("healthy", mesh=name, seed=42, bundle=bundle, out_dir=None)
            for name in ("coarse", "medium", "fine")}

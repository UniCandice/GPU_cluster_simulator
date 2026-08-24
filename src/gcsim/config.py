"""Configuration loading and the seed-derivation scheme.

Two things live here:

1.  Typed views over the YAML in `configs/`. These are deliberately plain
    dataclasses with a `from_dict` classmethod rather than a schema library, so
    the mapping from YAML key to simulator constant is readable in one pass.

2.  `derive_rng`, the reproducibility mechanism. Every stochastic quantity in
    the simulator draws from a generator keyed by a *stable string* -- the
    entity's identity, never its index. That means adding a scenario, adding a
    rack, or reordering a loop does not perturb any other entity's stream, so a
    healthy run and a faulted run are directly diffable: every shared stream is
    bit-identical and the only differences are consequences of the injection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def _stable_key(key: str) -> int:
    """Map a string to a stable 32-bit integer.

    Python's builtin ``hash`` is randomised per process, which would silently
    destroy reproducibility across runs. blake2b is stable everywhere.
    """
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest(), "big")


def derive_rng(master_seed: int, key: str) -> np.random.Generator:
    """Return the generator for `key` under `master_seed`.

    Keys are entity identities, e.g. ``"gpu:r1n2g5"``, ``"node:r1n2"``,
    ``"rack:r1"``, ``"job:jitter"``. Independent keys give independent streams
    and the same (seed, key) pair always gives the same stream.
    """
    return np.random.default_rng(np.random.SeedSequence(entropy=[master_seed, _stable_key(key)]))


# ---------------------------------------------------------------------------
# Cluster configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuSpec:
    model: str
    memory_gb: float
    base_clock_mhz: float
    min_clock_mhz: float
    clock_step_mhz: float
    idle_power_w: float
    max_power_w: float
    board_power_cap_w: float
    power_clock_exponent: float
    power_ema_alpha: float
    spin_occupancy: float
    spin_utilisation: float
    thermal_resistance_c_per_w: float
    thermal_time_constant_s: float
    thermal_slowdown_c: float
    thermal_hw_slowdown_c: float
    thermal_hysteresis_c: float
    thermal_derate_per_c: float
    seconds_per_cell_update: float
    occupancy_half_cells: float
    kernel_launch_overhead_s: float
    kernels_per_inner_iteration: int
    memory_bandwidth_gbps: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GpuSpec":
        return cls(**{f: d[f] for f in cls.__dataclass_fields__})


@dataclass(frozen=True)
class LinkSpec:
    bandwidth_gbps: float
    latency_us: float
    ber: float
    count: int = 1

    @property
    def total_bandwidth_gbps(self) -> float:
        return self.bandwidth_gbps * self.count


@dataclass(frozen=True)
class InterconnectSpec:
    intranode: LinkSpec
    nic: LinkSpec
    leaf_uplink: LinkSpec
    buffer_packets: int
    packet_bytes: float
    max_utilisation: float
    retransmit_penalty: float
    spine_count: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InterconnectSpec":
        return cls(
            intranode=LinkSpec(**d["intranode"]),
            nic=LinkSpec(**d["nic"]),
            leaf_uplink=LinkSpec(
                bandwidth_gbps=d["leaf_uplink"]["bandwidth_gbps"],
                latency_us=d["leaf_uplink"]["latency_us"],
                ber=d["leaf_uplink"]["ber"],
                count=d["leaf_uplink"]["count_per_leaf"],
            ),
            buffer_packets=d["buffer_packets"],
            packet_bytes=d["packet_bytes"],
            max_utilisation=d["max_utilisation"],
            retransmit_penalty=d["retransmit_penalty"],
            spine_count=d["spine_count"],
        )


@dataclass(frozen=True)
class CoolingSpec:
    base_inlet_temp_c: float
    coupling_c_per_kw: float
    nominal_efficiency: float


@dataclass(frozen=True)
class StorageSpec:
    capacity_gbps: float
    base_read_latency_ms: float
    base_write_latency_ms: float
    max_utilisation: float
    dirty_drain_gbps: float


@dataclass(frozen=True)
class HostSpec:
    cpu_cores: int
    memory_gb: float
    idle_power_w: float


@dataclass(frozen=True)
class TelemetrySpec:
    sample_interval_s: float
    straggler_rel_threshold: float


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    racks: int
    nodes_per_rack: int
    gpus_per_node: int
    nics_per_node: int
    gpu: GpuSpec
    interconnect: InterconnectSpec
    cooling: CoolingSpec
    storage: StorageSpec
    host: HostSpec
    telemetry: TelemetrySpec

    @property
    def gpus_per_rack(self) -> int:
        return self.nodes_per_rack * self.gpus_per_node

    @property
    def n_nodes(self) -> int:
        return self.racks * self.nodes_per_rack

    @property
    def n_gpus(self) -> int:
        return self.n_nodes * self.gpus_per_node

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClusterConfig":
        return cls(
            name=d["name"],
            racks=d["racks"],
            nodes_per_rack=d["nodes_per_rack"],
            gpus_per_node=d["gpus_per_node"],
            nics_per_node=d["nics_per_node"],
            gpu=GpuSpec.from_dict(d["gpu"]),
            interconnect=InterconnectSpec.from_dict(d["interconnect"]),
            cooling=CoolingSpec(**d["cooling"]),
            storage=StorageSpec(**d["storage"]),
            host=HostSpec(**d["host"]),
            telemetry=TelemetrySpec(**d["telemetry"]),
        )


# ---------------------------------------------------------------------------
# Mesh and workload configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeshConfig:
    name: str
    dims: tuple[int, int, int]
    label: str
    note: str
    bytes_per_cell: float
    halo_fields: int
    halo_depth: int
    bytes_per_value: float
    inner_iterations: int
    output_bytes_per_cell: float
    dataload_bytes_per_cell: float

    @property
    def total_cells(self) -> int:
        nx, ny, nz = self.dims
        return nx * ny * nz

    @property
    def halo_bytes_per_boundary_cell(self) -> float:
        """Bytes exchanged per boundary cell, per inner iteration."""
        return self.halo_fields * self.halo_depth * self.bytes_per_value

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "MeshConfig":
        return cls(
            name=name,
            dims=tuple(d["dims"]),  # type: ignore[arg-type]
            label=d.get("label", name),
            note=d.get("note", ""),
            bytes_per_cell=d["bytes_per_cell"],
            halo_fields=d["halo_fields"],
            halo_depth=d["halo_depth"],
            bytes_per_value=d["bytes_per_value"],
            inner_iterations=d["inner_iterations"],
            output_bytes_per_cell=d["output_bytes_per_cell"],
            dataload_bytes_per_cell=d["dataload_bytes_per_cell"],
        )


@dataclass(frozen=True)
class AllocationConfig:
    """Which slice of the fixed cluster the job occupies.

    The cluster itself never changes shape; this only chooses how many of its
    GPUs the job runs on and, optionally, which pool they are drawn from.
    `racks` and `nodes` are mutually exclusive ways of restricting the pool;
    with neither, the pool is the whole cluster. The `placement` strategy on
    WorkloadConfig then distributes the ranks over that pool.
    """
    n_ranks: int
    racks: tuple[int, ...] | None = None
    nodes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.racks is not None and self.nodes is not None:
            raise ValueError("allocation: `racks` and `nodes` are mutually exclusive")
        if self.n_ranks < 1:
            raise ValueError(f"allocation: n_ranks must be >= 1, got {self.n_ranks}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AllocationConfig":
        return cls(
            n_ranks=int(d["n_ranks"]),
            racks=tuple(int(r) for r in d["racks"]) if "racks" in d else None,
            nodes=tuple(str(n) for n in d["nodes"]) if "nodes" in d else None,
        )


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    iterations: int
    output_interval: int
    allreduce_values: int
    placement: str
    #: Absent means the job occupies every GPU -- exactly the historical
    #: behaviour, preserved bit-for-bit.
    allocation: AllocationConfig | None = None


@dataclass(frozen=True)
class Injection:
    """A perturbation of physical or workload state at a given timestep.

    Injections never touch telemetry. `type` selects a handler in `faults.py`.
    """
    at_iteration: int
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Injection":
        params = {k: v for k, v in d.items() if k not in ("at_iteration", "type")}
        return cls(at_iteration=d["at_iteration"], type=d["type"], params=params)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    fault: bool
    label: str
    description: str
    tier: str
    injections: tuple[Injection, ...]

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ScenarioConfig":
        return cls(
            name=name,
            fault=bool(d.get("fault", False)),
            label=d.get("label", name),
            description=" ".join(d.get("description", "").split()),
            tier=d.get("tier", "none"),
            injections=tuple(Injection.from_dict(i) for i in d.get("injections", [])),
        )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimConfig:
    """Everything one simulation run needs."""
    cluster: ClusterConfig
    mesh: MeshConfig
    workload: WorkloadConfig
    scenario: ScenarioConfig
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.scenario.name}__{self.mesh.name}__seed{self.seed}"


@dataclass(frozen=True)
class ConfigBundle:
    """The parsed contents of `configs/`, before a run is specialised."""
    cluster: ClusterConfig
    meshes: dict[str, MeshConfig]
    default_mesh: str
    workload: WorkloadConfig
    scenarios: dict[str, ScenarioConfig]
    scenario_order: tuple[str, ...]

    def build(self, scenario: str, mesh: str | None = None, seed: int = 42) -> SimConfig:
        if scenario not in self.scenarios:
            raise KeyError(f"unknown scenario {scenario!r}; have {sorted(self.scenarios)}")
        mesh_name = mesh or self.default_mesh
        if mesh_name not in self.meshes:
            raise KeyError(f"unknown mesh {mesh_name!r}; have {sorted(self.meshes)}")
        return SimConfig(
            cluster=self.cluster,
            mesh=self.meshes[mesh_name],
            workload=self.workload,
            scenario=self.scenarios[scenario],
            seed=seed,
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(config_dir: Path | str | None = None) -> ConfigBundle:
    """Load `configs/{cluster,meshes,workload,scenarios}.yaml`."""
    d = Path(config_dir) if config_dir is not None else CONFIG_DIR

    cluster = ClusterConfig.from_dict(_read_yaml(d / "cluster.yaml"))

    mesh_doc = _read_yaml(d / "meshes.yaml")
    meshes = {n: MeshConfig.from_dict(n, m) for n, m in mesh_doc["meshes"].items()}

    wl = _read_yaml(d / "workload.yaml")
    workload = WorkloadConfig(
        name=wl["name"],
        iterations=wl["iterations"],
        output_interval=wl["output_interval"],
        allreduce_values=wl["allreduce_values"],
        placement=wl["placement"],
        allocation=(AllocationConfig.from_dict(wl["allocation"])
                    if wl.get("allocation") else None),
    )

    sc_doc = _read_yaml(d / "scenarios.yaml")
    scenarios = {n: ScenarioConfig.from_dict(n, s) for n, s in sc_doc["scenarios"].items()}
    order = tuple(sc_doc.get("order", sorted(scenarios)))

    return ConfigBundle(
        cluster=cluster,
        meshes=meshes,
        default_mesh=mesh_doc.get("default_mesh", next(iter(meshes))),
        workload=workload,
        scenarios=scenarios,
        scenario_order=order,
    )

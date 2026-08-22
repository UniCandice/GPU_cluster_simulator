"""gcsim -- a seeded, reproducible simulator of distributed GPU CFD workload
performance and reliability.

The design in one sentence: a bulk-synchronous CFD job is decomposed onto a
128-GPU cluster, its phase costs are derived from mesh geometry and network
topology, and every telemetry stream is an *observation* of the resulting
physical state rather than an independently generated signal.

Read `TELEMETRY.md` for the full stream reference and `README.md` for the model,
assumptions and limitations.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

_LAZY = {
    "SimConfig": "gcsim.config",
    "load_config": "gcsim.config",
    "derive_rng": "gcsim.config",
    "build_cluster": "gcsim.topology",
    "partition": "gcsim.mesh",
    "run_scenario": "gcsim.scenarios",
    "run_matrix": "gcsim.scenarios",
    "RunResult": "gcsim.scenarios",
}

__all__ = [*_LAZY, "__version__"]


def __getattr__(name: str) -> Any:
    """Import submodules on first use.

    Keeps ``import gcsim.mesh`` working while the package is being built up, and
    keeps the import graph acyclic: nothing has to import the package root.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'gcsim' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)

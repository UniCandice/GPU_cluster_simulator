"""Running scenarios, singly and as a matrix."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from gcsim.config import ConfigBundle, SimConfig, load_config
from gcsim.engine.simulator import Simulator
from gcsim.metrics import diagnose
from gcsim.telemetry import write_run

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


@dataclass
class RunResult:
    config: SimConfig
    frames: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    path: Path | None = None
    wall_seconds: float = 0.0

    @property
    def run_id(self) -> str:
        return self.config.run_id

    def frame(self, name: str) -> pd.DataFrame:
        return self.frames[name]


def run_scenario(scenario: str, mesh: str | None = None, seed: int = 42,
                 bundle: ConfigBundle | None = None,
                 out_dir: Path | str | None = None,
                 record_ticks: bool = False) -> RunResult:
    """Simulate one (scenario, mesh, seed) and optionally persist it."""
    bundle = bundle or load_config()
    cfg = bundle.build(scenario=scenario, mesh=mesh, seed=seed)

    t0 = time.perf_counter()
    sim = Simulator(cfg, record_ticks=record_ticks)
    output = sim.run()
    wall = time.perf_counter() - t0

    #  The diagnosis reads telemetry only -- never the scenario's `fault` label.
    #  Storing both side by side is what makes the diagnosis view scoreable.
    verdict = diagnose(output.frames)

    #  `wall` is deliberately NOT persisted. It measures the machine that ran the
    #  simulation, not the simulation, and putting it in summary.json would make
    #  two identical runs differ on disk -- destroying the byte-for-byte
    #  reproducibility claim for the sake of a number the CLI can just print.
    summary = {**output.summary, "diagnosis": verdict.to_dict()}

    path = None
    if out_dir is not None:
        path = write_run(output.frames, summary, Path(out_dir) / cfg.run_id)

    return RunResult(config=cfg, frames=output.frames, summary=summary,
                     path=path, wall_seconds=wall)


def run_matrix(scenarios: Iterable[str] | None = None,
               meshes: Iterable[str] | None = None,
               seed: int = 42,
               bundle: ConfigBundle | None = None,
               out_dir: Path | str | None = DEFAULT_RUNS_DIR,
               progress: bool = True) -> list[RunResult]:
    """Run every (scenario, mesh) combination.

    Order is deterministic -- scenario order from the config, then mesh order --
    so `runs/` and the dashboard are reproducible down to the row ordering.
    """
    bundle = bundle or load_config()
    scenario_names = list(scenarios) if scenarios is not None else list(bundle.scenario_order)
    mesh_names = list(meshes) if meshes is not None else list(bundle.meshes)

    results: list[RunResult] = []
    total = len(scenario_names) * len(mesh_names)
    for i, scenario in enumerate(scenario_names):
        for j, mesh in enumerate(mesh_names):
            n = i * len(mesh_names) + j + 1
            if progress:
                print(f"[{n:2d}/{total}] {scenario} / {mesh} ...", end="", flush=True)
            result = run_scenario(scenario, mesh=mesh, seed=seed,
                                  bundle=bundle, out_dir=out_dir)
            if progress:
                s = result.summary
                print(f" {result.wall_seconds:5.1f}s  "
                      f"iter={s['mean_iteration_time_s'] * 1e3:7.1f}ms  "
                      f"eff={s['parallel_efficiency'] * 100:5.1f}%  "
                      f"verdict={s['diagnosis']['verdict']}")
            results.append(result)
    return results


def summaries_frame(results: list[RunResult]) -> pd.DataFrame:
    """Flatten every run's summary into one comparison table."""
    rows = []
    for r in results:
        s = dict(r.summary)
        d = s.pop("diagnosis", {})
        s["diagnosis_verdict"] = d.get("verdict")
        s["diagnosis_tier"] = d.get("tier")
        s["diagnosis_confidence"] = d.get("confidence")
        s["diagnosis_slowdown_pct"] = d.get("slowdown_pct")
        s["throttle_reasons"] = ",".join(s.get("throttle_reasons", []))
        rows.append(s)
    return pd.DataFrame(rows)

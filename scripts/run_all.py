#!/usr/bin/env python
"""Run the whole study and build the dashboard.

    python scripts/run_all.py --seed 42

Simulates every scenario on every mesh (6 x 3 = 18 runs), writes one Parquet
directory per run under `runs/`, prints the comparison tables, and renders the
self-contained HTML dashboard.

Nothing here is stochastic beyond the seed, so two invocations produce
byte-identical output. That is asserted in tests/test_reproducibility.py and is
worth re-checking by hand:

    python scripts/run_all.py --seed 42
    cp -r runs runs_a && python scripts/run_all.py --seed 42 && diff -r runs runs_a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from gcsim.config import load_config  # noqa: E402
from gcsim.metrics import (counter_conservation, mesh_scaling_table,  # noqa: E402
                           straggler_attribution)
from gcsim.scenarios import run_matrix, summaries_frame  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--dashboard", type=Path, default=ROOT / "dashboard" / "index.html")
    ap.add_argument("--meshes", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--no-dashboard", action="store_true")
    args = ap.parse_args(argv)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    fmt = lambda v: f"{v:,.4f}"  # noqa: E731

    bundle = load_config()
    print(f"cluster: {bundle.cluster.racks} racks x {bundle.cluster.nodes_per_rack} nodes "
          f"x {bundle.cluster.gpus_per_node} GPUs = {bundle.cluster.n_gpus} ranks")
    print(f"workload: {bundle.workload.iterations} timesteps, "
          f"output every {bundle.workload.output_interval}\n")

    results = run_matrix(scenarios=args.scenarios, meshes=args.meshes, seed=args.seed,
                         bundle=bundle, out_dir=args.runs)
    df = summaries_frame(results)

    # --- 1. did the diagnosis get it right? --------------------------------
    print("\n" + "=" * 100)
    print("DIAGNOSIS vs GROUND TRUTH   (the classifier reads telemetry only, never `fault`)")
    print("=" * 100)
    cols = ["run_id", "is_fault", "diagnosis_verdict", "diagnosis_tier",
            "diagnosis_slowdown_pct", "mean_iteration_time_s", "mean_rank_spread_s",
            "max_temperature_c", "throttle_reasons"]
    print(df[cols].to_string(index=False, float_format=fmt))
    agree = ((df["is_fault"] & (df["diagnosis_verdict"] == "HARDWARE_FAULT"))
             | (~df["is_fault"] & (df["diagnosis_verdict"] != "HARDWARE_FAULT")))
    print(f"\n  agreed on {agree.sum()}/{len(df)} runs")

    # --- 2. the mesh study --------------------------------------------------
    print("\n" + "=" * 100)
    print("MESH PARTITIONING vs GPU UTILISATION   (healthy runs; same cluster, no fault)")
    print("=" * 100)
    healthy = [r.summary for r in results if r.config.scenario.name == "healthy"]
    if healthy:
        print(mesh_scaling_table(healthy).to_string(index=False, float_format=fmt))
        print("\n  Compute scales with subdomain VOLUME, halo traffic with SURFACE AREA.")
        print("  Refining the mesh raises occupancy and parallel efficiency and lowers the")
        print("  relative cost of communication -- with no change to the cluster at all.")
        print("  `coarse` also divides neither grid extent, so it carries real load")
        print("  imbalance on perfectly healthy hardware.")

    # --- 3. straggler attribution ------------------------------------------
    ref = next((r for r in results
                if r.config.scenario.name == "straggler" and r.config.mesh.name == "medium"),
               None)
    if ref is not None:
        print("\n" + "=" * 100)
        print("STRAGGLER ATTRIBUTION   (straggler / medium)")
        print("=" * 100)
        table = straggler_attribution(ref.frames["rank_performance"], top_n=6)
        print(table.to_string(index=False, float_format=fmt))
        print("\n  The culprit is the rank with the LARGEST excess and the SMALLEST wait.")
        print("  Every other rank looks idle -- reading the wait column naively blames")
        print("  the 127 victims instead of the one cause.")
        print("\n  Note how weakly EXCESS resolves it here. This scenario is episodic: a")
        print("  small cohort carries the fault, but each member is derated only about a")
        print("  tenth of the run, so a whole-run average drags its excess down to within")
        print("  touching distance of the fleet's fastest silicon. The steps-as-straggler")
        print("  column separates them exactly, which is why the diagnosis localises by")
        print("  counting timesteps. Compare gpu_degradation, where the fault never lets")
        print("  go and the top row stands an order of magnitude clear on excess alone.")

        print("\n  counter conservation (leaf uplink bytes vs traffic that left the rack):")
        print(counter_conservation(ref.frames).to_string(index=False, float_format=fmt))

    # --- 4. dashboard -------------------------------------------------------
    if not args.no_dashboard:
        from gcsim.dashboard.build import build_dashboard
        print("\nbuilding dashboard ...", end="", flush=True)
        path, stamp = build_dashboard(runs_dir=args.runs, out_path=args.dashboard)
        print(f" {path}  ({path.stat().st_size / 1e6:.1f} MB)  --  Generated: {stamp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

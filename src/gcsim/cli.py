"""Command line interface.

    python -m gcsim run     --scenario thermal --mesh medium --seed 42
    python -m gcsim matrix  --seed 42
    python -m gcsim mesh-study
    python -m gcsim list
    python -m gcsim dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from gcsim.config import load_config
from gcsim.metrics import mesh_scaling_table, straggler_attribution
from gcsim.scenarios import DEFAULT_RUNS_DIR, run_matrix, run_scenario, summaries_frame


def _print_summary(result) -> None:
    s = result.summary
    d = s["diagnosis"]
    print(f"\n{s['run_id']}   ({result.wall_seconds:.1f}s wall, "
          f"{s['runtime_s']:.0f}s simulated)")
    print(f"  mesh              {s['mesh_dims']}  ->  {s['mesh_grid']} grid, "
          f"{s['mesh_cells_mean']:,.0f} cells/rank, imbalance {s['mesh_imbalance']:.4f}")
    print(f"  iteration time    {s['mean_iteration_time_s'] * 1e3:.1f} ms   "
          f"({s['throughput_iters_per_s']:.2f} it/s)")
    print(f"  parallel eff.     {s['parallel_efficiency'] * 100:.1f}%   "
          f"comm fraction {s['comm_fraction'] * 100:.1f}%")
    print(f"  occupancy / util  {s['mean_sm_occupancy_pct']:.1f}% / "
          f"{s['mean_utilization_pct']:.1f}%")
    print(f"  temperature       mean {s['mean_temperature_c']:.1f} C, "
          f"max {s['max_temperature_c']:.1f} C   "
          f"throttled {s['throttled_sample_fraction'] * 100:.1f}% of samples "
          f"{s['throttle_reasons'] or ''}")
    print(f"  rank spread       {s['mean_rank_spread_s'] * 1e3:.1f} ms")
    truth = "FAULT" if s["is_fault"] else "not a fault"
    print(f"\n  DIAGNOSIS  {d['verdict']} / tier={d['tier']} "
          f"(confidence {d['confidence']})     [ground truth: {truth}]")
    for line in d["evidence"]:
        print(f"    - {line}")
    if d["localisation"]:
        print(f"    localised to: {d['localisation']}")
    print(f"    discriminating channels: {', '.join(sorted(set(d['discriminators'])))}")


def cmd_run(args: argparse.Namespace) -> int:
    bundle = load_config(args.config_dir)
    out = None if args.no_write else (args.out or DEFAULT_RUNS_DIR)
    result = run_scenario(args.scenario, mesh=args.mesh, seed=args.seed,
                          bundle=bundle, out_dir=out, record_ticks=args.record_ticks)
    _print_summary(result)
    if result.path:
        print(f"\n  written to {result.path}")
    if args.stragglers:
        print("\n  straggler attribution (top ranks by excess busy time):")
        table = straggler_attribution(result.frames["rank_performance"])
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    bundle = load_config(args.config_dir)
    out = None if args.no_write else (args.out or DEFAULT_RUNS_DIR)
    results = run_matrix(scenarios=args.scenarios, meshes=args.meshes, seed=args.seed,
                         bundle=bundle, out_dir=out)
    df = summaries_frame(results)
    cols = ["run_id", "mesh", "is_fault", "mean_iteration_time_s", "parallel_efficiency",
            "mean_sm_occupancy_pct", "mean_rank_spread_s", "max_temperature_c",
            "diagnosis_verdict", "diagnosis_tier"]
    print("\n" + df[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    correct = ((df["is_fault"] & (df["diagnosis_verdict"] == "HARDWARE_FAULT"))
               | (~df["is_fault"] & (df["diagnosis_verdict"] != "HARDWARE_FAULT"))).sum()
    print(f"\n  diagnosis agreed with ground truth on {correct}/{len(df)} runs")
    return 0


def cmd_mesh_study(args: argparse.Namespace) -> int:
    bundle = load_config(args.config_dir)
    summaries = []
    for mesh in (args.meshes or list(bundle.meshes)):
        r = run_scenario("healthy", mesh=mesh, seed=args.seed, bundle=bundle, out_dir=None)
        summaries.append(r.summary)
    table = mesh_scaling_table(summaries)
    print("\nMesh partitioning vs GPU utilisation (healthy runs only)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n  Compute scales with subdomain VOLUME, halo traffic with SURFACE AREA.")
    print("  Refining the mesh therefore raises occupancy and parallel efficiency and")
    print("  lowers the relative cost of communication -- with no change to the cluster.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    bundle = load_config(args.config_dir)
    print("\nscenarios:")
    for name in bundle.scenario_order:
        sc = bundle.scenarios[name]
        tag = "FAULT" if sc.fault else "not a fault"
        print(f"  {name:18s} [{tag:11s}] tier={sc.tier:9s} {sc.label}")
    print("\nmeshes:")
    for name, m in bundle.meshes.items():
        marker = " (default)" if name == bundle.default_mesh else ""
        print(f"  {name:18s} {m.label}{marker}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from gcsim.dashboard.build import build_dashboard, open_in_browser
    paths, stamp = build_dashboard(runs_dir=args.runs or DEFAULT_RUNS_DIR, out_path=args.out)
    for path in paths:
        print(f"dashboard written to {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Generated: {stamp}  -- hard-refresh the page if the header shows an older stamp")
    if not args.no_open and open_in_browser(paths[-1]):
        print(f"  opened {paths[-1].name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gcsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", type=Path, default=None)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="simulate one scenario")
    r.add_argument("--scenario", required=True)
    r.add_argument("--mesh", default=None)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--no-write", action="store_true")
    r.add_argument("--record-ticks", action="store_true",
                   help="include SAMPLE_TICK rows in the event trace")
    r.add_argument("--stragglers", action="store_true", help="print straggler attribution")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("matrix", help="simulate every scenario on every mesh")
    m.add_argument("--scenarios", nargs="*", default=None)
    m.add_argument("--meshes", nargs="*", default=None)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--no-write", action="store_true")
    m.set_defaults(func=cmd_matrix)

    s = sub.add_parser("mesh-study", help="mesh partitioning vs GPU utilisation")
    s.add_argument("--meshes", nargs="*", default=None)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_mesh_study)

    l = sub.add_parser("list", help="list scenarios and meshes")
    l.set_defaults(func=cmd_list)

    d = sub.add_parser("dashboard", help="build the HTML dashboard from runs/")
    d.add_argument("--runs", type=Path, default=None)
    d.add_argument("--out", type=Path, default=None)
    d.add_argument("--no-open", action="store_true",
                   help="build but do not open the result in a browser")
    d.set_defaults(func=cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

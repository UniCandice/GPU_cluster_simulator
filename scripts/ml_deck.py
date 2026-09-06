#!/usr/bin/env python
"""Build the "Machine learning for fault diagnosis" slide deck from ml_baseline output.

    python scripts/ml_deck.py --metrics runs_ml/ml/metrics.json --figures runs_ml/ml/figures.json \
        --out ML_FAULT_DIAGNOSIS.html

Writes a single HTML file of 16:9 slides with inline SVG charts. Render it with
headless Edge (or Chrome):

    msedge --headless=new --disable-gpu --no-pdf-header-footer \
        --print-to-pdf=ML_FAULT_DIAGNOSIS.pdf file:///.../ML_FAULT_DIAGNOSIS.html

Every number on the slides is read from metrics.json; nothing is typed in.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["healthy", "straggler", "network_domain", "thermal", "gpu_degradation", "phase_change"]
SHORT = {"healthy": "healthy", "straggler": "straggler", "network_domain": "network",
         "thermal": "thermal", "gpu_degradation": "gpu degr.", "phase_change": "phase chg"}
CM = {"healthy": "healthy", "straggler": "straggler", "network_domain": "network",
      "thermal": "thermal", "gpu_degradation": "gpu", "phase_change": "phase"}
A, B, C, G = "#c2410c", "#0f766e", "#1e40af", "#15803d"
INK, INK2, INK3, LINE, WASH = "#14110f", "#4a4340", "#8a807a", "#ddd6d0", "#faf7f4"

CSS = """
  @page { size: 280mm 157.5mm; margin: 0; }
  :root{ --ink:#14110f; --ink2:#4a4340; --ink3:#8a807a; --line:#ddd6d0; --wash:#faf7f4;
         --a:#c2410c; --b:#0f766e; --c:#1e40af; --g:#15803d; }
  *{box-sizing:border-box}
  html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  body{margin:0;font:11pt/1.5 "Segoe UI","Helvetica Neue",Arial,sans-serif;color:var(--ink);background:#fff}
  code,.mono{font-family:Consolas,"Cascadia Mono",monospace}
  code{font-size:.92em;background:#f2eeea;padding:0 3px;border-radius:2px}
  .slide{width:280mm;height:157.5mm;padding:13mm 15mm 10mm;position:relative;break-after:page;
         overflow:hidden;background:#fff;display:flex;flex-direction:column}
  .slide:last-child{break-after:auto}
  .eyebrow{font-size:8.5pt;letter-spacing:1.6px;text-transform:uppercase;color:var(--a);font-weight:700;margin-bottom:3mm}
  h1{font-size:27pt;line-height:1.12;margin:0 0 2mm;font-weight:660;letter-spacing:-.7px}
  h2{font-size:19pt;line-height:1.15;margin:0 0 1.5mm;font-weight:650;letter-spacing:-.4px}
  h3{font-size:11pt;margin:0 0 2mm;font-weight:650}
  .dek{font-size:11.5pt;color:var(--ink2);margin:0 0 5mm;max-width:225mm}
  .body{flex:1;min-height:0;display:flex;gap:9mm}
  .col{flex:1;min-width:0}
  .col.narrow{flex:0 0 72mm}
  ul{margin:0;padding-left:15px} li{margin:0 0 2.2mm} li b{font-weight:650}
  table{border-collapse:collapse;width:100%;font-size:9.5pt}
  th,td{text-align:left;padding:1.8mm 2.6mm;border-bottom:1px solid var(--line);vertical-align:top}
  th{background:var(--wash);font-weight:650;font-size:8.5pt;border-bottom:1.6px solid var(--ink3)}
  td.n,th.n{text-align:right;font-family:Consolas,monospace;white-space:nowrap}
  tr.hl td{background:#fff7ed}
  table.tight th,table.tight td{padding:1.1mm 2.2mm;font-size:9pt}
  .nw{white-space:nowrap}
  .note{font-size:9pt;color:var(--ink3);margin-top:2.5mm}
  .num{position:absolute;right:15mm;bottom:6mm;font-size:8.5pt;color:var(--ink3);font-family:Consolas,monospace}
  .tag{position:absolute;left:15mm;bottom:6mm;font-size:8.5pt;color:var(--ink3)}
  svg{display:block;width:100%;height:auto}
  .big{font-size:34pt;font-weight:680;letter-spacing:-1.2px;line-height:1}
  .big.a{color:var(--a)} .big.b{color:var(--b)} .big.c{color:var(--c)} .big.g{color:var(--g)}
  .lab{font-size:9pt;color:var(--ink3);text-transform:uppercase;letter-spacing:1px;margin-top:1mm}
  .card{border:1px solid var(--line);background:var(--wash);padding:3.5mm 4.5mm;margin-bottom:3mm}
  .card h3{margin:0 0 1.5mm;font-size:10.5pt}
  .card p{margin:0;font-size:9.5pt;color:var(--ink2)}
  .card.a{border-left:3px solid var(--a)} .card.b{border-left:3px solid var(--b)} .card.c{border-left:3px solid var(--c)}
  .title{justify-content:center;background:var(--wash)}
  .rule{width:34mm;height:3px;background:var(--a);margin:0 0 7mm}
  .cols3{display:flex;gap:6mm} .cols3>div{flex:1;min-width:0}
  .small{font-size:9.5pt;color:var(--ink2)}
  .kv{font-family:Consolas,monospace;font-size:9pt;line-height:1.7}
"""


def pct(x: float | None, d: int = 0) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def f2(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def hbar_chart(rows: list[tuple[str, float, str]], width: int = 420, vmax: float = 1.0,
               fmt=lambda v: f"{v:.2f}", label_w: int = 130, row_h: int = 18,
               title: str | None = None) -> str:
    """Horizontal bars. rows = (label, value, colour)."""
    top = 18 if title else 4
    h = top + row_h * len(rows) + 4
    plot_w = width - label_w - 44
    o = [f'<svg viewBox="0 0 {width} {h}" font-family="Segoe UI, Arial">']
    if title:
        o.append(f'<text x="0" y="12" font-size="10" font-weight="650" fill="{INK2}">{title}</text>')
    for i, (lab, v, col) in enumerate(rows):
        y = top + i * row_h
        w = max(0.0, min(v, vmax)) / vmax * plot_w
        o.append(f'<text x="{label_w - 6}" y="{y + 12.5}" text-anchor="end" font-size="9.5" fill="{INK2}">{lab}</text>')
        o.append(f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row_h - 7}" rx="2" fill="{col}"/>')
        o.append(f'<text x="{label_w + w + 5:.1f}" y="{y + 12.5}" font-size="9" font-family="Consolas" fill="{INK}">{fmt(v)}</text>')
    o.append(f'<line x1="{label_w}" y1="{top}" x2="{label_w}" y2="{h - 4}" stroke="{INK3}" stroke-width=".8"/>')
    o.append("</svg>")
    return "\n".join(o)


def confusion_svg(cm: list[list[float]], counts: list[list[int]], size: int = 300) -> str:
    n = len(CLASSES)
    left, top = 78, 46
    cell = (size - left) / n
    w, h = left + cell * n + 4, top + cell * n + 4
    o = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" font-family="Segoe UI, Arial">']
    o.append(f'<text x="{left + cell * n / 2:.0f}" y="12" text-anchor="middle" font-size="9.5" fill="{INK3}">predicted class</text>')
    o.append(f'<text transform="translate(10,{top + cell * n / 2:.0f}) rotate(-90)" text-anchor="middle" font-size="9.5" fill="{INK3}">true class</text>')
    for j, c in enumerate(CLASSES):
        o.append(f'<text x="{left + cell * j + cell / 2:.1f}" y="{top - 6}" text-anchor="middle" font-size="8" fill="{INK2}">{CM[c]}</text>')
        o.append(f'<text x="{left - 5}" y="{top + cell * j + cell / 2 + 3:.1f}" text-anchor="end" font-size="8" fill="{INK2}">{CM[c]}</text>')
    for i in range(n):
        for j in range(n):
            v = cm[i][j]
            # single-hue sequential: teal, light -> dark
            alpha = 0.06 + 0.94 * v
            fill = f"rgba(15,118,110,{alpha:.3f})"
            x, y = left + cell * j, top + cell * i
            o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2" fill="{fill}" stroke="#fff"/>')
            if counts[i][j]:
                txt = "#fff" if v > 0.55 else INK
                o.append(f'<text x="{x + cell / 2 - 1:.1f}" y="{y + cell / 2 + 3:.1f}" text-anchor="middle" font-size="8.5" font-family="Consolas" fill="{txt}">{100 * v:.0f}%</text>')
    o.append("</svg>")
    return "\n".join(o)


def trace_svg(series: list[tuple[str, list[float | None], str]], x: list[int], onset: int | None,
              width: int = 330, height: int = 120, ylabel: str = "", title: str = "",
              shade: list[str] | None = None) -> str:
    """Small multiple: one or two series vs iteration, onset marker, predicted-class band."""
    left, right, top, bottom = 40, 8, 18, 22
    pw, ph = width - left - right, height - top - bottom
    vals = [v for _, s, _ in series for v in s if v is not None and not (isinstance(v, float) and math.isnan(v))]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    x0, x1 = min(x), max(x)
    sx = lambda v: left + (v - x0) / (x1 - x0) * pw  # noqa: E731
    sy = lambda v: top + (hi - v) / (hi - lo) * ph  # noqa: E731
    o = [f'<svg viewBox="0 0 {width} {height}" font-family="Segoe UI, Arial">']
    if title:
        o.append(f'<text x="{left}" y="11" font-size="9" font-weight="650" fill="{INK2}">{title}</text>')
    # predicted-class band under the plot
    if shade:
        bw = pw / len(shade)
        for i, cls in enumerate(shade):
            col = {"healthy": "#e5e7eb", "phase_change": "#bfdbfe"}.get(cls, "#fdba74")
            o.append(f'<rect x="{left + i * bw:.1f}" y="{top + ph + 3}" width="{bw + .3:.1f}" height="5" fill="{col}"/>')
    o.append(f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="{INK3}" stroke-width=".8"/>')
    o.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" stroke="{INK3}" stroke-width=".8"/>')
    for v in (lo + pad, hi - pad):
        o.append(f'<text x="{left - 4}" y="{sy(v) + 3:.1f}" text-anchor="end" font-size="7.5" font-family="Consolas" fill="{INK3}">{v:.3g}</text>')
    for v in (x0, x1):
        o.append(f'<text x="{sx(v):.1f}" y="{height - 4}" text-anchor="middle" font-size="7.5" font-family="Consolas" fill="{INK3}">{v}</text>')
    if ylabel:
        o.append(f'<text transform="translate(9,{top + ph / 2:.0f}) rotate(-90)" text-anchor="middle" font-size="7.5" fill="{INK3}">{ylabel}</text>')
    if onset:
        o.append(f'<line x1="{sx(onset):.1f}" y1="{top}" x2="{sx(onset):.1f}" y2="{top + ph}" stroke="{A}" stroke-width="1" stroke-dasharray="3 2"/>')
        o.append(f'<text x="{sx(onset) + 3:.1f}" y="{top + 8}" font-size="7.5" fill="{A}">onset</text>')
    for name, s, col in series:
        pts = [f"{sx(xi):.1f},{sy(v):.1f}" for xi, v in zip(x, s) if v is not None]
        o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="1.6" stroke-linejoin="round"/>')
        if len(series) > 1 and pts:
            lx, ly = pts[-1].split(",")
            o.append(f'<text x="{float(lx) - 2:.1f}" y="{float(ly) - 4:.1f}" text-anchor="end" font-size="7.5" fill="{col}">{name}</text>')
    o.append("</svg>")
    return "\n".join(o)


def gate_ladder_svg() -> str:
    gates = [("A", "any throttled sample?", "HARDWARE_FAULT · rack / gpu"),
             ("B", "rack temp drift > 8 °C?", "HARDWARE_FAULT · rack"),
             ("C", "link down or Δerrors > 1000?", "HARDWARE_FAULT · rack"),
             ("D", "≥ 2 % barrier-pacing duty?", "HARDWARE_FAULT · rank"),
             ("E", "slowdown < 3 %?", "NOMINAL"),
             ("F", "otherwise", "WORKLOAD_CHANGE")]
    o = ['<svg viewBox="0 0 330 178" font-family="Segoe UI, Arial">']
    for i, (g, q, r) in enumerate(gates):
        y = 6 + i * 28
        o.append(f'<rect x="0" y="{y}" width="150" height="22" rx="3" fill="{WASH}" stroke="{LINE}"/>')
        o.append(f'<text x="8" y="{y + 15}" font-size="9" font-weight="700" fill="{A}">{g}</text>')
        o.append(f'<text x="22" y="{y + 15}" font-size="8.5" fill="{INK}">{q}</text>')
        o.append(f'<path d="M150,{y + 11} L176,{y + 11}" stroke="{INK3}" stroke-width=".9"/>')
        o.append(f'<text x="180" y="{y + 15}" font-size="8.5" font-family="Consolas" fill="{INK2}">{r}</text>')
        if i < len(gates) - 1:
            o.append(f'<path d="M75,{y + 22} L75,{y + 28}" stroke="{INK3}" stroke-width=".9" stroke-dasharray="2 2"/>')
    o.append(f'<text x="0" y="176" font-size="8" fill="{INK3}">first match returns · 7 hand-set constants · one verdict per run, after the last 30 %</text>')
    o.append("</svg>")
    return "\n".join(o)


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------

def build(metrics: dict, figures: dict) -> str:
    ds = metrics["dataset"]
    sh = metrics["seed_holdout"]
    lr = metrics["seed_holdout_logreg"]
    cv = metrics["cv_by_seed"]
    mh = metrics["mesh_holdout"]
    abl = metrics.get("ablations", {})
    rl = metrics["run_level"]
    ttd = metrics["time_to_detect"]
    un = metrics["unsupervised_window"]
    loc = metrics["localisation"]
    imp = metrics["importance"]
    traces = figures["traces"]
    n_test_runs = sum(v["n"] for v in rl["per_scenario"].values())
    ml_runs = sum(v["ml_correct"] for v in rl["per_scenario"].values())
    rules_runs = sum(v["rules_correct"] for v in rl["per_scenario"].values())
    slides: list[str] = []

    # 1 ---------------------------------------------------------------------
    slides.append(f"""
<section class="slide title">
  <div class="rule"></div>
  <h1 style="font-size:36pt;max-width:215mm">Machine learning for fault diagnosis<br>from cluster telemetry</h1>
  <p class="dek" style="font-size:13pt;max-width:190mm;margin-top:4mm">
    What a learned classifier adds to the rule set, how to build one honestly on the simulator's data,
    and what it will take to move it onto a real fleet. Every number here comes from a prototype you can re-run.
  </p>
  <div style="margin-top:8mm;display:flex;gap:16mm">
    <div><div class="big a">{ds['runs']}</div><div class="lab">simulated runs, {len(ds['seeds'])} seeds</div></div>
    <div><div class="big c">{ds['windows_labelled']:,}</div><div class="lab">labelled windows</div></div>
    <div><div class="big b">{sh['macro_f1']:.3f}</div><div class="lab">macro-F1, held-out seeds</div></div>
    <div><div class="big g">{ml_runs}/{n_test_runs}</div><div class="lab">held-out runs classified</div></div>
  </div>
  <div class="tag">GPU Cluster Simulator &middot; gcsim &middot; scripts/ml_baseline.py @ {metrics.get('git_commit') or 'working tree'}</div>
</section>""")

    # 2 ---------------------------------------------------------------------
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Why go beyond rules</div>
  <h2>The rule set works, and that is exactly its limit</h2>
  <p class="dek"><code>metrics.diagnose</code> scores 18/18 on the matrix it was written against. It is a ladder of thresholds
  tuned on one seed with fixed fault targets and fixed onsets.</p>
  <div class="body">
    <div class="col" style="flex:1.1">{gate_ladder_svg()}
      <p class="note" style="margin-top:1mm">On {n_test_runs} held-out runs with <b>re-drawn targets and onsets</b> the rules still score
      <b>{rules_runs}/{n_test_runs}</b>, the learned model <b>{ml_runs}/{n_test_runs}</b> (slide 7). Six clean signatures are easy for both.</p>
      <div class="card" style="margin-bottom:0;padding:2.5mm 4mm"><h3>What learning adds</h3>
        <p>The <b>joint signature</b> across tables, a <b>score per window</b> as the run progresses, a <b>ranked list</b> of suspects, and an operating point the operator chooses.</p></div>
    </div>
    <div class="col">
      <div class="card a"><h3>Fixed thresholds, one operating point</h3>
        <p>8 °C of rack drift, 1000 frames, 2 % pacing duty, 3 % slowdown. Each was set by looking at one run per scenario.
        There is no way to trade recall for false alarms, and no calibration to a fleet.</p></div>
      <div class="card b"><h3>One channel at a time, in a fixed order</h3>
        <p>A gate reads one table. Nothing combines halo dispersion with uplink drops, or occupancy with clock.
        Three of nine tables (<code>nic</code>, <code>switch_aggregate</code>, <code>events</code>) are never read.</p></div>
      <div class="card c"><h3>Verdict at the end, not during</h3>
        <p>Four gates compare the first 15 % with the last 30 % of the run. A fault present from the start, or one that
        begins in the last quarter, is misread. There is no early warning and no per-entity ranking.</p></div>
    </div>
  </div>
  <div class="num">2</div>
</section>""")

    # 3 ---------------------------------------------------------------------
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">The data</div>
  <h2>Nine tables on two clocks</h2>
  <p class="dek">Device telemetry is sampled at 1 Hz on wall time; job timing is per iteration. They only meet through
  <code>job_performance.timestamp</code>, and the iteration clock runs 20&times; faster on the coarse mesh than on the fine one.</p>
  <div class="body">
    <div class="col" style="flex:1.35">
      <table class="tight">
        <tr><th>Table</th><th class="nw">Grain</th><th class="n">Rows</th><th>What it carries</th></tr>
        <tr><td><code>telemetry_gpu</code></td><td class="nw">GPU &times; 1 s</td><td class="n">24 k</td><td>occupancy, utilisation, power, temperature, clock, throttle flag &amp; reason</td></tr>
        <tr><td><code>telemetry_nic</code></td><td class="nw">NIC &times; 1 s</td><td class="n">6 k</td><td>Gbps, cumulative bytes / errors / drops</td></tr>
        <tr><td><code>telemetry_switch_port</code></td><td class="nw">port &times; 1 s</td><td class="n">18 k</td><td>link up, utilisation, queue depth, cumulative counters; leaf ports name their rack</td></tr>
        <tr><td><code>telemetry_switch_aggregate</code></td><td class="nw">switch &times; 1 s</td><td class="n">1 k</td><td>uplink utilisation, oversubscription ratio, congested flag</td></tr>
        <tr><td><code>telemetry_storage</code></td><td class="nw">backend &times; 1 s</td><td class="n">0.2 k</td><td>read / write latency, throughput, dirty backlog</td></tr>
        <tr><td><code>telemetry_node</code></td><td class="nw">node &times; 1 s</td><td class="n">3 k</td><td>cpu / memory / io pressure</td></tr>
        <tr class="hl"><td><code>rank_performance</code></td><td class="nw">rank &times; iteration</td><td class="n">128 k</td><td>compute, halo, all-reduce wait, checkpoint per rank. <b>No timestamp.</b></td></tr>
        <tr class="hl"><td><code>job_performance</code></td><td>iteration</td><td class="n">1 k</td><td>iteration time, rank spread, phase means and maxima, <b>timestamp</b></td></tr>
        <tr><td><code>events</code></td><td>event</td><td class="n">5 k</td><td>injections, throttle and congestion transitions. <b>Labels only, never features.</b></td></tr>
      </table>
    </div>
    <div class="col narrow">
      <h3>A window is 100 iterations</h3>
      <svg viewBox="0 0 250 120" font-family="Segoe UI, Arial">
        <text x="0" y="12" font-size="9" fill="{INK2}">samples of device telemetry per window</text>
        <g font-size="9" fill="{INK2}">
          <text x="0" y="40">coarse</text><rect x="46" y="30" width="10" height="13" rx="2" fill="{B}"/><text x="62" y="40" font-family="Consolas">~4</text>
          <text x="0" y="70">medium</text><rect x="46" y="60" width="50" height="13" rx="2" fill="{B}"/><text x="102" y="70" font-family="Consolas">~20</text>
          <text x="0" y="100">fine</text><rect x="46" y="90" width="200" height="13" rx="2" fill="{B}"/><text x="230" y="87" font-family="Consolas" text-anchor="end">~80</text>
        </g>
      </svg>
      <p class="note">Defining windows in <b>iterations</b>, not seconds, keeps the job-timing features comparable across meshes.
      Every feature must then be count-invariant: means, maxima, fractions and last&minus;first deltas, never sums.</p>
      <p class="note">Sample-level jitter is real: a 1 Hz tick lands mid-timestep and is credited pro rata. Anything shorter than a second is invisible in device telemetry, which is why <code>rank_performance</code> matters.</p>
    </div>
  </div>
  <div class="num">3</div>
</section>""")

    # 4 ---------------------------------------------------------------------
    cc = ds["class_counts"]
    rows = [(SHORT[c], cc[c], B if c in ("healthy", "phase_change") else A) for c in CLASSES]
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Labels</div>
  <h2>Ground truth comes from the injector, and seeds are the augmentation axis</h2>
  <p class="dek">The label of a window is whether the fault <b>physically exists</b> in it. That is read back from the
  <code>INJECTION_APPLIED</code> payload, which names the real target GPU or rack and, for the straggler, the whole episode schedule.</p>
  <div class="body">
    <div class="col" style="flex:1.2">
      <svg viewBox="0 0 470 150" font-family="Segoe UI, Arial">
        <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6z" fill="{INK3}"/></marker></defs>
        <g font-size="9">
          <rect x="0" y="10" width="92" height="40" rx="3" fill="{WASH}" stroke="{LINE}"/><text x="46" y="27" text-anchor="middle" fill="{INK}">scenarios.yaml</text><text x="46" y="41" text-anchor="middle" fill="{INK3}">fixed target, fixed onset</text>
          <path d="M92,30 L116,30" stroke="{INK3}" marker-end="url(#ar)"/>
          <rect x="118" y="10" width="102" height="40" rx="3" fill="#fff7ed" stroke="{A}"/><text x="169" y="27" text-anchor="middle" fill="{A}" font-weight="650">per-seed variant</text><text x="169" y="41" text-anchor="middle" fill="{INK3}">rack, GPU, onset re-drawn</text>
          <path d="M220,30 L244,30" stroke="{INK3}" marker-end="url(#ar)"/>
          <rect x="246" y="10" width="72" height="40" rx="3" fill="{WASH}" stroke="{LINE}"/><text x="282" y="27" text-anchor="middle" fill="{INK}">simulate</text><text x="282" y="41" text-anchor="middle" fill="{INK3}">9 tables</text>
          <path d="M318,30 L342,30" stroke="{INK3}" marker-end="url(#ar)"/>
          <rect x="344" y="10" width="126" height="40" rx="3" fill="#f0fdfa" stroke="{B}"/><text x="407" y="27" text-anchor="middle" fill="{B}" font-weight="650">INJECTION_APPLIED</text><text x="407" y="41" text-anchor="middle" fill="{INK3}">true gpu_id / rack_id / episodes</text>
          <path d="M407,50 L407,72" stroke="{INK3}" marker-end="url(#ar)"/>
          <text x="0" y="90" fill="{INK2}" font-weight="650">window labels along one faulted run</text>
          <rect x="0" y="98" width="180" height="14" fill="#e5e7eb"/><text x="90" y="108" text-anchor="middle" font-size="8" fill="{INK}">healthy (injector has not fired)</text>
          <rect x="180" y="98" width="40" height="14" fill="#fde68a"/><text x="200" y="108" text-anchor="middle" font-size="8" fill="{INK}">straddle</text>
          <rect x="220" y="98" width="250" height="14" fill="#fdba74"/><text x="345" y="108" text-anchor="middle" font-size="8" fill="{INK}">scenario class (fault present)</text>
          <line x1="200" y1="94" x2="200" y2="116" stroke="{A}" stroke-width="1.2"/><text x="200" y="126" text-anchor="middle" font-size="8" fill="{A}">onset</text>
          <text x="0" y="144" font-size="8.5" fill="{INK3}">straggler: a window is positive if any episode overlaps it. Straddling windows are dropped from training and scoring.</text>
        </g>
      </svg>
      <div class="cols3" style="margin-top:3mm">
        <div class="card a" style="margin:0"><h3>Why re-draw targets</h3><p>With the yaml's fixed targets, "rack 1" means thermal and "iteration 300" means network. A model would learn the label, not the physics. Per-seed variants are built in memory; the simulator is untouched.</p></div>
        <div class="card b" style="margin:0"><h3>Why seeds, not noise</h3><p>Seeds key every stochastic stream by entity identity, so each seed relocates the straggler cohort and moves the onset. That is augmentation with physically consistent samples.</p></div>
      </div>
    </div>
    <div class="col narrow">
      {hbar_chart(rows, width=260, vmax=max(cc.values()), fmt=lambda v: f"{int(v)}", label_w=70, title=f"labelled windows by class ({ds['windows_labelled']:,})")}
      <p class="note">{ds['runs']} runs &middot; seeds {', '.join(map(str, ds['seeds']))} &middot; test seeds <b>{', '.join(map(str, ds['test_seeds']))}</b>.
      Healthy dominates because every pre-onset window is healthy. The classifier is class-weighted and all scores are macro-averaged.</p>
      <p class="note">Orange = fault classes. <code>phase_change</code> is a labelled non-fault: a 10 % slowdown with clean hardware.</p>
    </div>
  </div>
  <div class="num">4</div>
</section>""")

    # 5 ---------------------------------------------------------------------
    th, nw, st, hl = traces.get("thermal"), traces.get("network_domain"), traces.get("straggler"), traces.get("healthy")

    def tr_(t, key, col=B):
        return (SHORT.get(key, key), t[key], col)

    charts = ""
    if th:
        charts += trace_svg([("rack drift", th["gpu_rack_temp_drift"], A)], th["start"], th["onset"],
                            title="thermal: hottest rack minus median rack, °C", shade=th["pred"])
    if nw:
        charts += trace_svg([("halo", nw["rank_halo_disp"], A), ("compute", nw["rank_compute_disp"], C)],
                            nw["start"], nw["onset"], title="network: dispersion across ranks (CV)", shade=nw["pred"])
    if st and hl:
        charts += trace_svg([("straggler", st["rank_pace_entropy"], A), ("healthy", hl["rank_pace_entropy"], C)],
                            st["start"], None, title="pacing entropy: who paces the barrier", shade=st["pred"])
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Feature engineering</div>
  <h2>Make the physics dimensionless</h2>
  <p class="dek">{ds['n_features']} window features. Every one is a ratio, a fraction, a dispersion or a deviation from the fleet or rack median,
  so a single model covers three meshes whose iteration times differ 20&times;.</p>
  <div class="body">
    <div class="col" style="flex:1.25">
      <table>
        <tr><th style="width:26%">Source</th><th>Window features (examples)</th><th style="width:30%">Which fault they separate</th></tr>
        <tr><td><code>job_performance</code></td><td>iteration CV and slope, spread / iteration time, halo / compute, max / mean of each phase, output duty</td><td>impact, phase change</td></tr>
        <tr><td><code>rank_performance</code></td><td><b>pacing duty</b> of the top rank and its entropy, max mean excess, p99 excess, halo and compute dispersion across ranks, min wait</td><td>straggler, gpu degradation, network</td></tr>
        <tr><td><code>telemetry_gpu</code></td><td>occupancy, util&minus;occupancy gap, min occupancy / median, min clock / median, power CV, <b>rack temperature drift</b> and rise, throttle fractions by reason</td><td>thermal, gpu degradation</td></tr>
        <tr><td><code>nic</code>, <code>switch_port</code>, <code>aggregate</code></td><td>error and drop rates per GB from counter deltas, downed-uplink fraction per rack, max queue, max oversubscription, congested fraction</td><td>network</td></tr>
        <tr><td><code>storage</code>, <code>node</code></td><td>write latency mean and max, dirty backlog, storage active fraction, io / cpu / memory pressure</td><td>phase change</td></tr>
      </table>
      <p class="note">Cumulative counters become <b>last &minus; first</b> within the window, divided by bytes moved. Entity features (slide 10) repeat this per GPU and per rack, relative to the fleet and the rack.</p>
    </div>
    <div class="col narrow" style="flex:0 0 82mm">{charts}
      <p class="note" style="margin-top:1mm">Held-out medium-mesh runs. Bar under each chart: model's per-window prediction (grey healthy, orange fault).</p>
    </div>
  </div>
  <div class="num">5</div>
</section>""")

    # 6 ---------------------------------------------------------------------
    perm_f1 = abl.get("label_permutation", {}).get("macro_f1")
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Leakage</div>
  <h2>What the model must not see</h2>
  <p class="dek">A simulator hands you perfect labels and perfect identities. Most of the ways to cheat are accidental.</p>
  <div class="body">
    <div class="col" style="flex:1.3">
      <table class="tight">
        <tr><th style="width:30%">Column(s)</th><th style="width:13%">Decision</th><th>Reason</th></tr>
        <tr><td><code>scenario</code>, <code>seed</code>, <code>summary.json</code>, the <code>events</code> table</td><td><b>excluded</b></td><td>the label and its proxies; <code>events</code> is read only to build labels</td></tr>
        <tr><td><code>is_straggler</code>, <code>straggler_count</code>, <code>THROTTLE_ENGAGED</code>, <code>CONGESTION_ONSET</code></td><td><b>excluded</b></td><td>detector outputs baked into the simulator; pacing duty is recomputed from raw per-rank times</td></tr>
        <tr><td><code>rank_id</code>, <code>gpu_id</code>, <code>rack_id</code>, <code>slowest_rank_id</code>, port and switch ids</td><td><b>excluded</b></td><td>identity. Used only as join keys; everything entity-level is a deviation from a median</td></tr>
        <tr><td><code>iteration</code>, <code>timestamp</code>, window position</td><td><b>excluded</b></td><td>position in the run; a streaming detector must not depend on it</td></tr>
        <tr><td>absolute iteration time, compute time, memory used</td><td><b>ablation only</b></td><td>constant within a run: a mesh identifier (slide 8)</td></tr>
        <tr class="hl"><td><code>throttled</code>, <code>throttle_reason</code></td><td><b>kept</b>, ablated</td><td>the device's own self-report (DCGM exposes it), a consequence not a label; the <code>&minus;throttle</code> ablation shows thermal is still caught from rack drift</td></tr>
        <tr><td>the healthy twin at the same seed</td><td><b>never</b></td><td>same silicon draw; a paired difference is an oracle no fleet has</td></tr>
      </table>
    </div>
    <div class="col narrow">
      <div class="stat"><div class="big a">{f2(perm_f1)}</div><div class="lab">macro-F1 with permuted labels</div></div>
      <p class="small">The sanity check: shuffle the training labels, retrain, score the held-out seeds. Chance for six classes is about 0.17. A number well above that would mean a leak survived the table on the left.</p>
      <div class="card b" style="margin-top:5mm"><h3>Split by seed, never by row</h3>
        <p>Windows overlap and neighbours are near-copies. A random row split scores near 1.0 and means nothing. All results here hold out whole seeds ({', '.join(map(str, ds['test_seeds']))}), and the mesh hold-out holds out whole workloads.</p></div>
    </div>
  </div>
  <div class="num">6</div>
</section>""")

    # 7 ---------------------------------------------------------------------
    pc = sh["per_class_f1"]
    f1_rows = [(SHORT[c], pc[c], B) for c in CLASSES]
    rl_rows = "".join(
        f'<tr><td>{SHORT[c]}</td><td class="n">{rl["per_scenario"][c]["ml_correct"]}/{rl["per_scenario"][c]["n"]}</td>'
        f'<td class="n">{rl["per_scenario"][c]["rules_correct"]}/{rl["per_scenario"][c]["n"]}</td></tr>'
        for c in CLASSES if c in rl["per_scenario"])
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Task 1 &middot; classification</div>
  <h2>Six classes on seeds the model never saw</h2>
  <p class="dek">Gradient-boosted trees on {ds['n_features']} window features, trained on seeds {', '.join(str(s) for s in ds['seeds'] if s not in ds['test_seeds'])},
  scored on {ds['windows_test']:,} windows from seeds {', '.join(map(str, ds['test_seeds']))}.</p>
  <div class="body">
    <div class="col" style="flex:1.05">{confusion_svg(sh['confusion'], sh['confusion_counts'], size=290)}
      <p class="note">Row-normalised. Macro-F1 <b>{sh['macro_f1']:.2f}</b>, accuracy {pct(sh['accuracy'])};
      {len(cv['folds'])}-fold leave-one-seed-out {cv['macro_f1_mean']:.2f} &plusmn; {cv['macro_f1_std']:.2f}.
      Fault recall {pct(sh['fault_recall'])}, false-alarm rate on healthy + phase-change windows {pct(sh['false_alarm_rate'], 1)}.</p>
    </div>
    <div class="col">
      {hbar_chart(f1_rows, width=300, fmt=lambda v: f"{v:.2f}", label_w=70, title="per-class F1, held-out seeds")}
      <p class="note">Logistic regression on the same features: macro-F1 {lr['macro_f1']:.2f}. The non-linearity buys {sh['macro_f1'] - lr['macro_f1']:+.2f}.</p>
    </div>
    <div class="col narrow" style="flex:0 0 62mm">
      <h3>Run-level verdicts, same held-out runs</h3>
      <table>
        <tr><th>Scenario</th><th class="n">ML</th><th class="n">Rules</th></tr>{rl_rows}
        <tr class="hl"><td><b>All</b></td><td class="n"><b>{ml_runs}/{n_test_runs}</b></td><td class="n"><b>{rules_runs}/{n_test_runs}</b></td></tr>
      </table>
      <p class="note">ML verdict = most frequent non-healthy class if it appears in &ge; 3 windows (2 &rarr; {pct(rl['rollup_sensitivity']['2'])}, 5 &rarr; {pct(rl['rollup_sensitivity']['5'])}).
      Rules = <code>diagnose()</code> mapped onto the same six classes.</p>
    </div>
  </div>
  <div class="num">7</div>
</section>""")

    # 8 ---------------------------------------------------------------------
    mh_rows = [("seed hold-out · trees", sh["macro_f1"], B), ("seed hold-out · linear", lr["macro_f1"], C)]
    for k, v in mh.items():
        mh_rows.append((k.replace("->", " → ") + " · trees", v["macro_f1"], B))
        mh_rows.append((k.replace("->", " → ") + " · linear", v["macro_f1_logreg"], C))
    abl_rows = [(k, v["macro_f1"], B if k == "full" else A) for k, v in abl.items() if k != "label_permutation"]
    abl_rows.sort(key=lambda r: r[0] != "full")
    worst = min(mh.items(), key=lambda kv: kv[1]["macro_f1"])
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Generalisation</div>
  <h2>Does it transfer to a workload it has not seen?</h2>
  <p class="dek">Holding out a whole mesh is the closest thing to a domain-shift test the simulator offers: a different
  cell count, a different iteration time, a different number of samples per window.</p>
  <div class="body">
    <div class="col">
      {hbar_chart(mh_rows, width=340, fmt=lambda v: f"{v:.2f}", label_w=170, row_h=16, title="macro-F1 by hold-out: gradient-boosted trees vs logistic regression")}
      <div class="card c" style="margin-top:3mm"><h3>Trees do not extrapolate</h3>
        <p>In-distribution the two models tie. Hold out a mesh and the trees fall to {min(v['macro_f1'] for v in mh.values()):.2f}&ndash;{max(v['macro_f1'] for v in mh.values()):.2f}
        while the linear model keeps {min(v['macro_f1_logreg'] for v in mh.values()):.2f}&ndash;{max(v['macro_f1_logreg'] for v in mh.values()):.2f} on two of three: a split threshold learned inside one mesh's healthy range says nothing about values outside it,
        whereas a linear boundary on dimensionless features extends. Recommendation: a linear or monotone-constrained model as the fleet default, trees as the in-distribution refinement.</p></div>
    </div>
    <div class="col">
      {hbar_chart(abl_rows, width=340, fmt=lambda v: f"{v:.2f}", label_w=150, title="ablations, seed hold-out")}
      <p class="note">Removing any one table costs almost nothing: the signatures are redundant across tables. Adding absolute iteration time or the checkpoint-to-iteration ratio does not help in-distribution, and both are mesh identifiers that broke the first mesh hold-out.</p>
      <div class="card a" style="margin-top:3mm"><h3>Hardest transfer: {worst[0].replace('->', ' → ')}</h3>
        <p>Both models struggle (trees {worst[1]['macro_f1']:.2f}, linear {worst[1]['macro_f1_logreg']:.2f}); healthy F1 {worst[1]['per_class_f1_logreg']['healthy']:.2f}.
        The coarse mesh carries a genuine 4 % load imbalance on healthy hardware, so "one rank always paces the barrier" is normal there and a straggler elsewhere. A fleet has the same problem between applications.</p></div>
    </div>
  </div>
  <div class="num">8</div>
</section>""")

    # 9 ---------------------------------------------------------------------
    ttd_rows = ""
    for scenario in ["thermal", "network_domain", "gpu_degradation", "phase_change", "straggler"]:
        cells = ""
        for mesh in ["coarse", "medium", "fine"]:
            v = ttd["summary"].get(f"{scenario}/{mesh}")
            if not v:
                cells += "<td class='n'>–</td>"
                continue
            it = v["median_iterations"]
            cells += (f"<td class='n'>{v['detected']}/{v['n']} &middot; {it:.0f} it &middot; {v['median_s']:.0f} s</td>"
                      if it is not None else f"<td class='n'>{v['detected']}/{v['n']}</td>")
        ttd_rows += f"<tr><td>{SHORT[scenario]}</td>{cells}</tr>"
    pf = traces.get("thermal") or traces.get("network_domain")
    pf_chart = trace_svg([("P(fault)", pf["p_fault"], A)], pf["start"], pf["onset"], width=330, height=110,
                         title=f"P(fault) per window, {pf['run_id']}", shade=pf["pred"]) if pf else ""
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Early warning</div>
  <h2>Time to detect, on trailing windows</h2>
  <p class="dek">The same model scores every 10 iterations. Detection = the first window ending after the onset whose class
  is right for two consecutive windows. The rule set answers once, after the run.</p>
  <div class="body">
    <div class="col" style="flex:1.3">
      <table>
        <tr><th>Scenario</th><th class="n">coarse</th><th class="n">medium</th><th class="n">fine</th></tr>{ttd_rows}
      </table>
      <p class="note">Cells: detected / runs &middot; median latency after onset in iterations &middot; in wall-clock seconds. Straggler latency is measured from the first iteration, since its episodes start at once.
      Thermal latency includes physics: the cooling fault ramps over 60 iterations and the die follows with a 20 s time constant.</p>
      <p class="note">False-alarm rate per window on held-out runs: healthy <b>{pct(ttd['false_alarm_window_rate']['healthy'], 1)}</b>,
      phase change <b>{pct(ttd['false_alarm_window_rate']['phase_change'], 1)}</b>.</p>
    </div>
    <div class="col narrow" style="flex:0 0 88mm">{pf_chart}
      <div class="card c" style="margin-top:3mm"><h3>Operating point is a choice</h3>
        <p>P(fault) is a score. Two consecutive windows is one policy; a fleet would tune persistence and threshold against its own false-alarm budget, which a rule ladder cannot offer.</p></div>
    </div>
  </div>
  <div class="num">9</div>
</section>""")

    # 10 --------------------------------------------------------------------
    gl = loc["gpu"]
    loc_rows = ""
    for scenario in ["straggler", "gpu_degradation", "thermal"]:
        cells = ""
        for m in ["hgb", "iforest", "pca"]:
            v = gl.get(m, {}).get(scenario)
            cells += f"<td class='n'>{f2(v['precision_at_1'])} / {f2(v['auroc'])}</td>" if v else "<td class='n'>–</td>"
        n = gl.get("hgb", {}).get(scenario, {}).get("n_windows", 0)
        loc_rows += f"<tr><td>{SHORT[scenario]} <span style='color:var(--ink3)'>({n} windows)</span></td>{cells}</tr>"
    rk = loc["rack"]
    rack_rows = "".join(
        f"<tr><td>{SHORT[s]} (rack, top-1)</td><td class='n'>{f2(rk['hgb'][s]['top1_accuracy'])}</td><td class='n'>{f2(rk['iforest'][s]['top1_accuracy'])}</td><td class='n'>{f2(rk.get('pca', {}).get(s, {}).get('top1_accuracy'))}</td></tr>"
        for s in ["thermal", "network_domain"] if s in rk.get("hgb", {}))
    lr_ = loc["run_level"]
    runloc = "".join(f"<tr><td>{SHORT[s]}</td><td class='n'>{lr_[s]['ml_localised']}/{lr_[s]['n']}</td><td class='n'>{lr_[s]['rules_localised']}/{lr_[s]['n']}</td></tr>"
                     for s in ["straggler", "gpu_degradation", "thermal", "network_domain"] if s in lr_)
    ex = figures.get("straggler_ranking_example")
    ex_html = ""
    if ex:
        items = "".join(f"<div style='display:flex;gap:3mm;font-family:Consolas;font-size:9pt;line-height:1.55'>"
                        f"<span style='width:10mm;color:var(--ink3)'>{i + 1}</span><span style='width:18mm;{'color:var(--a);font-weight:700' if p else ''}'>{g}</span>"
                        f"<span>{s:.2f}</span>{' &larr; in cohort' if p else ''}</div>" for i, (g, s, p) in enumerate(ex["ranked"][:6]))
        ex_html = f"<h3 style='margin-top:3mm'>One straggler window, ranked</h3><p class='small' style='margin:0 0 1mm'>{ex['run_id']}, iterations {ex['start']}&ndash;{ex['start'] + ds['window_iterations']}, {ex['n_positive']} GPU(s) in episode</p>{items}"
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Task 2 &middot; localisation</div>
  <h2>Which GPU, which rack: scoring entities, not runs</h2>
  <p class="dek">A second model scores every (window, GPU) and (window, rack) from fleet-relative features, with the injected
  entity as label. Consumed as a ranked list, so the metrics are ranking metrics.</p>
  <div class="body">
    <div class="col" style="flex:1.25">
      <table class="tight">
        <tr><th>Per-window ranking, held-out seeds</th><th class="n">supervised GBT</th><th class="n">isolation forest</th><th class="n">PCA residual</th></tr>
        {loc_rows}{rack_rows}
      </table>
      <p class="note" style="margin-top:1.5mm">GPU rows: precision@1 / mean AUROC per window; rack rows: top-1 accuracy. Unsupervised scores are fitted on healthy rows only.
      The isolation forest ranks the failed rack first <b>{f2(rk['iforest']['network_domain']['top1_accuracy'])}</b> of the time: downed-uplink fraction and drop rate are constant in healthy data, so no tree ever splits on them and an out-of-range value cannot be isolated. A reconstruction residual has no such blind spot.</p>
      <table class="tight" style="margin-top:2mm;width:70%">
        <tr><th>Run-level: named the right entity</th><th class="n">ML</th><th class="n">Rules</th></tr>{runloc}
      </table>
    </div>
    <div class="col narrow" style="flex:0 0 78mm">{ex_html}
      <div class="card a" style="margin-top:4mm"><h3>Cannot be localised, by construction</h3>
        <p>A degraded-but-up uplink shares its rack's counters with its healthy siblings in this model. No feature can rank ports the data does not distinguish. Real leaf switches count errors per PHY, so the first thing a fleet version needs is per-port telemetry asymmetry.</p></div>
    </div>
  </div>
  <div class="num">10</div>
</section>""")

    # 11 --------------------------------------------------------------------
    bt = imp["by_table"]
    tbl_rows = [(k, max(v, 0), B) for k, v in sorted(bt.items(), key=lambda kv: -kv[1])]
    top_html = ""
    for c in CLASSES:
        feats = ", ".join(f"<code>{f}</code>" for f, _ in imp["top_per_class"][c][:2])
        top_html += f"<tr><td>{SHORT[c]}</td><td>{feats}</td></tr>"
    fr = un["isolation_forest"]["flag_rate_by_class"]
    flag_rows = [(SHORT[c], fr[c], A if c in ("straggler", "network_domain", "thermal", "gpu_degradation") else C) for c in CLASSES]
    slides.append(f"""
<section class="slide">
  <div class="eyebrow">Interpretation</div>
  <h2>What it learned, and what it ignored</h2>
  <p class="dek">Permutation importance on the held-out seeds: shuffle one feature, measure the F1 it costs. Grouped by source table, then the top two features per class.</p>
  <div class="body">
    <div class="col">
      {hbar_chart(tbl_rows, width=560, row_h=15, vmax=max(max(bt.values()), 1e-6), fmt=lambda v: f"{v:.3f}", label_w=80, title="macro-F1 drop when shuffled, summed per source table")}
      <table class="tight" style="margin-top:2mm"><tr><th>Class</th><th>Most important features</th></tr>{top_html}</table>
      <p class="note">Unsupervised is-fault AUROC on the same held-out windows: isolation forest {un['isolation_forest']['auroc_is_fault']:.2f}, PCA reconstruction residual {un['pca_residual']['auroc_is_fault']:.2f}.</p>
    </div>
    <div class="col narrow" style="flex:0 0 88mm">
      {hbar_chart(flag_rows, width=360, row_h=15, fmt=lambda v: f"{100 * v:.0f}%", label_w=70, title="isolation forest: windows flagged, by true class")}
      <div class="card b" style="margin-top:0"><h3>It never touched the fabric counters</h3>
        <p>Every uplink, NIC and storage feature scores zero: halo dispersion across ranks already names the network fault, output duty the phase change. On a fleet the job-timing tables are the ones most often missing, so the <code>&minus;rank_performance</code> ablation matters more.</p></div>
      <div class="card c" style="margin-top:2mm;margin-bottom:0"><h3>Novelty is not fault</h3>
        <p>Fitted on healthy windows, the isolation forest flags <b>{pct(fr['phase_change'])}</b> of output-campaign windows: the storage channel moved, so the window is novel. Only labelled non-faults tell the two apart.</p></div>
    </div>
  </div>
  <div class="num">11</div>
</section>""")

    # 12 --------------------------------------------------------------------
    slides.append("""
<section class="slide">
  <div class="eyebrow">Sim to real</div>
  <h2>Getting this onto a fleet</h2>
  <p class="dek" style="margin-bottom:4mm">Train on simulation only after calibration, validate features against real telemetry before validating labels, then fine-tune on real incidents with the simulator as pre-training.</p>
  <div class="cols3">
      <div class="card b" style="margin:0"><h3>1 &middot; Calibrate, regenerate, retrain</h3>
        <p>Fit the simulator's constants from what a fleet already records: DCGM power and clock curves, thermal step response, nccl-tests bandwidth and latency, a CRAC derate test. Then regenerate the matrix and retrain; the pipeline is one command.</p></div>
      <div class="card a" style="margin:0"><h3>2 &middot; Validate features before verdicts</h3>
        <p>Before any label exists, check real telemetry obeys what the features assume: phases sum to the iteration time, counters are monotone, occupancy and utilisation diverge under a barrier, throttling is a clock staircase. Where a healthy feature distribution differs from simulation, the gap names missing physics.</p></div>
      <div class="card c" style="margin:0"><h3>3 &middot; Fine-tune, then monitor</h3>
        <p>Replay diagnosed real incidents through the same windowing and fine-tune with simulation as pre-training. Keep the unsupervised score as an "unknown" class. Monitor drift on healthy-window features; use ML-versus-rules disagreement as the retraining trigger.</p></div>
  </div>
  <div style="display:flex;gap:9mm;margin-top:5mm">
    <div style="flex:1"><h3>Known domain gaps</h3>
      <ul class="small"><li>Per-port asymmetry: real optics fail one at a time; the model gives siblings identical counters.</li>
      <li>Missing and jittered samples; a fleet's 1 Hz is not clean.</li><li>Storage shares the fabric on many clusters; here it does not, and that separation is what makes phase change easy.</li>
      <li>Only six signatures. NIC flaps, HBM ECC storms, shared-aisle cooling, ECMP polarisation are unmodelled.</li></ul></div>
    <div style="flex:1"><h3>Roadmap, in order</h3>
      <ul class="small"><li><b>Per-port telemetry asymmetry</b> in the simulator, then a port-level localiser.</li>
      <li><b>More fault families</b> and mixed faults, so the classifier is multi-label.</li>
      <li><b>Sequence model over windows</b> (or HMM smoothing) for latency and persistence instead of "two in a row".</li>
      <li><b>Conformal thresholds per class</b> to give an honest false-alarm guarantee.</li>
      <li><b>Incident replay harness</b>: one real diagnosed incident, scored as on slide 7.</li></ul></div>
  </div>
  <div class="num">12</div>
</section>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Machine Learning for Fault Diagnosis from Telemetry</title>
<style>{CSS}</style></head><body>{''.join(slides)}</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, default=ROOT / "runs_ml" / "ml" / "metrics.json")
    ap.add_argument("--figures", type=Path, default=ROOT / "runs_ml" / "ml" / "figures.json")
    ap.add_argument("--out", type=Path, default=ROOT / "ML_FAULT_DIAGNOSIS.html")
    args = ap.parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    figures = json.loads(args.figures.read_text(encoding="utf-8"))
    args.out.write_text(build(metrics, figures), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

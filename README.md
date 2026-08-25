# GPU Cluster Simulator

A seeded, reproducible simulator of distributed GPU workload performance and reliability. A
bulk-synchronous CFD job runs on a simulated 128-GPU cluster; the simulator models how workload
behaviour, cluster topology, resource contention and infrastructure faults combine to produce
observable telemetry.

**The cluster is flexibly configured.** Everything that defines an experiment lives in YAML, not
code: the cluster shape (racks × nodes × GPUs, link bandwidths, thermal constants) in
`configs/cluster.yaml`, the workload and its mesh resolutions in `configs/workload.yaml`, and the
six fault scenarios in `configs/scenarios.yaml`. The job does not have to fill the cluster: an
optional `allocation` block runs it on any slice — a rank count, a set of racks, or a list of
nodes — with `packed` or `scatter` placement, while unallocated GPUs sit idle and still appear in
telemetry. Change the YAML and the topology, routing, partitioning and telemetry all follow;
nothing is hard-coded to one cluster.

**Get started in two commands** (full details in [Quick start](#quick-start) below):

```bash
python -m pip install -e .
python scripts/run_all.py --seed 42   # 18 runs + HTML dashboard, opens in your browser
```

**What it is for.** Beyond the study shipped in this repo, the simulator is built as a foundation
for fleet work:

- **Validate against real fleet telemetry.** Every model constant is chosen to be fittable from
  measurements a real fleet already produces — DCGM counters, microbenchmarks, thermal step
  responses. The [Calibration](#calibration) section lists, constant by constant, which
  measurement pins it. Once calibrated, the simulator becomes a testable model of the fleet
  rather than a toy.
- **Simulate the fleet for analysis.** With a calibrated model you can ask what-if questions
  offline that are expensive or risky to ask in production: how a placement strategy changes
  collective latency, how a mesh refinement shifts the compute/communication balance, how a
  degraded uplink or a failing cooling loop propagates into job-level slowdown — all seeded and
  byte-reproducible.
- **Create datasets for GPU fleet intelligence.** Each run writes labelled Parquet telemetry:
  realistic multi-table observations plus per-run ground truth (which fault, which tier, which
  GPU or rack). That is exactly the training and benchmark data that anomaly detectors, fault
  classifiers and diagnosis models need but that real fleets rarely have labels for. The
  built-in telemetry-only diagnosis (18/18 against ground truth here) is the rule-based baseline
  any learned model should beat.

**The governing rule:** every telemetry value is *derived from simulated physical state*. No field
is ever written by a scenario. Fault injectors perturb physical state only — link bandwidth,
cooling efficiency, achievable SM throughput, memory bandwidth — and telemetry is an observation of
the consequences. This is enforced structurally (`faults.py` has no import path to any
telemetry-producing module) and asserted by a test.

---

## Quick start

```bash
python -m pip install -e .        # numpy, pandas, pyarrow, pyyaml — no compiler, no GPU
python -m pytest -q               # 124 tests, ~50 s

python scripts/run_all.py --seed 42
```

`run_all.py` simulates all 6 scenarios on all 3 meshes (18 runs, ~30 s total), writes one Parquet
directory per run under `runs/`, prints the comparison tables, and renders
**`dashboard/index.html`** — a single self-contained file with no external dependencies, which
**opens in your browser automatically** when the run finishes. Pass `--no-open` to suppress that
(the file is still written), or just double-click it later.

With runs from more than one seed on disk you get one page per seed —
`dashboard/index_seed7.html`, `index_seed42.html` — and `index.html` becomes a small chooser
listing every seed with its run count. A page never mixes seeds, and each page's header links
to the other seeds and back to the chooser.

Individual pieces:

```bash
python -m gcsim list                                          # scenarios and meshes
python -m gcsim run --scenario thermal --mesh medium --stragglers
python -m gcsim mesh-study                                    # the partitioning study
python -m gcsim matrix --seed 7                               # the full matrix at another seed
python -m gcsim dashboard                                     # rebuild the HTML from runs/
```

`run` and `matrix` also finish by building the dashboard for whatever `runs/` then holds and
opening it -- a single scenario renders fine, with the unsimulated combinations greyed out.
`--no-dashboard` writes the Parquet only; `--no-open` builds without launching a browser. A
custom `--out` runs directory gets a sibling `dashboard/` of its own, never the repo's.

---

## The model

### Cluster

4 racks × 4 nodes × 8 GPUs = **128 GPUs, one rank per GPU**. Two-tier fabric, one leaf switch per
rack.

| Link class | Where | Bandwidth | Latency |
|---|---|---|---|
| intra-node | GPU↔GPU, NVLink-class | 2400 Gbps | 2 µs |
| NIC | node↔leaf, 2 per node | 200 Gbps | 5 µs |
| leaf uplink | leaf↔spine, 8 per leaf | 200 Gbps | 12 µs |

The uplink bundle matches the downlink capacity, so the leaf is **non-blocking** and the host NIC
is the intended bottleneck. That leaves the uplink path real headroom in health — without it,
`congested` and the queue counters would be pinned high in the baseline and would mean nothing when
a fault arrived.

A rack is simultaneously a **network domain** (one leaf) and a **thermal domain** (one CRAC,
shared cold-aisle inlet). That co-location is why rack-scoped faults appear as a *contiguous block*
of 32 ranks rather than a scatter.

**Routing** determines both cost and which counters move:

| Pair | Path | Switch hops |
|---|---|---|
| same node | GPU → GPU | 0 |
| same rack | NIC → leaf → NIC | 1 |
| cross rack | NIC → leaf → spine → leaf → NIC | 3 |

`latency = Σ hop latency`; `effective_bw = min(hop bandwidth)` — the bottleneck, not the mean.

### Workload

Deliberately **not a physics model**. One outer timestep is:

```
HALO_EXCHANGE  →  COMPUTE  →  ALLREDUCE  →  [OUTPUT every 100 steps]
```

preceded by a one-off `DATA_LOAD`. 1000 timesteps. The CFD framing exists to motivate domain
decomposition, which is the actual subject.

The case is a **Taylor–Green vortex** — a cube, periodic on all three axes. That is not decoration:
it is the geometry the partitioner actually implements, so every rank owns exactly six halo faces
and none sits on a wall. A walled case (a lid-driven cavity, or Rayleigh–Bénard with its plates in
z) would give boundary ranks five faces and hand healthy runs a load imbalance that has nothing to
do with partitioning — destroying the control the whole straggler study is measured against.

The barrier relationship holds by construction:

```
iteration_time = max(rank arrival times) + collective + output
wait[r]        = max(arrival) − arrival[r]
```

`wait` is the load-bearing quantity in the whole simulator. A slow rank has `wait ≈ 0` while its
127 peers accumulate wait — **the culprit looks busy and every victim looks idle.** The four phase
columns sum exactly to the timestep for every rank, every step; a test asserts it.

### Mesh partitioning — the centrepiece

A uniform Cartesian box is decomposed onto a 3D process grid, chosen by minimising subdomain
surface area: **8 × 4 × 4** for 128 ranks. With x-fastest rank ordering and packed placement this
puts ±x on NVLink, ±y inside the rack, and ±z across racks — so the *largest* halo faces ride the
*fastest* link and only the two smallest ever cross a spine.

Three uniform meshes, same cluster, no fault anywhere:

| Mesh | Dims | Cells/rank | Imbalance | SM occupancy | Parallel efficiency |
|---|---|---|---|---|---|
| coarse | 250³ | 122,070 | **1.0404** | 29.6% | 27.7% |
| medium | 600³ | 1,687,500 | 1.0000 | 79.1% | 78.0% |
| fine | 1000³ | 7,812,500 | 1.0000 | 89.9% | 89.5% |

Two independent effects, both derived rather than tuned:

- **Compute scales with volume, communication with surface area.** Refining the mesh raises
  cells-per-rank faster than halo-face cells, so the compute-to-communication ratio improves.
- **Small subdomains cannot fill the device.** Achieved occupancy follows
  `cells / (cells + occupancy_half_cells)`, so a rank holding 122k cells reaches about a third of
  peak throughput.

`coarse` is chosen so 250 divides *neither* grid extent. The resulting 4% per-rank cell-count
spread is **real load imbalance on perfectly healthy hardware** — the control case for every
straggler scenario.

Note what utilisation does across that table: **flat at ~99.9%** while occupancy runs 30% → 90%.
A fleet dashboard showing only utilisation would report the coarse mesh as fully busy while it
wasted two-thirds of the hardware.

### The causal loop

Everything else in the simulator is a one-way function. This is the only place an output feeds back
into an input:

```
work → occupancy → power → temperature → throttle → clock
          ↑                                           │
          └───────────────────────────────────────────┘
                (applies to the NEXT interval)
```

- `P = P_idle·leakage + (P_max − P_idle)·occupancy·(f/f_base)^2.2`
- `dT/dt = (P·R_th − (T − T_inlet)) / τ`, with `T_inlet` a **rack** property
- `T > 84 °C` (with hysteresis) → clock steps down → compute slows → peers' wait rises

The one-interval lag is why the thermal scenario produces a *ramp* and a clock *staircase* rather
than a step. The loop is self-limiting: throttled GPUs draw less power, which lowers their own rack
inlet.

**Power follows occupancy, not utilisation.** A rank spinning on a barrier keeps a kernel resident,
so `utilization_pct` stays pinned near 100 while occupancy, power and temperature fall. That
divergence between two channels most models treat as one number is the clearest fingerprint of a
synchronisation stall, and it only exists because the two are computed from different quantities.

---

## Scenarios

| Id | Fault? | Physical perturbation |
|---|---|---|
| `healthy` | no | — |
| `straggler` | **yes** (rank) | a cohort of 3 GPUs carries stale processes that wake intermittently from the first timestep, stealing 20–45% of a victim's SM throughput for a few timesteps at a time, then releasing it completely |
| `network_domain` | **yes** (rack) | rack 2's uplink bundle fails to 1 of 8; the survivor carries a 2% frame error rate |
| `thermal` | **yes** (rack) | rack 1 cooling degrades to 30% over 60 timesteps |
| `gpu_degradation` | **yes** (GPU) | one GPU loses 15% memory bandwidth; RAS caps its clock at 90% |
| `phase_change` | **no** | at step 500 the job switches to a full-field output campaign: every 20 steps, 4× the data |

### Signature matrix (medium mesh)

The two halves answer different questions, and a cell does not mean the same thing in each. Above
the divider it is the **direction** a run's late window moved relative to its early one. Below it,
whether the condition is **present** anywhere in the run — no baseline involved. That distinction
turns out to matter more than it sounds like it should; see the `straggler` column.

| Channel | `healthy` | `straggler` | `network_domain` | `thermal` | `gpu_degradation` | `phase_change` |
|---|---|---|---|---|---|---|
| *Early → late shift* | | | | | | |
| Timestep duration | — | — | ▲ | ▲ | ▲ | ▲ |
| **Rank spread** | — | — | **▲** | **▲** | **▲** | — |
| **Compute time** | — | — | — | — | — | — |
| **Halo exchange time** | — | — | **▲** | — | — | — |
| GPU utilisation | — | — | — | — | — | — |
| SM occupancy | — | — | ▼ | ▼ | ▼ | ▼ |
| Board power | — | — | ▼ | ▼ | ▼ | ▼ |
| **One rack drifting thermally** | — | — | — | **▲** | — | — |
| **Storage write latency** | — | — | — | — | — | **▲** |
| **Node io pressure** | **▼** | **▼** | **▼** | **▼** | **▼** | **▲** |
| *Present in the run — no baseline needed* | | | | | | |
| **Ranks pacing the barrier** | — | **▲** | — | **▲** | **▲** | — |
| **Throttling** | — | — | — | **▲** | **▲** | — |
| **Throttle reason** | **none** | **none** | **none** | **THERMAL** | **RELIABILITY** | **none** |
| **Uplink down** | — | — | **▲** | — | — | — |
| **Link errors / drops** | — | — | **▲** | — | — | — |

**The discriminating rows mostly work by staying still.** Every scenario except `healthy` slows the
job, so the top row separates nothing. What separates them is the pattern of what did *not* move:

- `phase_change` is the only slowdown with a **tight rank spread** and clean device counters. It
  costs ~10% throughput — enough to trip any threshold detector — and no hardware channel moves.
- `network_domain` is the only one where **compute time is untouched**: the loss is on the wire.
- `gpu_degradation` is the only device that **reports its own condition**, through
  `throttle_reason = RELIABILITY`. A stolen-SM straggler leaves no fingerprint on any counter.
- **Read the `straggler` column against the divider.** Its entire top half is dashes, and that is
  the point. A shift row is blind to a fault which was *already running during the early window* —
  this scenario's episodes start at the first timestep, as a job inheriting stale processes from
  whatever ran before it would, so both windows move together and the comparison reports **+1.3%**.
  Above the line the run is indistinguishable from a healthy cluster.

  It only appears **below** the line, on **ranks pacing the barrier**. That is also why `diagnose`
  checks who is pacing *before* it looks at how much the job slowed: waiting for the slowdown to
  move would mean never finding this fault at all.

  Split the same run by whether an episode was running and it is unmistakable: **+49%** on
  iteration time and **11×** on rank spread, with the job 13.6% slower overall than healthy. The
  information was never missing; it was averaged away.
- **Intermittency also breaks mean-based attribution.** Each culprit is slow for ~10% of the run,
  so its *mean* excess busy time falls to within touching distance of the fleet's fastest silicon
  — which is genuinely slow, just not faulty. Ranking by the mean therefore promotes healthy
  ranks. Counting the timesteps a rank actually **paced the barrier** separates them exactly, and
  that is what the diagnosis localises on: it recovers all three injected GPUs, and only those,
  from timing telemetry alone, on a run where the headline slowdown is 1.3%.

### Fault severity depends on the workload

The same cooling failure produces three different outcomes:

| Mesh | Rack power | Peak temp | Throttled | Throughput cost |
|---|---|---|---|---|
| coarse | ~9.9 kW | 55 °C | none | +0.6% |
| medium | ~19.6 kW | 89 °C | 32 GPUs | +8.9% |
| fine | ~21 kW | 91 °C | 32 GPUs | +13.8% |

A communication-bound coarse-mesh job draws far less power, so the same failure never reaches the
slowdown threshold. It is still a failed CRAC — the rack has drifted 14 °C from its peers — and the
classifier catches it on that drift rather than waiting for a throttle bit.

---

## Reproducibility

One master seed. Every stochastic stream is keyed by the entity's **stable identity**
(`"gpu:r1n2g5"`), never its index, via `numpy.random.SeedSequence`. Consequences:

- The same seed reproduces byte-identical output across all nine tables.
- Adding a scenario, adding a rack or reordering a loop cannot perturb an unrelated GPU's stream.
- A healthy run and a faulted run are **directly diffable**: they agree exactly up to the injection
  timestep, so every later difference is a consequence of the fault rather than a different roll.

There is exactly one stochastic input in the model — fixed per-GPU manufacturing variation (clock
headroom, leakage) plus a small per-timestep efficiency jitter. Everything else is deterministic
given that and the workload.

To check by hand:

```bash
python scripts/run_all.py --seed 42 && cp -r runs runs_a
python scripts/run_all.py --seed 42 && diff -r runs runs_a
# -> 180 files byte-identical
```

Wall-clock timing is deliberately **not** written to `summary.json`. It measures the machine that
ran the simulation rather than the simulation itself, and persisting it would make two identical
runs differ on disk for the sake of a number the CLI can simply print.

---

## Layout

```
configs/            cluster.yaml, meshes.yaml, workload.yaml, scenarios.yaml
src/gcsim/
  config.py         typed config + the seed-derivation scheme
  topology.py       racks/nodes/GPUs/NICs/switches/ports/channels
  routing.py        route(src, dst) -> path; latency = sum, bandwidth = min
  mesh.py           process grid, block partition, halo geometry, imbalance
  placement.py      rank -> GPU (packed | scatter)
  workload.py       phases and mutable workload state
  models/           compute, power, thermal, network, storage
  faults.py         injectors -- PHYSICAL STATE ONLY, no telemetry imports
  engine/           event queue, trace, main loop
  samplers.py       the six telemetry samplers, on their own 1 Hz clock
  telemetry.py      schemas, invariant declarations, Parquet I/O
  metrics.py        attribution + the rule-based diagnosis
  dashboard/        payload builder + the self-contained HTML template
scripts/run_all.py  the whole study, end to end
tests/              124 tests
```

`TELEMETRY.md` documents all nine streams: schema, causal origin, and what each shows per scenario.

---

## Tests

```bash
python -m pytest -q
```

Beyond the usual unit coverage, the tests that carry the argument:

- **`test_faults_cannot_reach_telemetry`** walks the import graph and asserts `faults.py` has no
  path to `telemetry`, `samplers` or `metrics`. If it fails, a scenario has gained the ability to
  write its own signature and nothing the simulator produces means anything.
- **Counter conservation** — bytes on a leaf's uplink ports equal the halo traffic whose route
  genuinely left that rack, recomputed from the flow table rather than read back from the same
  accounting path. This is the difference between switch telemetry *derived from routed flows* and
  switch telemetry invented alongside them.
- **Phase-sum exactness** — the four phase columns sum to the timestep for all 128 ranks × 1000
  steps, so `wait` is genuinely barrier slack and not a free parameter.
- **Barrier-stall fingerprint** — peers of a straggler show high utilisation with *falling*
  occupancy and power, and the culprit is the one GPU that runs *hotter*.
- **Straggler amplification** — the job grows by the victim's excess over the *previous pacer*, not
  by its own slowdown. Getting this wrong overstates the cost of every straggler.
- **Degradation vs straggler** — asserts the persistent fault owns the barrier in *every* late
  timestep while the episodic one keeps handing it back, and that `throttle_reason` and occupancy
  still separate the two devices.
- **Episodes restore exactly** — a timestep outside every straggler episode is not merely close to
  the healthy run, it is bit-identical to it, so an episode leaves no residue and the fault cannot
  quietly accumulate.
- **The window comparison misses the straggler** — asserted as a negative, so the scenario cannot
  quietly acquire a baseline later and make the count-based detector look unnecessary.
- **`phase_change` is not a fault** — throughput drops while `throttled` is never set, link errors
  stay at the noise floor, and rank spread stays tight.

---

## Assumptions

- **One rank per GPU, single job.** No other job competes for the cluster -- but the job no
  longer has to *be* the cluster: an optional `allocation` block in `workload.yaml` runs it on
  a subset (`n_ranks`, optionally restricted to listed `racks` or `nodes`), with the existing
  `placement` strategy distributing the ranks over that pool. Unallocated GPUs sit visibly
  idle in telemetry -- idle power, near-inlet temperature, ~0% utilisation -- rather than
  disappearing. Targeted faults honour the slice: if a scenario's configured target hosts no
  rank, the injection retargets to a seed-chosen member of the allocated set and records the
  substitution in the event trace, so a fault can never fire into an idle GPU and silently do
  nothing. Absent, the job occupies every GPU, byte-for-byte the historical behaviour.
- The domain is **triply periodic**, so every rank has exactly six neighbours and none is privileged
  by sitting on a wall. Any imbalance therefore comes from partitioning alone.
- `seconds_per_cell_update` represents a full outer timestep *inclusive of all 20 inner
  linear-solver iterations*. The halo is exchanged once per inner iteration, which is what makes
  communication matter at all for an implicit solver.
- Storage traffic uses a **separate storage fabric** and does not contend with halo traffic. This
  one is load-bearing — it is what keeps `phase_change`'s fabric counters flat — so it is written
  up properly under [Known limitations](#known-limitations) rather than left as a bare assumption.
- The job is at **thermal equilibrium** when observed; the cold-start transient is not modelled.
- Constants are plausible order-of-magnitude values for an H100-SXM-class node. They are **not
  calibrated** against real telemetry — see below.

## Known limitations

- **The network model is flow-level and the collective is closed-form** — detailed in
  [the subsection below](#the-network-model-in-detail), because the boundary is load-bearing.
- **Thermal model is a first-order RC per GPU** with no rack airflow recirculation and no
  hot-aisle coupling between racks.
- **No failure/restart path.** Nothing crashes, no checkpoint is ever recovered from, and there is
  no job requeue.
- **Sampled telemetry inherits real-fleet coarseness.** At a 1 s interval against a ~190 ms
  timestep, a sampler averages over several timesteps and cannot resolve the phase structure inside
  them. Sub-interval stalls are genuinely invisible — a limitation of the model *and* of real
  monitoring.
- **Queue depth and utilisation are reported on different bases** (burst high-water mark vs window
  average). They look inconsistent on purpose: a bulk-synchronous job moves its entire halo in a
  burst occupying a few percent of the sample window, so a link can average 4% and still queue
  hard. Averaging the queue away would hide exactly that.
- **Congestion never causes drops in the halo exchange**, because a closed-loop application-paced
  workload cannot overload a link — the flows simply take longer. Drops here come from physical
  frame errors. That is correct for this workload but would not hold for an open-loop one.
- **Byte counters record offered bytes at both ends.** `rx_bytes` equals `tx_bytes` across a link
  even when frames are being dropped, because both sides are credited from the same flow. A real
  switch would show `rx < tx` across a lossy link, with the shortfall visible as drops. The
  simplification is deliberate: it is what makes the counter-conservation invariant hold *exactly*
  rather than approximately, so a violation means a genuine accounting bug and not accumulated
  loss. Drops and errors are still counted separately, so nothing is hidden — they just are not
  subtracted from the byte totals.
- **Storage traffic does not contend with the fabric.** Field output and the one-off mesh load are
  costed by `StorageModel` — bandwidth, queue depth, writeback backlog — and never enter the flow
  solver that carries the halo. The separation is structural, not incidental: `storage.py` imports
  nothing from `network.py`. Real clusters are usually not built this way. Checkpoint traffic
  crosses the same leaf and spine as the halo, so on the medium mesh an 8.6 GB field write (34.6 GB
  once `phase_change` quadruples it) landing while 144 MB per rank of halo is in flight would slow
  both, and the fabric counters would move during an output campaign.

  Here they do not, and that has a visible consequence worth being honest about: `phase_change`
  raises storage latency and node I/O pressure while leaving **every fabric channel flat**, which
  is exactly what makes it separable from `network_domain` in the signature matrix. The
  simplification is therefore load-bearing rather than harmless — it buys a clean discrimination
  that a contended model would blur. The extension is well defined: build the output write as a
  `FlowSet` aimed at the storage targets and pass it to `Fabric.solve` alongside `halo_flows` in
  the same timestep, so the two compete for the same links. That is a deliberate next step, not an
  oversight, and it would need the `phase_change` signature re-derived rather than assumed.
- The **diagnosis is a small rule set**, not a detector anyone should deploy. It exists to make the
  discriminating channels explicit and checkable, and it is scored against ground truth on the
  dashboard (18/18 on the current matrix).

### The network model, in detail

The fabric is an **analytic flow-level model**, not a packet-level simulator. The mesh
decomposition and the halo traffic matrix are modelled faithfully; what is approximated is the
rate math on shared links, and the collective.

#### What *is* modelled per rank

The halo exchange is compiled into one flow per (rank, direction) — 768 concurrent transfers for
a 128-rank job (`build_halo_flows`). Each flow's destination comes from the periodic Cartesian
neighbour map, its size from that rank's actual face area (`face_cells ×
halo_bytes_per_boundary_cell × inner_iterations`), and its path from the rank→GPU placement
through the router (intranode / intra-rack / cross-rack). Contention is computed over the flows
that genuinely share each channel, and each rank completes when all six sends and six receives
finish. Ragged partitions and placement strategy (`packed` vs `scatter`) therefore change the
result for real reasons, not by assertion.

#### Flow-level, not packet-level

Transfers are solved as flows with a closed-form rate, never as packets:

```
share(channel)   = available_capacity / concurrent_flows
bottleneck(flow) = min(share(h) for h in hops(flow))
duration(flow)   = latency(path) + bytes / bottleneck(flow)
```

Queueing and loss are *derived* from utilisation (`queue = ρ²/(1−ρ)`, loss = overflow past the
port buffer) rather than simulated. Consequences:

- No packet-scale dynamics — no microbursts, incast, PFC/pause frames, or TCP/RoCE
  congestion-control behaviour. A burst overflowing a shallow buffer at low mean utilisation
  cannot happen here.
- No intra-timestep ordering. All flows are treated as starting together, so axis-serialised
  halo exchanges or compute/communication overlap are not represented.
- The traffic matrix is static: the `FlowSet` is built once per run and is not part of the solve
  cache key, so adaptive meshing or a time-varying communication pattern would require extending
  that key.

#### Two-pass approximation, not an exact fixed point

Durations → utilisation → loss → goodput → durations is circular. The solver stops after **two
passes** (solve clean, then re-solve at the goodput congestion left) rather than iterating to
convergence. It is also not true max-min fairness: every flow takes `capacity / count` at each
hop, with no progressive filling, so a flow bottlenecked elsewhere still reserves an unused
share on its other hops. Good enough for this symmetric, repeating workload; approximate under
strongly asymmetric traffic.

#### Uplink bundles are aggregated — fault localisation is bundle-level

Leaf uplink members are fused into one logical channel whose capacity is the sum of live members
and whose error rate is their **mean** (`_capacity_and_error`). Per-port telemetry is
reconstructed afterwards by spreading the channel's bytes, drops and errors **evenly** across
live members (`accumulate`) — which is what ECMP does to bytes in aggregate. Therefore:

- **Hard-down port** — localisable from raw `telemetry_switch_port` (`link_up = False`,
  `capacity_gbps = 0`, counters flatline). Note that `diagnose()` reports only the affected
  *domain* and a count of down ports; it does not name the port IDs in `localisation`.
- **Degraded-but-up port** — *not* localisable by construction. One marginal optic among N
  survivors produces identical counters on every live member. Real switches count errors per
  receiving PHY, so a real fleet can single out the bad optic where this model cannot.

No ECMP hash imbalance, no elephant-flow pinning, no per-member asymmetry of any kind. Routing
is static and health-independent (mirroring ECMP over a LAG), so there is no rerouting around
failures; `MAX_HOPS` is capped at 4.

#### The intranode fabric has no internal topology

`intranode:{node}` is a single flat channel with no member ports: it always runs at nominal
rate, never errors, and cannot be degraded by any injection. There is no NVSwitch-vs-cube-mesh
distinction and no per-link NVLink contention, so under `packed` placement the ±x exchanges are
effectively free and fault-free.

#### Collectives are a closed-form cost, not a simulated collective

`allreduce_time_s` receives a rank **count** and a byte count — plus the halo solution, used for
exactly one term below — never the decomposition, placement, or router. It returns a single
scalar:

- `steps = 2·⌈log₂P⌉` (double-binary tree), a function of rank count only.
- Per-step latency is the worst pairwise latency among the GPUs the ranks actually occupy
  (`Router.worst_latency_us`), so a job confined to one node or rack is priced at 2 µs or 10 µs
  per step rather than the 34 µs spine crossing. The shipped 128-GPU job always spans racks, so
  for every scenario in this repository the value is the cross-rack bound. It is still a single
  worst-case figure per step — tree levels are not tiered by distance.
- The ring bandwidth term uses the **nominal** uplink bundle capacity from config, not the live
  degraded capacity, so downing uplinks does not shrink it.

Consequences:

- No NCCL algorithm/protocol selection (ring vs tree vs CollNet; LL / LL128 / Simple), so none
  of the performance cliffs at NCCL's size thresholds appear, and no channel/chunk pipelining is
  modelled.
- **No flows are created for the collective**, so it never contends with halo traffic or with
  itself, and `accumulate` is never called for it — the collective's bytes appear in **zero**
  NIC or switch-port counters. (This is also what keeps the counter-conservation invariant an
  exact statement about halo bytes.)
- **It is not simulated per rank.** Every rank receives the identical `allreduce_s`. All
  per-rank variation in `allreduce_wait_s` comes from the barrier wait (`arrival − busy`), which
  originates in the compute and halo phases.
- The one real coupling: the maximum uplink queueing delay from the halo solution is added to
  each tree step, which is why a fabric fault lengthens the collective and not just the halo
  exchange.

This is defensible because these payloads are small and latency-dominated. Making the collective
topology-aware or per-rank would mean building it as a `FlowSet` and pushing it through
`solve`/`accumulate` like the halo exchange — a structural change, not a parameter tweak.

## Calibration

None of the constants are fitted. How each *would* be, from real fleet telemetry:

| Constant | Fit from |
|---|---|
| `seconds_per_cell_update` | per-timestep wall time vs cells-per-rank across job sizes |
| `occupancy_half_cells` | DCGM `PROF_SM_OCCUPANCY` vs subdomain size on a strong-scaling sweep |
| `kernel_launch_overhead_s` | the intercept of that same sweep as subdomain size → 0 |
| link bandwidths, latencies | point-to-point and collective microbenchmarks (osu, nccl-tests) |
| `R_th`, `τ` | thermal step response — start a job on an idle rack and fit the exponential |
| `coupling_c_per_kw` | inlet temperature vs rack power across the fleet, or a CRAC derate test |
| power model exponent | board power vs clock at fixed occupancy, from DCGM power + clock |
| storage `ρ`, base latency | filesystem queue-depth and latency counters during checkpoint bursts |

The straggler model would be validated against measured `wait_ms` distributions: the claim that a
slow rank shows near-zero wait while its peers accumulate it is directly checkable in any real
profile, and the amplification relationship (job cost = victim's excess over the *previous* pacer)
is a falsifiable prediction.

## Extending it

- **A new scenario** is a YAML block plus one function in `faults.py` decorated with
  `@handler("name")`. Nothing in the engine changes.
- **A new mesh** is a block in `meshes.yaml`.
- **A different cluster shape** is `cluster.yaml`; the process grid, placement and routing all
  follow from it.
- **A new telemetry channel** means a field in `SCHEMAS`, a line in the relevant sampler, and — if
  it is a counter or a bounded gauge — an entry in `CUMULATIVE_COLUMNS` or `BOUNDED_COLUMNS`, which
  makes the invariant tests cover it automatically.
- **Putting storage traffic on the fabric** is the one extension that changes results rather than
  adding to them. Output bytes would become a `FlowSet` solved alongside `halo_flows`, so a field
  write and a halo exchange contend for the same links. Expect `phase_change` to stop being clean
  on the fabric channels, and re-derive its signature column rather than porting the current one
  across. See [Known limitations](#known-limitations).

## Prior art

The placement/proximity and communication-latency modelling is informed by *Dally: a
network-placement sensitive cluster scheduler for deep learning* (Sharma et al., Penn State,
arXiv:2401.16492), which makes the same core argument — that job placement relative to network
topology dominates communication cost — and describes `ArtISt-sim`, a data-driven DDL cluster
simulator built for the same reason this one was.

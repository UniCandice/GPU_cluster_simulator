# Telemetry reference

Every stream the simulator emits: schema, causal origin, and the signature each shows under each
scenario.

**Governing rule.** Every field below is *derived from simulator state*. No field is ever written by
a scenario config. Fault injectors perturb physical state only (link bandwidth, cooling efficiency,
achievable SM throughput, memory bandwidth); telemetry is an observation of the consequences. The
rule is enforced structurally — `faults.py` may import `topology` and `workload` and nothing else —
and asserted by `tests/test_scenarios.py::test_faults_cannot_reach_telemetry`.

---

## 1. Streams at a glance

| Stream | Grain | Cadence | Emitted by |
|---|---|---|---|
| `telemetry_gpu` | per GPU | sample interval (1 s) | GPU state sampler |
| `telemetry_node` | per node | sample interval | node resource sampler |
| `telemetry_nic` | per NIC | sample interval | NIC counter sampler |
| `telemetry_switch_port` | per switch port | sample interval | switch port accounting |
| `telemetry_switch_aggregate` | per switch | sample interval | switch rollup |
| `telemetry_storage` | per storage backend | sample interval | storage queue model |
| `rank_performance` | per rank per timestep | per timestep | event handler |
| `job_performance` | per job per timestep | per timestep | event handler |
| `events` | per event | event-driven | event queue trace |

Every table carries `scenario`, `seed` and `timestamp`, so runs are directly comparable.

**Sampling cadence is deliberately independent of event resolution.** Real fleets poll DCGM at about
1 Hz while a timestep here lasts ~40–800 ms, so a sample is a time-average over several timesteps
and cannot resolve the phase structure inside them. Sample ticks are scheduled on the same event
queue as phase boundaries, so a tick landing mid-timestep genuinely sees a partial timestep. This is
a real limitation of real monitoring, and inheriting it honestly is more useful than pretending the
exporter sees everything.

Counters that are cumulative on real hardware are cumulative here; rates are computed by
differencing against the previous sample, exactly as a collector does.

---

## 2. GPU telemetry

```
scenario, seed, timestamp, gpu_id, node_id, rack_id,
utilization_pct, sm_occupancy_pct,
memory_used_gb, memory_total_gb,
power_w, temperature_c,
clock_mhz, throttled, throttle_reason
```

**Causal origin**

```
workload phase -> resource demand -> occupancy
occupancy + clock -> power (idle + dynamic, clamped to the board cap)
power + cooling   -> temperature   [dT/dt = (P*R_th - (T - T_inlet)) / tau]
temperature (with hysteresis) -> throttled + throttle_reason
throttled -> clock_mhz -> effective performance -> rank compute time
```

`throttle_reason` is an enum — `NONE`, `THERMAL`, `POWER_CAP`, `RELIABILITY` — not a boolean. The
reason is what makes the stream diagnostically useful: it is the only channel separating
`straggler` from `gpu_degradation`.

**Why utilisation and SM occupancy are separate.** A rank spinning at a barrier keeps a kernel
resident, so `utilization_pct` stays pinned near 100 while doing no useful work. `sm_occupancy_pct`
is the *time-weighted* occupancy over the sample window — what a profiler reports — so it tracks
work actually retired and falls. That divergence is the clearest fingerprint of a synchronisation
stall, and it exists only because the two are computed from different quantities. A model deriving
power from utilisation could not produce it.

**Clock quantisation.** Clocks move in 15 MHz steps, so throttling appears as a staircase rather
than a smooth ramp — visually distinctive, and true of real parts.

Constants: `R_th ≈ 0.06 °C/W`, `τ = 20 s`, software slowdown 84 °C with 3 °C hysteresis, hardware
slowdown 90 °C, board cap 700 W, base clock 1755 MHz.

---

## 3. Node telemetry

```
scenario, seed, timestamp, node_id, rack_id,
cpu_pressure, memory_pressure, io_pressure          # normalised 0-1
```

**Causal origin.** Mesh loading and output staging drive `cpu_pressure` and `io_pressure`;
`memory_pressure` follows the filesystem writeback backlog held in host page cache. If outputs
arrive faster than the filesystem drains them, dirty pages accumulate and the pressure keeps
climbing — which is how a legitimate output campaign shows up here while every GPU channel stays
flat.

---

## 4. NIC telemetry

```
scenario, seed, timestamp, node_id, rack_id, nic_id, capacity_gbps,
tx_gbps, rx_gbps,                     # sample-interval derived
tx_bytes, rx_bytes,                   # cumulative, monotonic
tx_errors, rx_errors,                 # cumulative, monotonic
tx_drops, rx_drops                    # cumulative, monotonic
```

NICs are full duplex. Counters accumulate from flows that genuinely traversed the NIC, obtained from
`route(src_gpu, dst_gpu)` — so intra-node exchanges, which are the *largest* faces in the
decomposition, never appear here at all. A test asserts that NIC tx bytes equal exactly the halo
traffic whose route left the node.

---

## 5. Switch telemetry

### Per port

```
scenario, seed, timestamp,
switch_id, switch_tier,               # "leaf" | "spine"
domain_id,                            # rack id on leaf ports; null on spine ports
port_id, port_role,                   # "downlink" | "uplink"
peer_id, capacity_gbps, link_up,
tx_gbps, rx_gbps,
tx_bytes, rx_bytes, tx_errors, rx_errors, tx_drops, rx_drops,
utilisation_pct, queue_depth
```

### Per switch

```
scenario, seed, timestamp, switch_id, switch_tier, domain_id,
aggregate_tx_gbps, aggregate_rx_gbps,
uplink_utilisation_pct, oversubscription_ratio, max_queue_depth, congested
```

**Causal origin**

```
concurrent flows on a link -> link load -> queue_depth
queue_depth -> queueing latency (and drops once the buffer overflows)
physical frame errors -> retransmission -> lost goodput
-> effective bandwidth down -> communication duration up
-> rank wait time -> job runtime
```

`congested` and every drop counter are *consequences* of load and link health, never scenario
inputs. `oversubscription_ratio` is itself an observable: when uplinks fail it jumps by exactly the
factor of capacity lost, which localises the fault to a domain with no reference to ground truth.

**Two conventions worth knowing.**

- *Spine ports mirror their leaf partner.* Each cable is accounted once, on its leaf side; the spine
  side reports the mirror (leaf tx = spine rx). Counting both ends independently would double the
  fabric's apparent traffic and break conservation.
- *Utilisation is a window average; queue depth is the burst high-water mark.* They look
  inconsistent on purpose. A bulk-synchronous job moves its whole halo in a burst occupying a few
  percent of the sample window, so a link can average 4% utilisation and still queue hard.
  Averaging the queue away would hide exactly that. `congested` is therefore scoped to the **uplink**
  path only — a downlink queues deeply during every healthy exchange, because the host NIC is the
  intended bottleneck.

**Fault tier is recoverable from telemetry.** A degraded *leaf uplink* affects only traffic entering
or leaving that one rack; a degraded *spine* would affect cross-domain traffic globally while
leaving every intra-rack exchange untouched. The two are distinguishable in the port counters
without knowing the ground truth.

---

## 6. Storage telemetry

```
scenario, seed, timestamp, backend_id,
read_latency_ms, write_latency_ms, throughput_gbps,
queue_depth, utilisation_pct, dirty_backlog_gb
```

**Causal origin**

```
output demand -> concurrent writes from 16 nodes -> queue_depth
-> latency rising super-linearly as rho -> 1        [M/M/1-style base/(1-rho)]
-> output duration -> job timestep time
and: each write leaves a writeback backlog that drains at a bounded rate,
     so if outputs outpace the drain the BASELINE between them rises too
```

Field output is a **legitimate phase, not a fault**. Storage latency spikes every 100 timesteps in
the healthy baseline by design — a detector calibrated on a flat baseline would fire on it.

The backlog mechanism is what makes `phase_change` a *sustained* rise rather than a train of
isolated spikes.

---

## 7. Performance tables

### `rank_performance` — per rank per timestep

```
scenario, seed, iteration, rank_id, gpu_id,
compute_time_s, halo_wait_s, allreduce_wait_s, checkpoint_time_s,
total_time_s, is_straggler
```

The four phase columns sum **exactly** to `total_time_s` for every rank at every timestep, and
`total_time_s` is identical across ranks because the job is synchronised. Both are asserted. If they
did not hold, `wait` would be a free parameter rather than the barrier slack it is supposed to be,
and the whole attribution story would rest on nothing.

### `job_performance` — per timestep

```
scenario, seed, iteration, timestamp, iteration_time_s,
compute_max_s, compute_mean_s, halo_max_s, halo_mean_s,
allreduce_s, checkpoint_s, slowest_rank_id, fastest_rank_id,
rank_spread_s, sync_overhead_s, wait_total_s, straggler_count,
throughput_iters_per_s, cumulative_runtime_s
```

The barrier relationship holds by construction:

```
iteration_time = max(rank completion times) + collective + output
```

`rank_spread_s` is the headline diagnostic — tight under healthy and workload-change runs, wide
under any fault hitting a subset of ranks.

### `events`

```
scenario, seed, timestamp, event_type, rank_id, gpu_id, payload
```

The full event-queue trace: phase boundaries, injections, throttle transitions, straggler changes,
congestion onset. Timestamps are monotonic by construction (the recorder raises otherwise). This is
the audit trail that lets a reviewer walk back from a number on the dashboard to the occurrence that
caused it.

---

## 8. Scenario signature matrix

Read down a column for what a fault does; read across a row for which stream discriminates. Every
cell is derived from telemetry, comparing each run's early window against its late one.

| Signal | `healthy` | `straggler` | `network_domain` | `thermal` | `gpu_degradation` | `phase_change` |
|---|---|---|---|---|---|---|
| Timestep duration | — | ▲ | ▲ | ▲ | ▲ | ▲ |
| **Rank spread** | tight | **wide** | wide | wide | **wide** | **tight** |
| **Victim compute time** | — | ▲ | **unchanged** | ▲ (lagged) | ▲ | uniform |
| **Peer compute time** | — | unchanged | unchanged | unchanged | unchanged | uniform |
| Peer wait time | low | ▲ | ▲ | ▲ | ▲ | low |
| GPU utilisation | steady | **high (spinning)** | high | high | high | uniform |
| SM occupancy | steady | ▼ | ▼ | ▼ | ▼ | ▼ |
| GPU temperature | stable | victim ▲, peers ▼ | normal | **ramps, throttles** | victim ▲ | normal |
| **Rack thermal drift** | — | — | — | **▲ one rack** | — | — |
| **Throttle flag** | false | **false** | **false** | **true** | **true** | **false** |
| **Throttle reason** | NONE | **NONE** | **NONE** | **THERMAL** | **RELIABILITY** | **NONE** |
| Clock | nominal | nominal | nominal | **stepped down** | **capped** | nominal |
| **Cross-domain comm** | baseline | baseline | **▲** | baseline | baseline | baseline |
| **Intra-domain comm** | baseline | baseline | **unchanged** | baseline | baseline | baseline |
| Uplink utilisation | moderate | moderate | **saturated** | moderate | moderate | moderate |
| Switch queue / drops | ~0 | ~0 | **▲** | ~0 | ~0 | ~0 |
| Link errors | noise floor | noise floor | **▲▲** | noise floor | noise floor | noise floor |
| Active uplinks | 8 | 8 | **1** | 8 | 8 | 8 |
| **Storage latency** | periodic spikes | periodic | periodic | periodic | periodic | **sustained ▲** |
| Node io pressure | low | low | low | low | low | **▲** |
| Affected geometry | — | **1 rank** | **32, contiguous** | **32, contiguous** | **1 GPU** | **all 128** |
| *ground truth* `fault` | false | true | true | true | true | **false** |

**The discriminating rows mostly work by staying still.** A naive detector fires on `phase_change`
because throughput moves sharply. What separates it from a genuine fault is the *absence* of
movement everywhere else: no throttling, no link errors, no queue growth, and a rank spread that
never widens. Likewise, what separates a fabric fault from a compute fault is that compute durations
are untouched; and what separates `gpu_degradation` from `straggler` — near-identical on spread,
wait and throughput — is `throttle_reason` plus the occupancy of the pacing rank.

A note on that last pair, because it is the sharpest case. Both victims run at the same reduced
throughput. The difference is *why*: the straggler's SMs are healthy but time-sliced away by another
process, so its occupancy is exactly what its subdomain predicts; the degraded GPU's SMs are stalled
on memory, so it retires less per resident cycle. One mechanism, two observables — and the device
itself reports the second one through the RAS governor.

---

## 9. Invariants asserted in tests

- Utilisation, occupancy and pressure in `[0, 1]`; `memory_used <= memory_total`;
  `power_w <= board cap`; `min_clock <= clock_mhz <= base_clock`; temperature bounded.
- `throttled` and `throttle_reason != NONE` always agree.
- All `*_bytes`, `*_errors`, `*_drops` counters monotonic non-decreasing, per entity.
- No nulls outside the declared nullable columns (`domain_id` on spine ports; `rank_id`/`gpu_id` on
  job-wide events) — and those are null *only* where they should be.
- Phase times sum exactly to the timestep, per rank per timestep; `total_time_s` identical across
  ranks; every phase time non-negative.
- Event timestamps monotonic; the trace opens with `SIM_START` and closes with `SIM_END`.
- Telemetry timestamps land exactly on the sample interval, and that interval is coarser than the
  timestep.
- One row per entity per sample, for every stream.
- **Counter conservation:** bytes recorded on a leaf's uplink ports equal the halo traffic whose
  route genuinely left that rack, recomputed from the flow table rather than read back from the same
  accounting path. Globally, cross-rack tx equals cross-rack rx. Spine counters equal their leaf
  partners'. Intra-node traffic appears in no fabric counter at all.
- Same seed reproduces byte-identical output across all nine streams.

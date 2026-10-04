# Diagnosing a pipeline that isn't working

Start with `diagnose_msight_pipeline`. It measures live data flow on every
topic (subscribing for up to ~10s) and walks the node graph — publisher →
topic → subscriber — so it works for any topology, including nodes added with
`add_msight_node`, not just the fixed demo pipeline.

## Reading its output

- `root_causes` — ranked, upstream-most first. Lead with `root_causes[0]`,
  quote its `evidence`, offer its `suggested_fix`.
- `affected` — nodes that are broken *only as a consequence* of a root cause.
  Mention them, but never as causes.
- `nodes` — per-node `health`, with measured `in_rate_hz` / `out_rate_hz`.

Per-node `health`:

| health | meaning |
|---|---|
| `DEAD` | process/container not running |
| `UNREGISTERED` | running, but never appeared in the node registry |
| `STARVED` | running, but nothing arrives on its input topic — an upstream problem |
| `STALLED` | getting input (or it's a source), but publishing nothing — this node itself is broken |
| `OK` | data flowing in and out |

Root-cause kinds also include `MISSING_PUBLISHER` (a node subscribes to a topic
no registered node publishes — its upstream was removed, crashed and
deregistered, or never started) and `TOPIC_MISMATCH` (same, but a published
topic name is a near-match, e.g. a sensor-name typo).

## Things that will mislead you if you reason from raw fields

- **Heartbeat age is not evidence on its own.** msight_core nodes update their
  heartbeat every N *messages processed*, not on a timer, so a node starved of
  input goes stale exactly like a dead one.
- **Failures cascade.** By default a node with no heartbeat for ~15s marks
  itself ERROR and kills itself. Kill the detector and the viewer can die ~15–30s
  later too. The upstream-most failure is the cause; the rest are effects —
  which is why `root_causes` is ordered by graph position, not by which node
  you noticed first.
- **Slow is not stuck.** RF-DETR on some hosts emits one detection every few
  seconds. The diagnosis waits long enough to see one; don't conclude a topic
  is dead from a single quick look at logs.

## What it can't see

- A node that crashed *before* registering and was never started through the
  control plane leaves no trace at all. If the user expected a node that isn't
  in `nodes`, say so — "it isn't registered" is itself the finding.
- A source legitimately publishing slower than about one message per 10s
  (e.g. an idle LiDAR with nothing sending to it) reads as `STALLED`. Check
  whether that source is actually expected to be producing right now.

## After the root cause is known

Use `get_msight_logs` on the root-cause node for the specific error (most
useful for `STALLED`), and fix just that node with `remove_msight_node` +
`add_msight_node` (corrected config) — the healthy nodes don't need touching.
If the root cause is one of the fixed pipeline's nodes, restarting the
pipeline is the simpler fix.

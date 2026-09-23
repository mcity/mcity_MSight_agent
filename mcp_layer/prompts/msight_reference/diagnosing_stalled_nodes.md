# Diagnosing a silently-stalled node

A node can report `status: "RUNNING"` in `get_msight_status` while producing
nothing at all — the process is alive and heartbeating, but no data is
actually flowing through it. There is no topic-level "is data flowing" signal
in this system (MSight_Vision's own topic registry only records that a topic
exists, never when it was last used) — heartbeat and log recency are the only
ground truth available, so use both together:

0. **Before checking any node's health, confirm every stage the symptom
   depends on is actually present in `get_msight_status`'s `services` list
   at all.** A missing stage produces the exact same symptom as a stalled
   one (e.g. "the viewer looks frozen") but with a completely different
   cause and fix — `remove_msight_node` (called directly, or via the
   dashboard, which calls the same tool) leaves no error anywhere; the node
   just silently stops appearing. The nodes upstream and downstream of the
   gap will keep reporting `RUNNING` with perfectly fresh heartbeats, since
   they themselves are fine — only checking their status can make a missing
   middle stage look like a false "everything's healthy" reading. For the
   demo pipeline specifically, `services` should list all three of
   `video_source`, `rfdetr_detector`, and `detection_viewer`; if any one of
   them is absent, that's the entire answer -- `add_msight_node` it back
   rather than investigating heartbeats on the two that are still there.

1. **Check `seconds_since_heartbeat`** (from `get_msight_status`). If this is
   large relative to what it should be, the process itself is dead or
   unresponsive — that's not a "stalled" node, that's a crashed one. Check
   its logs for a crash, then `remove_msight_node` + `add_msight_node` to
   restart it.

2. **If the heartbeat is fresh, check `seconds_since_last_line`**
   (per-service) or `log_freshness` (all-services) from `get_msight_logs`.
   A healthy, actively-processing node logs continuously — RF-DETR detection
   nodes, for example, log one line per frame processed. If the heartbeat is
   fresh but the last log line is old and getting older each time you check,
   the node is alive but stuck — genuinely "alive but not doing its job."

3. **The single most common real cause of an alive-but-stuck node: a
   `publish_topic`/`subscribe_topic` mismatch between two nodes that are
   supposed to be connected.** Pull the full topology with `get_msight_status`
   and check, for the suspect node and whatever should be feeding it, that
   the upstream node's `publish_topic` string is byte-for-byte identical to
   the downstream node's `subscribe_topic` string (a stray sensor-name
   typo, or a sensor_name that changed on one node but not the other, is
   enough — Redis pub/sub doesn't error on a topic nobody's publishing to,
   it just silently delivers nothing).

4. If topics genuinely match and the node is still stuck, check whether
   its actual upstream source is producing anything at all — walk back to
   the chain's source node (usually `video_source`) and apply the same
   heartbeat/log-recency check there. A stalled source (e.g. an RTSP stream
   that dropped) will silently stall everything downstream of it, each of
   which will otherwise look individually healthy.

5. Once you've identified the actually-broken node, fix and restart it with
   `remove_msight_node` + `add_msight_node` (corrected config) rather than
   restarting the whole pipeline — the other nodes don't need to be touched.

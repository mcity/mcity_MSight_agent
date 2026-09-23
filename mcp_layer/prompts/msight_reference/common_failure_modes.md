# Common MSight failure modes

Symptom → likely cause → what to do. These are real error strings this
system actually produces, not guesses.

## "Unknown node_type" for `downloaded_image_player_fps` or `road_user_list_aggregator`
These aren't missing from the catalog by oversight — both are genuinely
broken upstream in `msight_core` today (source files missing from the
installed package; `road_user_list_aggregator`'s node class isn't even
exported). Tell the user plainly these node types aren't available on this
deployment; don't try workarounds.

## A GPU-needing node runs but immediately looks wrong / never on rfdetr_detector specifically after this system's own GPU-absent fallback
If `rfdetr_detector` is added on a host with no GPU, it now automatically
falls back to a locally-built CPU-only image (first time only — a one-time
build, can take several minutes; check `get_msight_logs` if a first
`add_msight_node`/`start_msight_pipeline` call seems to hang). This is
expected, not an error — subsequent starts reuse the cached image and are
fast.

## "MSIGHT_VISION_PATH is not set in .env" / "does not exist or is not a directory"
The agent's `.env` doesn't point at a valid MSight_Vision checkout. Not
something you can fix from inside a conversation — tell the user plainly and
stop; don't retry the same call.

## A node with `needs_gpu` (currently only `rfdetr_detector`) fails to start, or starts and immediately errors
`add_msight_node`/`start_msight_pipeline` only add `--gpus all` when
`nvidia-smi` reports a working GPU — on a host with none, the node still
runs, but against the same GPU-built image (there's no automatic CPU-only
image swap for individually-added nodes). If the container's own logs show
a CUDA/driver error, this is very likely a GPU-less host — tell the user
this node type needs a GPU on this deployment, don't keep retrying with the
same config.

## "docker: conflicting options" / a `docker run` error you don't recognize
Surfaced verbatim from Docker — read it literally rather than guessing.
Two known ones already handled with friendlier messages upstream: a
missing NVIDIA Container Toolkit despite `nvidia-smi` working, and port
6379 already bound by something other than this system's own Redis.

## "'<name>' not found in .../venv/bin -- has MSight_Vision's venv been installed?"
A Supervisor-backed node type (`frame_annotator`, `video_aggregator`,
`video_local_dumper`, `aws_video_pusher`) whose binary isn't installed in
MSight_Vision's own venv. Check `MSight_Vision/install.sh` was actually run
on this host — not something a node config change fixes.

## "'<name>' exited immediately (code N): <log tail>"
A Supervisor-backed node crashed within ~0.5s of starting — almost always a
bad config value (wrong topic name format, missing required flag) rather
than an infrastructure problem. The log tail in the error message usually
names the exact argparse or runtime error; read it before retrying rather
than guessing a different config.

## A node you added doesn't show up / doesn't do what you expected, but `add_msight_node` returned "ok"
`add_msight_node` is idempotent by name — if you call it again for a name
that's already running, it returns success without checking whether the
config you just passed matches what's actually running. If you need to
change a running node's config, `remove_msight_node` it first, then
`add_msight_node` again with the new config — don't assume a second
`add_msight_node` call updates it in place.

## Redis-related errors when adding a node
The control plane auto-starts its own Redis container if nothing's
listening on the configured host/port — but only when that host is
`localhost`/`127.0.0.1`. If `MSIGHT_REDIS_MESSAGE_BROKER_HOST` points
somewhere else and it's unreachable, this surfaces as a plain connection
error — that's a remote broker being down, not something this system can
self-heal.

## Status looks fine but nothing seems to be happening
This is a *symptom*, not a specific error — see
`get_msight_reference(topic="diagnosing_stalled_nodes")` rather than
treating "no error" as "working."

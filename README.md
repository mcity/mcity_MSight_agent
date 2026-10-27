# MSight Agent

A natural-language agent for running and debugging [MSight Vision](https://github.com/michigan-traffic-lab/MSight_Vision), Mcity's roadside camera-perception pipeline (video source → RF-DETR detection → live web viewer). You talk to it in a chat window; it starts, reconfigures, records and troubleshoots the live pipeline for you. An auto-labeling workflow is built in to produce labeled datasets for the detector.

<p align="center">
  <img alt="Ubuntu Version" src="https://img.shields.io/badge/Ubuntu-24.04-blue"/>
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.12-blue"/>
</p>

## What it does

- **Run the pipeline.** A one-click demo on a sample video, or your own video file, folder or RTSP stream. Starting your own source always shows a summary and waits for your confirmation.
- **Camera calibration.** Upload `intrinsics.json` and a pixel-to-world `.npz` map from the chat. Files are validated on upload (keys, shape, image size), and the pipeline refuses to start if the map can't cover the video's frames.
- **Build custom pipelines.** Add or remove any of 29 MSight node types (extra viewers, LiDAR, trackers, sinks) by asking. Input topics and ports are checked before a node starts, and viewers come back with their URL.
- **Record & archive.** Save the annotated video locally or push it to S3.
- **Debug it.** "The viewer is frozen" triggers a live diagnosis: it measures data flow on every topic and walks the publisher → topic → subscriber graph to separate the root cause from knock-on effects. It also reports which nodes run on GPU vs CPU and why, what was added or removed recently, and can run read-only host commands (drivers, ports, disk, Docker). It never changes the pipeline unless you ask.
- **Dashboard.** A web dashboard (port 8003) for status, nodes, logs and recording, alongside the chat.
- **Auto-labeling.** Select a dataset, configure a detection model, run auto-labeling, review in CVAT or Label Studio, and localize detections to lat/lon. See [auto_labeling/README.md](auto_labeling/README.md).

The agent never modifies MSight Vision itself: it drives a separate checkout through Docker, MSight's own launch commands and its Redis message bus.

## How it fits together

```
browser chat (5225) ──► chat server (8001) ──► agent loop ──► MCP server (8000) ──► control plane ──► Docker / host processes
dashboard (8003)    ──► MSight API ─────────────────────────►        ▲ same tools          │
dataset uploads ──► ingest server (8002)                                                   └──► Redis (MSight topics + node registry)
```

| Process | Port | Role |
|---|---|---|
| `mcp_server` | 8000 | All tools (MSight pipeline, nodes, diagnosis, auto-labeling). Owns the control plane, the single source of truth for what's running. |
| `chat_server` | 8001 | Runs the agent: builds the prompt and live state hint, calls the LLM, executes its tool calls, streams replies. Also serves calibration/video uploads and recording downloads. |
| `ingest_server` | 8002 | Dataset uploads for auto-labeling. |
| `msight_api` | 8003 | REST API + built dashboard. Calls the same MCP tools as the chat. |
| `client_chat` | 5225 | Serves the chat web page (without `--ui`: a terminal chat against the same server). |

The LLM is your choice: OpenAI, Anthropic Claude, Google Gemini or Groq.

## Setup

**Prerequisites** (Ubuntu 24.04):

- Python 3.12, `git`, `ffmpeg`, `pciutils` (`sudo apt install python3.12-venv git ffmpeg pciutils`; pciutils lets the agent tell "no GPU" from "GPU present but driver broken")
- Docker Engine with the Compose plugin, and your user in the `docker` group (`sudo usermod -aG docker $USER`, then log out and back in)
- GPU (optional): NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Without a GPU, RF-DETR runs on a CPU-only image automatically (much slower).
- Node.js 22.12+ or 24 (only to build the dashboard; Ubuntu's own `nodejs` package is too old). With [nvm](https://github.com/nvm-sh/nvm): `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash`, open a new shell, then `nvm install 24`
- Redis is optional: if nothing is listening on `localhost:6379`, the agent starts a Redis container itself.

**1. This repo:**

```bash
git clone git@github.com:mcity/mcity_MSight_agent.git
cd mcity_MSight_agent
python3.12 -m venv .venv          # 3.12 explicitly: several pinned packages don't support 3.13+
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# fiftyone requires sse-starlette<1 but mcp requires >=1.6.1, so fiftyone is
# installed on its own with --no-deps.
.venv/bin/pip install "fiftyone==1.4.1" --no-deps
```

**2. MSight Vision**, cloned outside this repo:

```bash
git clone https://github.com/michigan-traffic-lab/MSight_Vision.git ~/MSight_Vision
cd ~/MSight_Vision
python3 -m venv venv && venv/bin/pip install -e .   # for host-process nodes (recording, LiDAR, trackers)
cp env_sample .env                                  # its own settings; the agent overrides the source
```

**3. `.env`** in this repo's root:

```bash
LLM_PROVIDER=openai            # openai, claude, gemini or groq
OPENAI_API_KEY=sk-...          # and/or ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY

MSIGHT_VISION_PATH=/home/<you>/MSight_Vision
MSIGHT_DEMO_VIDEO_PATH=/path/to/demo_video.mp4   # optional: enables the one-click demo

# Optional, for auto-labeling and S3 archiving
HF_TOKEN=...                           # Hugging Face (models, datasets)
LS_URL=... LS_TOKEN=...                # Label Studio
CVAT_URL=... CVAT_ACCESS_TOKEN=...     # CVAT
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
```

**4. Dashboard** (one-time build):

```bash
cd msight_agent/frontend/dashboard && npm install && npx quasar build
```

**First pipeline start is slow:** Docker pulls the `michigantrafficlab/msight-vision` image (several GB) and RF-DETR downloads its weights (~500 MB) from Hugging Face into `MSight_Vision/examples/rfdetr/models/`. Later starts take seconds.

## Running

Start each in its own terminal, **from the repo root**, MCP server first:

```bash
.venv/bin/python3 -m msight_agent.mcp_server                                            # 8000
.venv/bin/python3 -m uvicorn msight_agent.chat_server:app --port 8001                   # 8001
.venv/bin/python3 -m uvicorn msight_agent.ingest_server:app --host 0.0.0.0 --port 8002  # 8002
.venv/bin/python3 -m msight_agent.client_chat --ui                                      # 5225, opens the chat
.venv/bin/python3 -m msight_agent.msight_api                                            # 8003, dashboard
```

Then open the chat at http://localhost:5225 and the dashboard at http://localhost:8003. Use `.venv/bin/python3` as shown rather than an activated shell or bare `uvicorn`, so another project's environment can't get picked up.

**Quick check:** in the chat, say "hi" then "run a demo". A viewer link appears; detections should show within a minute (longer on the first run). "Is it actually working?" runs a live diagnosis.

**Common problems:**

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'msight_agent'` | Run from the repo root, with `python -m`. |
| `permission denied ... docker.sock` | Your user isn't in the `docker` group yet (see prerequisites), or you haven't logged back in. |
| `address already in use` on 8000–8003 or 5225 | Another copy of a server is still running; stop it first. |
| Demo says no demo video | `MSIGHT_DEMO_VIDEO_PATH` isn't set, or the file doesn't exist. |
| Dashboard page shows only JSON / 404 | The dashboard hasn't been built (setup step 4). |

## Example session

```
You:   hi
Agent: Welcome to MSight Agent! Would you like to:
       1. Run a demo  2. Run your own pipeline  3. Manage or troubleshoot something directly

You:   run a demo
Agent: [explains the pipeline] The pipeline is running — live detections at http://localhost:9010.

You:   add a second viewer
Agent: 'detection_viewer_2' started on detection/gs_mcity_1. Open it at http://localhost:9011.

You:   [deletes rfdetr_detector from the dashboard] the viewer is stuck
Agent: Root cause: detection_viewer subscribes to 'detection/gs_mcity_1', which was published by
       'rfdetr_detector' — removed 11s ago. Want me to add it back or restart the pipeline?

You:   is it running on CPU or GPU?
Agent: rfdetr_detector runs on the GPU (RTX 4090); the source and viewer are CPU-only.
```

## Deployment

`deployment/` holds the AWS setup: `Docker/` is a small Flask app that launches a remote EC2 sandbox through CloudFormation, and `Full-install/` and `Using-AMI/` are the templates it submits. Those boot an instance and start the same five processes from the repo root. There is no Dockerfile for running the agent itself in a local container yet.

## Development

```bash
.venv/bin/python3 -m pytest msight_agent/tests          # agent unit tests (no services needed)
cd auto_labeling && ../.venv/bin/python3 -m pytest tests  # engine tests
.venv/bin/python3 msight_agent/evals/fault_injection_eval.py --runs 2   # live fault-injection eval (needs the stack running, ~45 min)
```

The eval injects real faults (killed or deleted nodes, topic typos, port conflicts, bad configs) into a running pipeline, asks the agent about the symptom, and scores both the diagnosis and the agent's reply.

## Repository structure

```
.
├── msight_agent/               # The MSight agent
│  ├── mcp_server.py            # MCP tool server (8000)
│  ├── chat_server.py           # Chat backend (8001)
│  ├── ingest_server.py         # Dataset uploads (8002)
│  ├── msight_api.py            # REST API + dashboard (8003)
│  ├── client_chat.py           # Chat page (5225)
│  ├── agent/                   # The chat agent (workflow-independent)
│  │   ├── loop.py              # One user turn: LLM call -> tool calls -> repeat
│  │   ├── guards.py            # In-code turn rules (grounding, consent, measure-first)
│  │   ├── dispatch.py          # Runs a turn's tool calls
│  │   ├── routing.py           # Injection / HardStop / FallThrough, WRITE_TOOLS, helpers
│  │   ├── state.py             # Pydantic session state, WORKFLOW_SPECS, valid tools per step
│  │   ├── prompts.py           # System prompt assembly
│  │   ├── state_hints.py       # Per-turn SESSION_STATE hint
│  │   ├── tool_schema.py       # Tool definitions the LLM sees
│  │   ├── llm_clients.py       # OpenAI / Claude / Gemini / Groq
│  │   ├── workflow_tools.py    # MCP tools: select/switch/reset workflow, datasets
│  │   └── prompts/             # base_prompt.txt, state_hints.txt
│  ├── msight/                  # MSight Vision live pipeline (primary workflow)
│  │   ├── runtime/             # Control plane, executors, diagnosis, node catalog, calibration
│  │   ├── tools/               # MCP tools: pipeline, nodes, record_archive, calibration,
│  │   │                        #   host_inspect, reference
│  │   ├── handlers.py          # Chat-side handlers (consent gate, deferred recording)
│  │   ├── hints.py             # MSight part of the state hint
│  │   ├── routes.py            # Upload/download endpoints
│  │   └── prompts/             # msight_pipeline.txt + reference/*.md
│  ├── auto_labeling/           # Agent side of auto-labeling: tools/, handlers.py, hints.py, prompts/
│  ├── shared/                  # MCP instance, MCP connection, repo paths, host lookup
│  ├── frontend/                # chat/ (web page) and dashboard/ (Quasar app)
│  ├── tests/                   # Unit tests
│  └── evals/                   # Live fault-injection eval
├── auto_labeling/              # Auto-labeling engine the agent runs (see auto_labeling/README.md)
├── upstream/                   # Our MSight node meant to move into MSight Vision (frame annotator)
├── deployment/                 # AWS CloudFormation / launcher
├── archive/                    # Retired data-engine material (cloud scripts, generated docs, Colab notebook)
├── data/                       # Runtime data, gitignored: recordings, uploads, logs, annotation tasks
├── requirements.txt            # Full pinned environment
└── requirements_prod.txt       # Subset installed on AWS instances
```

## Acknowledgements

Built on the [Mcity Data Engine](https://github.com/mcity/mcity_data_engine) (see [auto_labeling/README.md](auto_labeling/README.md) for its background and citation) and [MSight Vision](https://github.com/michigan-traffic-lab/MSight_Vision). Mcity thanks Amazon Web Services (AWS) for the cloud infrastructure behind the deployment.

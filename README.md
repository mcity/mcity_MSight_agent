# MSight Agent

A natural-language agent for running and debugging [MSight Vision](https://github.com/michigan-traffic-lab/MSight_Vision), Mcity's roadside camera-perception pipeline (video source → RF-DETR detection → live web viewer). You talk to it in a chat window; it starts, reconfigures, records and troubleshoots the live pipeline for you. An auto-labeling workflow is built in to produce labeled datasets for the detector.

<p align="center">
  <img alt="Ubuntu Version" src="https://img.shields.io/badge/Ubuntu-24.04-blue"/>
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.12-blue"/>
</p>

## What it does

- **Run the pipeline.** A one-click demo on a sample video, or your own video file, folder or RTSP stream (shown as a summary for you to confirm first).
- **Camera calibration.** Upload your camera's intrinsics and pixel-to-world map from the chat; they're validated against the video before the pipeline starts.
- **Build custom pipelines.** Add or remove any of 25 MSight node types (extra sources, detectors, viewers, LiDAR, trackers, sinks) by asking. Nodes are validated before they start, and viewers come back with their URL.
- **Record & archive.** Save the annotated video locally or push it to S3.
- **Debug it.** Describe a symptom ("the viewer is frozen") and the agent measures live data flow to find the root cause. It can also report GPU/CPU usage and check the host (drivers, ports, disk, Docker), without changing anything unless you ask.
- **Asks before destructive actions.** Stopping the pipeline, removing nodes or resetting the session only happen at your request.
- **Dashboard.** A web dashboard for status, nodes, logs and recording, alongside the chat.
- **Auto-labeling.** Run a detection model on a dataset, review the labels in CVAT or Label Studio, and import them back. See [auto_labeling/README.md](auto_labeling/README.md).

The agent never modifies MSight Vision itself: it drives a separate checkout through Docker, MSight's launch commands and its Redis message bus.

## How it fits together

```
browser chat (5225) ──► chat server (8001) ──► agent loop ──► MCP server (8000) ──► control plane ──► Docker / host processes
dashboard (8003)    ──► MSight API ─────────────────────────►        ▲ same tools          │
dataset uploads ──► ingest server (8002)                                                   └──► Redis (MSight topics + node registry)
```

| Process | Port | Role |
|---|---|---|
| `mcp_server` | 8000 | Every tool the agent and dashboard use; tracks what's running |
| `chat_server` | 8001 | The agent itself, plus file uploads and downloads |
| `ingest_server` | 8002 | Dataset uploads for auto-labeling |
| `msight_api` | 8003 | REST API and dashboard |
| `client_chat` | 5225 | Chat web page (a terminal chat without `--ui`) |

The LLM is your choice: OpenAI, Anthropic Claude, Google Gemini or Groq.

## Setup

**Prerequisites** (Ubuntu 24.04):

- Python 3.12, `git`, `ffmpeg`, `pciutils`: `sudo apt install python3.12-venv git ffmpeg pciutils`
- Docker Engine with the Compose plugin, with your user in the `docker` group (`sudo usermod -aG docker $USER`, then log out and back in)
- Optional GPU: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Without one, detection runs on the CPU (much slower).
- Node.js 22.12+ to build the dashboard (Ubuntu's `nodejs` package is too old; [nvm](https://github.com/nvm-sh/nvm) works)
- Redis is optional: if none is running on `localhost:6379`, the agent starts one in Docker.

**1. This repo**

```bash
git clone git@github.com:mcity/mcity_MSight_agent.git
cd mcity_MSight_agent
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt --no-deps
```

`requirements.txt` pins every package; `--no-deps` skips pip's resolver, which rejects one declared version range the pinned set works fine with.

**2. MSight Vision**, cloned outside this repo

```bash
git clone https://github.com/michigan-traffic-lab/MSight_Vision.git ~/MSight_Vision
cd ~/MSight_Vision
python3 -m venv venv && venv/bin/pip install -e .
cp env_sample .env
```

**3. `.env`** in this repo's root

```bash
LLM_PROVIDER=openai            # openai, claude, gemini or groq
OPENAI_API_KEY=sk-...          # and/or ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY

MSIGHT_VISION_PATH=/home/<you>/MSight_Vision
MSIGHT_DEMO_VIDEO_PATH=/path/to/demo_video.mp4   # optional: enables the demo

# Optional, for auto-labeling and S3 archiving
HF_TOKEN=...
LS_URL=... LS_TOKEN=...                # Label Studio
CVAT_URL=... CVAT_ACCESS_TOKEN=...     # CVAT
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
```

**4. Dashboard** (one-time build)

```bash
cd msight_agent/frontend/dashboard && npm install && npx quasar build
```

The first pipeline start takes a while: Docker pulls the MSight Vision image (several GB) and RF-DETR downloads its weights. Later starts take seconds.

## Running

Start each in its own terminal from the repo root, MCP server first:

```bash
.venv/bin/python3 -m msight_agent.mcp_server                                            # 8000
.venv/bin/python3 -m uvicorn msight_agent.chat_server:app --port 8001                   # 8001
.venv/bin/python3 -m uvicorn msight_agent.ingest_server:app --host 0.0.0.0 --port 8002  # 8002
.venv/bin/python3 -m msight_agent.client_chat --ui                                      # 5225
.venv/bin/python3 -m msight_agent.msight_api                                            # 8003
```

Open the chat at http://localhost:5225 and the dashboard at http://localhost:8003. To check everything works, say "hi" then "run a demo": a viewer link appears, with detections within a minute.

**Common problems**

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'msight_agent'` | Run from the repo root, with `python -m`. |
| `permission denied ... docker.sock` | Add your user to the `docker` group and log back in. |
| `address already in use` on 8000–8003 or 5225 | Another copy of that server is still running. |
| Demo says there's no demo video | `MSIGHT_DEMO_VIDEO_PATH` isn't set, or the file doesn't exist. |
| Dashboard shows only JSON or a 404 | Build the dashboard (setup step 4). |
| Chat can't reach the MSight backend | Start the MCP server; the other servers reconnect on their own. |

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

`deployment/` launches the agent on AWS EC2 through CloudFormation: `Docker/` is a small launcher app, and `Full-install/` and `Using-AMI/` are the templates it submits.

## Development

```bash
.venv/bin/python3 -m pytest msight_agent/tests                                 # agent unit tests
cd auto_labeling && ../.venv/bin/python3 -m pytest tests                         # engine tests
.venv/bin/python3 msight_agent/evals/fault_injection_eval.py                     # live eval: diagnosis
.venv/bin/python3 -m msight_agent.evals.agent_behaviour_eval --provider openai   # live eval: behaviour
```

The live evals need the MCP server, chat server and dashboard API running, and take over the pipeline while they run. The fault-injection eval breaks a running pipeline and scores the agent's diagnosis; the behaviour eval plays multi-turn conversations and scores what actually happened (nodes, topics, files), so it can also compare LLMs.

Every LLM call is logged with its full prompt and response to `data/logs/agent/prompts.jsonl`.

### Contributing

`main` is protected: changes go through pull requests, and the `lint` and `agent-tests` checks must pass.

## Repository structure

```
.
├── msight_agent/          # The agent
│  ├── agent/              # Agent loop, tool dispatch, guards, session state, prompts, LLM clients
│  ├── msight/             # MSight pipeline: control plane, node catalog, diagnosis, tools
│  ├── auto_labeling/      # Agent side of auto-labeling
│  ├── shared/             # MCP connection, paths, shared helpers
│  ├── frontend/           # Chat page and dashboard
│  ├── tests/              # Unit tests
│  ├── evals/              # Live evals
│  └── *_server.py, msight_api.py, client_chat.py   # The five processes
├── auto_labeling/         # Auto-labeling engine (see auto_labeling/README.md)
├── upstream/              # MSight node intended to move into MSight Vision
├── deployment/            # AWS deployment
├── archive/               # Retired data-engine material
├── data/                  # Runtime data (gitignored)
├── requirements.txt       # Full pinned environment
└── requirements_prod.txt  # Subset installed on AWS instances
```

## Acknowledgements

Built on the [Mcity Data Engine](https://github.com/mcity/mcity_data_engine) (see [auto_labeling/README.md](auto_labeling/README.md) for its background and citation) and [MSight Vision](https://github.com/michigan-traffic-lab/MSight_Vision). Mcity thanks Amazon Web Services (AWS) for the cloud infrastructure behind the deployment.

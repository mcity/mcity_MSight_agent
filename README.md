## Acknowledgements
Mcity would like to thank Amazon Web Services (AWS) for their pivotal role in providing the cloud infrastructure on which the Data Engine depends. We couldn’t have done it without their tremendous support!

# Agentic Mcity Data Engine

![mcity_dataengine](https://github.com/user-attachments/assets/4b80c882-7522-4a06-8b15-c4e294b95b56)

<p align="center">
  <img alt="Test Results" src="https://github.com/mcity/mcity_data_engine/actions/workflows/tests_documentation.yml/badge.svg"/>
  <img alt="Test Results for UofM Cluster" src="https://github.com/mcity/mcity_data_engine/actions/workflows/lighthouse_build.yml/badge.svg"/>
  <img alt="Ubuntu Version" src="https://img.shields.io/badge/Ubuntu-24.04-blue"/>
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.12-blue"/>
  <img alt="PyTorch Version" src="https://img.shields.io/badge/PyTorch-2.5-blue"/>
  <img alt="CUDA Version" src="https://img.shields.io/badge/CUDA-12.4-blue"/>
  <img alt="Visitors" src="https://visitor-badge.laobi.icu/badge?page_id=mcity.mcity_data_engine"/>
</p>

<p align='center'>

<a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/mcity/mcity_data_engine/blob/main/fish_eye_8k_colab.ipynb">
<picture>
  <source srcset="https://github.com/user-attachments/assets/26c12ccc-327a-4702-a49d-76bfeb83bc62" width="15%">
  <img alt="Mcity Data Engine Google Colab Demo" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://github.com/mcity/mcity_data_engine/wiki">
<picture>
  <source srcset="https://github.com/user-attachments/assets/e3e1cd10-5195-4db7-9147-11b75e078662" width="15%">
  <img alt="Mcity Data Engine Wiki" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://mcity.github.io/mcity_data_engine/">
<picture>
  <source srcset="https://github.com/user-attachments/assets/b93f0c88-172d-4eed-8dac-3fdb82436f71"
width="15%">
  <img alt="Mcity Data Engine Docs" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://wandb.ai/mcity">
<picture>
  <source srcset="https://github.com/user-attachments/assets/2e54c0ba-26b7-42cf-b33f-903ddfd55ae9" width="15%">
  <img alt="Mcity Data Engine Logs" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://huggingface.co/mcity-data-engine">
<picture>
  <source srcset="https://github.com/user-attachments/assets/5b925a76-d0a2-46ad-8d95-b9296d6a5b46" width="15%">
  <img alt="Mcity Data Engine Models" src="">
</picture>
</a>
</p>

The Agentic MCity Data Engine introduces a conversational AI layer that sits seamlessly on top of the core data engine, enabling natural language interaction with complex computer vision workflows. Built using the Model Context Protocol (MCP), the agent acts as an intelligent orchestrator that guides users through workflow configuration and execution without requiring deep technical knowledge.

This repository (`mcity_MSight_agent`) currently exposes two workflows through that agentic layer:
- **Auto Labeling** — dataset selection, model configuration, and training/inference against the Mcity Data Engine's own FiftyOne-based pipeline, with export to CVAT or Label Studio for review.
- **MSight Pipeline** — natural-language control of [MSight_Vision](https://github.com/michigan-traffic-lab/MSight_Vision), a third-party, open-source roadside camera-perception pipeline (RF-DETR detection, tracking, live web viewer). The agent does not vendor or modify MSight_Vision — it drives a separate, independently-checked-out copy entirely through Docker Compose and MSight_Vision's own CLI entry points. See [MSight Pipeline Workflow](#msight-pipeline-workflow) below.

<div align="center">
  <picture>
    <source srcset="https://github.com/user-attachments/assets/19f326be-6588-457a-92d4-b7ec08f7491b" width="75%">
    <img alt="Agentic Mcity Data Engine Architecture" src="">
  </picture>
  <p><em>Figure 1. The Agentic Mcity Data Engine bridges programmatic and natural-language workflows through an LLM-agnostic MCP layer.</em></p>
</div>


On February 24, 2025, Daniel Bogdoll, a research scholar at Mcity, gave a presentation on the first release of the Mcity Data Engine in Ann Arbor, Michigan. The recording provides insight into the general architecture, its features and ecosystem integrations, and demonstrates successful data curation and model training for improved Vulnerable Road User (VRU) detection:
<div align="center">
      <a href="https://www.youtube.com/watch?v=ciT8YwQCHwo">
         <img src="https://github.com/user-attachments/assets/dcd2cd42-9cc0-4cf0-abab-a4d4ebd14198" style="width:60%;">
      </a>
</div>

## Key Features of the Agentic Implementation:
The Agentic Mcity Data Engine extends the Mcity Data Engine with an LLM-agnostic orchestration layer powered by the Model Context Protocol (MCP).
This layer transforms each workflow — Auto Labeling and MSight Pipeline — into structured, callable tools that can be accessed either through natural-language interaction or programmatic APIs.

<div align="center"> <picture> <source srcset="https://github.com/user-attachments/assets/3a5c751d-a386-4170-bd44-a29783fc92d6" width="80%"> <img alt="Agentic Mcity Data Engine Detailed Architecture" src=""> </picture> <p><em>Figure 2. Agentic Mcity Data Engine architecture – detailed interaction between user, LLM, chat server, MCP tool server.</em></p> </div>

Natural Language Configuration: Configure complex workflows through conversational commands instead of manually editing Python config files. The agent translates natural language requests into correct configuration settings, validates parameters, maintains context across conversation turns, and guides users through multi-step workflow setup with intelligent prompts and error prevention.

### Core Components:

**- User Interface :** A unified entry point for interaction—users can chat via a natural-language web UI or send direct HTTP API requests from the terminal.

**- Chat Server:**  A FastAPI service (port 8001) acting as the bridge between the user, LLM, and backend MCP services.
It maintains multi-turn chat history, handles tool invocations, streams Server-Sent Event (SSE) logs, and supports both web-UI and programmatic clients.

**- LLM Layer (Model-Agnostic):** Connects to OpenAI, Anthropic Claude, Google Gemini, or Groq models.
The LLM interprets user instructions, determines the appropriate workflow tool call, and sends structured requests back to the chat server for execution.

**- MCP Server:** A FastAPI-based backend (port 8000) exposing 30+ tools across both workflows — Auto Labeling (dataset selection, ingestion, model configuration, CVAT/Label Studio export, Voxel51 visualization) and MSight Pipeline (Docker Compose lifecycle control, camera-calibration status, video record & archive) — plus shared infrastructure used by both.

**- Data Ingestion Server:** A dedicated service (port 8002) for uploading and preprocessing datasets.
It supports drag-and-drop ingestion of images, videos, and annotations in COCO, YOLO, or CVAT-XML formats, automatically converting them into FiftyOne-compatible datasets.
This server streams conversion logs and progress via SSE and updates datasets.yaml dynamically to register new datasets for use across workflows.

**- Data Engine Core:** The underlying Mcity Data Engine handling data selection, labeling, training, validation, and visualization.
The agentic layer orchestrates these modules programmatically via MCP instead of relying on static configuration editing.

### MSight Pipeline Workflow

The `msight_pipeline` workflow lets the agent start, stop, and monitor MSight_Vision's live RF-DETR detection pipeline — a separate `redis` + `video_source` + `rfdetr_detector` + `detection_viewer` Docker Compose stack — from a natural-language conversation, without ever touching MSight_Vision's own source:

- **Requires an independent MSight_Vision checkout.** Set `MSIGHT_VISION_PATH` in `.env` to point at a working copy of [MSight_Vision](https://github.com/michigan-traffic-lab/MSight_Vision) (its own `git clone`, with `docker-compose.yml` and a Python venv with MSight's CLI entry points installed). This repo never vendors or edits that checkout — it only invokes `docker compose` against it and shells out to its CLI binaries (`msight_launch_image_to_video_aggregator`, `msight_launch_video_local_dumper`, `msight_launch_aws_video_pusher`) for the record/archive nodes.
- **Demo or custom source:** either point the agent at the developer's own test video/stream (`MSIGHT_DEMO_VIDEO_PATH` in `.env`) for a zero-setup demo, or supply your own `video_input` (file/folder) or `rtsp_url`.
- **Explicit run confirmation:** for a custom source, the first `start_msight_pipeline` call never launches anything — it returns a consent summary (source, calibration status, recording/archiving selection) that the user must confirm before any container actually starts.
- **Camera calibration:** upload your own `intrinsics.json` + calibration `.npz` (built with [camera_calibration2](https://github.com/michigan-traffic-lab/camera_calibration2)) via the web UI's upload button; the agent validates the file formats and reports live calibration status (`default` / `user_calibrated` / `missing` / `partial`) every turn.
- **Record & Archive:** independent, opt-in local video recording and S3 archiving, run as tracked host subprocesses alongside the Docker-based detection pipeline.
- **Friendly error handling:** known Docker/host failure modes (missing NVIDIA Container Toolkit, a host process already bound to Redis's port, a malformed `.env` line) are translated into actionable messages instead of raw `docker compose` output.

## Online Demo: Data Selection with Embeddings

To get a first feel for the Mcity Data Engine, we provide an online demo in a [Google Colab](https://colab.research.google.com/github/mcity/mcity_data_engine/blob/main/fish_eye_8k_colab.ipynb) environment. We will load the [Fisheye8K dataset](https://huggingface.co/datasets/Voxel51/fisheye8k) and demonstrate the Mcity Data Engine workflow [Embedding Selection](https://github.com/mcity/mcity_data_engine/wiki/Workflows#embedding-selection). This workflow leverages a set of models to compute image embeddings which are used to determine both representative and rare samples. The dataset is then visualized in the Voxel51 UI, highlighting how often a sample was picked by the workflow.

Note that most of the Mcity Data Engine workflows require a more powerful GPU, so the possibilities within the Colab environment are limited. Other workflows may not work.

Online demo on Google Colab: [Mcity Data Engine Web Demo](https://colab.research.google.com/github/mcity/mcity_data_engine/blob/main/fish_eye_8k_colab.ipynb
)

## Local Execution

At least one GPU is required for many of the Mcity Data Engine workflows. Check the hardware setups we have tested in the [**Wiki**](https://github.com/mcity/mcity_data_engine/wiki/Environments). To download the repository and install the requirements run:
```
git clone git@github.com:mcity/mcity_MSight_agent.git
cd mcity_MSight_agent

# Use python3.12 explicitly -- plain `python3` may resolve to a newer default
# (3.13/3.14) on your system, which several pinned packages (e.g. numba==0.61.0)
# do not yet support.
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r agent_requirements.txt

# fiftyone requires sse-starlette<1, but mcp (via fastmcp) requires >=1.6.1 --
# no version satisfies both, so fiftyone is installed separately with --no-deps
# rather than forcing the wrong one onto the rest of the stack.
pip install "fiftyone==1.4.1" --no-deps
```

Login with your [Weights and Biases](https://wandb.ai/) and [Hugging Face](https://huggingface.co/) accounts:
```
wandb login
hf auth login
```

### Agentic Implementation Guide:

**Configuration**: Create a .env file with your LLM API key.
```
# Choose your LLM provider (openai, claude, gemini, or groq)
LLM_PROVIDER=openai

# Add your API key
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
# OR
GEMINI_API_KEY=...
# OR
GROQ_API_KEY=...

# Required only for the MSight Pipeline workflow — path to an independent
# MSight_Vision checkout (see "MSight Pipeline Workflow" above).
MSIGHT_VISION_PATH=/path/to/your/MSight_Vision

# Optional — enables a one-click "Demo" source in the MSight Pipeline workflow
# without asking the user for a video path.
MSIGHT_DEMO_VIDEO_PATH=/path/to/a/demo/video/or/folder
```

**Launch the Agentic Interface:**
Start all four components one after the other in separate terminals:
```
# Terminal 1: MCP Tool Server (port 8000)
python mcp_layer/mcp_server.py

# Terminal 2: Chat Server (port 8001)
uvicorn mcp_layer.chat_server:app --port 8001 --reload

# Terminal 3: Data Ingestion Server (port 8002)
uvicorn mcp_layer.ingest_server:app --host 0.0.0.0 --port 8002 --reload

# Terminal 4: Launch Web UI (port 5225)
python mcp_layer/client_chat.py --ui
```

The web interface will open automatically at http://localhost:5225.

**Docker / AWS Deployment:**
`agent_deployment/` does **not** contain a Dockerfile for running the agent itself in a
local container. Instead:
- `agent_deployment/Docker/` is a small Flask app (`App.py` + `cloudFormationManager.py` +
  `redisManager.py`) that provides a web form for launching a *remote* AWS EC2 sandbox
  running this repo via CloudFormation. `docker-compose.yml` there runs that launcher
  app itself (plus a Redis instance it depends on) — it does not run the data engine.
  Copy `agent_deployment/Docker/env_sample` to `agent_deployment/Docker/.env` and fill in
  your AWS credentials before running `docker compose up` in that directory.
- `agent_deployment/Full-install/` and `agent_deployment/Using-AMI/` contain the
  CloudFormation templates (`deploy-agent.yml`, `deploy-agent-AWS-AMI.yml`) that the
  Flask app submits — these boot a plain EC2 instance (or one from a pre-baked AMI) and
  run the same "Launch the Agentic Interface" steps above via cloud-init.

If you're looking to containerize the agent itself for local testing, that setup doesn't
exist yet in this repo.

**Talk to Task: Example Run**
```
User: Hi, I would like to explore the data engine for processing my data.

Agent: Great! I can help you with Auto Labeling — training or running an object
       detection model on your dataset. Which dataset would you like to use?

User: fisheye8k

Agent: Dataset 'fisheye8k' confirmed. Do you have credentials configured for
       CVAT or Label Studio? I detected Label Studio credentials — using
       Label Studio for annotation. How would you like to annotate your
       dataset?
          - Manual Labeling: export your dataset for manual annotation.
          - Auto Generated Labeling: run a detection model to generate
            predictions automatically, then review/correct them.

User: Auto Generated Labeling.

Agent: Here are the available model sources and models: [...]
       Which model would you like to use?

User: rfdetr_2xlarge

Agent: rfdetr_2xlarge (roboflow) has been configured.
       Here are the current hyperparameters: [...]
       Would you like to modify any of these hyperparameters, or are you
       ready to start?

User: Let's go with the defaults.

Agent: Here's a summary of what will be run: [...] Shall I proceed with
       auto-labeling?

User: Yes, proceed.

[Data Engine Server]: Running auto-labeling — live logs are streamed here and to Weights and Biases.

Agent: Auto-labeling completed. Predictions have been exported to Label Studio
       for review. Let me know when you're done and I'll import your labels
       back, or I can launch Voxel51 so you can explore the results now.
```

**MSight Pipeline: Example Run**
```
User: I want to run the MSight camera detection pipeline.

Agent: Welcome to MSight Agent! Would you like to:
          - Demo: see it running immediately with a test video.
          - Run your own pipeline: set up your own camera calibration,
            video/RTSP source, recording & archiving, and optionally
            auto-labeling — in any order.

User: I want to run my own pipeline.

Agent: Here's your checklist — do these in any order, come back anytime:
          1. Camera calibration (currently: default demo calibration)
          2. Video/RTSP source (currently: not set)
          3. Record & Archive (available once a real source is running)
          4. Auto Labeling (optional, if you don't have a labeled dataset yet)
       Which would you like to start with?

User: Use this video file: /data/gs_mcity_1/2026-07-01.mp4

Agent: Here's what will run:
          - Source: /data/gs_mcity_1/2026-07-01.mp4
          - Calibration: default demo calibration (no user upload yet)
          - Recording: off · Archiving: off
       Shall I start the pipeline with these settings?

User: Yes, go ahead.

Agent: MSight_Vision pipeline started. Watch live detections at
       http://localhost:9010. You can also check status, view logs,
       stop it, or restart with a different input — just ask.

User: How's it doing?

Agent: All 4 containers are running (redis, video_source, rfdetr_detector,
       detection_viewer). No errors in the last 200 log lines.
```

### Notebooks and Submodules

To exclude the output of jupyter notebooks from git tracking, add the following lines to your ```.git/config``` :

```
[filter "strip-notebook-output-engine"]
    clean = <your_path>/mcity_MSight_agent/.venv/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout
    smudge = cat
    required = true
```

and those to ```.git/modules/mcity_data_engine_scripts/config```

```
[filter "strip-notebook-output-scripts"]
    clean = <your_path>/mcity_MSight_agent/.venv/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout
    smudge = cat
    required = true
```

In order to keep the submodules updated, add the following lines to the top of your ```.git/hooks/pre-commit```:

```
git submodule update --recursive --remote
git add .gitmodules $(git submodule foreach --quiet 'echo $name')
```

## Repository Structure
```
.
├── main.py                     # Entry point of the framework → Terminal 1
├── session_v51.py              # Script to launch Voxel51 session → Terminal 2
├── workflows/                  # Workflows for the Mcity Data Engine
├── config/                     # Local configuration files
├── utils/                      # General-purpose utility functions
├── cloud/                      # Scripts run in the cloud to pre-process data
├── docs/                       # Documentation generated with `pdoc`
├── tests/                      # Tests using Pytest
├── custom_models/              # External models with containerized environments
├── MSight/                      # Offline dataset geolocation (not the live pipeline — see MSight/README.md)
│  ├── localize_dataset.py      # Fisheye ground-contact + lat/lon localization
│  ├── utils/
│  │   ├── fiftyone_to_msight_det.py  # FiftyOne detections -> MSight detection objects
│  │   └── load_locamaps.py     # Calibration NPZ/intrinsics loading + pixel localizer
│  ├── data/                    # Hardcoded single-camera (Ashley/Huron) calibration files
│  └── install.sh               # Installs msight_base/msight_core into the active venv
├── mcp_layer/                   # Agentic (MCP) layer
│  ├── mcp_server.py            # MCP tool registry (port 8000)
│  ├── chat_server.py           # FastAPI chat endpoint (port 8001) + /msight/upload_calibration
│  ├── ingest_server.py         # File upload & processing (port 8002)
│  ├── client_chat.py           # Web/terminal client (port 5225)
│  ├── chat_pipeline.py         # Tool dispatch, state routing, reply formatting
│  ├── pipeline_handlers/       # Per-workflow handler mixins for ChatPipeline
│  │   ├── auto_labeling.py
│  │   └── msight_pipeline.py
│  ├── pipeline_common.py       # Shared routing/sentinel primitives for the handlers above
│  ├── progress_relay.py        # Relays live docker-compose build output to the active /chat/stream
│  ├── validate_workflow_state.py # Pydantic workflow state machine (both workflows)
│  ├── host_utils.py            # AWS IMDSv2 host/IP resolution for the viewer URL
│  ├── mcptools/                # Tool implementations
│  │   ├── __init__.py
│  │   ├── workflow_selector.py
│  │   ├── auto_labeling.py
│  │   ├── data_ingest.py
│  │   ├── cvat_export.py
│  │   ├── label_studio_export.py
│  │   ├── msight_docker.py     # start/stop/status/logs — docker compose control for MSight_Vision
│  │   ├── msight_record_archive.py # Local recording + S3 archiving as tracked host subprocesses
│  │   ├── msight_calibration_helper.py # Fisheye-intrinsics auto-detect (not yet registered)
│  │   └── v51.py               # Voxel51 integration
│  ├── llm_clients.py           # Multi-LLM support (OpenAI, Claude, Gemini, Groq)
│  ├── tool_schema.py           # Tool definitions exposed to the LLM
│  ├── prompts/                 # System + per-workflow prompt text
│  │   ├── state_hints.txt      # Per-turn SESSION_STATE hint fragments
│  │   └── workflows/
│  │       ├── auto_labeling.txt
│  │       └── msight_pipeline.txt
│  ├── tests/                   # Pytest coverage for state routing/tool filtering
│  └── ui/                      # Web interface assets
│      └── index.html
├── agent_deployment/            # AWS CloudFormation / Docker launcher for the agent
├── mcity_data_engine_scripts/   # Experiment scripts and one-time operations (Mcity internal)
├── .vscode                     # Settings for VS Code IDE
├── .github/workflows/          # GitHub Action workflows
├── .gitignore                  # Files and directories to be ignored by Git
├── .gitattributes              # Rules for handling files like Notebooks during commits
├── .gitmodules                 # Configuration for managing Git submodules
├── .secret                     # Secret tokens (not tracked by Git)
└── requirements.txt            # Python dependencies (pip install -r requirements.txt)
```

## Training

Training runs are logged with [Weights and Biases (WandB)](https://wandb.ai/mcity/mcity-data-engine).

In order to change the standard WandB directory, run

```
echo 'export WANDB_DIR="<your_path>/mcity_MSight_agent/logs"' >> ~/.profile
source ~/.profile
```

## Contribution

Contributions are very welcome! The Mcity Data Engine is a blueprint for data curation and model training and will not support every use case out of the box. Please find instructions on how to contribute here:

- [Contribute new workflow](https://github.com/mcity/mcity_data_engine/wiki/Workflows#how-to-add-a-new-workflow)
- [Contribute new dataset](https://github.com/mcity/mcity_data_engine/wiki/Datasets#how-to-add-a-new-dataset)
- [Contribute new model](https://github.com/mcity/mcity_data_engine/wiki/Models)

Special thanks to these amazing people for contributing to the Mcity Data Engine! 🙌

<a href="https://github.com/mcity/mcity_data_engine/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mcity/mcity_data_engine" />
</a>

## Citation

If you use the Mcity Data Engine in your research, feel free to cite the project:

```bibtex
@article{bogdoll2025mcitydataengine,
  title={Mcity Data Engine},
  author={Bogdoll, Daniel and Anata, Rajanikant Patnaik and Stevens, Gregory},
  journal={GitHub. Note: https://github.com/mcity/mcity_data_engine},
  year={2025}
}
```

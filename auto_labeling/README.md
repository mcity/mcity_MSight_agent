# Auto-labeling engine

The engine behind the MSight agent's auto-labeling workflow. It comes from the [Mcity Data Engine](https://github.com/mcity/mcity_data_engine): FiftyOne-based dataset handling, model training and inference, and export for human review. The agent drives it from the chat; you rarely need to run it by hand.

## How the agent uses it

- The agent's tools (`msight_agent/auto_labeling/tools/`) write the user's choices into `config/config.py` (selected dataset, model source, hyperparameters), then run `python main.py` **from this folder** and stream its logs into the chat.
- The agent also reads `config/datasets.yaml` and `utils/dataset_loader.py` directly, to list and load datasets. `msight_agent/shared/paths.py` puts this folder on `sys.path` for that.
- `session_v51.py` is launched on demand to open a Voxel51 session on a dataset.

## Layout

```
auto_labeling/
├── main.py              # Entry point: runs the workflows selected in config/config.py
├── session_v51.py       # Voxel51 session launcher
├── config/              # config.py (engine settings + runtime selections), datasets.yaml
├── workflows/           # auto_labeling.py, data_ingest.py, keypoint models
├── utils/               # Dataset loading, logging, W&B helpers
├── localization/        # Offline lat/lon localization of detections (see localization/README.md)
├── tests/               # Engine tests (run from this folder: pytest tests)
├── output/              # Trained models, exports, detections (gitignored)
└── logs/                # Engine logs (gitignored)
```

The engine resolves its paths relative to this folder (`config/datasets.yaml`, `output/models/...`), so always run it from here.

## Running it directly

```bash
cd auto_labeling
../.venv/bin/python3 main.py          # runs SELECTED_WORKFLOW from config/config.py
```

Login once for experiment tracking and model hosting:

```bash
wandb login
hf auth login
```

Training runs are logged with [Weights and Biases](https://wandb.ai/mcity/mcity-data-engine). To change the WandB directory: `export WANDB_DIR="<repo>/auto_labeling/logs"`.

## Notebook outputs

To keep Jupyter notebook outputs out of git, add this to `.git/config`:

```
[filter "strip-notebook-output-engine"]
    clean = <your_path>/mcity_MSight_agent/.venv/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout
    smudge = cat
    required = true
```

## Background

The Mcity Data Engine was presented by Daniel Bogdoll in Ann Arbor on February 24, 2025: [recording](https://www.youtube.com/watch?v=ciT8YwQCHwo). It covers the architecture, ecosystem integrations, and data curation and model training for Vulnerable Road User (VRU) detection. More in its [wiki](https://github.com/mcity/mcity_data_engine/wiki), [docs](https://mcity.github.io/mcity_data_engine/) and [models](https://huggingface.co/mcity-data-engine). The original Colab demo is kept in `archive/fish_eye_8k_colab.ipynb`.

## Citation

```bibtex
@article{bogdoll2025mcitydataengine,
  title={Mcity Data Engine},
  author={Bogdoll, Daniel and Anata, Rajanikant Patnaik and Stevens, Gregory},
  journal={GitHub. Note: https://github.com/mcity/mcity_data_engine},
  year={2025}
}
```

"""Dataset ingestion must not leave config.py's SELECTED_DATASET out of step with the
chat's session state (that sent main.py an empty dataset name)."""
import pytest

from msight_agent.agent import state as state_mod
from msight_agent.agent import workflow_tools
from msight_agent.agent.state import config_selected_dataset
from msight_agent.agent.workflow_tools import write_selected_dataset
from msight_agent.auto_labeling.tools import data_ingest

CONFIG = '''SELECTED_WORKFLOW = ["auto_labeling"]
SELECTED_DATASET = {
    "name": "",
    "n_samples": None,
    "custom_view": None,
}

WORKFLOWS = {
    "auto_labeling": {
        "epochs": 15,
    },
    "data_ingest": {
        "dataset_name": 'custom_dataset',
        "annotation_format": 'auto',
        "dataset_dir": '',
        "split_percentages": [0.7, 0.2, 0.1],
        "fps": 2,
    }
}
'''


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.py"
    path.write_text(CONFIG)
    for module in (state_mod, workflow_tools, data_ingest):
        monkeypatch.setattr(module, "CONFIG_PATH", path)
    return path


def _ingest_patch(original: str) -> str:
    patched = data_ingest._set_selected_workflow(original)
    return data_ingest._patch_data_ingest_block(patched, "/data/job_1", 2, [0.7, 0.2, 0.1], "custom_dataset4", "auto")


def test_selection_made_during_ingest_survives_the_restore(config):
    original = config.read_text()
    config.write_text(_ingest_patch(original))
    write_selected_dataset("custom_dataset4")          # user picks the dataset mid-upload
    data_ingest._restore_ingest_patch(original)

    restored = config.read_text()
    assert config_selected_dataset() == "custom_dataset4"
    assert 'SELECTED_WORKFLOW = ["auto_labeling"]' in restored
    assert "\"dataset_dir\": ''" in restored             # ingest's own patch is undone


def test_restore_without_concurrent_changes_is_exact(config):
    original = config.read_text()
    config.write_text(_ingest_patch(original))
    data_ingest._restore_ingest_patch(original)
    assert config.read_text() == original


def test_write_and_read_selected_dataset(config):
    assert config_selected_dataset() == ""
    write_selected_dataset("fisheye8k")
    assert config_selected_dataset() == "fisheye8k"
    assert '"n_samples": None' in config.read_text()

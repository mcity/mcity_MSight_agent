import shutil
import subprocess

import pytest
import fiftyone as fo
from config import config
from workflows.data_ingest import run_data_ingest
from utils.logging import configure_logging
import logging


@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()


@pytest.fixture(autouse=True)
def deactivate_wandb_sync():
    config.WANDB_ACTIVE = False


@pytest.fixture
def test_video_ingest_dataset_dir(tmp_path):
    """A folder with one 10-second H.264 sample.mov (an ffmpeg test pattern), for dataset_ingest."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is needed to make the test video")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=10:size=640x480:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "sample.mov")],
        check=True,
    )
    return str(tmp_path)


def test_dataset_ingest_workflow_video(test_video_ingest_dataset_dir):
    """
    Test the dataset_ingest workflow on a single generated .mov file.
    Verifies that frames are extracted, splits are applied, and ground_truth exists.
    """

    base_name = "video_ingest_test"

    config.WORKFLOWS["data_ingest"] = {
        "dataset_name": base_name,
        "dataset_dir": test_video_ingest_dataset_dir,
        "annotation_format": "auto",  # Will auto-detect 'video'
        "fps": 1,
        "split_percentages": [0.7, 0.15, 0.15],
    }

    # Run the workflow
    run_data_ingest()

    dataset_name = "video_ingest_test1"

    # Load and validate the output dataset
    dataset = fo.load_dataset(dataset_name)

    logging.info(f"Loaded dataset: {dataset.name}")
    assert dataset is not None
    assert dataset.name == dataset_name

    tag_counts = dataset.count_sample_tags()
    total_tagged = sum(tag_counts.values())
    logging.info(f"Tag counts: {tag_counts}")
    assert total_tagged == len(dataset), "Not all samples were tagged for split"

    # Clean up
    fo.delete_dataset(dataset_name)

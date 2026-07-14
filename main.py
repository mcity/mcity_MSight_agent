import datetime
import logging
import os
import signal
import sys
import time
import warnings
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

import torch

IGNORE_FUTURE_WARNINGS = True
if IGNORE_FUTURE_WARNINGS:
    warnings.simplefilter("ignore", category=FutureWarning)

import gc

import fiftyone as fo
import torch.multiprocessing as mp
from tqdm import tqdm

from config.config import (
    SELECTED_DATASET,
    SELECTED_WORKFLOW,
    V51_ADDRESS,
    V51_PORT,
    V51_REMOTE,
    WORKFLOWS,
)
from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO
from utils.dataset_loader import load_dataset
from utils.logging import configure_logging
from utils.sidebar_groups import arrange_fields_in_groups
from utils.wandb_helper import wandb_close, wandb_init
from workflows.auto_labeling import (
    CustomCoDETRObjectDetection,
    CustomRFDETRObjectDetection,
    HuggingFaceObjectDetection,
    UltralyticsObjectDetection,
)
from workflows.rfdetr_keypoint import RFDETRKeypointDetection
# from workflows.vitpose_keypoint import (
#     ViTPoseKeypointDetection,
#     RoIKeypointDetection,
#     download_vitpose_weights,
# )
from workflows.data_ingest import run_data_ingest

wandb_run = None  # Init globally to make sure it is available

def signal_handler(sig, frame):
    """Handle Ctrl+C signal by cleaning up resources and exiting."""
    logging.error("You pressed Ctrl+C!")
    try:
        wandb_close(exit_code=1)
        #cleanup_memory()
    except:
        pass
    sys.exit(0)


def workflow_auto_labeling_ultralytics(dataset, run_config, wandb_activate=True):
    """Auto-labeling workflow using Ultralytics models with optional training and inference."""
    try:
        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name=run_config["model_name"],
            project_name="Auto Labeling Ultralytics",
            dataset_name=run_config["v51_dataset_name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        detector = UltralyticsObjectDetection(dataset=dataset, config=run_config)

        # Check if all selected modes are supported
        SUPPORTED_MODES = ["train", "inference"]
        for mode in run_config["mode"]:
            if mode not in SUPPORTED_MODES:
                logging.error(f"Selected mode {mode} is not supported.")

        if SUPPORTED_MODES[0] in run_config["mode"]:
            logging.info(f"Training model {run_config['model_name']}")
            detector.train()
        if SUPPORTED_MODES[1] in run_config["mode"]:
            logging.info(f"Running inference for model {run_config['model_name']}")
            detector.inference()

    except Exception as e:
        logging.error(f"An error occurred with model {run_config['model_name']}: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_exit_code)

    return True


def workflow_auto_labeling_hf(dataset, hf_dataset, run_config, wandb_activate=True):
    """Auto-labeling using Hugging Face models on a dataset, including training and/or inference based on the provided configuration."""
    try:
        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name=run_config["model_name"],
            project_name="Auto Labeling Hugging Face",
            dataset_name=run_config["v51_dataset_name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        detector = HuggingFaceObjectDetection(
            dataset=dataset,
            config=run_config,
        )
        SUPPORTED_MODES = ["train", "inference"]

        # Check if all selected modes are supported
        for mode in run_config["mode"]:
            if mode not in SUPPORTED_MODES:
                logging.error(f"Selected mode {mode} is not supported.")
        if SUPPORTED_MODES[0] in run_config["mode"]:
            logging.info(f"Training model {run_config['model_name']}")
            detector.train(hf_dataset)
        if SUPPORTED_MODES[1] in run_config["mode"]:
            logging.info(f"Running inference for model {run_config['model_name']}")
            detector.inference(inference_settings=run_config["inference_settings"])

    except Exception as e:
        logging.error(f"An error occurred with model {run_config['model_name']}: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_exit_code)

    return True


def workflow_auto_labeling_custom_codetr(
    dataset, dataset_info, run_config, wandb_activate=True
):
    """Auto labeling workflow using Co-DETR model supporting training and inference modes."""

    try:
        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name=run_config["config"],
            project_name="Co-DETR Auto Labeling",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        mode = run_config["mode"]

        detector = CustomCoDETRObjectDetection(dataset, dataset_info, run_config)
        detector.convert_data()
        if "train" in mode:
            detector.update_config_file(
                dataset_name=dataset_info["name"],
                config_file=run_config["config"],
                max_epochs=run_config["epochs"],
            )
            detector.train(
                run_config["config"], run_config["n_gpus"], run_config["container_tool"]
            )
        if "inference" in mode:
            detector.run_inference(
                dataset,
                run_config["config"],
                run_config["n_gpus"],
                run_config["container_tool"],
                run_config["inference_settings"],
            )
    except Exception as e:
        logging.error(f"Error during CoDETR training: {e}")
        wandb_exit_code = 1
    finally:
        wandb_close(wandb_exit_code)

    return True


def cleanup_memory(do_extensive_cleanup=False):
    """Clean up memory after workflow execution. 'do_extensive_cleanup' recommended for multiple training sessions in a row."""
    logging.info("Starting memory cleanup")
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    if do_extensive_cleanup:

        # Clear any leftover tensors
        n_deleted_torch_objects = 0
        for obj in tqdm(
            gc.get_objects(), desc="Deleting objects from Python Garbage Collector"
        ):
            try:
                if torch.is_tensor(obj):
                    del obj
                    n_deleted_torch_objects += 1
            except:
                pass

        logging.info(f"Deleted {n_deleted_torch_objects} torch objects")

        # Final garbage collection
        gc.collect()


class WorkflowExecutor:
    """Orchestrates the execution of multiple data processing workflows in sequence."""

    def __init__(
        self,
        workflows: List[str],
        selected_dataset: str,
        dataset: fo.Dataset,
        dataset_info: Dict,
    ):
        """Initializes with specified workflows, dataset selection, and dataset metadata."""
        self.workflows = workflows
        self.selected_dataset = selected_dataset
        self.dataset = dataset
        self.dataset_info = dataset_info

    def execute(self) -> bool:
        """Execute all configured workflows in sequence and handle errors."""
        if len(self.workflows) == 0:
            logging.error("No workflows selected.")
            return False

        logging.info(f"Selected workflows: {self.workflows}")
        for workflow in self.workflows:
            logging.info(
                f"Running workflow {workflow} for dataset {self.selected_dataset}"
            )
            try:
                if workflow == "auto_labeling":

                    # Config
                    SUPPORTED_MODEL_SOURCES = [
                        "hf_models_objectdetection",  # [0]
                        "ultralytics",                # [1]
                        "custom_codetr",              # [2]
                        "roboflow",                   # [3]
                        "roboflow_keypoint",          # [4]
                        "vitpose",                    # [5] standalone ViTPose fine-tuning
                        "roi_keypoint",               # [6] two-stage: RF-DETR + ViTPose
                    ]

                    # Common parameters between models
                    config_autolabel = WORKFLOWS["auto_labeling"]
                    mode = config_autolabel["mode"]
                    epochs = config_autolabel["epochs"]
                    selected_model_source = config_autolabel["model_source"]

                    # Check if all selected modes are supported
                    for model_source in selected_model_source:
                        if model_source not in SUPPORTED_MODEL_SOURCES:
                            logging.error(
                                f"Selected model source {model_source} is not supported."
                            )

                    if SUPPORTED_MODEL_SOURCES[0] in selected_model_source:
                        # Hugging Face Models
                        # Single GPU mode (https://github.com/huggingface/transformers/issues/28740)
                        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                        hf_models = config_autolabel["hf_models_objectdetection"]

                        # Dataset Conversion
                        try:
                            logging.info("Converting dataset into Hugging Face format.")
                            pytorch_dataset = FiftyOneTorchDatasetCOCO(self.dataset)
                            pt_to_hf_converter = TorchToHFDatasetCOCO(pytorch_dataset)
                            hf_dataset = pt_to_hf_converter.convert()
                        except Exception as e:
                            logging.error(f"Error during dataset conversion: {e}")

                        for MODEL_NAME in (
                            pbar := tqdm(hf_models, desc="Auto Labeling Models")
                        ):
                            # Status Update
                            pbar.set_description(
                                f"Processing Hugging Face model {MODEL_NAME}"
                            )

                            # Config
                            config_model = config_autolabel[
                                "hf_models_objectdetection"
                            ][MODEL_NAME]

                            run_config = {
                                "mode": mode,
                                "model_name": MODEL_NAME,
                                "v51_dataset_name": self.selected_dataset,
                                "epochs": epochs,
                                "early_stop_threshold": config_autolabel[
                                    "early_stop_threshold"
                                ],
                                "early_stop_patience": config_autolabel[
                                    "early_stop_patience"
                                ],
                                "learning_rate": config_autolabel["learning_rate"],
                                "weight_decay": config_autolabel["weight_decay"],
                                "max_grad_norm": config_autolabel["max_grad_norm"],
                                "batch_size": config_model.get("batch_size", 1),
                                "image_size": config_model.get("image_size", None),
                                "n_worker_dataloader": config_autolabel[
                                    "n_worker_dataloader"
                                ],
                                "inference_settings": config_autolabel[
                                    "inference_settings"
                                ],
                            }

                            # Workflow
                            workflow_auto_labeling_hf(
                                self.dataset,
                                hf_dataset,
                                run_config,
                            )

                    if SUPPORTED_MODEL_SOURCES[1] in selected_model_source:
                        # Ultralytics Models
                        config_ultralytics = config_autolabel["ultralytics"]
                        models_ultralytics = config_ultralytics["models"]
                        export_dataset_root = config_ultralytics["export_dataset_root"]

                        # Export data into necessary format
                        if "train" in mode:
                            try:
                                UltralyticsObjectDetection.export_data(
                                    self.dataset,
                                    self.dataset_info,
                                    export_dataset_root,
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error during Ultralytics dataset export: {e}"
                                )

                        for model_name in (
                            pbar := tqdm(
                                models_ultralytics, desc="Ultralytics training"
                            )
                        ):
                            pbar.set_description(f"Ultralytics model {model_name}")
                            run_config = {
                                "mode": mode,
                                "model_name": model_name,
                                "v51_dataset_name": self.dataset_info["name"],
                                "epochs": epochs,
                                "patience": config_autolabel["early_stop_patience"],
                                "batch_size": models_ultralytics[model_name][
                                    "batch_size"
                                ],
                                "img_size": models_ultralytics[model_name]["img_size"],
                                "export_dataset_root": export_dataset_root,
                                "inference_settings": config_autolabel[
                                    "inference_settings"
                                ],
                                "multi_scale": config_ultralytics["multi_scale"],
                                "cos_lr": config_ultralytics["cos_lr"],
                            }

                            workflow_auto_labeling_ultralytics(self.dataset, run_config)

                    if SUPPORTED_MODEL_SOURCES[2] in selected_model_source:
                        # Custom Co-DETR
                        config_codetr = config_autolabel["custom_codetr"]
                        run_config = {
                            "export_dataset_root": config_codetr["export_dataset_root"],
                            "container_tool": config_codetr["container_tool"],
                            "n_gpus": config_codetr["n_gpus"],
                            "mode": config_autolabel["mode"],
                            "epochs": config_autolabel["epochs"],
                            "inference_settings": config_autolabel[
                                "inference_settings"
                            ],
                            "config": None,
                        }
                        codetr_configs = config_codetr["configs"]

                        for config in (
                            pbar := tqdm(
                                codetr_configs, desc="Processing Co-DETR configurations"
                            )
                        ):
                            pbar.set_description(f"Co-DETR model {config}")
                            run_config["config"] = config
                            workflow_auto_labeling_custom_codetr(
                                self.dataset, self.dataset_info, run_config
                            )

                    if SUPPORTED_MODEL_SOURCES[3] in selected_model_source:

                        config_rfdetr = config_autolabel["roboflow"]

                        # Shared config parameters
                        shared_config = {
                            "epochs": config_autolabel["epochs"],
                            "learning_rate": config_autolabel["learning_rate"],
                            "weight_decay": config_autolabel["weight_decay"],
                            "early_stop_patience": config_autolabel["early_stop_patience"],
                            "early_stop_threshold": config_autolabel["early_stop_threshold"],
                        }

                        run_config = {
                            "export_dataset_root": config_rfdetr["export_dataset_root"],
                            "mode": config_autolabel["mode"],
                            "inference_settings": config_autolabel["inference_settings"],
                            "config": None,
                            # RF-DETR specific parameters
                            "batch_size": config_rfdetr["batch_size"],
                            "grad_accum_steps": config_rfdetr["grad_accum_steps"],
                            "lr_encoder": config_rfdetr["lr_encoder"],
                            "resolution": config_rfdetr["resolution"],
                            "use_ema": config_rfdetr["use_ema"],
                            "gradient_checkpointing": config_rfdetr["gradient_checkpointing"],
                            "early_stopping_min_delta": config_rfdetr["early_stopping_min_delta"],
                            "early_stopping_use_ema": config_rfdetr["early_stopping_use_ema"],
                        }

                        rfdetr_configs = config_rfdetr["configs"]

                        for config in (
                            pbar := tqdm(rfdetr_configs, desc="Processing RF-DETR configurations")
                        ):
                            pbar.set_description(f"RF-DETR model {config}")
                            run_config["config"] = config

                            try:
                                wandb_exit_code = 0
                                wandb_run = wandb_init(
                                    run_name=config,
                                    project_name="RF-DETR Auto Labeling",
                                    dataset_name=self.dataset_info["name"],
                                    config=run_config,
                                    wandb_activate=True,
                                )

                                detector = CustomRFDETRObjectDetection(
                                    self.dataset, self.dataset_info, run_config
                                )

                                # Convert data to RF-DETR format (only needed for training)
                                if "train" in mode:
                                    detector.convert_data()

                                # Training
                                if "train" in mode:
                                    logging.info(f"Training RF-DETR model: {config}")
                                    detector.train(run_config, shared_config)

                                # Inference
                                if "inference" in mode:
                                    logging.info(f"Running inference for RF-DETR model: {config}")
                                    fallback_hf_map = config_rfdetr.get("fallback_hf_repo", {})
                                    rfdetr_inference_settings = {
                                        **config_autolabel["inference_settings"],
                                        "class_names": config_rfdetr.get("class_names"),
                                        "fallback_hf_repo": fallback_hf_map.get(config),
                                    }
                                    detector.inference(
                                        inference_settings=rfdetr_inference_settings
                                    )

                            except Exception as e:
                                logging.error(f"Error during RF-DETR workflow with {config}: {e}")
                                wandb_exit_code = 1
                            finally:
                                wandb_close(wandb_exit_code)

                    if SUPPORTED_MODEL_SOURCES[4] in selected_model_source:
                        # RF-DETR with keypoint head
                        config_rfdetr_kp = config_autolabel["roboflow_keypoint"]

                        shared_config = {
                            "epochs": config_autolabel["epochs"],
                            "learning_rate": config_autolabel["learning_rate"],
                            "weight_decay": config_autolabel["weight_decay"],
                            "early_stop_patience": config_autolabel["early_stop_patience"],
                            "early_stop_threshold": config_autolabel["early_stop_threshold"],
                        }

                        run_config_kp = {
                            "export_dataset_root": config_rfdetr_kp["export_dataset_root"],
                            "mode": config_autolabel["mode"],
                            "inference_settings": config_autolabel["inference_settings"],
                            "config": None,
                            # FiftyOne data-path selector — MUST be forwarded so that
                            # RFDETRKeypointDetection.train() uses the FO-native loader
                            # (prefetch_fo_split → FiftyOneKeypointDataset) instead of
                            # the COCO-export path.  Without this flag, fo_native
                            # defaults to False and all keypoint visibilities are 0
                            # (COCO fallback), causing OKS=nan for the entire run.
                            "fo_native": config_rfdetr_kp.get("fo_native", False),
                            "detection_field": config_rfdetr_kp.get("detection_field", "ground_truth"),
                            "target_label": config_rfdetr_kp.get("target_label", "pedestrian"),
                            "class_names": config_rfdetr_kp.get("class_names", ["pedestrian"]),
                            "num_classes": config_rfdetr_kp.get("num_classes", 1),
                            # Keypoint-specific
                            "keypoint_field": config_rfdetr_kp["keypoint_field"],
                            "keypoint_names": config_rfdetr_kp["keypoint_names"],
                            "num_keypoints": len(config_rfdetr_kp["keypoint_names"]),
                            "kp_xy_coef": config_rfdetr_kp.get("kp_xy_coef", 5.0),
                            "kp_vis_coef": config_rfdetr_kp.get("kp_vis_coef", 1.0),
                            "freeze_backbone_epochs": config_rfdetr_kp.get("freeze_backbone_epochs", 5),
                            "freeze_bbox_head": config_rfdetr_kp.get("freeze_bbox_head", False),
                            # RF-DETR parameters
                            "batch_size": config_rfdetr_kp.get("batch_size", 8),
                            "lr_encoder": config_rfdetr_kp.get("lr_encoder", None),
                            "resolution": config_rfdetr_kp.get("resolution", 560),
                            "pretrain_weights": config_rfdetr_kp.get("pretrain_weights", None),
                        }

                        for config in (
                            pbar := tqdm(
                                config_rfdetr_kp["configs"],
                                desc="Processing RF-DETR Keypoint configurations",
                            )
                        ):
                            pbar.set_description(f"RF-DETR Keypoint model {config}")
                            run_config_kp["config"] = config

                            try:
                                wandb_exit_code = 0
                                wandb_run = wandb_init(
                                    run_name=config,
                                    project_name="RF-DETR Keypoint Auto Labeling",
                                    dataset_name=self.dataset_info["name"],
                                    config=run_config_kp,
                                    wandb_activate=True,
                                )

                                detector_kp = RFDETRKeypointDetection(
                                    self.dataset, self.dataset_info, run_config_kp
                                )

                                detector_kp.convert_data()

                                if "train" in mode:
                                    logging.info(f"Training RF-DETR keypoint model: {config}")
                                    detector_kp.train(run_config_kp, shared_config)

                                if "inference" in mode:
                                    logging.info(f"Running keypoint inference: {config}")
                                    detector_kp.inference(
                                        inference_settings=config_autolabel["inference_settings"]
                                    )

                            except Exception as e:
                                logging.error(f"Error during RF-DETR keypoint workflow with {config}: {e}")
                                wandb_exit_code = 1
                            finally:
                                wandb_close(wandb_exit_code)


                    if SUPPORTED_MODEL_SOURCES[5] in selected_model_source:
                        # ── vitpose: standalone ViTPose-B fine-tuning ───────────────
                        config_vp = config_autolabel["vitpose"]
                        kp_names_vp = config_vp.get("keypoint_names", ["ankle_center"])

                        shared_config_vp = {
                            "epochs":              config_autolabel["epochs"],
                            "learning_rate":       config_autolabel["learning_rate"],
                            "weight_decay":        config_autolabel["weight_decay"],
                            "early_stop_patience": config_autolabel["early_stop_patience"],
                        }
                        run_config_vp = {
                            "mode":                     config_autolabel["mode"],
                            "vitpose_pretrain_weights": config_vp.get("vitpose_pretrain_weights", "usyd-community/vitpose-base-simple"),
                            "vitpose_save_dir":         config_vp.get("vitpose_save_dir", "output/models/vitpose/"),
                            "detection_field":          config_vp.get("detection_field", "ground_truth"),
                            "keypoint_field":           config_vp.get("keypoint_field", "pedestrian_points"),
                            "target_label":             config_vp.get("target_label", "pedestrian"),
                            "keypoint_names":           kp_names_vp,
                            "num_keypoints":            len(kp_names_vp),
                            "kp_sigma":                 config_vp.get("kp_sigma", 0.089),
                            "batch_size":               config_vp.get("batch_size", 32),
                            "freeze_backbone":          config_vp.get("freeze_backbone", True),
                            "freeze_backbone_epochs":   config_vp.get("freeze_backbone_epochs", 5),
                        }

                        try:
                            wandb_exit_code = 0
                            wandb_run = wandb_init(
                                run_name="vitpose-finetune",
                                project_name="ViTPose Fine-Tuning",
                                dataset_name=self.dataset_info["name"],
                                config=run_config_vp,
                                wandb_activate=True,
                            )
                            vp_detector = ViTPoseKeypointDetection(
                                self.dataset, self.dataset_info, run_config_vp
                            )
                            if "train" in mode:
                                logging.info("Downloading ViTPose-B pretrained weights…")
                                vp_detector.download_weights()
                                logging.info("Fine-tuning ViTPose-B on GT RoI crops…")
                                vp_detector.train(run_config_vp, shared_config_vp)
                            if "inference" in mode:
                                logging.info("ViTPose inference with GT boxes…")
                                vp_detector.inference(
                                    inference_settings=config_autolabel["inference_settings"]
                                )
                        except Exception as e:
                            logging.error(f"Error in vitpose workflow: {e}")
                            wandb_exit_code = 1
                        finally:
                            wandb_close(wandb_exit_code)

                    if SUPPORTED_MODEL_SOURCES[6] in selected_model_source:
                        # ── roi_keypoint: RF-DETR detect → ViTPose predict ──────────
                        config_roi   = config_autolabel["roi_keypoint"]
                        kp_names_roi = config_roi.get("keypoint_names", ["ankle_center"])

                        run_config_roi = {
                            "mode":                config_autolabel["mode"],
                            "rfdetr_model":        config_roi.get("rfdetr_model", "rfdetr_2xlarge"),
                            "pretrain_weights":    config_roi.get("pretrain_weights", None),
                            "detection_threshold": config_roi.get("detection_threshold", 0.3),
                            "vitpose_weights":     config_roi.get("vitpose_weights", None),
                            "target_label":        config_roi.get("target_label", "pedestrian"),
                            "keypoint_names":      kp_names_roi,
                            "num_keypoints":       len(kp_names_roi),
                            "kp_sigma":            config_roi.get("kp_sigma", 0.089),
                        }

                        try:
                            wandb_exit_code = 0
                            wandb_run = wandb_init(
                                run_name=f"roi_kp-{config_roi.get('rfdetr_model','rfdetr_2xlarge')}",
                                project_name="RoI Keypoint Inference",
                                dataset_name=self.dataset_info["name"],
                                config=run_config_roi,
                                wandb_activate=True,
                            )
                            roi_detector = RoIKeypointDetection(
                                self.dataset, self.dataset_info, run_config_roi
                            )
                            if "inference" in mode:
                                logging.info(
                                    "Running RoI Keypoint inference "
                                    "(RF-DETR detect → ViTPose predict)…"
                                )
                                roi_detector.inference(
                                    inference_settings=config_autolabel["inference_settings"]
                                )
                        except Exception as e:
                            logging.error(f"Error in roi_keypoint workflow: {e}")
                            wandb_exit_code = 1
                        finally:
                            wandb_close(wandb_exit_code)

                elif workflow == "vitpose_download":
                    cfg = WORKFLOWS["vitpose_download"]
                    download_vitpose_weights(
                        model_name=cfg.get("vitpose_model", "usyd-community/vitpose-base-simple"),
                        save_dir=cfg.get("save_dir", "output/models/vitpose/"),
                    )

                elif workflow == "data_ingest":
                    dataset = run_data_ingest()

                    dataset_info = {
                        "name": "custom_dataset",
                        "v51_type": "FiftyOneDataset",
                        "splits": ["train", "val", "test"],
                    }

                    self.dataset = dataset
                    self.dataset_info = dataset_info
                    self.selected_dataset = "custom_dataset"

                    logging.info(f"Data ingestion completed successfully.")


                else:
                    logging.error(
                        f"Workflow {workflow} not found. Check available workflows in config.py."
                    )
                    return False

                #cleanup_memory()  # Clean after each workflow
                logging.info(f"Completed workflow {workflow} and cleaned up memory")

            except Exception as e:
                logging.error(f"Workflow {workflow}: An error occurred: {e}")
                wandb_close(exit_code=1)
                #cleanup_memory()  # Clean up even after failure

        return True


def main():
    """Executes the data processing workflow, loads dataset, and launches Voxel51 visualization interface."""
    time_start = time.time()
    configure_logging()

    # Signal handler for CTRL + C
    signal.signal(signal.SIGINT, signal_handler)

    # Execute workflows
    if "data_ingest" in SELECTED_WORKFLOW:
        executor = WorkflowExecutor(
            SELECTED_WORKFLOW,
            SELECTED_DATASET["name"],
            dataset=None,
            dataset_info=None,
        )
        executor.execute()

        # FIX: Pull back outputs after ingestion
        dataset = executor.dataset
        dataset_info = executor.dataset_info

    else:
        dataset, dataset_info = load_dataset(SELECTED_DATASET)

        executor = WorkflowExecutor(
            SELECTED_WORKFLOW,
            SELECTED_DATASET["name"],
            dataset,
            dataset_info,
        )
        executor.execute()


    if dataset is not None:
        dataset.reload()
        dataset.save()

        arrange_fields_in_groups(dataset)
        logging.info(f"Launching Voxel51 session for dataset {dataset_info['name']}.")

        # Dataset stats
        logging.debug(dataset)
        logging.debug(dataset.stats(include_media=True))

        # V51 UI launch
        session = fo.launch_app(
            dataset, address=V51_ADDRESS, port=V51_PORT, remote=V51_REMOTE
        )
    else:
        logging.info("Skipping Voxel51 session.")



    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    cleanup_memory()
    main()

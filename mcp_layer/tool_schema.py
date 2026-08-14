tools = [

        {
            "type": "function",
            "function": {
                "name": "send_reply",
                "description": (
                    "Use this to send a plain text reply to the user when no other tool is needed. "
                    "Call this instead of responding with text directly. "
                    "Parameters: 'message' (required) — the reply text; "
                    "'source' (optional) — set when your reply provides knowledge (explains, compares, "
                    "or recommends based on ML expertise or tool documentation); omit when your reply "
                    "drives the workflow (confirms an action, reports state, or presents options). "
                    "See the SOURCE TAG RULE in the system prompt for full classification criteria and examples."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The reply text to send to the user."
                        },
                        "source": {
                            "type": "string",
                            "description": (
                                "Source attribution — set when your reply provides knowledge "
                                "(explains, describes, compares, or recommends). "
                                "Omit when your reply drives the workflow (confirms, presents options, reports state)."
                            )
                        }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_intro",
                "description": (
                    "Show an informational message to the user BEFORE a slower action's own "
                    "progress/log output streams in — currently only used for msight_pipeline's "
                    "Demo start (see STEP 2a): explain what MSight Pipeline does and what's about "
                    "to run, THEN call start_msight_pipeline as your very next tool call in this "
                    "same turn. Unlike send_reply, this does NOT end your turn — you MUST keep "
                    "going and actually call the tool you just described; calling send_intro alone "
                    "with no follow-up tool call leaves the user with an explanation but nothing "
                    "actually happening. Do not use this anywhere else — every other reply goes "
                    "through send_reply as usual."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The informational text to show, e.g. what MSight Pipeline does and what's about to run."
                        }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "confirm_export",
                "description": (
                    "Call this ONLY after the user has explicitly confirmed they want to export "
                    "(e.g., responded 'yes', 'proceed', 'go ahead', 'sounds good'). "
                    "This records their consent and unlocks the export tool for this session. "
                    "After calling this, immediately call export_to_cvat or export_to_label_studio "
                    "with the classes from SESSION_STATE manual_classes. "
                    "Do NOT call this speculatively — only call it when the user has actually confirmed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "confirm_run",
                "description": (
                    "Call this ONLY after the user has explicitly confirmed they want to proceed "
                    "(e.g., responded 'yes', 'proceed', 'go ahead', 'start it', 'run it') in response "
                    "to a pre-run confirmation summary. This is shared across workflows — what it "
                    "unlocks depends on which one is active: in auto_labeling, it unlocks "
                    "run_auto_labeling and you should call that next; in msight_pipeline, it unlocks "
                    "start_msight_pipeline and you should call that again with the same source you "
                    "already stated. Do NOT call this speculatively — only call it when the user has "
                    "actually confirmed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "select_workflow",
                "description": (
                    "Call this ONLY at the very start of a session when the user picks a workflow "
                    "for the first time and no workflow is currently active. "
                    "Do NOT call this if SESSION_STATE already shows a workflow — "
                    "use switch_workflow instead if the user wants to change to a different workflow."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_name": {
                            "type": "string",
                            "description": "The workflow name the user selected. Must come from the user's message — do not infer or guess."
                        }
                    },
                    "required": ["workflow_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_datasets",
                "description": "Lists all available datasets from datasets.yaml (plus fixed defaults).",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "switch_workflow",
                "description": (
                    "Call this when the user wants to start over or restart from scratch within the "
                    "current workflow — pass the current workflow name ('auto_labeling') to reset all "
                    "state back to dataset selection. "
                    "Do NOT call this if the user is answering a pending in-workflow question — e.g. replying "
                    "'auto labeling' or 'auto' to a Manual vs Auto Generated Labeling prompt means the annotation "
                    "method (call set_labeling_path instead), not the 'Auto Labeling' workflow by name. "
                    "Only treat a message as a workflow switch when no other pending question is awaiting an answer. "
                    "If the run is mid-export/mid-training (locked), this call is blocked and returns a message "
                    "telling you to confirm with the user that they want to discard all progress — once they "
                    "explicitly confirm (e.g. 'yes', 'discard it', 'start over anyway'), call this again with "
                    "confirm_restart=true to force the reset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_name": {
                            "type": "string",
                            "description": "The exact workflow name the user stated. Must come from the user's message — do not infer or guess."
                        },
                        "confirm_restart": {
                            "type": "boolean",
                            "description": "Set to true ONLY after the user has explicitly confirmed they want to discard an in-progress (locked) run. Omit or leave false otherwise."
                        }
                    },
                    "required": ["workflow_name"]
                }
            }
        },
       {
            "type": "function",
            "function": {
                "name": "set_selected_dataset",
                "description": (
                    "REQUIRED: Update SELECTED_DATASET in config.py. "
                    "Call this immediately whenever the user provides a dataset name — "
                    "whether from ingestion or from the existing list. "
                    "No downstream tool (run_auto_labeling, export_to_cvat, etc.) works correctly without this."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "The name of the dataset to select."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "configure_auto_labeling",
                "description": (
                    "Enable the selected model source and model inside config.py for the auto_labeling workflow. "
                    "ACCURACY GUARD: only call this when the user has explicitly named a specific model "
                    "(e.g. 'use yolo12n', 'rfdetr_2xlarge', 'I want the DETR model'). "
                    "A user describing their use case ('I want to detect cars and pedestrians'), "
                    "asking what to use ('what model should I use?'), or saying 'sure' to a vague suggestion "
                    "is NOT an explicit model selection. In those cases, recommend with send_reply and ask "
                    "'Which model would you like to use?' — then wait for a specific answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_source": {
                            "type": "string",
                            "description": (
                                "The model source to enable. Derive it from the model name: "
                                "rfdetr_* → 'roboflow'; "
                                "yolo* (yolo11n, yolo12x, etc.) → 'ultralytics'; "
                                "facebook/*, microsoft/*, SenseTime/*, hustvl/*, jozhang97/*, Omnifact/* → 'hf_models_objectdetection'; "
                                "co_deformable_detr_*.py or co_dino_*.py → 'custom_codetr'."
                            )
                        },
                        "selected_model": {
                            "type": "string",
                            "description": "The exact model name or config file the user typed (e.g. 'rfdetr_2xlarge', 'yolo11n', 'facebook/detr-resnet-50')"
                        }
                    },
                    "required": ["selected_source", "selected_model"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_model_sources_and_models",
                "description": (
                    "Lists valid model sources and models for auto_labeling. "
                    "Only call this to show the list for the first time. "
                    "Do NOT call this again if the user is asking for a recommendation or says they're unsure "
                    "which model to pick (e.g. 'I don't know which to use', 'what do you suggest?') — "
                    "the list has already been shown; use send_reply to recommend a specific model with brief "
                    "reasoning instead, then ask them to confirm."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_auto_labeling_hyperparams",
                "description": (
                    "Update hyperparameters like mode, epochs, learning_rate, etc. for auto_labeling workflow. "
                    "ACCURACY GUARD — TWO valid triggers only: "
                    "(1) user directly instructs a change ('set epochs to 5', 'change learning rate to 0.001') — act immediately; "
                    "(2) user explicitly confirms a recommendation you already made ('yes apply those', 'go ahead', 'sounds good', 'yes') — act after that confirmation. "
                    "NOT a trigger — any request for advice or suggestions, regardless of phrasing: "
                    "'what do you recommend?', 'what would you suggest?', 'what's best for my use case?', 'any recommendations?'. "
                    "In those cases: give the recommendation with send_reply, ask 'Would you like me to apply these?', then wait. "
                    "Do NOT call this tool speculatively or before receiving explicit confirmation. "
                    "If the user accepts current defaults without changes, do NOT call this tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pipeline mode(s): inference."
                        },
                        "epochs": {"type": "integer", "description": "Number of training epochs."},
                        "early_stop_patience": {"type": "integer", "description": "Patience for early stopping."},
                        "early_stop_threshold": {"type": "number", "description": "Improvement threshold for early stopping."},
                        "learning_rate": {"type": "number", "description": "Learning rate for the optimizer."},
                        "weight_decay": {"type": "number", "description": "Weight decay (L2 penalty)."},
                        "max_grad_norm": {"type": "number", "description": "Max norm for gradient clipping."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_msight_localization_config",
                "description": (
                    "Configure MSight geolocation for the auto_labeling workflow's detections — "
                    "optional, only relevant on the 'auto' labeling path (it geolocates model "
                    "predictions, which only exist after auto-generated labeling). Only call this "
                    "after the user has explicitly confirmed they want to geolocate detections — "
                    "never call speculatively. "
                    "IMPORTANT LIMITATION: the calibration (camera intrinsics + lat/lon map) is "
                    "currently hardcoded to the Ashley/Huron intersection camera — there is no "
                    "per-dataset or per-camera calibration selection. Only offer this if the "
                    "dataset's footage is actually from that camera; if the user asks to geolocate "
                    "a dataset from a different camera/location, tell them that calibration isn't "
                    "supported yet rather than calling this tool. detection_field should default to "
                    "'pred_od_<model_name>-<dataset_name>' using the currently configured model and "
                    "dataset, unless the user names a different field explicitly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "detection_field": {
                            "type": "string",
                            "description": "FiftyOne field holding the fo.Detections to geolocate, e.g. 'pred_od_rfdetr_2xlarge-mydataset'."
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "True to enable localization (default), false to save the config but leave it off."
                        }
                    },
                    "required": ["detection_field"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "launch_voxel51_session",
                "description": "Launch the Voxel51 session for a specific dataset. Always pass dataset_name explicitly — never call without it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the FiftyOne dataset to visualize. After import_from_cvat, use the labeled name (e.g. 'custom_dataset15_labeled')."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reset_workflow_state",
                "description": (
                    "Fully resets workflow, dataset, and all in-progress state so the user can "
                    "leave the current workflow entirely — the next greeting will show the full "
                    "workflow list again. Use this when the user is done with the workflow itself "
                    "(e.g. says 'that's all', 'thanks, I'm finished', or explicitly asks to exit or "
                    "start over from the top). "
                    "This is a bigger action than switch_workflow(confirm_restart=true): to restart "
                    "the CURRENT auto_labeling run while staying in auto_labeling (e.g. after it "
                    "locks mid-export/mid-training), use switch_workflow(workflow_name='auto_labeling', "
                    "confirm_restart=true) instead — that's the targeted action for that case, not this one."
                ),
                "parameters": {
                "type": "object",
                "properties": {},
                "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_auto_labeling",
                "description": (
                    "Run main.py for auto_labeling workflow and stream logs in real time. "
                    "IRREVERSIBLE: training consumes time and compute and cannot be undone once started. "
                    "Only call this after the user has given an explicit run confirmation — "
                    "valid phrases: 'Run the workflow', 'Start training', 'Let's begin', or equivalent. "
                    "Do not call this based on inferred intent or because the workflow appears ready."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "export_to_cvat",
                "description": "Export a FiftyOne dataset to CVAT for annotation. Use ONLY for the Manual Labeling path with with_predictions=False. For Auto Generated Labeling, this tool is called automatically by the system after run_auto_labeling completes — do NOT call it yourself for Auto Generated Labeling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the FiftyOne dataset to export."
                        },
                        "with_predictions": {
                            "type": "boolean",
                            "description": "If true, exports model predictions for Auto Generated Labeling. If false, exports images only for Manual Labeling."
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Label names to pre-configure in CVAT for the Manual Labeling path (e.g. ['car', 'pedestrian', 'cyclist']). Only used when with_predictions=False. Ask the user for these before calling export."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "import_from_cvat",
                "description": "Download completed annotations from CVAT for a previously uploaded dataset and save as a new labeled FiftyOne dataset named <dataset_name>_labeled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the original FiftyOne dataset that was uploaded to CVAT."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_labeling_backend",
                "description": "Check which annotation backends (CVAT, Label Studio) are configured in .env and return the active one. Call this when the backend has not yet been detected for the session (e.g., after a workflow switch mid-session).",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_labeling_backend",
                "description": "Set the active annotation backend for this session. Call this when the user names a backend as their choice, including softly-phrased decisions that still name a specific backend ('CVAT', 'use CVAT', 'I prefer Label Studio', 'can I use CVAT instead', 'let's switch to Label Studio', 'actually use CVAT'). Only treat it as a non-selection when the user asks for information WITHOUT naming a backend they want ('what's the difference?', 'tell me more', 'which is better?') — those get send_reply, re-asking which they prefer. Do NOT call this if only one backend is configured — it is selected automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "string",
                            "enum": ["cvat", "label_studio"],
                            "description": "The annotation backend to use. 'cvat' for CVAT, 'label_studio' for Label Studio."
                        }
                    },
                    "required": ["backend"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_labeling_path",
                "description": (
                    "Call this IMMEDIATELY when the user names their labeling path at Step 3b, "
                    "or explicitly asks to switch between manual and auto generated labeling later. "
                    "Use 'manual' when the user says 'Manual Labeling', 'annotate myself', etc. "
                    "Use 'auto' when the user says 'Auto Generated Labeling', 'run a model', etc. "
                    "This MUST be called before any export or model tool — it unlocks the correct "
                    "downstream tools and persists the path so backend switches do not lose it. "
                    "Do NOT call this for unrelated requests (backend changes, dataset changes, questions) — "
                    "it only sets manual vs. auto, never re-call it with the path that is already active."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "enum": ["manual", "auto"],
                            "description": "'manual' for Manual Labeling, 'auto' for Auto Generated Labeling."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "export_to_label_studio",
                "description": "Export a FiftyOne dataset to Label Studio for annotation. Use ONLY when the active backend is Label Studio. For Manual Labeling use with_predictions=False. For Auto Generated Labeling, this is called automatically after run_auto_labeling — do NOT call it yourself.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the FiftyOne dataset to export."
                        },
                        "with_predictions": {
                            "type": "boolean",
                            "description": "If true, attaches model predictions for Auto Generated Labeling. If false, uploads images only for Manual Labeling."
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Label names to pre-configure in Label Studio for the Manual Labeling path (e.g. ['car', 'pedestrian', 'cyclist']). Only used when with_predictions=False. Ask the user for these before calling export."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "import_from_label_studio",
                "description": "Download completed annotations from Label Studio for a previously exported dataset and save as a new FiftyOne dataset named <dataset_name>_labeled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the original FiftyOne dataset that was exported to Label Studio."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "start_msight_pipeline",
                "description": (
                    "Start the MSight_Vision RF-DETR camera-detection pipeline (redis, "
                    "video_source, rfdetr_detector, detection_viewer) as detached Docker "
                    "containers, driven from whatever MSight_Vision checkout MSIGHT_VISION_PATH "
                    "points at (this repo does not vendor a copy of MSight_Vision's docker files). "
                    "Provide exactly one of video_input (path to a video file or folder of .mp4 "
                    "files, must exist on this host — not a URL or upload) or rtsp_url (a live "
                    "RTSP stream URL; the primary intended mode for real deployments, though still "
                    "work-in-progress in MSight_Vision so treat failures here as less predictable "
                    "than the file-path mode). Requires Docker Engine + Compose plugin on this host. "
                    "GPU vs CPU mode is auto-detected (nvidia-smi) — no GPU means this repo's own "
                    "CPU compose override is applied automatically, nothing to specify. "
                    "Returns a friendly error for known failure modes instead of raw docker output: "
                    "missing/misconfigured MSIGHT_VISION_PATH, a malformed line in MSight_Vision's "
                    "own .env, or port 6379 already bound by a host redis-server. On success, "
                    "includes the web viewer URL (port 9010)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_input": {"type": "string", "description": "Absolute path to a video file or folder of .mp4 files on this host."},
                        "rtsp_url": {"type": "string", "description": "Live RTSP stream URL. Takes priority over video_input if both are set."},
                        "sensor_name": {"type": "string", "description": "Logical camera/sensor name. Optional — MSight_Vision defaults to gs_mcity_1 if omitted."},
                        "build": {"type": "boolean", "description": "Rebuild Docker images from source before starting (default false — a plain `docker compose up -d` using the already-built/published image, no rebuild output). Only set true if MSight_Vision's own source has changed and the running image is stale; a rebuild prints noisy 'exporting layers'-style BuildKit output that looks alarming but is normal — avoid triggering it unless actually needed."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "select_msight_mode",
                "description": (
                    "Call this exactly once, right after the user answers STEP 1's Demo-vs-"
                    "run-your-own-pipeline question — before showing the checklist (custom) or "
                    "collecting a source (Demo). Records which mode is active so the consent "
                    "behavior around start_msight_pipeline is applied consistently for the rest "
                    "of the session: Demo starts immediately once a source is known, no "
                    "confirmation exchange; 'run your own pipeline' always shows a consent "
                    "summary first. Do not call this again just to keep going within the mode "
                    "already active — only if the user explicitly switches modes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["demo", "custom"], "description": "'demo' or 'custom' (custom = 'run your own pipeline')."}
                    },
                    "required": ["mode"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop_msight_pipeline",
                "description": "Stop the MSight_Vision pipeline containers (docker compose down). Does not delete the MSight_Vision checkout or its .env.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "remove_volumes": {"type": "boolean", "description": "If true, also removes anonymous volumes (docker compose down -v). Default false."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_msight_status",
                "description": "Report the current state (running/exited/etc.) of each MSight_Vision pipeline container, plus the web viewer URL if reachable.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_msight_logs",
                "description": "Fetch recent MSight_Vision pipeline logs, e.g. to diagnose why detections aren't showing up in the viewer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "Optional: one of redis, video_source, rfdetr_detector, detection_viewer. Omit for logs from all services."},
                        "tail": {"type": "integer", "description": "Number of trailing log lines to fetch per service (default 200, max 2000)."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_msight_calibration_status",
                "description": (
                    "Check whether the user's own camera calibration is in place for the live "
                    "MSight_Vision pipeline, or if it's still the shipped demo calibration. Call "
                    "this if the user asks about calibration status directly — SESSION_STATE's "
                    "msight_checklist already reports the same thing every turn "
                    "(calibration=default/user-uploaded/missing/partial/unknown), so you usually "
                    "don't need to call this explicitly; prefer reading that first."
                ),
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "start_msight_recording",
                "description": (
                    "Start local video recording — records the ANNOTATED feed (bounding boxes, "
                    "class labels, confidence scores drawn in by a frame-annotator node this "
                    "process launches), never raw/unannotated video: recording exactly what the "
                    "user fed in, with no detection output, isn't useful. Tracked host subprocesses "
                    "(not Docker containers). Safe to call before start_msight_pipeline too: if the "
                    "pipeline isn't running yet, this doesn't launch anything — it saves the request "
                    "and returns status='deferred'; recording is launched automatically, with no "
                    "further tool call needed from you, the moment the pipeline containers start — "
                    "but since it needs rfdetr_detector's detection output (not just raw camera "
                    "frames) to draw anything, it may sit idle for a few seconds after that while "
                    "the detector finishes loading its model, same as any other startup wait. No "
                    "AWS credentials needed; this only writes to local disk. Does not require "
                    "start_msight_archiving to be active or vice versa — independent sinks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensor_name": {
                            "type": "string",
                            "description": "Sensor name to subscribe to. Should match the sensor_name used in start_msight_pipeline, if one was given. Optional — defaults to gs_mcity_1."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "start_msight_archiving",
                "description": (
                    "Start pushing recorded video to an S3 bucket — pushes the ANNOTATED feed "
                    "(bounding boxes/labels/scores drawn in), never raw video, same reasoning as "
                    "start_msight_recording. A tracked host subprocess, independent of "
                    "start_msight_recording (does not require it to be active first) and safe to "
                    "call before start_msight_pipeline too: if the pipeline isn't running yet, this "
                    "doesn't launch anything — it saves the request and returns status='deferred'; "
                    "archiving is launched automatically, with no further tool call needed from "
                    "you, the moment the pipeline containers start — but may sit idle briefly after "
                    "that until rfdetr_detector actually begins publishing detections. Requires "
                    "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY to already be configured in this "
                    "host's .env — if the tool returns an error about missing credentials, tell "
                    "the user plainly rather than guessing at a fix."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "s3_bucket": {
                            "type": "string",
                            "description": "Name of the S3 bucket to push recorded video to. Must come from the user — never invent or guess a bucket name."
                        },
                        "s3_prefix": {
                            "type": "string",
                            "description": "Optional S3 key prefix (folder path) within the bucket."
                        }
                    },
                    "required": ["s3_bucket"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop_msight_recording",
                "description": "Stop local video recording. Independent of archiving — if archiving is still active, the shared upstream aggregator node keeps running for it.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop_msight_archiving",
                "description": "Stop pushing video to S3. Independent of local recording — if recording is still active, the shared upstream aggregator node keeps running for it.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_msight_record_archive_status",
                "description": (
                    "Report whether local recording and/or S3 archiving are currently running. "
                    "SESSION_STATE's msight_checklist already reports recording/archiving status "
                    "every turn from the same underlying state, so you usually don't need to call "
                    "this explicitly; prefer reading that first."
                ),
                "parameters": {"type": "object", "properties": {}}
            }
        },

]
"""Safe ML and campaign-control tools exposed to the GoTrain agent."""

import json
import time

import torch
from strands import tool

from agent.feedback import build_campaign_report, build_feedback
from agent.campaign import CampaignRunner
from agent.experiments import (
    finalize_training,
    get_best_experiment,
    get_budget_status,
    load_experiments,
    load_campaign_experiments,
    reserve_training,
)

from ml import train_model as run_training
from ml.inspect_dataset import inspect_dataset as inspect_ml_dataset
from ml.model_registry import (
    get_model_spec,
    list_eligible_models as eligible_model_metadata,
    list_model_options,
    model_runtime_status,
    validate_model_for_dataset,
)
from ml.registry import list_dataset_specs


@tool
def agent_capabilities() -> dict:
    """Return GoTrain's implemented capabilities and registered datasets."""
    return {
        "agent": "GoTrain",
        "status": "integration_ready",
        "available": [
            "Claude Sonnet via Amazon Bedrock",
            "Strands tool calling",
            "GPU inspection",
            "registry-driven classification and segmentation",
            "validation-only model selection",
            "Feedback experiments E0-E3",
            "Adaptive E3 diagnostics",
            "Campaign-isolated experiment history",
            "Persistent experiment reservations",
            "Budget and duplicate checks",
            "Guarded training tool factory",
            "real autonomous multi-experiment campaigns",
        ],
        "datasets": list_dataset_specs(),
        "pending": ["human-authorized final one-time test evaluation"],
    }


@tool
def inspect_gpu() -> dict:
    """Check GPU availability, name, memory, and CUDA version."""
    if not torch.cuda.is_available():
        return {"gpu_available": False, "message": "CUDA GPU is not available."}
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(device),
        "total_memory_gb": round(total_bytes / (1024 ** 3), 2),
        "free_memory_gb": round(free_bytes / (1024 ** 3), 2),
        "cuda_version": torch.version.cuda,
    }


@tool
def list_supported_datasets() -> dict:
    """List datasets whose contracts are registered and safe for training."""
    return {"datasets": list_dataset_specs()}


@tool
def list_eligible_models(dataset_id: str) -> dict:
    """Return factual eligible and deferred model metadata for a registered dataset."""
    return {
        "dataset_id": dataset_id,
        "eligible": eligible_model_metadata(dataset_id),
        "all_options": list_model_options(dataset_id),
    }


@tool
def inspect_dataset(dataset_id: str) -> dict:
    """Inspect one registered dataset's train/validation data without using test data."""
    return inspect_ml_dataset(dataset_id=dataset_id)


@tool
def train_registered_model(
    dataset_id: str,
    model_id: str,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.0001,
    pretrained: bool = True,
    seed: int = 42,
    image_size: int | None = None,
) -> dict:
    """Train one safe experiment for a registered classification or segmentation dataset.

    Inspect the dataset first. The registry selects the compatible task type, model,
    channels, class count, and metric. This tool never accesses the test split.
    """
    selected_model = validate_model_for_dataset(model_id, dataset_id)
    model_spec = get_model_spec(selected_model)
    runtime = model_runtime_status(selected_model)
    if not runtime["eligible"]:
        return {
            "status": "rejected",
            "error_type": "MODEL_NOT_ELIGIBLE",
            "message": runtime["reason"],
        }
    if not 1 <= epochs <= 20:
        return {"status": "rejected", "message": "epochs must be between 1 and 20."}
    if not 1 <= batch_size <= model_spec.conservative_batch_size:
        return {
            "status": "rejected",
            "message": (
                f"batch_size must be between 1 and "
                f"{model_spec.conservative_batch_size} for {selected_model}."
            ),
        }
    return run_training(
        dataset_id=dataset_id,
        config={
            "model": selected_model,
            "pretrained": pretrained,
            "input_mode": "registry",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": "auto",
            "mixed_precision": True,
            "image_size": image_size,
        },
    )


# ============================================
# TOOL 3: E3 ADAPTIVE DIAGNOSTICS
# ============================================

def make_diagnostics_tool(experiment: dict):
    """
    Create an E3-only diagnostic tool.

    Only approved diagnostics from validation
    experiments are accessible.
    """

    if experiment.get("evaluation_split") != "validation":
        raise ValueError(
            "Diagnostics must come from validation data."
        )

    full_feedback = build_feedback(
        version="E3",
        experiment=experiment,
        remaining_budget=0,
        request_details=True,
    )

    diagnostics = full_feedback.get(
        "detailed_diagnostics", {}
    )

    requested = []
    max_requests = 2

    @tool
    def request_diagnostics(diagnostic_type: str) -> dict:
        """
        Retrieve additional validation diagnostics.

        Args:
            diagnostic_type: per_class_metrics,
                confusion_matrix,
                learning_rate_history,
                or gradient_norms.
        """

        allowed = [
            "per_class_metrics",
            "confusion_matrix",
            "learning_rate_history",
            "gradient_norms",
        ]

        if diagnostic_type not in allowed:
            return {
                "status": "error",
                "message": "Unsupported diagnostic type.",
                "allowed": allowed,
            }

        if diagnostic_type not in diagnostics:
            return {
                "status": "unavailable",
                "message": "Diagnostic was not recorded.",
                "available": list(diagnostics.keys()),
            }

        if diagnostic_type in requested:
            return {
                "status": "duplicate",
                "message": "Diagnostic already requested.",
            }

        if len(requested) >= max_requests:
            return {
                "status": "budget_exhausted",
                "message": "Maximum diagnostic requests reached.",
            }

        data = diagnostics[diagnostic_type]

        if len(json.dumps(data, default=str)) > 4000:
            return {
                "status": "too_large",
                "message": "Diagnostic exceeds response size limit.",
            }

        requested.append(diagnostic_type)

        return {
            "status": "success",
            "experiment_id": experiment.get("experiment_id"),
            "diagnostic_type": diagnostic_type,
            "data": data,
            "requests_remaining": max_requests - len(requested),
        }

    return request_diagnostics


# ============================================
# TOOL 4: EXPERIMENT HISTORY (E2/E3)
# ============================================

def make_history_tool(
    version: str,
    dataset: str,
    seed: int,
    campaign_id: str,
):
    """
    Create a history tool restricted to one
    feedback version, campaign, dataset, and seed.
    """

    if version not in ("E2", "E3"):
        raise ValueError(
            "Enhanced history is available only in E2/E3."
        )

    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id is required.")

    @tool
    def get_history(limit: int = 10) -> dict:
        """
        Retrieve recent experiments from this trial.

        Args:
            limit: Number of recent experiments (1 to 10).
        """

        if type(limit) is not int or not 1 <= limit <= 10:
            return {
                "status": "rejected",
                "error_code": "INVALID_LIMIT",
                "message": "Limit must be between 1 and 10.",
            }

        records = load_experiments(
            version,
            campaign_id=campaign_id,
            dataset=dataset,
            seed=seed,
        )

        # Sort chronologically before applying limit.
        records = sorted(
            records,
            key=lambda run: (
                run.get(
                    "started_at",
                    run.get("recorded_at", ""),
                ),
                str(run.get("experiment_id", "")),
            ),
        )

        recent = records[-limit:]

        return {
            "status": "success",
            "campaign_id": campaign_id,
            "feedback_version": version,
            "dataset": dataset,
            "seed": seed,
            "count": len(recent),
            "experiments": [
                {
                    "experiment_id": run.get("experiment_id"),
                    "configuration": run.get("configuration"),
                    "validation_score": run.get(
                        "validation_score"
                    ),
                    "status": run.get("status"),
                    "runtime_seconds": run.get(
                        "runtime_seconds"
                    ),
                }
                for run in recent
            ],
        }

    return get_history


# ============================================
# TOOL 5: GUARDED ML TRAINING
# ============================================

def make_training_tool(
    train_fn,
    version: str,
    dataset: str,
    seed: int,
    campaign_id: str,
    max_experiments: int = 3,
    max_gpu_seconds=None,
    allowed_values=None,
):
    """
    Wrap the existing ML trainer with budget,
    duplicate, and campaign protections.

    train_fn must accept:
        train_fn(
            configuration,
            evaluation_split="validation",
            max_runtime_seconds=None,
        )

    It must return a JSON-serializable dictionary:
        {
            "status": "completed",
            "evaluation_split": "validation",
            "validation_score": 0.82,
            "train_loss": [...],
            "val_loss": [...],
            "validation_metric_curve": [...]
        }

    The actual trainer is responsible for enforcing
    the validation/test dataset separation and any
    hard runtime deadline.
    """

    if not callable(train_fn):
        raise ValueError("train_fn must be callable.")

    if version not in ("E0", "E1", "E2", "E3"):
        raise ValueError("Invalid feedback version.")

    @tool
    def train_model(configuration: dict) -> dict:
        """
        Train a model using the approved experiment budget.

        Args:
            configuration: Architecture and hyperparameters
                to use for the next training attempt.
        """

        # 1. Reserve before touching GPU.
        reservation = reserve_training(
            version=version,
            campaign_id=campaign_id,
            dataset=dataset,
            seed=seed,
            configuration=configuration,
            max_experiments=max_experiments,
            max_gpu_seconds=max_gpu_seconds,
            allowed_values=allowed_values,
        )

        if reservation["status"] != "reserved":
            return reservation

        experiment_id = reservation["experiment_id"]

        # 2. Only the training callback performs GPU work.
        started = time.perf_counter()

        try:
            result = train_fn(
                configuration,
                evaluation_split="validation",
                max_runtime_seconds=reservation[
                    "max_runtime_seconds"
                ],
            )

            if not isinstance(result, dict):
                raise ValueError(
                    "Trainer must return a dictionary."
                )

            if result.get("evaluation_split") != "validation":
                raise ValueError(
                    "Trainer did not return validation-only results."
                )

            if any(
                key.lower().startswith("test")
                for key in result
            ):
                raise ValueError(
                    "Trainer returned prohibited test results."
                )

            if result.get("status", "completed") != "completed":
                raise RuntimeError(
                    result.get("error", "Training failed.")
                )

            result = {
                **result,
                "status": "completed",
            }

        except Exception as exc:
            result = {
                "status": "failed",
                "evaluation_split": "validation",
                "error": str(exc),
            }

        # 3. Record success or failure permanently.
        result["runtime_seconds"] = (
            time.perf_counter() - started
        )

        final_record = finalize_training(
            version=version,
            campaign_id=campaign_id,
            dataset=dataset,
            seed=seed,
            experiment_id=experiment_id,
            result=result,
        )

        # 4. Prepare the appropriate observation profile.
        history = load_experiments(
            version,
            campaign_id=campaign_id,
            dataset=dataset,
            seed=seed,
        )

        previous_history = [
            run for run in history
            if run.get("experiment_id") != experiment_id
        ]

        budget = get_budget_status(
            version,
            campaign_id,
            dataset,
            seed,
            max_experiments=max_experiments,
            max_gpu_seconds=max_gpu_seconds,
        )

        feedback = build_feedback(
            version=version,
            experiment=final_record,
            remaining_budget=budget["remaining_experiments"],
            history=previous_history,
        )

        # Do not return the full underlying record:
        # E0/E1 must not receive E2/E3-only information.
        return {
            "status": final_record["status"],
            "campaign_id": campaign_id,
            "experiment_id": experiment_id,
            "feedback": feedback,
        }

    return train_model


def _make_dataset_training_tool(
    *,
    version: str,
    dataset_id: str,
    seed: int,
    campaign_id: str,
    max_experiments: int = 3,
    allowed_values=None,
):
    """Connect campaign controls to the real registry-driven ML trainer.

    GPU-time deadlines are intentionally not accepted here because the current
    trainer cannot safely interrupt an epoch at a wall-clock deadline. Experiment
    count, duplicate configuration, validation-only use, and persistent history
    are enforced by ``make_training_tool``.
    """
    dataset_spec = next(
        item for item in list_dataset_specs() if item["dataset_id"] == dataset_id
    )

    def train_registered_configuration(
        configuration: dict,
        *,
        evaluation_split: str,
        max_runtime_seconds=None,
    ) -> dict:
        if evaluation_split != "validation":
            raise ValueError("Only validation evaluation is supported.")
        if max_runtime_seconds is not None:
            raise ValueError(
                "A hard GPU-time deadline is not supported by the current trainer; "
                "use an experiment-count budget."
            )

        model_id = configuration["architecture"]
        selected_model = validate_model_for_dataset(model_id, dataset_id)
        runtime = model_runtime_status(selected_model)
        if not runtime["eligible"]:
            raise ValueError(runtime["reason"])

        training_config = {
            "model": selected_model,
            "input_mode": "registry",
            "epochs": configuration["epochs"],
            "batch_size": configuration["batch_size"],
            "learning_rate": configuration["learning_rate"],
            "weight_decay": configuration.get("weight_decay", 0.0001),
            "pretrained": configuration.get("pretrained", True),
            "image_size": configuration.get("image_size"),
            "seed": seed,
            "device": "auto",
            "mixed_precision": True,
        }
        result = run_training(dataset_id=dataset_id, config=training_config)
        if result.get("status") != "completed":
            raise RuntimeError(result.get("message", "Registry training failed."))

        with open(result["metrics_path"], encoding="utf-8") as metrics_file:
            history = json.load(metrics_file)
        metric_key = f"val_{result['primary_metric']}"
        return {
            "status": "completed",
            "evaluation_split": "validation",
            "validation_score": result["best_val_score"],
            "train_loss": [epoch["train_loss"] for epoch in history],
            "val_loss": [epoch["val_loss"] for epoch in history],
            "validation_metric_curve": [epoch[metric_key] for epoch in history],
            "higher_is_better": True,
            "detailed_diagnostics": {
                "learning_rate_history": [epoch["learning_rate"] for epoch in history]
            },
            "ml_experiment_id": result["experiment_id"],
            "checkpoint_path": result["checkpoint_path"],
            "native_resolution": result["native_resolution"],
            "effective_resolution": result["effective_resolution"],
            "primary_metric": dataset_spec["primary_metric"],
        }

    return make_training_tool(
        train_fn=train_registered_configuration,
        version=version,
        dataset=dataset_id,
        seed=seed,
        campaign_id=campaign_id,
        max_experiments=max_experiments,
        max_gpu_seconds=None,
        allowed_values=allowed_values,
    )


def execute_registered_training(dataset_id, configuration, seed, training_fn=run_training):
    """Run the real ML API and translate only its factual artifacts."""
    selected_model = validate_model_for_dataset(configuration["architecture"], dataset_id)
    model_spec = get_model_spec(selected_model)
    runtime = model_runtime_status(selected_model)
    if not runtime["eligible"]:
        raise ValueError(runtime["reason"])
    if configuration["batch_size"] > model_spec.conservative_batch_size:
        raise ValueError(
            f"batch_size exceeds the {model_spec.conservative_batch_size} limit "
            f"for {selected_model}."
        )
    training_config = {
        "model": selected_model,
        "input_mode": "registry",
        "epochs": configuration["epochs"],
        "batch_size": configuration["batch_size"],
        "learning_rate": configuration["learning_rate"],
        "weight_decay": configuration.get("weight_decay", 0.0001),
        "pretrained": configuration.get("pretrained", True),
        "image_size": configuration.get("image_size"),
        "seed": seed,
        "device": "auto",
        "mixed_precision": True,
    }
    result = training_fn(dataset_id=dataset_id, config=training_config)
    if result.get("status") != "completed":
        return {
            "status": "failed",
            "evaluation_split": "validation",
            "error": result.get("message", "Real ML training failed."),
            "ml_experiment_id": result.get("experiment_id"),
            "training_seconds": result.get("training_seconds"),
            "log_path": result.get("log_path"),
            "native_resolution": result.get("native_resolution"),
            "effective_resolution": result.get("effective_resolution"),
        }
    with open(result["metrics_path"], encoding="utf-8") as metrics_file:
        history = json.load(metrics_file)
    metric_key = f"val_{result['primary_metric']}"
    return {
        "status": "completed",
        "evaluation_split": "validation",
        "validation_score": result["best_val_score"],
        "train_loss": [epoch["train_loss"] for epoch in history],
        "val_loss": [epoch["val_loss"] for epoch in history],
        "validation_metric_curve": [epoch[metric_key] for epoch in history],
        "higher_is_better": True,
        "detailed_diagnostics": {
            "learning_rate_history": [epoch["learning_rate"] for epoch in history]
        },
        "ml_experiment_id": result["experiment_id"],
        "checkpoint_path": result["checkpoint_path"],
        "metrics_path": result["metrics_path"],
        "native_resolution": result["native_resolution"],
        "effective_resolution": result["effective_resolution"],
        "primary_metric": result["primary_metric"],
        "training_seconds": result["training_seconds"],
        "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb"),
        "device": result["device"],
    }


def make_registered_training_tool(
    *,
    version: str,
    campaign_id: str,
    seed: int,
    max_experiments: int,
    required_dataset: str | None = None,
    training_fn=run_training,
    attempt_log: list[dict] | None = None,
):
    """Create the only production tool that launches campaign GPU training."""
    runner = CampaignRunner(
        version=version,
        campaign_id=campaign_id,
        seed=seed,
        max_experiments=max_experiments,
        executor=lambda dataset_id, configuration, run_seed: execute_registered_training(
            dataset_id, configuration, run_seed, training_fn=training_fn
        ),
    )

    @tool
    def run_campaign_experiment(
        dataset_id: str,
        model_id: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 0.0001,
        pretrained: bool = True,
        image_size: int | None = None,
    ) -> dict:
        """Run one real train/validation experiment within the campaign budget."""
        if required_dataset is not None and dataset_id != required_dataset:
            response = {
                "status": "rejected",
                "error_code": "DATASET_NOT_ALLOWED",
                "message": f"This campaign is restricted to {required_dataset}.",
            }
            if attempt_log is not None:
                attempt_log.append({"configuration": None, "response": response})
            return response
        configuration = {
            "architecture": model_id,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "pretrained": pretrained,
            "image_size": image_size,
        }
        try:
            if not 1 <= epochs <= 20:
                raise ValueError("epochs must be between 1 and 20.")
            if batch_size < 1:
                raise ValueError("batch_size must be positive.")
            if learning_rate <= 0 or weight_decay < 0:
                raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
            if image_size is not None and image_size < 32:
                raise ValueError("image_size must be at least 32 when explicitly set.")
            selected_model = validate_model_for_dataset(model_id, dataset_id)
            model_spec = get_model_spec(selected_model)
            runtime = model_runtime_status(selected_model)
            if not runtime["eligible"]:
                raise ValueError(runtime["reason"])
            if batch_size > model_spec.conservative_batch_size:
                raise ValueError(
                    f"batch_size exceeds limit {model_spec.conservative_batch_size}."
                )
        except Exception as exc:
            response = {
                "status": "rejected",
                "error_code": "INVALID_MODEL_CONFIGURATION",
                "message": str(exc),
            }
            if attempt_log is not None:
                attempt_log.append({"configuration": configuration, "response": response})
            return response

        response = runner.run(dataset_id, configuration)
        if attempt_log is not None:
            attempt_log.append({"configuration": configuration, "response": response})
        return response

    return run_campaign_experiment


def make_campaign_history_tool(version: str, campaign_id: str, seed: int):
    @tool
    def get_campaign_history(limit: int = 10) -> dict:
        """Return persisted real runs from this campaign, newest last."""
        if not 1 <= limit <= 50:
            return {"status": "rejected", "message": "limit must be 1..50"}
        records = load_campaign_experiments(version, campaign_id, seed)[-limit:]
        return {
            "status": "success",
            "campaign_id": campaign_id,
            "experiments": [
                {
                    "experiment_id": run["experiment_id"],
                    "dataset_id": run["dataset"],
                    "configuration": run["configuration"],
                    "status": run["status"],
                    "validation_score": run.get("validation_score"),
                    "primary_metric": run.get("primary_metric"),
                    "runtime_seconds": run.get("runtime_seconds"),
                    "checkpoint_path": run.get("checkpoint_path"),
                    "error": run.get("error"),
                }
                for run in records
            ],
        }
    return get_campaign_history


def make_campaign_diagnostics_tool(version: str, campaign_id: str, seed: int):
    @tool
    def request_campaign_diagnostics(experiment_id: int, diagnostic_type: str) -> dict:
        """Return one available real diagnostic for an E3 campaign experiment."""
        if version != "E3":
            return {"status": "rejected", "message": "Diagnostics require E3."}
        records = load_campaign_experiments(version, campaign_id, seed)
        record = next(
            (run for run in records if run["experiment_id"] == experiment_id), None
        )
        if record is None:
            return {"status": "unavailable", "message": "Unknown experiment ID."}
        diagnostics = record.get("detailed_diagnostics") or {}
        if diagnostic_type not in diagnostics:
            return {
                "status": "unavailable",
                "message": "The real trainer did not record this diagnostic.",
                "available": sorted(diagnostics),
            }
        return {
            "status": "success",
            "experiment_id": experiment_id,
            "diagnostic_type": diagnostic_type,
            "data": diagnostics[diagnostic_type],
        }
    return request_campaign_diagnostics


def make_campaign_report_tool(version: str, campaign_id: str, seed: int):
    @tool
    def get_campaign_report() -> dict:
        """Return mode-filtered factual results and the best real checkpoint."""
        records = load_campaign_experiments(version, campaign_id, seed)
        datasets = sorted({run["dataset"] for run in records})
        best = None
        if len(datasets) == 1:
            best = get_best_experiment(version, campaign_id, datasets[0], seed)
        return build_campaign_report(version, campaign_id, records, best)
    return get_campaign_report

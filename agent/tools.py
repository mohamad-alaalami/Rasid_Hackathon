"""Safe ML and campaign-control tools exposed to the GoTrain agent."""

import json
import time

import torch
from strands import tool

from agent.feedback import build_campaign_report, build_feedback
from agent.autonomy import AutonomyPolicy, scope_rejection
from agent.campaign import CampaignRunner
from agent.policies import (
    DEFAULT_POLICY,
    check_campaign_policy,
    check_policy,
    launch_with_decision,
    record_rejection,
    validate_decision_record,
)
from agent.experiments import (
    _configuration_key,
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
from ml.components import PipelineError
from ml.config import TrainingConfig
from ml.registry import list_dataset_specs
from ml.solution import build_solution_spec, solution_exists, solution_id


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
            "composable pipeline components (scope S1)",
            "sandboxed generated custom losses (scope S2, Linux isolation runtime required)",
            "versioned, immutable solution specifications",
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
            "early_stopping": configuration.get("early_stopping", False),
            "early_stopping_patience": configuration.get(
                "early_stopping_patience", 3
            ),
            "early_stopping_min_delta": configuration.get(
                "early_stopping_min_delta", 0.0
            ),
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
            "requested_epochs": result.get(
                "requested_epochs", configuration["epochs"]
            ),
            "completed_epochs": result.get("completed_epochs", len(history)),
            "stopped_early": result.get("stopped_early", False),
            "stop_reason": result.get("stop_reason"),
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


def campaign_training_config(dataset_id, configuration, seed):
    """Translate a campaign configuration into ml.train_model's config mapping."""
    selected_model = validate_model_for_dataset(configuration["architecture"], dataset_id)
    training_config = {
        "model": selected_model,
        "input_mode": "registry",
        "epochs": configuration["epochs"],
        "batch_size": configuration["batch_size"],
        "learning_rate": configuration["learning_rate"],
        "weight_decay": configuration.get("weight_decay", 0.0001),
        "pretrained": configuration.get("pretrained", True),
        "image_size": configuration.get("image_size"),
        "early_stopping": configuration.get("early_stopping", False),
        "early_stopping_patience": configuration.get("early_stopping_patience", 3),
        "early_stopping_min_delta": configuration.get(
            "early_stopping_min_delta", 0.0
        ),
        "seed": seed,
        "device": "auto",
        "mixed_precision": True,
    }
    if configuration.get("pipeline") is not None:
        training_config["pipeline"] = configuration["pipeline"]
    return training_config


def campaign_solution(dataset_id, configuration, seed):
    """Validate every pipeline constraint and return (solution_id, spec)."""
    cfg = TrainingConfig.from_mapping(campaign_training_config(dataset_id, configuration, seed))
    cfg.validate_for_dataset(dataset_id)
    spec = build_solution_spec(dataset_id, cfg)
    return solution_id(spec), spec


def execute_registered_training(dataset_id, configuration, seed, training_fn=run_training,
                                solution_context=None):
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
    training_config = campaign_training_config(dataset_id, configuration, seed)
    extra = {} if solution_context is None else {"solution_context": solution_context}
    result = training_fn(dataset_id=dataset_id, config=training_config, **extra)
    if result.get("status") != "completed":
        return {
            "status": "failed",
            "evaluation_split": "validation",
            "error": result.get("message", "Real ML training failed."),
            "ml_experiment_id": result.get("experiment_id"),
            "solution_id": result.get("solution_id"),
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
        "requested_epochs": result.get(
            "requested_epochs", configuration["epochs"]
        ),
        "completed_epochs": result.get("completed_epochs", len(history)),
        "stopped_early": result.get("stopped_early", False),
        "stop_reason": result.get("stop_reason"),
        "solution_id": result.get("solution_id"),
    }


def check_run_arguments(dataset_id, model_id, epochs, batch_size, learning_rate, weight_decay,
                        image_size, early_stopping_patience, early_stopping_min_delta):
    """Deterministic argument, eligibility, and batch-limit checks; raises ValueError."""
    if not 1 <= epochs <= 20:
        raise ValueError("epochs must be between 1 and 20.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    if image_size is not None and image_size < 32:
        raise ValueError("image_size must be at least 32 when explicitly set.")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative.")
    selected_model = validate_model_for_dataset(model_id, dataset_id)
    model_spec = get_model_spec(selected_model)
    runtime = model_runtime_status(selected_model)
    if not runtime["eligible"]:
        raise ValueError(runtime["reason"])
    if batch_size > model_spec.conservative_batch_size:
        raise ValueError(
            f"batch_size exceeds limit {model_spec.conservative_batch_size}."
        )


def make_registered_training_tool(
    *,
    version: str,
    campaign_id: str,
    seed: int,
    max_experiments: int,
    required_dataset: str | None = None,
    training_fn=run_training,
    attempt_log: list[dict] | None = None,
    selection_policy: str = DEFAULT_POLICY,
    launch_gate=None,
    autonomy: AutonomyPolicy | None = None,
):
    """Create the only production tool that launches campaign GPU training.

    P0 exposes the original signature. P1 adds a required ``decision_record``
    that is validated before any reservation; both share one CampaignRunner.
    ``launch_gate`` (HITL) can block a valid launch before reservation.
    Scope S1 (``autonomy``) adds ``pipeline``, ``parent_solution_id``, and
    ``change_description``; the default policy (A2/S0) keeps the original tool.
    """
    check_policy(selection_policy)
    policy = autonomy or AutonomyPolicy()
    # Lineage is not part of the configuration (it does not change training),
    # so it reaches the trainer through this map keyed by configuration.
    lineage: dict[str, dict] = {}

    def solution_context(configuration):
        return {
            "campaign_id": campaign_id,
            "feedback_version": version,
            "selection_policy": selection_policy,
            "autonomy": policy.to_dict(),
            "proposer": "claude",
            **lineage.pop(_configuration_key(configuration), {}),
        }

    runner = CampaignRunner(
        version=version,
        campaign_id=campaign_id,
        seed=seed,
        max_experiments=max_experiments,
        executor=lambda dataset_id, configuration, run_seed: execute_registered_training(
            dataset_id, configuration, run_seed, training_fn=training_fn,
            solution_context=solution_context(configuration),
        ),
    )

    def attempt(dataset_id, model_id, epochs, batch_size, learning_rate,
                weight_decay, pretrained, image_size, early_stopping,
                early_stopping_patience, early_stopping_min_delta,
                decision_record=None, pipeline=None, parent_solution_id=None,
                change_description=None):
        identity = {
            "version": version,
            "campaign_id": campaign_id,
            "seed": seed,
            "dataset_id": dataset_id,
            "policy": selection_policy,
            "configuration": None,
            "decision_record": decision_record,
            "origin": "agent",
        }

        def reject(response):
            if attempt_log is not None:
                attempt_log.append({
                    "configuration": identity["configuration"],
                    "decision_record": identity["decision_record"],
                    "response": response,
                })
            return record_rejection(response, **identity)

        if required_dataset is not None and dataset_id != required_dataset:
            return reject({
                "status": "rejected",
                "error_code": "DATASET_NOT_ALLOWED",
                "message": f"This campaign is restricted to {required_dataset}.",
            })
        configuration = {
            "architecture": model_id,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "pretrained": pretrained,
            "image_size": image_size,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        }
        if pipeline is not None:
            configuration["pipeline"] = pipeline
        identity["configuration"] = configuration
        out_of_scope = scope_rejection(policy, pipeline)
        if out_of_scope:
            return reject(out_of_scope)
        mismatch = check_campaign_policy(campaign_id, seed, selection_policy)
        if mismatch:
            return reject(mismatch)
        if selection_policy == "P1":
            normalized, error = validate_decision_record(decision_record)
            if error:
                return reject(error)
            identity["decision_record"] = normalized
        try:
            check_run_arguments(dataset_id, model_id, epochs, batch_size, learning_rate,
                                weight_decay, image_size, early_stopping_patience,
                                early_stopping_min_delta)
        except Exception as exc:
            return reject({
                "status": "rejected",
                "error_code": "INVALID_MODEL_CONFIGURATION",
                "message": str(exc),
            })
        if pipeline is not None or parent_solution_id or policy.requires_proposal:
            try:
                sid, _ = campaign_solution(dataset_id, configuration, seed)
            except PipelineError as exc:
                return reject({
                    "status": "rejected",
                    "error_code": "INVALID_PIPELINE",
                    "errors": exc.errors,
                    "message": "Unsupported pipeline combination; no budget was used.",
                })
            except Exception as exc:
                return reject({"status": "rejected", "error_code": "INVALID_MODEL_CONFIGURATION",
                               "message": str(exc)})
            if parent_solution_id and not solution_exists(parent_solution_id):
                return reject({"status": "rejected", "error_code": "UNKNOWN_PARENT_SOLUTION",
                               "message": f"No solution {parent_solution_id!r} exists."})
            blocked = policy.check_launch(sid, campaign_id)
            if blocked:
                return reject(blocked)
            if parent_solution_id or change_description:
                lineage[_configuration_key(configuration)] = {
                    "parent_solution_id": parent_solution_id,
                    "change_description": change_description,
                }

        approval = None
        if launch_gate is not None:
            approval, blocked = launch_gate.check(dataset_id, configuration)
            if blocked:
                return reject(blocked)
        response = launch_with_decision(
            lambda: runner.run(dataset_id, configuration), **identity
        )
        if launch_gate is not None:
            launch_gate.record_launch(approval, response)
        if attempt_log is not None:
            attempt_log.append({
                "configuration": configuration,
                "decision_record": identity["decision_record"],
                "response": response,
            })
        return response

    if policy.composes_pipelines:
        return _composable_tool(attempt, selection_policy)

    if selection_policy == "P0":
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
            early_stopping: bool = False,
            early_stopping_patience: int = 3,
            early_stopping_min_delta: float = 0.0,
        ) -> dict:
            """Run one real train/validation experiment within the campaign budget."""
            return attempt(dataset_id, model_id, epochs, batch_size, learning_rate,
                           weight_decay, pretrained, image_size, early_stopping,
                           early_stopping_patience, early_stopping_min_delta)

        return run_campaign_experiment

    @tool
    def run_campaign_experiment(
        decision_record: dict,
        dataset_id: str,
        model_id: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 0.0001,
        pretrained: bool = True,
        image_size: int | None = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.0,
    ) -> dict:
        """Run one real train/validation experiment within the campaign budget.

        Args:
            decision_record: Required object with nonempty string fields
                observation, action, expected_signal, and cost_justification.
                Malformed records are rejected before any training or budget use.
            dataset_id: Registered dataset to train on.
            model_id: Eligible model from list_eligible_models.
            epochs: Training epochs, 1 to 20. Each run starts from pretrained
                weights; checkpoint resumption is not supported.
            batch_size: Positive batch size within the model limit.
            learning_rate: Positive learning rate.
            weight_decay: Nonnegative weight decay.
            pretrained: Use the registry's verified pretrained weights.
            image_size: Optional square resize; native resolution when omitted.
            early_stopping: Stop when the primary validation metric stalls;
                epochs then becomes the maximum.
            early_stopping_patience: Epochs without improvement before stopping (>=1).
            early_stopping_min_delta: Minimum improvement that resets patience (>=0).
        """
        return attempt(dataset_id, model_id, epochs, batch_size, learning_rate,
                       weight_decay, pretrained, image_size, early_stopping,
                       early_stopping_patience, early_stopping_min_delta,
                       decision_record)

    return run_campaign_experiment


def _composable_tool(attempt, selection_policy):
    """Scope S1 variants of run_campaign_experiment (P0 and P1)."""
    if selection_policy == "P0":
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
            early_stopping: bool = False,
            early_stopping_patience: int = 3,
            early_stopping_min_delta: float = 0.0,
            pipeline: dict | None = None,
            parent_solution_id: str | None = None,
            change_description: str | None = None,
        ) -> dict:
            """Run one real train/validation experiment within the campaign budget.

            Args:
                pipeline: Optional composed components (see list_pipeline_components):
                    input_view, normalization, augmentations, loss, optimizer, scheduler,
                    dropout. Omitted keys keep the defaults; unsupported combinations are
                    rejected before any budget is used.
                parent_solution_id: Optional earlier solution this one changes.
                change_description: Optional description of what changed and why.
            """
            return attempt(dataset_id, model_id, epochs, batch_size, learning_rate,
                           weight_decay, pretrained, image_size, early_stopping,
                           early_stopping_patience, early_stopping_min_delta, None,
                           pipeline, parent_solution_id, change_description)

        return run_campaign_experiment

    @tool
    def run_campaign_experiment(
        decision_record: dict,
        dataset_id: str,
        model_id: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 0.0001,
        pretrained: bool = True,
        image_size: int | None = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.0,
        pipeline: dict | None = None,
        parent_solution_id: str | None = None,
        change_description: str | None = None,
    ) -> dict:
        """Run one real train/validation experiment within the campaign budget.

        Args:
            decision_record: Required object with nonempty string fields
                observation, action, expected_signal, and cost_justification.
                Malformed records are rejected before any training or budget use.
            pipeline: Optional composed components (see list_pipeline_components):
                input_view, normalization, augmentations, loss, optimizer, scheduler,
                dropout. Omitted keys keep the defaults; unsupported combinations are
                rejected before any budget is used.
            parent_solution_id: Optional earlier solution this one changes.
            change_description: Optional description of what changed and why.
        """
        return attempt(dataset_id, model_id, epochs, batch_size, learning_rate,
                       weight_decay, pretrained, image_size, early_stopping,
                       early_stopping_patience, early_stopping_min_delta,
                       decision_record, pipeline, parent_solution_id, change_description)

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

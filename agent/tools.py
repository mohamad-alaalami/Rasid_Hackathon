
# agent/tools.py

"""
GoTrain Strands tools.

- Agent capability inspection
- GPU inspection
- E3 adaptive diagnostics
- E2/E3 campaign-isolated history
- Guarded training integration
"""

import json
import time

import torch

from strands import tool

from agent.feedback import build_feedback
from agent.experiments import (
    load_experiments,
    reserve_training,
    finalize_training,
    get_budget_status,
)


# ============================================
# TOOL 1: AGENT CAPABILITIES
# ============================================

@tool
def agent_capabilities() -> dict:
    """Return the current capabilities of GoTrain."""

    return {
        "agent": "GoTrain",
        "status": "integration_ready",
        "available": [
            "Claude Sonnet via Amazon Bedrock",
            "Strands tool calling",
            "GPU hardware inspection",
            "Feedback experiments E0-E3",
            "Adaptive E3 diagnostics",
            "Campaign-isolated experiment history",
            "Persistent experiment reservations",
            "Budget and duplicate checks",
            "Guarded training tool factory",
        ],
        "pending": [
            "Connect teammate's real ML training function",
            "Verify training-side test split isolation",
            "End-to-end automatic optimization",
        ],
    }


# ============================================
# TOOL 2: GPU INSPECTION
# ============================================

@tool
def inspect_gpu() -> dict:
    """Check GPU availability, memory, and CUDA."""

    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "message": "CUDA GPU is not available.",
        }

    device = torch.cuda.current_device()

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(device),
        "total_memory_gb": round(
            total_bytes / (1024 ** 3), 2
        ),
        "free_memory_gb": round(
            free_bytes / (1024 ** 3), 2
        ),
        "cuda_version": torch.version.cuda,
    }


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

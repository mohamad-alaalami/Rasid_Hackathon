
# agent/feedback.py

"""
GoTrain Feedback Experiment System

E0: Minimal feedback
E1: Training diagnostics
E2: Experiment memory
E3: Adaptive detailed diagnostics
"""

FEEDBACK_VERSIONS = {
    "E0": "Minimal feedback",
    "E1": "Training diagnostics",
    "E2": "Experiment memory",
    "E3": "Adaptive diagnostics",
}


def summarize_curve(values, higher_is_better=False):
    """Summarize epoch-by-epoch training metrics."""

    if not values:
        return None

    values = [float(v) for v in values]

    best_value = (
        max(values) if higher_is_better else min(values)
    )

    best_epoch = values.index(best_value) + 1

    if values[-1] < values[0]:
        trend = "decreasing"
    elif values[-1] > values[0]:
        trend = "increasing"
    else:
        trend = "stable"

    return {
        "first": values[0],
        "last": values[-1],
        "best": best_value,
        "best_epoch": best_epoch,
        "trend": trend,
        "epochs_recorded": len(values),
    }


def build_feedback(
    version,
    experiment,
    remaining_budget,
    history=None,
    request_details=False,
):
    """
    Build the information Claude receives
    after a training experiment.

    Only validation results may be used.
    """

    if version not in FEEDBACK_VERSIONS:
        raise ValueError(
            f"Unknown feedback version: {version}"
        )

    if remaining_budget < 0:
        raise ValueError(
            "Remaining budget cannot be negative."
        )

    if experiment.get("status") == "completed":
        if experiment.get("evaluation_split") != "validation":
            raise ValueError(
                "Completed experiments must use "
                "validation results only."
            )

    if request_details and version != "E3":
        raise ValueError(
            "Detailed diagnostics are only allowed in E3."
        )

    history = history or []

    # E0: Minimal feedback
    feedback = {
        "configuration": experiment.get("configuration"),
        "primary_metric": experiment.get("primary_metric"),
        "validation_score": experiment.get(
            "validation_score"
        ),
        "status": experiment.get("status"),
        "remaining_budget": remaining_budget,
    }

    # E1: Add training diagnostics
    if version in ["E1", "E2", "E3"]:

        feedback["training_diagnostics"] = {
            "train_loss": summarize_curve(
                experiment.get("train_loss", [])
            ),
            "validation_loss": summarize_curve(
                experiment.get("val_loss", [])
            ),
            "validation_metric_curve": summarize_curve(
                experiment.get("validation_metric_curve", []),
                higher_is_better=experiment.get(
                    "higher_is_better", True
                )
            ),
        }

    # E2: Add memory from previous experiments
    if version in ["E2", "E3"]:

        feedback["experiment_memory"] = [
            {
                "experiment_id": run.get("experiment_id"),
                "configuration": run.get("configuration"),
                "validation_score": run.get(
                    "validation_score"
                ),
                "status": run.get("status"),
            }
            for run in history[-10:]
        ]

    # E3: Allow requesting additional diagnostics
    if version == "E3":

        diagnostics = experiment.get("detailed_diagnostics") or {}
        feedback["detailed_diagnostics_available"] = sorted(diagnostics)

        if request_details:
            allowed_fields = [
                "per_class_metrics",
                "confusion_matrix",
                "learning_rate_history",
                "gradient_norms",
            ]

            feedback["detailed_diagnostics"] = {
                key: diagnostics[key]
                for key in allowed_fields
                if key in diagnostics
            }

    return feedback

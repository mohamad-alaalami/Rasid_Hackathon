"""Strands tools for composable pipelines and versioned solutions.

Added to a campaign only when its autonomy policy is not the default (scope S1
or authority A1), so default campaigns keep exactly their historical tools.
Solution inspection follows the E-profile history rules: E0/E1 see only the
most recently executed solution; E2/E3 see every solution in the campaign.
"""

from __future__ import annotations

from strands import tool

from agent.autonomy import AutonomyPolicy
from agent.experiments import load_campaign_experiments
from agent.tools import campaign_solution, check_run_arguments
from ml.components import PipelineError, describe_configuration_space
from ml.model_registry import compatible_models, get_model_spec
from ml.registry import get_dataset_spec
from ml.solution import campaign_solution_ids, load_solution, solution_exists
from ml.solution import propose_solution as record_proposal

HISTORY_MODES = {"E2", "E3"}


def _summary(sid, campaign_id, detail=False):
    solution = load_solution(sid)
    spec = solution["spec"]
    proposal = next((p for p in solution["proposals"] if p.get("campaign_id") == campaign_id),
                    solution["proposals"][0] if solution["proposals"] else {})
    runs = [run for run in solution["runs"] if run.get("campaign_id") == campaign_id]
    summary = {
        "solution_id": sid,
        "parent_solution_id": proposal.get("parent_solution_id"),
        "change_description": proposal.get("change_description"),
        "model": spec["model"]["model_id"],
        "input_view": spec["inputs"]["input_view"],
        "band_names": spec["inputs"]["band_names"],
        "normalization": spec["inputs"]["normalization"],
        "pipeline": spec["pipeline"],
        "training": spec["training"],
        "runs": [
            {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "primary_metric": (run.get("validation") or {}).get("primary_metric"),
                "best_val_score": (run.get("validation") or {}).get("best_val_score"),
                "best_epoch": (run.get("validation") or {}).get("best_epoch"),
                "training_seconds": (run.get("resources") or {}).get("training_seconds"),
                "completed_epochs": (run.get("resources") or {}).get("completed_epochs"),
                "checkpoint_path": (run.get("artifacts") or {}).get("checkpoint_path"),
                "checkpoint_sha256": (run.get("artifacts") or {}).get("checkpoint_sha256"),
                "error": run.get("error"),
            }
            for run in runs
        ],
    }
    if detail:
        summary["spec"] = spec
    return summary


def make_pipeline_tools(*, version, campaign_id, seed, autonomy: AutonomyPolicy,
                        selection_policy="P0", required_dataset=None):
    tools = []

    if autonomy.composes_pipelines:
        @tool
        def list_pipeline_components(dataset_id: str) -> dict:
            """List supported pipeline components, their parameters and defaults, and
            which input views, normalizations, and components each eligible model can
            use on this dataset."""
            try:
                dataset = get_dataset_spec(dataset_id)
            except ValueError as exc:
                return {"status": "rejected", "message": str(exc)}
            models = [get_model_spec(model_id) for model_id in compatible_models(dataset_id)]
            return {"status": "success", **describe_configuration_space(dataset, models)}

        tools.append(list_pipeline_components)

    @tool
    def propose_solution(
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
        """Validate and record a candidate solution without training or using budget.

        Returns its immutable solution_id. Launch it with run_campaign_experiment
        using the same arguments.

        Args:
            pipeline: Optional composed components (scope S1 only).
            parent_solution_id: Optional earlier solution this one changes.
            change_description: Optional description of what changed and why.
        """
        if required_dataset is not None and dataset_id != required_dataset:
            return {"status": "rejected", "error_code": "DATASET_NOT_ALLOWED",
                    "message": f"This campaign is restricted to {required_dataset}."}
        if pipeline is not None and not autonomy.composes_pipelines:
            return {"status": "rejected", "error_code": "SCOPE_NOT_PERMITTED",
                    "message": "Composing pipeline components requires scope S1."}
        configuration = {
            "architecture": model_id, "learning_rate": learning_rate, "epochs": epochs,
            "batch_size": batch_size, "weight_decay": weight_decay, "pretrained": pretrained,
            "image_size": image_size, "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        }
        if pipeline is not None:
            configuration["pipeline"] = pipeline
        try:
            check_run_arguments(dataset_id, model_id, epochs, batch_size, learning_rate,
                                weight_decay, image_size, early_stopping_patience,
                                early_stopping_min_delta)
            _, spec = campaign_solution(dataset_id, configuration, seed)
        except PipelineError as exc:
            return {"status": "rejected", "error_code": "INVALID_PIPELINE", "errors": exc.errors}
        except Exception as exc:
            return {"status": "rejected", "error_code": "INVALID_MODEL_CONFIGURATION",
                    "message": str(exc)}
        if parent_solution_id and not solution_exists(parent_solution_id):
            return {"status": "rejected", "error_code": "UNKNOWN_PARENT_SOLUTION",
                    "message": f"No solution {parent_solution_id!r} exists."}
        sid = record_proposal(spec, {
            "campaign_id": campaign_id, "feedback_version": version,
            "selection_policy": selection_policy, "autonomy": autonomy.to_dict(),
            "proposer": "claude", "parent_solution_id": parent_solution_id,
            "change_description": change_description,
        })
        return {"status": "proposed", "solution_id": sid, "spec": spec,
                "message": "Validated and recorded; no training was started."}

    @tool
    def inspect_solutions(solution_id: str | None = None) -> dict:
        """Inspect solutions from this campaign: specification, lineage, checkpoints,
        and validation-only results. Pass solution_id for the full specification."""
        records = load_campaign_experiments(version, campaign_id, seed)
        executed = [record.get("solution_id") for record in records if record.get("solution_id")]
        if version in HISTORY_MODES:
            visible = list(dict.fromkeys(campaign_solution_ids(campaign_id) + executed))
        else:
            visible = executed[-1:]
        if solution_id is not None:
            if solution_id not in visible:
                return {"status": "rejected", "error_code": "NOT_VISIBLE",
                        "message": f"{solution_id!r} is not visible under feedback mode {version}.",
                        "visible": visible}
            return {"status": "success", "solution": _summary(solution_id, campaign_id, True)}
        return {"status": "success", "feedback_version": version,
                "solutions": [_summary(sid, campaign_id) for sid in visible]}

    if autonomy.composes_pipelines or autonomy.requires_proposal:
        tools += [propose_solution, inspect_solutions]
    return tools

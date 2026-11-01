"""Dependency-free campaign state machine shared by production tools and tests."""

from __future__ import annotations

import time

from agent.experiments import (
    finalize_training,
    get_best_experiment,
    get_budget_status,
    load_campaign_experiments,
    reserve_training,
)
from agent.feedback import build_feedback


class CampaignRunner:
    def __init__(self, *, version, campaign_id, seed, max_experiments, executor,
                 max_gpu_seconds=None):
        self.version = version
        self.campaign_id = campaign_id
        self.seed = seed
        self.max_experiments = max_experiments
        self.executor = executor
        # Cumulative wall-clock training-time budget (seconds). Each launch reserves
        # all remaining time and the executor receives it as max_runtime_seconds.
        self.max_gpu_seconds = max_gpu_seconds

    def run(self, dataset_id, configuration, admit=None):
        reservation = reserve_training(
            self.version,
            self.campaign_id,
            dataset_id,
            self.seed,
            configuration,
            max_experiments=self.max_experiments,
            max_gpu_seconds=self.max_gpu_seconds,
            admit=admit,
        )
        if reservation["status"] != "reserved":
            return reservation
        started = time.perf_counter()
        deadline = {} if self.max_gpu_seconds is None else {
            "max_runtime_seconds": reservation["max_runtime_seconds"]}
        try:
            result = self.executor(dataset_id, configuration, self.seed, **deadline)
            if not isinstance(result, dict):
                raise TypeError("Training executor must return a dictionary.")
            if result.get("evaluation_split") != "validation":
                raise ValueError("Training executor must return validation-only results.")
            if result.get("status") not in {"completed", "failed"}:
                raise ValueError("Training executor returned an invalid status.")
        except Exception as exc:
            result = {
                "status": "failed",
                "evaluation_split": "validation",
                "error": str(exc),
            }
        result["runtime_seconds"] = time.perf_counter() - started
        record = finalize_training(
            self.version,
            self.campaign_id,
            dataset_id,
            self.seed,
            reservation["experiment_id"],
            result,
        )
        history = load_campaign_experiments(
            self.version, self.campaign_id, self.seed
        )
        budget = get_budget_status(
            self.version,
            self.campaign_id,
            dataset_id,
            self.seed,
            max_experiments=self.max_experiments,
            max_gpu_seconds=self.max_gpu_seconds,
        )
        return {
            "status": record["status"],
            "campaign_id": self.campaign_id,
            "experiment_id": record["experiment_id"],
            "feedback": build_feedback(
                self.version,
                record,
                budget["remaining_experiments"],
                history=[run for run in history if run["experiment_id"] != record["experiment_id"]],
            ),
            "error": record.get("error"),
        }

    def report(self):
        records = load_campaign_experiments(
            self.version, self.campaign_id, self.seed
        )
        datasets = sorted({record["dataset"] for record in records})
        best = (
            get_best_experiment(
                self.version, self.campaign_id, datasets[0], self.seed
            )
            if len(datasets) == 1 else None
        )
        return {
            "campaign_id": self.campaign_id,
            "feedback_version": self.version,
            "dataset_id": datasets[0] if len(datasets) == 1 else None,
            "experiments": records,
            "best_experiment_id": best.get("experiment_id") if best else None,
            "best_validation_score": best.get("validation_score") if best else None,
            "best_checkpoint_path": best.get("checkpoint_path") if best else None,
        }

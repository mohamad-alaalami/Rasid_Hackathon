"""Pure aggregation and reporting for real E0-E3 comparisons."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

MODES = ("E0", "E1", "E2", "E3")


def summarize_trial(*, mode, seed, campaign_id, records, attempts, agent_seconds,
                    decision_trace, token_usage=None, estimated_cost_usd=None):
    completed = [record for record in records if record["status"] == "completed"]
    first = records[0] if records else None
    first_score = first.get("validation_score") if first else None
    best = max(completed, key=lambda record: record["validation_score"], default=None)
    best_score = best.get("validation_score") if best else None
    improvement = (
        best_score - first_score
        if best_score is not None and first_score is not None else None
    )
    rejected = [attempt for attempt in attempts if attempt["response"].get("status") == "rejected"]
    duplicate_attempts = sum(
        attempt["response"].get("error_code") == "DUPLICATE_CONFIGURATION"
        for attempt in rejected
    )
    training_seconds = sum(
        float(record.get("training_seconds") or record.get("runtime_seconds") or 0)
        for record in records
    )
    autonomous_training_seconds = sum(
        float(record.get("training_seconds") or record.get("runtime_seconds") or 0)
        for record in records[1:]
    )
    return {
        "mode": mode,
        "seed": seed,
        "campaign_id": campaign_id,
        "initial_validation_score": first_score,
        "best_validation_score": best_score,
        "validation_improvement": improvement,
        "best_experiment_id": best.get("experiment_id") if best else None,
        "best_checkpoint_path": best.get("checkpoint_path") if best else None,
        "successful_runs": len(completed),
        "failed_runs": sum(record["status"] == "failed" for record in records),
        "total_experiments": len(records),
        "total_gpu_training_seconds": training_seconds,
        "duplicate_attempts": duplicate_attempts,
        "invalid_attempts": len(rejected) - duplicate_attempts,
        "agent_wall_seconds": agent_seconds,
        "agent_overhead_seconds": max(0.0, agent_seconds - autonomous_training_seconds),
        "token_usage": token_usage,
        "estimated_cost_usd": estimated_cost_usd,
        "decision_trace": decision_trace,
        "experiments": records,
        "attempts": attempts,
    }


def _mean_sd(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"mean": None, "stdev": None}
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_comparison(trials, metadata):
    for seed in metadata["seeds"]:
        matched = [trial for trial in trials if trial["seed"] == seed]
        modes_present = {trial["mode"] for trial in matched}
        if modes_present != set(MODES):
            raise ValueError(f"Seed {seed} does not contain all four feedback modes.")
        configurations = {
            json.dumps(trial["initial_configuration"], sort_keys=True)
            for trial in matched
        }
        campaign_ids = {trial["campaign_id"] for trial in matched}
        if len(configurations) != 1:
            raise ValueError(f"Seed {seed} has unmatched initial configurations.")
        if len(campaign_ids) != len(MODES):
            raise ValueError(f"Seed {seed} reuses a campaign ID across modes.")
    modes = {}
    for mode in MODES:
        subset = [trial for trial in trials if trial["mode"] == mode]
        modes[mode] = {
            "trials": len(subset),
            "validation_improvement": _mean_sd(
                trial["validation_improvement"] for trial in subset
            ),
            "best_validation_metric": _mean_sd(
                trial["best_validation_score"] for trial in subset
            ),
            "successful_runs": _mean_sd(trial["successful_runs"] for trial in subset),
            "total_gpu_training_seconds": _mean_sd(
                trial["total_gpu_training_seconds"] for trial in subset
            ),
            "agent_overhead_seconds": _mean_sd(
                trial["agent_overhead_seconds"] for trial in subset
            ),
            "estimated_cost_usd": _mean_sd(
                trial["estimated_cost_usd"] for trial in subset
            ),
            "failed_runs": sum(trial["failed_runs"] for trial in subset),
            "duplicate_attempts": sum(trial["duplicate_attempts"] for trial in subset),
            "invalid_attempts": sum(trial["invalid_attempts"] for trial in subset),
            "token_usage_available": all(trial["token_usage"] is not None for trial in subset),
            "estimated_cost_available": all(
                trial["estimated_cost_usd"] is not None for trial in subset
            ),
        }
    eligible = [
        mode for mode in MODES
        if modes[mode]["validation_improvement"]["mean"] is not None
    ]
    winner = None
    if eligible:
        winner = max(
            eligible,
            key=lambda mode: (
                modes[mode]["validation_improvement"]["mean"],
                -modes[mode]["failed_runs"],
                -modes[mode]["invalid_attempts"],
                -(modes[mode]["estimated_cost_usd"]["mean"] or 0.0),
                -(modes[mode]["total_gpu_training_seconds"]["mean"] or float("inf")),
            ),
        )
    return {
        "metadata": metadata,
        "selection_criterion": (
            "maximize mean validation improvement; ties favor fewer failures, "
            "fewer invalid attempts, then lower available estimated cost and GPU time"
        ),
        "winner": winner,
        "modes": modes,
        "trials": trials,
    }


def write_reports(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for mode, summary in report["modes"].items():
        rows.append({
            "mode": mode,
            "trials": summary["trials"],
            "improvement_mean": summary["validation_improvement"]["mean"],
            "improvement_stdev": summary["validation_improvement"]["stdev"],
            "best_metric_mean": summary["best_validation_metric"]["mean"],
            "successful_runs_mean": summary["successful_runs"]["mean"],
            "gpu_seconds_mean": summary["total_gpu_training_seconds"]["mean"],
            "agent_overhead_seconds_mean": summary["agent_overhead_seconds"]["mean"],
            "estimated_cost_usd_mean": summary["estimated_cost_usd"]["mean"],
            "failed_runs": summary["failed_runs"],
            "duplicate_attempts": summary["duplicate_attempts"],
            "invalid_attempts": summary["invalid_attempts"],
        })
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# GoTrain Feedback Comparison",
        "",
        f"Selection criterion: {report['selection_criterion']}.",
        "",
        "| Mode | Improvement mean ± SD | Best metric mean | Successful runs mean | GPU seconds mean | Cost USD mean | Failures | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {_fmt(row['improvement_mean'])} ± {_fmt(row['improvement_stdev'])} "
            f"| {_fmt(row['best_metric_mean'])} | {_fmt(row['successful_runs_mean'])} "
            f"| {_fmt(row['gpu_seconds_mean'])} | {_fmt(row['estimated_cost_usd_mean'])} "
            f"| {row['failed_runs']} | {row['invalid_attempts']} |"
        )
    lines.extend(["", f"**Selected mode:** {report['winner'] or 'unavailable'}", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "json": str(output / "comparison.json"),
        "csv": str(output / "comparison.csv"),
        "markdown": str(output / "comparison.md"),
    }


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"

"""Pure aggregation and reporting for real P0-vs-P1 selection-policy comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from agent.feedback_comparison import _fmt, _mean_sd
from agent.policies import SELECTION_POLICIES


def _summary(subset):
    return {
        "trials": len(subset),
        "best_validation_metric": _mean_sd(t["best_validation_score"] for t in subset),
        "validation_improvement": _mean_sd(t["validation_improvement"] for t in subset),
        "total_gpu_training_seconds": _mean_sd(t["total_gpu_training_seconds"] for t in subset),
        "agent_overhead_seconds": _mean_sd(t["agent_overhead_seconds"] for t in subset),
        "successful_runs": sum(t["successful_runs"] for t in subset),
        "failed_runs": sum(t["failed_runs"] for t in subset),
        "total_experiments": sum(t["total_experiments"] for t in subset),
        "rejected_decision_records": sum(t["rejected_decision_records"] for t in subset),
        "duplicate_attempts": sum(t["duplicate_attempts"] for t in subset),
        "invalid_configuration_attempts": sum(
            t["invalid_attempts"] - t["rejected_decision_records"] for t in subset
        ),
    }


def aggregate_policy_comparison(trials, metadata):
    """Check frozen common conditions, then summarize each policy descriptively."""
    for seed in metadata["seeds"]:
        matched = [trial for trial in trials if trial["seed"] == seed]
        if sorted(trial["selection_policy"] for trial in matched) != sorted(SELECTION_POLICIES):
            raise ValueError(f"Seed {seed} must contain exactly one P0 and one P1 trial.")
        for field in ("initial_configuration", "feedback_version", "dataset", "max_experiments"):
            if len({json.dumps(trial[field], sort_keys=True) for trial in matched}) != 1:
                raise ValueError(f"Seed {seed} has unmatched {field}.")
        if len({trial["campaign_id"] for trial in matched}) != len(matched):
            raise ValueError(f"Seed {seed} reuses a campaign ID across policies.")
    policies = {
        policy: _summary([trial for trial in trials if trial["selection_policy"] == policy])
        for policy in SELECTION_POLICIES
    }
    means = {
        policy: policies[policy]["validation_improvement"]["mean"]
        for policy in SELECTION_POLICIES
    }
    leader = None
    if all(value is not None for value in means.values()) and len(set(means.values())) > 1:
        leader = max(means, key=means.get)
    matched_seeds = len(metadata["seeds"])
    return {
        "metadata": metadata,
        "policies": policies,
        "descriptive_leader": leader,
        "winner_claimed": False,
        "statistical_note": (
            f"{matched_seeds} matched seed(s) per policy; no significance test was run. "
            "The leader only has the higher mean validation improvement over the shared "
            "starting run and is not an established winner."
        ),
        "trials": trials,
    }


def write_policy_reports(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "policy_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# GoTrain Selection-Policy Comparison (P0 vs P1)",
        "",
        f"Feedback profile: {report['metadata']['feedback_version']} | "
        f"dataset: {report['metadata']['dataset']} | "
        f"budget: {report['metadata']['max_experiments']} experiments",
        "",
        "| Policy | Best metric mean ± SD | Improvement mean ± SD | GPU s mean | "
        "Completed | Failed | Rejected decisions | Invalid configs | Duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, summary in report["policies"].items():
        best, gain = summary["best_validation_metric"], summary["validation_improvement"]
        lines.append(
            f"| {policy} | {_fmt(best['mean'])} ± {_fmt(best['stdev'])} "
            f"| {_fmt(gain['mean'])} ± {_fmt(gain['stdev'])} "
            f"| {_fmt(summary['total_gpu_training_seconds']['mean'])} "
            f"| {summary['successful_runs']} | {summary['failed_runs']} "
            f"| {summary['rejected_decision_records']} "
            f"| {summary['invalid_configuration_attempts']} | {summary['duplicate_attempts']} |"
        )
    lines += [
        "",
        f"Descriptive leader: {report['descriptive_leader'] or 'none'}. {report['statistical_note']}",
        "",
        "## Decision traces",
    ]
    for trial in report["trials"]:
        lines += ["", f"### {trial['selection_policy']} · seed {trial['seed']} · `{trial['campaign_id']}`", ""]
        for step in trial["decision_trace"]:
            config = step["configuration"] or {}
            measured = step["measured"] or {}
            lines.append(
                f"{step['attempt']}. [{step['origin']}] {config.get('architecture')} "
                f"lr={config.get('learning_rate')} epochs={config.get('epochs')} "
                f"bs={config.get('batch_size')} -> {step['outcome']['status']}"
                + (f" ({step['outcome']['error_code']})" if step["outcome"].get("error_code") else "")
                + (f", measured {measured.get('primary_metric')}={_fmt(measured.get('validation_score'))}"
                   if measured.get("validation_score") is not None else "")
            )
            if step["decision_record"] and isinstance(step["decision_record"], dict):
                for field, text in step["decision_record"].items():
                    lines.append(f"   - *{field}*: {text}")
    (output / "policy_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": str(output / "policy_comparison.json"),
        "markdown": str(output / "policy_comparison.md"),
    }

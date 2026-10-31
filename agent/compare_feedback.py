"""Run a controlled, real-training comparison of feedback modes E0-E3."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from strands import Agent
from strands.models import BedrockModel

from agent.campaign import CampaignRunner
from agent.experiments import load_campaign_experiments
from agent.feedback_comparison import MODES, aggregate_comparison, summarize_trial, write_reports
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    execute_registered_training,
    inspect_dataset,
    inspect_gpu,
    list_eligible_models,
    list_supported_datasets,
    make_campaign_diagnostics_tool,
    make_campaign_history_tool,
    make_campaign_report_tool,
    make_registered_training_tool,
)

OUTPUT_ROOT = Path("artifacts/feedback_comparison")


def _json_safe(value, depth=0):
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value), depth + 1)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    for method in ("model_dump", "to_dict", "dict"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            try:
                return _json_safe(candidate(), depth + 1)
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value), depth + 1)
    return str(value)


def _extract_token_usage(agent_result):
    safe = _json_safe(agent_result)

    def find(node):
        if isinstance(node, dict):
            lowered = {str(key).lower(): value for key, value in node.items()}
            token_keys = [key for key in lowered if "token" in key]
            if token_keys:
                return {key: lowered[key] for key in token_keys}
            for value in node.values():
                found = find(value)
                if found:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = find(value)
                if found:
                    return found
        return None

    return find(safe)


def _campaign_id(run_id, repeat_index, mode, seed):
    return f"feedback-{run_id}-r{repeat_index}-s{seed}-{mode.lower()}"


def _comparison_prompt(args, mode, campaign_id, initial_feedback):
    history_rule = {
        "E0": "You have no training diagnostics and no history tool.",
        "E1": "You have training curves but no previous-experiment history tool.",
        "E2": "You may use the campaign history tool, but no detailed diagnostics.",
        "E3": "You may use campaign history and request only recorded diagnostics.",
    }[mode]
    return f"""
You are participating in a controlled real-training feedback comparison.
Campaign: {campaign_id}; mode: {mode}; dataset: {args.dataset}; seed: {args.current_seed}.
The total campaign budget is {args.max_experiments} experiments. Experiment 1 was
already run with the common configuration below. Its mode-appropriate real feedback is:

{json.dumps(initial_feedback, indent=2)}

{history_rule}
Inspect the dataset and eligible models, then independently choose and launch the
remaining experiments through run_campaign_experiment. Use only real tool results.
Do not repeat the common configuration. Respect native resolution, compatibility,
pretrained-weight availability, batch limits, and the fixed budget. Failed and rejected
attempts must remain visible. Finish by calling get_campaign_report and explain each
decision and the best real checkpoint. Never access test data or invent missing values.
"""


def run_mode(args, run_id, output_dir, repeat_index, mode, seed):
    campaign_id = _campaign_id(run_id, repeat_index, mode, seed)
    initial_configuration = {
        "architecture": args.initial_model,
        "learning_rate": args.initial_learning_rate,
        "epochs": args.initial_epochs,
        "batch_size": args.initial_batch_size,
        "weight_decay": args.initial_weight_decay,
        "pretrained": True,
        "image_size": None,
    }
    forced_runner = CampaignRunner(
        version=mode,
        campaign_id=campaign_id,
        seed=seed,
        max_experiments=args.max_experiments,
        executor=execute_registered_training,
    )
    initial_response = forced_runner.run(args.dataset, initial_configuration)

    attempts = []
    tools = [
        inspect_gpu,
        list_supported_datasets,
        list_eligible_models,
        inspect_dataset,
        make_registered_training_tool(
            version=mode,
            campaign_id=campaign_id,
            seed=seed,
            max_experiments=args.max_experiments,
            required_dataset=args.dataset,
            attempt_log=attempts,
        ),
        make_campaign_report_tool(mode, campaign_id, seed),
    ]
    if mode in {"E2", "E3"}:
        tools.append(make_campaign_history_tool(mode, campaign_id, seed))
    if mode == "E3":
        tools.append(make_campaign_diagnostics_tool(mode, campaign_id, seed))

    agent = Agent(
        model=BedrockModel(model_id=args.model_id, region_name=args.region),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    )
    args.current_seed = seed
    prompt = _comparison_prompt(args, mode, campaign_id, initial_response["feedback"])
    started = time.perf_counter()
    agent_result = agent(prompt)
    agent_seconds = time.perf_counter() - started
    decision_trace = {
        "prompt": prompt,
        "agent_result": _json_safe(agent_result),
        "tool_attempts": attempts,
    }
    records = load_campaign_experiments(mode, campaign_id, seed)
    trial = summarize_trial(
        mode=mode,
        seed=seed,
        campaign_id=campaign_id,
        records=records,
        attempts=attempts,
        agent_seconds=agent_seconds,
        decision_trace=decision_trace,
        token_usage=_extract_token_usage(agent_result),
        estimated_cost_usd=None,
    )
    trial["initial_configuration"] = initial_configuration
    trial["token_cost_note"] = (
        "Token usage unavailable from the installed Strands result object."
        if trial["token_usage"] is None
        else "Token usage recorded; cost unavailable because no pricing schedule was configured."
    )
    trial_dir = output_dir / f"repeat_{repeat_index}" / mode
    trial_dir.mkdir(parents=True, exist_ok=False)
    (trial_dir / "trial.json").write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
    (trial_dir / "decision_trace.json").write_text(
        json.dumps(decision_trace, indent=2) + "\n", encoding="utf-8"
    )
    return trial


def build_parser():
    parser = argparse.ArgumentParser(description="Compare real GoTrain feedback modes")
    parser.add_argument("--dataset", default="m-pv4ger")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="First trial seed; repeats increment it.")
    parser.add_argument("--max-experiments", type=int, default=3)
    parser.add_argument("--initial-model", default="resnet18_rgb")
    parser.add_argument("--initial-epochs", type=int, default=2)
    parser.add_argument("--initial-batch-size", type=int, default=16)
    parser.add_argument("--initial-learning-rate", type=float, default=0.0003)
    parser.add_argument("--initial-weight-decay", type=float, default=0.0001)
    parser.add_argument("--region", default="eu-central-1")
    parser.add_argument("--model-id", default="eu.anthropic.claude-sonnet-5-5")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    return parser


def main():
    args = build_parser().parse_args()
    if args.repeats < 1 or args.max_experiments < 3:
        raise SystemExit("--repeats must be >=1 and --max-experiments must be >=3")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    output_dir = Path(args.output_root) / run_id
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite comparison directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "dataset": args.dataset,
        "feedback_modes": list(MODES),
        "repeats": args.repeats,
        "seeds": [args.seed + index for index in range(args.repeats)],
        "max_experiments": args.max_experiments,
        "initial_configuration": {
            "architecture": args.initial_model,
            "epochs": args.initial_epochs,
            "batch_size": args.initial_batch_size,
            "learning_rate": args.initial_learning_rate,
            "weight_decay": args.initial_weight_decay,
            "pretrained": True,
            "image_size": None,
        },
        "claude_model": args.model_id,
        "region": args.region,
        "shared_constraints": {
            "epochs": "1..20",
            "batch_size": "positive and no greater than the selected model registry limit",
            "learning_rate": "positive",
            "weight_decay": "nonnegative",
            "resolution": "dataset native by default; explicit override >=32 only",
            "splits": ["train", "valid"],
        },
        "feedback_tool_access": {
            "E0": ["real training", "final report"],
            "E1": ["real training with curve feedback", "final report"],
            "E2": ["real training with curve feedback", "isolated history", "final report"],
            "E3": ["real training with curve feedback", "isolated history", "recorded diagnostics", "final report"],
        },
        "cost_estimation": "unavailable unless an external pricing schedule is supplied",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    trials = []
    for repeat_index in range(args.repeats):
        seed = args.seed + repeat_index
        for mode in MODES:
            print(f"Running repeat={repeat_index} seed={seed} mode={mode}", flush=True)
            trials.append(run_mode(args, run_id, output_dir, repeat_index, mode, seed))
    report = aggregate_comparison(trials, metadata)
    paths = write_reports(report, output_dir)
    print(json.dumps({"run_id": run_id, "winner": report["winner"], "reports": paths}, indent=2))


if __name__ == "__main__":
    main()

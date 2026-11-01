"""Run a controlled, real-training comparison of selection policies P0 and P1.

Both policies share one frozen E-profile, dataset, seed, forced first run,
model registry, training tool, evaluator, and experiment budget. Only the
selection policy differs. E-profile comparisons stay in compare_feedback.py.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from strands import Agent
from strands.models import BedrockModel

from agent.campaign import CampaignRunner
from agent.compare_feedback import _comparison_prompt, _extract_token_usage, _json_safe
from agent.experiments import FEEDBACK_VERSIONS, load_campaign_experiments
from agent.feedback_comparison import summarize_trial
from agent.policies import (
    SELECTION_POLICIES,
    build_decision_trace,
    build_system_prompt,
    launch_with_decision,
)
from agent.policy_comparison import aggregate_policy_comparison, write_policy_reports
from research.resource_accounting import CampaignSession, campaign_resource_report, load_prices
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

OUTPUT_ROOT = Path("artifacts/policy_comparison")


def _campaign_id(run_id, repeat_index, feedback, policy, seed):
    return f"policy-{run_id}-r{repeat_index}-s{seed}-{feedback.lower()}-{policy.lower()}"


def run_policy(args, run_id, output_dir, repeat_index, policy, seed):
    mode = args.feedback
    campaign_id = _campaign_id(run_id, repeat_index, mode, policy, seed)
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
    # The shared starting run is not an agent decision, so it carries no record.
    initial_response = launch_with_decision(
        lambda: forced_runner.run(args.dataset, initial_configuration),
        version=mode, campaign_id=campaign_id, seed=seed, dataset_id=args.dataset,
        policy=policy, configuration=initial_configuration, decision_record=None,
        origin="common_start",
    )

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
            selection_policy=policy,
        ),
        make_campaign_report_tool(mode, campaign_id, seed),
    ]
    if mode in {"E2", "E3"}:
        tools.append(make_campaign_history_tool(mode, campaign_id, seed))
    if mode == "E3":
        tools.append(make_campaign_diagnostics_tool(mode, campaign_id, seed))

    agent = Agent(
        model=BedrockModel(model_id=args.model_id, region_name=args.region),
        system_prompt=build_system_prompt(policy),
        tools=tools,
    )
    args.current_seed = seed
    # The user prompt is identical for both policies; P1 rules live in the system
    # prompt and the training tool schema.
    prompt = _comparison_prompt(args, mode, campaign_id, initial_response["feedback"])
    started = time.perf_counter()
    with CampaignSession(campaign_id, model_id=args.model_id, label="policy_comparison") as session:
        agent_result = session.call(agent, prompt)
    agent_seconds = time.perf_counter() - started
    decision_trace = build_decision_trace(mode, campaign_id, seed)
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
    trial.update({
        "selection_policy": policy,
        "feedback_version": mode,
        "dataset": args.dataset,
        "max_experiments": args.max_experiments,
        "initial_configuration": initial_configuration,
        "resources": campaign_resource_report(campaign_id, args.prices_data, mode, seed),
        "rejected_decision_records": sum(
            attempt["response"].get("error_code") == "INVALID_DECISION_RECORD"
            for attempt in attempts
        ),
    })
    trial_dir = output_dir / f"repeat_{repeat_index}" / policy
    trial_dir.mkdir(parents=True, exist_ok=False)
    (trial_dir / "trial.json").write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
    (trial_dir / "agent_transcript.json").write_text(
        json.dumps({"prompt": prompt, "agent_result": _json_safe(agent_result)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return trial


def build_parser():
    parser = argparse.ArgumentParser(description="Compare real GoTrain selection policies P0/P1")
    parser.add_argument("--dataset", default="m-pv4ger")
    parser.add_argument("--feedback", choices=FEEDBACK_VERSIONS, default="E3",
                        help="One frozen feedback profile shared by both policies.")
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
    parser.add_argument("--prices", help="JSON file of explicit USD prices for cost reports.")
    return parser


def main():
    args = build_parser().parse_args()
    args.prices_data = load_prices(args.prices)
    if args.repeats < 1 or args.max_experiments < 2:
        raise SystemExit("--repeats must be >=1 and --max-experiments must be >=2")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    output_dir = Path(args.output_root) / run_id
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite comparison directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "dataset": args.dataset,
        "feedback_version": args.feedback,
        "selection_policies": list(SELECTION_POLICIES),
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
        "gpu_time_budget": (
            "not enforced: the real trainer cannot stop at a wall-clock deadline; "
            "measured GPU seconds are reported per policy instead"
        ),
        "checkpoint_resumption": "unsupported; every run starts from registry pretrained weights",
        "prices": args.prices_data,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    trials = []
    for repeat_index in range(args.repeats):
        seed = args.seed + repeat_index
        for policy in SELECTION_POLICIES:
            print(f"Running repeat={repeat_index} seed={seed} feedback={args.feedback} "
                  f"policy={policy}", flush=True)
            trials.append(run_policy(args, run_id, output_dir, repeat_index, policy, seed))
    report = aggregate_policy_comparison(trials, metadata)
    paths = write_policy_reports(report, output_dir)
    print(json.dumps({
        "run_id": run_id,
        "descriptive_leader": report["descriptive_leader"],
        "statistical_note": report["statistical_note"],
        "reports": paths,
    }, indent=2))


if __name__ == "__main__":
    main()

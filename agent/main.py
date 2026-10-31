"""CLI for real GoTrain campaigns and explicit one-run debugging."""

from __future__ import annotations

import argparse
import json

from strands import Agent
from strands.models import BedrockModel

from agent.experiments import create_campaign_id, get_best_experiment, load_campaign_experiments
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    agent_capabilities,
    inspect_dataset,
    inspect_gpu,
    list_eligible_models,
    list_supported_datasets,
    make_campaign_diagnostics_tool,
    make_campaign_history_tool,
    make_campaign_report_tool,
    make_registered_training_tool,
    train_registered_model,
)


def _bedrock_model(model_id: str, region: str):
    return BedrockModel(model_id=model_id, region_name=region)


def create_campaign_agent(args, campaign_id: str) -> Agent:
    tools = [
        agent_capabilities,
        inspect_gpu,
        list_supported_datasets,
        list_eligible_models,
        inspect_dataset,
        make_registered_training_tool(
            version=args.feedback,
            campaign_id=campaign_id,
            seed=args.seed,
            max_experiments=args.max_experiments,
            required_dataset=args.dataset,
        ),
        make_campaign_history_tool(args.feedback, campaign_id, args.seed),
        make_campaign_report_tool(args.feedback, campaign_id, args.seed),
    ]
    if args.feedback == "E3":
        tools.append(make_campaign_diagnostics_tool(args.feedback, campaign_id, args.seed))
    return Agent(
        model=_bedrock_model(args.model_id, args.region),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    )


def campaign_prompt(args, campaign_id: str) -> str:
    constraint = args.dataset or "Choose one registered dataset yourself after inspection."
    return f"""
Run a real autonomous GoTrain campaign. Campaign ID: {campaign_id}.
Feedback mode: {args.feedback}. Maximum real experiments: {args.max_experiments}.
Dataset constraint: {constraint}

Inspect the GPU and registered datasets. Select exactly one dataset and inspect its
task, native resolution, channels, classes, and primary validation metric. Discover
runtime-eligible compatible pretrained models. Run only real experiments through
run_campaign_experiment. Native resolution is the default; resolution overrides must
be deliberate and explained. Respect model batch limits.

After every run, reason only from returned validation feedback. Change a justified
model or hyperparameter and never repeat a configuration. Use campaign history for
real comparisons. In E3, request only diagnostics actually recorded. Continue until
the budget is exhausted or another run is not justified. Finally call
get_campaign_report and report every configuration, metric, runtime, failure,
checkpoint, and the best validation checkpoint.

Never access a test split. Never invent or estimate missing results. A failed run is
recorded as failed and is not evidence of model quality.
"""


def run_campaign(args) -> str:
    campaign_id = args.campaign_id or create_campaign_id()
    print(json.dumps({"mode": "campaign", "campaign_id": campaign_id}, indent=2))
    create_campaign_agent(args, campaign_id)(campaign_prompt(args, campaign_id))
    records = load_campaign_experiments(args.feedback, campaign_id, args.seed)
    datasets = sorted({record["dataset"] for record in records})
    best = (
        get_best_experiment(args.feedback, campaign_id, datasets[0], args.seed)
        if len(datasets) == 1 else None
    )
    report = {
        "campaign_id": campaign_id,
        "feedback_version": args.feedback,
        "dataset_id": datasets[0] if len(datasets) == 1 else None,
        "experiments": records,
        "best_experiment_id": best.get("experiment_id") if best else None,
        "best_validation_score": best.get("validation_score") if best else None,
        "best_checkpoint_path": best.get("checkpoint_path") if best else None,
    }
    print("\nFINAL PERSISTED CAMPAIGN REPORT")
    print(json.dumps(report, indent=2))
    return campaign_id


def run_smoke(args) -> None:
    if not args.dataset or not args.model:
        raise SystemExit("smoke mode requires --dataset and --model")
    agent = Agent(
        model=_bedrock_model(args.model_id, args.region),
        system_prompt=SYSTEM_PROMPT,
        tools=[inspect_gpu, list_supported_datasets, list_eligible_models,
               inspect_dataset, train_registered_model],
    )
    agent(f"""
Run exactly one real debugging experiment for dataset {args.dataset} and model
{args.model}. Inspect eligibility first. Use epochs={args.smoke_epochs},
batch_size={args.smoke_batch_size}, native resolution, pretrained weights, and seed
{args.seed}. Report only actual tool output and never access the test split.
""")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GoTrain real autonomous ML agent")
    parser.add_argument("--mode", choices=("campaign", "smoke"), default="campaign")
    parser.add_argument("--feedback", choices=("E0", "E1", "E2", "E3"), default="E3")
    parser.add_argument("--max-experiments", type=int, default=5)
    parser.add_argument("--campaign-id", help="Resume an existing campaign ID.")
    parser.add_argument("--dataset", help="Optional registered dataset restriction.")
    parser.add_argument("--model", help="Required model ID only in smoke mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--region", default="eu-central-1")
    parser.add_argument("--model-id", default="eu.anthropic.claude-sonnet-5-5")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_experiments < 1:
        raise SystemExit("--max-experiments must be positive")
    run_campaign(args) if args.mode == "campaign" else run_smoke(args)


if __name__ == "__main__":
    main()

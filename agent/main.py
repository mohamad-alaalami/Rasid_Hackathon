"""CLI for real GoTrain campaigns and explicit one-run debugging."""

from __future__ import annotations

import argparse
import json

from strands import Agent
from strands.models import BedrockModel

from agent.experiments import create_campaign_id, get_best_experiment, load_campaign_experiments
from agent.hitl import (
    HITL_MODES,
    UNKNOWN_COST_POLICIES,
    HitlController,
    HitlError,
    awaiting_delivery,
    create_state,
    existing_state,
    pending_request,
    session_exists,
)
from agent.policies import SELECTION_POLICIES, build_decision_trace, build_system_prompt
from agent.prompts import SYSTEM_PROMPT
from agent.store_tools import make_store_tools, store_campaign_instructions
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
from research.resource_accounting import CampaignSession, campaign_resource_report, load_prices

# Campaign arguments restored from HITL state on --resume.
RESUMED_SETTINGS = ("feedback", "policy", "seed", "max_experiments", "dataset", "model_id", "region")


def _bedrock_model(model_id: str, region: str):
    return BedrockModel(model_id=model_id, region_name=region)


def setup_hitl(args, campaign_id: str, prices):
    """Return the campaign's HitlController, or None when HITL is off."""
    state = existing_state(campaign_id)
    if args.resume:
        if state is None:
            raise SystemExit(f"--resume needs a HITL campaign; none found for {campaign_id}.")
        settings = state["settings"]
        controller = HitlController(campaign_id, settings)
        try:
            controller.resume_target()
        except HitlError as exc:
            raise SystemExit(str(exc))
        for field in RESUMED_SETTINGS:
            setattr(args, field, settings[field])
        args.hitl = settings["hitl_mode"]
        return controller
    if state is not None:
        raise SystemExit(
            f"Campaign {campaign_id} uses HITL {state['settings']['hitl_mode']}; "
            "continue it with --resume."
        )
    if args.hitl == "off":
        return None
    if args.hitl_cost_threshold_usd is not None and (
        prices is None or prices.get("gpu_instance_usd_per_hour") is None
    ):
        raise SystemExit("--hitl-cost-threshold-usd needs --prices with gpu_instance_usd_per_hour.")
    if session_exists(campaign_id):
        raise SystemExit(f"A Strands session already exists for campaign {campaign_id}.")
    settings = {
        **{field: getattr(args, field) for field in RESUMED_SETTINGS},
        "hitl_mode": args.hitl,
        "failure_threshold": args.hitl_failure_threshold,
        "cost_threshold_usd": args.hitl_cost_threshold_usd,
        "unknown_cost_policy": args.hitl_unknown_cost,
        "prices": prices,
    }
    create_state(campaign_id, settings)
    return HitlController(campaign_id, settings)


def create_campaign_agent(args, campaign_id: str, hitl=None) -> Agent:
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
            selection_policy=args.policy,
            launch_gate=hitl,
        ),
        make_campaign_report_tool(args.feedback, campaign_id, args.seed),
    ]
    if args.feedback in {"E2", "E3"}:
        tools.append(make_campaign_history_tool(args.feedback, campaign_id, args.seed))
    if args.feedback == "E3":
        tools.append(make_campaign_diagnostics_tool(args.feedback, campaign_id, args.seed))
    tools.extend(make_store_tools(args.feedback))
    system_prompt = build_system_prompt(args.policy)
    extra = {}
    if hitl is not None:
        tools.append(hitl.make_tool())
        system_prompt += hitl.instructions()
        extra = hitl.agent_options()
    return Agent(
        model=_bedrock_model(args.model_id, args.region),
        system_prompt=system_prompt,
        tools=tools,
        **extra,
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
""" + store_campaign_instructions(args.feedback)


def run_campaign(args) -> str:
    campaign_id = args.campaign_id or create_campaign_id()
    prices = load_prices(args.prices)
    hitl = setup_hitl(args, campaign_id, prices)
    if hitl is not None and prices is None:
        prices = hitl.settings.get("prices")
    print(json.dumps(
        {"mode": "campaign", "campaign_id": campaign_id, "selection_policy": args.policy,
         "hitl": args.hitl},
        indent=2,
    ))
    agent = create_campaign_agent(args, campaign_id, hitl)
    with CampaignSession(campaign_id, model_id=args.model_id) as session:
        if hitl is None:
            session.call(agent, campaign_prompt(args, campaign_id))
        else:
            try:
                resuming = hitl.resume_target() if args.resume else None
                if resuming is not None:
                    hitl.verify_restored(agent, resuming)
                prompt = (
                    hitl.interrupt_response(resuming) if resuming
                    else campaign_prompt(args, campaign_id)
                )
                hitl.run(session, agent, prompt,
                         interactive=args.hitl_interactive, resuming=resuming)
            except HitlError as exc:
                untouched = resuming is not None and (
                    awaiting_delivery(existing_state(campaign_id)) or {}
                ).get("id") == resuming["id"]
                raise SystemExit(str(exc) + (
                    "\nThe decision was not delivered, so no training was launched; "
                    "campaign state, budget, and the pending decision are unchanged."
                    if untouched else ""
                ))
    resources = campaign_resource_report(campaign_id, prices)
    if hitl is not None:
        pending = pending_request(existing_state(campaign_id))
        if pending:
            print("\nCAMPAIGN PAUSED FOR HUMAN DECISION "
                  f"(consultation #{pending['id']}); no training job is running.")
            print(f"  python -m agent.hitl show --campaign-id {campaign_id}")
            print(f"  python -m agent.hitl respond --campaign-id {campaign_id} "
                  f"--request-id {pending['id']} --choice <id|delegate|stop>")
            print(f"  python -m agent.main --resume --campaign-id {campaign_id}")
            return campaign_id
    records = load_campaign_experiments(args.feedback, campaign_id, args.seed)
    datasets = sorted({record["dataset"] for record in records})
    best = (
        get_best_experiment(args.feedback, campaign_id, datasets[0], args.seed)
        if len(datasets) == 1 else None
    )
    report = {
        "campaign_id": campaign_id,
        "feedback_version": args.feedback,
        "selection_policy": args.policy,
        "dataset_id": datasets[0] if len(datasets) == 1 else None,
        "experiments": records,
        "best_experiment_id": best.get("experiment_id") if best else None,
        "best_validation_score": best.get("validation_score") if best else None,
        "best_checkpoint_path": best.get("checkpoint_path") if best else None,
        "decision_trace": build_decision_trace(args.feedback, campaign_id, args.seed),
        "resources": resources,
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
    parser.add_argument(
        "--policy", choices=SELECTION_POLICIES, default="P0",
        help="Experiment selection: P0 free, P1 evidence-grounded decision records.",
    )
    parser.add_argument(
        "--hitl", choices=HITL_MODES, default="off",
        help="Human-in-the-loop: off (autonomous), always (human picks every run), "
             "critical (human approves consequential decisions).",
    )
    parser.add_argument("--hitl-failure-threshold", type=int, default=2,
                        help="critical mode: consecutive failed runs that require approval.")
    parser.add_argument("--hitl-cost-threshold-usd", type=float,
                        help="critical mode: approval above this measured-cost estimate.")
    parser.add_argument("--hitl-unknown-cost", choices=UNKNOWN_COST_POLICIES, default="ask",
                        help="critical mode with a cost threshold: ask (default) or allow "
                             "when a run's cost cannot be estimated from measurements.")
    parser.add_argument("--hitl-interactive", action="store_true",
                        help="Answer consultations on this terminal between agent calls "
                             "instead of pausing (default: pause; answer with agent.hitl).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a paused HITL campaign given by --campaign-id.")
    parser.add_argument("--prices", help="JSON file of explicit USD prices for cost reports.")
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
    if args.resume and not args.campaign_id:
        raise SystemExit("--resume requires --campaign-id")
    if args.hitl_failure_threshold < 1:
        raise SystemExit("--hitl-failure-threshold must be positive")
    run_campaign(args) if args.mode == "campaign" else run_smoke(args)


if __name__ == "__main__":
    main()

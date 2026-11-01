"""Optional human-in-the-loop (HITL) control for GoTrain campaigns.

Modes:
- off: no HITL objects are created; campaigns behave exactly as before.
- always: every training launch needs an answered consultation.
- critical: launches need one only when a deterministic trigger fires:
  repeated failures, or a cost estimate above the threshold (an unknown
  estimate asks the human by default). Claude must also consult before
  proposing a code change and may consult when evidence is insufficient.

Pausing uses Strands' documented interrupt mechanism (strands-agents 1.63):
the consultation tool raises ``tool_context.interrupt(...)``, which stops the
agent loop before any further tool runs (HITL agents use the sequential tool
executor). The conversation and the pending interrupt are persisted by
Strands' ``FileSessionManager``; GoTrain persists the request and the human's
decision. Resuming restores the session in a new Agent and answers the
interrupt; the re-executed tool reads the decision from GoTrain's state, not
from the resume payload. The launch gate runs before reservation, so a blocked
launch never uses budget or a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from strands import ToolContext, tool
from strands.interrupt import InterruptException

from agent import experiments
from agent.experiments import _now, get_budget_status, load_campaign_experiments
from research.resource_accounting import estimate_run_cost

HITL_MODES = ("off", "always", "critical")
UNKNOWN_COST_POLICIES = ("ask", "allow")
TRIGGERS = (
    "before_experiment",
    "repeated_failures",
    "cost_threshold",
    "code_change",
    "insufficient_evidence",
)
DELEGATE, STOP = "delegate", "stop"
RUN_PARAMETERS = (
    "model_id", "epochs", "batch_size", "learning_rate", "weight_decay", "pretrained",
    "image_size", "early_stopping", "early_stopping_patience", "early_stopping_min_delta",
)
INTERRUPT_NAME = "gotrain_human_consultation"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class HitlError(RuntimeError):
    """A refused answer or resume; campaign state is left unchanged."""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _state_path(campaign_id):
    if not _SAFE_ID.match(campaign_id or ""):
        raise ValueError(f"Unsafe campaign ID for HITL state: {campaign_id!r}")
    return experiments.RESULTS_DIR / "hitl" / f"{campaign_id}.json"


def session_storage_dir():
    return experiments.RESULTS_DIR / "hitl" / "sessions"


def session_exists(campaign_id):
    """FileSessionManager stores each session in ``session_<id>/``."""
    return (session_storage_dir() / f"session_{campaign_id}").is_dir()


def load_state(campaign_id):
    path = _state_path(campaign_id)
    with experiments._locked_storage():
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def existing_state(campaign_id):
    """HITL state for any campaign ID; IDs unusable as file names never have one."""
    if not _SAFE_ID.match(campaign_id or ""):
        return None
    return load_state(campaign_id)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _update_state(campaign_id, change):
    """Read-modify-write the state under the shared experiment lock."""
    path = _state_path(campaign_id)
    with experiments._locked_storage():
        state = json.loads(path.read_text(encoding="utf-8"))
        result = change(state)
        _write_json(path, state)
        return result


def create_state(campaign_id, settings):
    path = _state_path(campaign_id)
    with experiments._locked_storage():
        if path.exists():
            raise ValueError(f"HITL state already exists for campaign {campaign_id}.")
        _write_json(path, {
            "campaign_id": campaign_id,
            "created_at": _now(),
            "settings": settings,
            "stopped_by_human": False,
            "requests": [],
        })


def pending_request(state):
    """A consultation waiting for the human."""
    return next((r for r in state["requests"] if r["status"] == "pending"), None)


def awaiting_delivery(state):
    """An answered consultation whose decision the agent has not received yet."""
    return next((r for r in state["requests"] if r["status"] == "answered"), None)


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

def _approval(request, choice):
    if choice == DELEGATE:
        return {"configuration": None, "consumed_by_experiment": None}
    option = next(item for item in request["options"] if item["id"] == choice)
    if option.get("configuration"):
        return {"configuration": option["configuration"], "consumed_by_experiment": None}
    return None


def answer_request(campaign_id, request_id, choice, note=None, via="cli"):
    """Record the human's answer to one specific pending consultation.

    Answers to unknown, already answered (duplicate), or superseded (stale)
    consultations are rejected without changing anything.
    """
    def change(state):
        request = next((r for r in state["requests"] if r["id"] == request_id), None)
        if request is None:
            raise HitlError(f"Consultation #{request_id} does not exist.")
        if request["status"] != "pending":
            raise HitlError(
                f"Consultation #{request_id} is already {request['status']}; "
                "duplicate or stale responses are rejected."
            )
        if request.get("interrupt_id") is None:
            raise HitlError(f"Consultation #{request_id} has not paused the agent yet.")
        valid = [item["id"] for item in request["options"]] + [DELEGATE, STOP]
        if choice not in valid:
            raise HitlError(f"choice must be one of {valid}")
        request.update({
            "status": "answered",
            "response": {"choice": choice, "note": note or None,
                         "answered_at": _now(), "via": via},
            "approval": None if choice == STOP else _approval(request, choice),
        })
        if choice == STOP:
            state["stopped_by_human"] = True
        return request
    return _update_state(campaign_id, change)


def agent_view(request):
    """What Claude learns from an answer: only the human's decision."""
    choice = request["response"]["choice"]
    view = {
        "status": "answered",
        "request_id": request["id"],
        "human_choice": choice,
        "human_note": request["response"]["note"],
    }
    if choice == DELEGATE:
        view["instruction"] = "The human delegated this decision to you. Use your own judgment."
    elif choice == STOP:
        view["instruction"] = (
            "The human stopped the campaign. Do not train again; call "
            "get_campaign_report and write the final report."
        )
    else:
        view["selected_option"] = next(o for o in request["options"] if o["id"] == choice)
        view["instruction"] = (
            "Carry out the selected option. If it has a configuration, launch exactly "
            "that configuration. Follow the human's note if present."
        )
    return view


# ---------------------------------------------------------------------------
# Human-facing display
# ---------------------------------------------------------------------------

def campaign_facts(settings, campaign_id):
    records = load_campaign_experiments(settings["feedback"], campaign_id, settings["seed"])
    completed = [run for run in records if run["status"] == "completed"]
    best = max(completed, key=lambda run: run["validation_score"], default=None)
    return {
        "experiments_used": len(records),
        "max_experiments": settings["max_experiments"],
        "completed": len(completed),
        "failed": sum(run["status"] == "failed" for run in records),
        "best": (
            f"{best.get('primary_metric') or 'score'}={best['validation_score']:.4f} "
            f"(experiment {best['experiment_id']})" if best else None
        ),
        "recent": [
            f"#{run['experiment_id']} {run['configuration'].get('architecture')} "
            f"{run['status']} score={run.get('validation_score')}"
            for run in records[-3:]
        ],
    }


def format_request(state, request):
    settings = state["settings"]
    facts = campaign_facts(settings, state["campaign_id"])
    lines = [
        "",
        "=" * 72,
        f"GoTrain consultation #{request['id']} | campaign {state['campaign_id']} "
        f"| HITL {settings['hitl_mode']}",
        f"Trigger: {request['trigger']}",
        "",
        "What happened (Claude):",
        f"  {request['situation']}",
        "",
        f"Recorded facts: {facts['experiments_used']}/{facts['max_experiments']} experiments "
        f"used, {facts['completed']} completed, {facts['failed']} failed; "
        f"best {facts['best'] or 'none yet'}",
    ]
    lines += [f"  {line}" for line in facts["recent"]]
    lines += ["", "Options:"]
    for option in request["options"]:
        lines.append(f"  [{option['id']}] {option['title']}")
        lines.append(f"      trade-offs: {option['tradeoffs']}")
        if option.get("configuration"):
            lines.append(f"      configuration: {json.dumps(option['configuration'])}")
            estimate = option.get("estimate") or {}
            usd = estimate.get("estimated_gpu_usd")
            seconds = estimate.get("estimated_training_seconds")
            lines.append(
                "      cost estimate: "
                + ("unknown" if seconds is None else f"~{seconds:,.0f} s max")
                + ("" if usd is None else f", ~${usd:,.4f} GPU")
                + f" [{estimate.get('basis')}]"
            )
    lines += [
        "",
        f"Claude recommends: [{request['recommended_option']}] {request['rationale']}",
        "=" * 72,
    ]
    return "\n".join(lines)


def prompt_terminal(text, option_ids, input_fn=input, output=sys.stdout):
    """Ask on the terminal between agent calls; returns (choice, note) or None to pause."""
    print(text, file=output)
    prompt = f"Choose {'/'.join(option_ids)}, d=delegate to Claude, s=stop campaign, p=pause: "
    while True:
        try:
            raw = input_fn(prompt).strip()
        except EOFError:
            return None
        if raw.lower() == "p":
            return None
        choice = {"d": DELEGATE, "s": STOP}.get(raw.lower(), raw)
        if choice in (*option_ids, DELEGATE, STOP):
            try:
                note = input_fn("Optional note for Claude (enter to skip): ").strip()
            except EOFError:
                note = ""
            return choice, note
        print(f"Unrecognized choice {raw!r}.", file=output)


# ---------------------------------------------------------------------------
# Controller: launch gate + consultation tool for one campaign
# ---------------------------------------------------------------------------

def _validate_consultation(trigger, situation, options, recommended_option, rationale):
    errors = {}
    if trigger not in TRIGGERS:
        errors["trigger"] = f"must be one of {list(TRIGGERS)}"
    if not isinstance(situation, str) or not situation.strip():
        errors["situation"] = "describe what happened and the current evidence"
    if not isinstance(rationale, str) or not rationale.strip():
        errors["rationale"] = "explain the recommendation"
    if not isinstance(options, list) or not 1 <= len(options) <= 5:
        errors["options"] = "provide 1 to 5 options"
        return errors
    ids = []
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            errors[f"options[{index}]"] = "must be an object"
            continue
        option_id = option.get("id")
        if not isinstance(option_id, str) or not option_id.strip() or option_id in (DELEGATE, STOP):
            errors[f"options[{index}].id"] = "nonempty id other than 'delegate'/'stop'"
        ids.append(option_id)
        for field in ("title", "tradeoffs"):
            if not isinstance(option.get(field), str) or not option[field].strip():
                errors[f"options[{index}].{field}"] = "required nonempty string"
        configuration = option.get("configuration")
        if configuration is not None:
            if not isinstance(configuration, dict) or not configuration:
                errors[f"options[{index}].configuration"] = "object of run arguments"
            elif set(configuration) - set(RUN_PARAMETERS):
                errors[f"options[{index}].configuration"] = (
                    f"unsupported keys {sorted(set(configuration) - set(RUN_PARAMETERS))}"
                )
        unknown = set(option) - {"id", "title", "tradeoffs", "configuration"}
        if unknown:
            errors[f"options[{index}]"] = f"unsupported fields {sorted(unknown)}"
    if len(set(ids)) != len(ids):
        errors["options"] = "option ids must be unique"
    if recommended_option not in ids:
        errors["recommended_option"] = "must be one of the option ids"
    return errors


def _matches(approved, configuration):
    """An approved option binds only the run arguments it specified."""
    for key, value in approved.items():
        actual = configuration.get("architecture" if key == "model_id" else key)
        if actual != value:
            return False
    return True


class HitlController:
    def __init__(self, campaign_id, settings):
        self.campaign_id = campaign_id
        self.settings = settings
        self.mode = settings["hitl_mode"]
        if self.mode not in HITL_MODES[1:]:
            raise ValueError("HitlController requires mode 'always' or 'critical'.")

    # -- deterministic launch gate -------------------------------------------------

    def _triggers(self, dataset_id, configuration):
        if self.mode == "always":
            return ["before_experiment"], {"before_experiment": {"hitl_mode": "always"}}
        triggers, details = [], {}
        records = load_campaign_experiments(
            self.settings["feedback"], self.campaign_id, self.settings["seed"]
        )
        trailing = 0
        for run in reversed(records):
            if run["status"] != "failed":
                break
            trailing += 1
        if trailing >= self.settings["failure_threshold"]:
            triggers.append("repeated_failures")
            details["repeated_failures"] = {
                "consecutive_failed_runs": trailing,
                "threshold": self.settings["failure_threshold"],
            }
        threshold = self.settings.get("cost_threshold_usd")
        if threshold is not None:
            estimate = estimate_run_cost(dataset_id, configuration, self.settings.get("prices"))
            reason = None
            if estimate["estimated_gpu_usd"] is None:
                if self.settings.get("unknown_cost_policy", "ask") == "ask":
                    reason = "estimate_unknown"
            elif estimate["estimated_gpu_usd"] > threshold:
                reason = "estimate_exceeds_threshold"
            if reason:
                triggers.append("cost_threshold")
                details["cost_threshold"] = {
                    "reason": reason,
                    "threshold_usd": threshold,
                    "unknown_cost_policy": self.settings.get("unknown_cost_policy", "ask"),
                    "estimate": estimate,
                }
        return triggers, details

    def check(self, dataset_id, configuration):
        """Return (approval_request_id, None) to launch or (None, rejection)."""
        state = load_state(self.campaign_id)
        if state["stopped_by_human"]:
            return None, {
                "status": "rejected",
                "error_code": "CAMPAIGN_STOPPED_BY_HUMAN",
                "message": "The human stopped this campaign. Write the final report.",
            }
        if pending_request(state) or awaiting_delivery(state):
            return None, {
                "status": "rejected",
                "error_code": "HUMAN_DECISION_PENDING",
                "message": "A consultation awaits the human's decision.",
            }
        budget = get_budget_status(
            self.settings["feedback"], self.campaign_id, dataset_id, self.settings["seed"],
            max_experiments=self.settings["max_experiments"],
        )
        if budget["remaining_experiments"] == 0:
            # Reservation rejects deterministically; never ask the human to
            # approve a run the budget forbids.
            return None, None
        triggers, details = self._triggers(dataset_id, configuration)
        if not triggers:
            return None, None
        approvals = [
            request for request in state["requests"]
            if request.get("approval") and request["approval"]["consumed_by_experiment"] is None
        ]
        for request in reversed(approvals):
            approved = request["approval"]["configuration"]
            if approved is None or _matches(approved, configuration):
                return request["id"], None
        if approvals:
            return None, {
                "status": "rejected",
                "error_code": "HUMAN_SELECTION_MISMATCH",
                "message": "Launch exactly the configuration the human selected.",
                "approved_configuration": approvals[-1]["approval"]["configuration"],
            }
        return None, {
            "status": "rejected",
            "error_code": "HUMAN_APPROVAL_REQUIRED",
            "triggers": triggers,
            "trigger_details": details,
            "message": (
                "Human approval is required before this launch. Call "
                f"request_human_consultation with trigger {triggers[0]!r}. "
                "No budget was used."
            ),
        }

    def record_launch(self, approval_request_id, response):
        """Consume the approval once its launch actually reserved an experiment."""
        if approval_request_id is None or response.get("status") == "rejected":
            return

        def change(state):
            request = next(r for r in state["requests"] if r["id"] == approval_request_id)
            request["approval"]["consumed_by_experiment"] = response.get("experiment_id")
        _update_state(self.campaign_id, change)

    # -- consultation lifecycle ----------------------------------------------------

    def open_request(self, tool_use_id, trigger, situation, options, recommended_option,
                     rationale):
        """Create the request, or return the existing one when Strands re-executes it."""
        state = load_state(self.campaign_id)
        existing = next((r for r in state["requests"] if r.get("tool_use_id") == tool_use_id), None)
        if existing is not None:
            return existing, None
        errors = _validate_consultation(trigger, situation, options, recommended_option, rationale)
        if errors:
            return None, {
                "status": "rejected",
                "error_code": "INVALID_CONSULTATION",
                "field_errors": errors,
                "message": "Fix the listed fields and call request_human_consultation again.",
            }
        stored_options = []
        for option in options:
            stored = {key: option[key] for key in ("id", "title", "tradeoffs")}
            if option.get("configuration"):
                stored["configuration"] = option["configuration"]
                if self.settings.get("dataset"):
                    stored["estimate"] = estimate_run_cost(
                        self.settings["dataset"], option["configuration"],
                        self.settings.get("prices"),
                    )
            stored_options.append(stored)

        def change(state):
            if pending_request(state) or awaiting_delivery(state):
                return None
            request = {
                "id": len(state["requests"]) + 1,
                "tool_use_id": tool_use_id,
                "interrupt_id": None,
                "created_at": _now(),
                "trigger": trigger,
                "situation": situation.strip(),
                "options": stored_options,
                "recommended_option": recommended_option,
                "rationale": rationale.strip(),
                "status": "pending",
            }
            state["requests"].append(request)
            return request

        request = _update_state(self.campaign_id, change)
        if request is None:
            return None, {"status": "rejected", "error_code": "HUMAN_DECISION_PENDING",
                          "message": "Another consultation already awaits the human."}
        return request, None

    def record_interrupt(self, request_id, interrupt_id):
        def change(state):
            request = next(r for r in state["requests"] if r["id"] == request_id)
            request["interrupt_id"] = request.get("interrupt_id") or interrupt_id
        _update_state(self.campaign_id, change)

    def deliver(self, request_id, response):
        """Hand the recorded decision to Claude exactly once."""
        def change(state):
            request = next(r for r in state["requests"] if r["id"] == request_id)
            if not isinstance(response, dict) or response.get("request_id") != request_id:
                return {"status": "rejected", "error_code": "STALE_HUMAN_RESPONSE",
                        "message": "The resume payload does not match this consultation."}
            if request["status"] != "answered":
                return {"status": "rejected", "error_code": "HUMAN_DECISION_PENDING",
                        "message": f"Consultation #{request_id} is {request['status']}."}
            request["status"] = "delivered"
            request["delivered_at"] = _now()
            return agent_view(request)
        return _update_state(self.campaign_id, change)

    def interrupt_response(self, request):
        """Strands resume payload. The decision itself is read from GoTrain's state."""
        return [{"interruptResponse": {
            "interruptId": request["interrupt_id"],
            "response": {"request_id": request["id"]},
        }}]

    def make_tool(self):
        controller = self

        @tool(context=True)
        def request_human_consultation(
            trigger: str,
            situation: str,
            options: list[dict],
            recommended_option: str,
            rationale: str,
            tool_context: ToolContext,
        ) -> dict:
            """Pause the campaign and ask the human to choose the next action.

            The campaign stops until the human answers; this tool then returns
            their decision.

            Args:
                trigger: before_experiment, repeated_failures, cost_threshold,
                    code_change, or insufficient_evidence.
                situation: What happened and the current evidence. Separate
                    observations from interpretation; say when evidence is thin.
                options: 1-5 feasible options, each {"id", "title", "tradeoffs"}
                    plus, for a training option, "configuration" with the exact
                    run_campaign_experiment arguments (model_id, epochs, ...).
                recommended_option: The id of the option you recommend.
                rationale: Why you recommend it.
            """
            request, error = controller.open_request(
                tool_context.tool_use["toolUseId"], trigger, situation, options,
                recommended_option, rationale,
            )
            if error:
                return error
            try:
                response = tool_context.interrupt(
                    INTERRUPT_NAME,
                    reason={"campaign_id": controller.campaign_id, "request_id": request["id"]},
                )
            except InterruptException as exc:
                controller.record_interrupt(request["id"], exc.interrupt.id)
                raise
            return controller.deliver(request["id"], response)

        return request_human_consultation

    def agent_options(self):
        """Strands Agent arguments for HITL: persisted session, sequential tools."""
        from strands.session.file_session_manager import FileSessionManager
        from strands.tools.executors import SequentialToolExecutor

        return {
            "session_manager": FileSessionManager(
                session_id=self.campaign_id, storage_dir=str(session_storage_dir())
            ),
            "tool_executor": SequentialToolExecutor(),
        }

    def instructions(self):
        common = """
Consultation format: in `situation`, state what happened and the evidence,
separating observations from interpretation and saying plainly when evidence is
insufficient. Offer feasible options with honest trade-offs; for training options
give the exact run_campaign_experiment arguments in `configuration`. Recommend one.
Calling request_human_consultation pauses the campaign until the human answers;
the tool then returns their decision. The human may pick an option, delegate the
decision back to you, or stop the campaign. Launch exactly a selected
configuration and follow any human note. Approval never overrides model, budget,
or data-split rules.
"""
        if self.mode == "always":
            return """
HUMAN-IN-THE-LOOP MODE: always.
Before EVERY training experiment, call request_human_consultation with trigger
"before_experiment" and let the human choose. Launches without an answered
consultation are rejected without using budget.""" + common
        return """
HUMAN-IN-THE-LOOP MODE: critical.
Act autonomously, but call request_human_consultation for consequential decisions:
- "code_change": before proposing any change to code or the training pipeline.
  You cannot change code yourself; the human decides.
- "repeated_failures" or "cost_threshold": when run_campaign_experiment returns
  HUMAN_APPROVAL_REQUIRED with that trigger (see its trigger_details).
- "insufficient_evidence": optionally, when the evidence is too weak to choose
  responsibly.""" + common

    # -- running the agent ---------------------------------------------------------

    def resume_target(self):
        """The answered consultation to deliver; refuses anything else."""
        state = load_state(self.campaign_id)
        pending = pending_request(state)
        if pending is not None:
            raise HitlError(
                f"Consultation #{pending['id']} still awaits the human. Answer it first:\n"
                f"  python -m agent.hitl show --campaign-id {self.campaign_id}"
            )
        request = awaiting_delivery(state)
        if request is None:
            raise HitlError(
                f"Nothing to resume for campaign {self.campaign_id}: no answered consultation "
                "is waiting to be delivered (it may already have been resumed)."
            )
        if not session_exists(self.campaign_id):
            raise HitlError(f"No saved Strands session for campaign {self.campaign_id}.")
        return request

    @staticmethod
    def verify_restored(agent, request):
        """Fail before any model call if Strands did not restore the paused session."""
        if not getattr(agent, "messages", None):
            raise HitlError("Strands restored no conversation for this campaign.")
        interrupt_state = getattr(agent, "_interrupt_state", None)
        if interrupt_state is not None and (
            not interrupt_state.activated
            or request["interrupt_id"] not in interrupt_state.interrupts
        ):
            raise HitlError("Strands did not restore the paused consultation interrupt.")

    def run(self, session, agent, prompt, *, interactive=False, resuming=None,
            input_fn=None, output=None):
        """Run until the campaign finishes or pauses; returns 'finished' or 'paused'."""
        input_fn = input_fn or input
        output = output or sys.stdout
        try:
            result = session.call(agent, prompt)
        except (KeyError, ValueError, TypeError) as exc:
            state = load_state(self.campaign_id)
            if resuming is not None and awaiting_delivery(state) and \
                    awaiting_delivery(state)["id"] == resuming["id"]:
                raise HitlError(
                    f"Strands rejected the resume ({type(exc).__name__}: {exc}). "
                    "Campaign state and the answered decision are unchanged."
                ) from exc
            raise
        while getattr(result, "stop_reason", None) == "interrupt":
            state = load_state(self.campaign_id)
            request = pending_request(state)
            raised = {interrupt.id for interrupt in (result.interrupts or [])}
            if request is None or request["interrupt_id"] not in raised:
                raise HitlError("The agent paused on an interrupt GoTrain did not record.")
            if not interactive:
                return "paused"
            answer = prompt_terminal(
                format_request(state, request),
                [option["id"] for option in request["options"]], input_fn, output,
            )
            if answer is None:
                return "paused"
            answered = answer_request(self.campaign_id, request["id"], *answer, via="terminal")
            result = session.call(agent, self.interrupt_response(answered))
        return "finished"


# ---------------------------------------------------------------------------
# CLI for paused campaigns
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Answer paused GoTrain HITL consultations")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="Show the campaign's pending consultation.")
    show.add_argument("--campaign-id", required=True)
    respond = sub.add_parser("respond", help="Answer the pending consultation.")
    respond.add_argument("--campaign-id", required=True)
    respond.add_argument("--request-id", type=int, required=True,
                         help="The consultation number shown by 'show'.")
    respond.add_argument("--choice", required=True,
                         help="An option id, 'delegate', or 'stop'.")
    respond.add_argument("--note", help="Optional instruction for Claude.")
    args = parser.parse_args(argv)

    state = existing_state(args.campaign_id)
    if state is None:
        raise SystemExit(f"No HITL state for campaign {args.campaign_id}.")
    if args.command == "show":
        request = pending_request(state)
        if request is None:
            waiting = awaiting_delivery(state)
            print(
                f"Consultation #{waiting['id']} is answered; resume the campaign."
                if waiting else "No pending consultation."
                + (" The human stopped this campaign." if state["stopped_by_human"] else "")
            )
            return
        print(format_request(state, request))
        print(f"\nAnswer with: python -m agent.hitl respond --campaign-id {args.campaign_id} "
              f"--request-id {request['id']} "
              f"--choice <{'|'.join(o['id'] for o in request['options'])}|delegate|stop>")
        return
    try:
        answered = answer_request(
            args.campaign_id, args.request_id, args.choice, args.note, via="cli"
        )
    except HitlError as exc:
        raise SystemExit(str(exc))
    print(f"Recorded choice {answered['response']['choice']!r} for consultation #{answered['id']}.")
    print(f"Resume with: python -m agent.main --resume --campaign-id {args.campaign_id}")


if __name__ == "__main__":
    main()

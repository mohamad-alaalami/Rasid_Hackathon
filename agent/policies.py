"""Experiment-selection policies P0 (free) and P1 (evidence-grounded).

The policy changes only what Claude must state before launching a run. Both
policies use the same training tool, model registry, budget, duplicate checks,
and validation evaluator. P1 decision records are stored in a separate audit
log that no agent tool reads, so P1 gains no extra observations over P0.
"""

from __future__ import annotations

import json
import os
import uuid

from agent import experiments
from agent.experiments import load_campaign_experiments
from agent.prompts import SYSTEM_PROMPT

SELECTION_POLICIES = ("P0", "P1")
DEFAULT_POLICY = "P0"
DECISION_FIELDS = ("observation", "action", "expected_signal", "cost_justification")
MAX_FIELD_CHARS = 1200

P1_INSTRUCTIONS = """
EXPERIMENT-SELECTION POLICY P1 (evidence-grounded):
Every run_campaign_experiment call must include a decision_record object with
exactly these four nonempty string fields:
- observation: what the evidence you currently have indicates. Quote concrete
  results when available and separate observations from interpretations. If the
  evidence is insufficient (for example, before any run), say so plainly instead
  of inventing a diagnosis.
- action: the run you are launching and what changes relative to the reference
  configuration (the best or most recent run). Several coordinated changes are
  allowed when justified.
- expected_signal: what observable validation outcome would support your
  reasoning, and what outcome would challenge it. Do not invent improvement
  percentages or confidence values; one noisy run is not proof or refutation.
- cost_justification: why this run is a reasonable use of the remaining budget.
  Use runtimes only if your tools actually reported them; otherwise say the cost
  is uncertain. Never fabricate GPU runtime predictions.

Executable actions are limited to what run_campaign_experiment supports: a new
run with an eligible model_id, epochs (1-20), batch_size (within the model
limit), learning_rate, weight_decay, pretrained, and an optional image_size.
Every run starts from the registry's pretrained initialization; resuming or
warm-starting from an earlier campaign checkpoint is NOT supported, so "train
longer" means a new run with more epochs. A malformed decision_record is
rejected without training and without consuming budget; fix it and resubmit.
The decision record never overrides deterministic validation of the run.
"""


def check_policy(policy):
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"Unknown selection policy: {policy!r}")
    return policy


def build_system_prompt(policy=DEFAULT_POLICY):
    """P0 keeps the existing system prompt byte-for-byte; P1 appends its rules."""
    check_policy(policy)
    return SYSTEM_PROMPT if policy == "P0" else SYSTEM_PROMPT + P1_INSTRUCTIONS


def validate_decision_record(record):
    """Return (normalized_record, None) or (None, structured rejection)."""
    if isinstance(record, str):
        try:
            record = json.loads(record)
        except ValueError:
            pass
    field_errors = {}
    if not isinstance(record, dict):
        field_errors["decision_record"] = "must be an object with the four required fields"
    else:
        for name in DECISION_FIELDS:
            value = record.get(name)
            if not isinstance(value, str) or not value.strip():
                field_errors[name] = "missing or empty; must be a nonempty string"
            elif len(value) > MAX_FIELD_CHARS:
                field_errors[name] = f"longer than {MAX_FIELD_CHARS} characters; be concise"
        for name in sorted(set(record) - set(DECISION_FIELDS)):
            field_errors[name] = "unsupported field"
    if field_errors:
        return None, {
            "status": "rejected",
            "error_code": "INVALID_DECISION_RECORD",
            "message": (
                "Policy P1 requires a decision_record with nonempty observation, "
                "action, expected_signal, and cost_justification. No training was "
                "launched and no budget was consumed; correct it and resubmit."
            ),
            "field_errors": field_errors,
            "required_fields": list(DECISION_FIELDS),
        }
    return {name: record[name].strip() for name in DECISION_FIELDS}, None


# ---------------------------------------------------------------------------
# Decision audit log: separate from E0-E3 experiment records and agent tools.
# ---------------------------------------------------------------------------

def _log_path():
    return experiments.RESULTS_DIR / "decisions.jsonl"


def _append(entry):
    with experiments._locked_storage():
        with _log_path().open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, allow_nan=False, default=str) + "\n")
            file.flush()
            os.fsync(file.fileno())


def load_decisions(campaign_id, seed=None):
    """Latest event per attempt for one campaign, in submission order."""
    with experiments._locked_storage():
        path = _log_path()
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    latest = {}
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["campaign_id"] == campaign_id and (seed is None or entry["seed"] == seed):
            latest[entry["attempt_id"]] = entry
    return sorted(latest.values(), key=lambda entry: (entry["submitted_at"], entry["attempt_id"]))


def check_campaign_policy(campaign_id, seed, policy):
    """Reject mixing policies inside one campaign, which would contaminate history."""
    used = {entry["selection_policy"] for entry in load_decisions(campaign_id, seed)}
    if used - {policy}:
        return {
            "status": "rejected",
            "error_code": "POLICY_MISMATCH",
            "message": (
                f"Campaign {campaign_id} already uses policy {sorted(used)}; "
                "start a new campaign ID to use another selection policy."
            ),
        }
    return None


def decision_context(version, campaign_id, seed):
    """Deterministic reference runs at decision time: previous run and incumbent."""
    records = load_campaign_experiments(version, campaign_id, seed)
    completed = [run for run in records if run["status"] == "completed"]
    incumbent = max(completed, key=lambda run: run["validation_score"], default=None)
    return {
        "previous_experiment_id": records[-1]["experiment_id"] if records else None,
        "incumbent_experiment_id": incumbent["experiment_id"] if incumbent else None,
        "incumbent_validation_score": incumbent["validation_score"] if incumbent else None,
    }


def _entry(*, version, campaign_id, seed, dataset_id, policy, configuration,
           decision_record, origin):
    return {
        "attempt_id": uuid.uuid4().hex,
        "campaign_id": campaign_id,
        "feedback_version": version,
        "selection_policy": check_policy(policy),
        "dataset": dataset_id,
        "seed": seed,
        "origin": origin,
        "submitted_at": experiments._now(),
        "configuration": configuration,
        "decision_record": decision_record,
        "context_at_decision": decision_context(version, campaign_id, seed),
    }


def _outcome(response):
    return {
        "status": response.get("status"),
        "experiment_id": response.get("experiment_id"),
        "error_code": response.get("error_code"),
        "message": response.get("message") or response.get("error"),
        "field_errors": response.get("field_errors"),
        "recorded_at": experiments._now(),
    }


def record_rejection(response, **identity):
    """Persist a rejected attempt; it never reached reservation or the GPU."""
    _append({**_entry(**identity), "outcome": _outcome(response)})
    return response


def launch_with_decision(launch, **identity):
    """Log the decision before training, run launch(), then log its outcome."""
    entry = _entry(**identity)
    _append({**entry, "outcome": {"status": "pending"}})
    try:
        response = launch()
    except Exception as exc:
        _append({**entry, "outcome": _outcome({"status": "failed", "error": str(exc)})})
        raise
    _append({**entry, "outcome": _outcome(response)})
    return response


def build_decision_trace(version, campaign_id, seed):
    """Join decisions with measured results for reports; never exposed to the agent.

    ``decision_record.expected_signal`` is Claude's stated expectation; the
    ``measured`` block holds only values recorded by the trainer/evaluator.
    """
    records = {
        run["experiment_id"]: run
        for run in load_campaign_experiments(version, campaign_id, seed)
    }
    trace = []
    for index, entry in enumerate(load_decisions(campaign_id, seed), start=1):
        run = records.get(entry["outcome"].get("experiment_id"))
        measured = None
        if run is not None:
            measured = {
                key: run.get(key)
                for key in (
                    "status", "validation_score", "primary_metric", "runtime_seconds",
                    "training_seconds", "peak_gpu_memory_mb", "device",
                    "checkpoint_path", "error",
                )
            }
        incumbent_score = entry["context_at_decision"]["incumbent_validation_score"]
        delta = (
            measured["validation_score"] - incumbent_score
            if measured and measured["validation_score"] is not None
            and incumbent_score is not None
            else None
        )
        trace.append({
            "attempt": index,
            **{key: entry[key] for key in (
                "selection_policy", "campaign_id", "feedback_version", "origin",
                "submitted_at", "decision_record", "context_at_decision", "configuration",
            )},
            "outcome": entry["outcome"],
            "measured": measured,
            "measured_delta_vs_incumbent": delta,
        })
    return trace

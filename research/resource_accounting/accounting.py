"""Resource accounting for GoTrain campaigns.

Sources, in order of trust:
- LLM tokens and latency come only from provider-reported values that Strands
  (1.63) accumulates on ``Agent.event_loop_metrics``: ``accumulated_usage``
  (Bedrock ``metadata.usage``) and ``accumulated_metrics["latencyMs"]``. These
  are cumulative per Agent instance, are not restored by session managers, and
  are recorded per call as before/after differences so nothing is counted twice.
- Model calls are counted with an ``AfterModelCallEvent`` hook (one event per
  request attempt, successful or failed). Event-loop cycles are not used: a
  cycle can contain retried model calls.
- Nothing is estimated from text or from Strands' projected token counts.
- Training time comes from the existing experiment records. Each experiment
  contributes exactly one value: the trainer's ``training_seconds`` when
  present, otherwise the campaign wrapper's ``runtime_seconds``. They overlap,
  so they are never added together.
- Costs are computed only from explicitly configured prices.

Anything that was not reported is ``None`` (unknown), never zero.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent import experiments
from agent.experiments import FEEDBACK_VERSIONS, load_experiments

PRICE_FIELDS = (
    "bedrock_input_usd_per_million_tokens",
    "bedrock_output_usd_per_million_tokens",
    "bedrock_cache_read_usd_per_million_tokens",
    "bedrock_cache_write_usd_per_million_tokens",
    "gpu_instance_usd_per_hour",
)
TOKEN_FIELDS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "total_tokens": "totalTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
    "cache_write_input_tokens": "cacheWriteInputTokens",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _number(value, allow_zero=True):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value >= 0 if allow_zero else value > 0)
    )


def _get(source, key):
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def load_prices(path):
    """Load an explicit USD price file; ``None`` means no prices configured."""
    if path is None:
        return None
    prices = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prices, dict):
        raise ValueError("Price file must contain a JSON object.")
    unknown = set(prices) - set(PRICE_FIELDS) - {"source", "currency"}
    if unknown:
        raise ValueError(f"Unknown price fields: {sorted(unknown)}")
    if prices.get("currency", "USD") != "USD":
        raise ValueError("Only USD prices are supported.")
    for field in PRICE_FIELDS:
        value = prices.get(field)
        if value is not None and not _number(value):
            raise ValueError(f"{field} must be a nonnegative number or null.")
    return prices


def _price(prices, field):
    return None if prices is None else prices.get(field)


# ---------------------------------------------------------------------------
# LLM usage from Strands
# ---------------------------------------------------------------------------

LLM_FIELDS = (*TOKEN_FIELDS, "model_invocations", "failed_model_calls", "llm_latency_seconds")


def read_cumulative_usage(source):
    """Cumulative provider-reported counters on a Strands Agent (or AgentResult).

    Returns ``None`` for every counter the SDK object does not expose.
    """
    metrics = _get(source, "event_loop_metrics") or _get(source, "metrics")
    usage = _get(metrics, "accumulated_usage")
    snapshot = {
        field: (_get(usage, key) if _number(_get(usage, key)) else None)
        for field, key in TOKEN_FIELDS.items()
    }
    latency_ms = _get(_get(metrics, "accumulated_metrics"), "latencyMs")
    snapshot["latency_ms"] = latency_ms if _number(latency_ms) else None
    return snapshot


class ModelCallCounter:
    """``AfterModelCallEvent`` callback: one event per model request attempt.

    Strands also fires this event, without calling the model, when a
    BeforeModelCallEvent hook cancels the call; GoTrain registers no such hook.
    """

    def __init__(self):
        self.succeeded = 0
        self.failed = 0

    def __call__(self, event):
        if getattr(event, "exception", None) is not None:
            self.failed += 1
        else:
            self.succeeded += 1


def attach_model_call_counter(agent):
    """Register a counter on the agent's hook registry; ``None`` if unsupported."""
    try:
        from strands.hooks import AfterModelCallEvent

        counter = ModelCallCounter()
        agent.hooks.add_callback(AfterModelCallEvent, counter)
        return counter
    except Exception:
        return None


def _delta(after, before):
    return after - before if after is not None and before is not None else None


def call_usage(before, after, calls_before, counter):
    """Provider usage of one ``agent(...)`` call from cumulative counters.

    Model calls that reported no input tokens (or no latency) mean the provider
    did not report them, so those values are unknown rather than zero.
    """
    usage = {field: _delta(after[field], before[field]) for field in TOKEN_FIELDS}
    latency_ms = _delta(after["latency_ms"], before["latency_ms"])
    succeeded = failed = None
    if counter is not None:
        succeeded = counter.succeeded - calls_before[0]
        failed = counter.failed - calls_before[1]
    # Zero is only trusted when the hook confirms no model call happened.
    calls_happened_or_unknown = succeeded is None or succeeded > 0
    if not usage["input_tokens"] and calls_happened_or_unknown:
        usage = dict.fromkeys(TOKEN_FIELDS)
    if not latency_ms and calls_happened_or_unknown:
        latency_ms = None
    usage["model_invocations"] = succeeded
    usage["failed_model_calls"] = failed
    usage["llm_latency_seconds"] = None if latency_ms is None else latency_ms / 1000.0
    return usage


def _sum_fields(entries):
    """Sum per-call (or per-session) usage; any unknown makes the total unknown."""
    totals = {}
    for field in LLM_FIELDS:
        values = [entry[field] for entry in entries]
        totals[field] = None if any(value is None for value in values) else sum(values)
    return totals


# ---------------------------------------------------------------------------
# Session log: one entry per agent process (fresh campaign or resume)
# ---------------------------------------------------------------------------

def _log_path():
    return experiments.RESULTS_DIR / "resources.jsonl"


def _append(entry):
    with experiments._locked_storage():
        with _log_path().open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())


def load_sessions(campaign_id):
    """All session entries for a campaign; each holds only its own calls."""
    with experiments._locked_storage():
        path = _log_path()
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    latest = {}
    for line in lines:
        if line.strip():
            entry = json.loads(line)
            if entry["campaign_id"] == campaign_id:
                latest[entry["session_id"]] = entry
    return sorted(latest.values(), key=lambda entry: entry["started_at"])


class CampaignSession:
    """Time one agent process and record provider-reported usage per call.

    Every ``agent(...)`` call (first prompt, interrupt responses, resumed
    agents) is measured as the change in the agent's cumulative counters, so
    repeated calls on one Agent and restored Agents are never double counted.
    """

    def __init__(self, campaign_id, *, model_id=None, label="campaign"):
        self.entry = {
            "campaign_id": campaign_id,
            "session_id": uuid.uuid4().hex,
            "label": label,
            "model_id": model_id,
            "started_at": _now(),
        }
        self.calls = []
        self.agent_call_seconds = 0.0
        self._counters = {}

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def call(self, agent, prompt):
        """Invoke the agent, timing it and capturing usage even if it raises."""
        if id(agent) not in self._counters:
            self._counters[id(agent)] = attach_model_call_counter(agent)
        counter = self._counters[id(agent)]
        before = read_cumulative_usage(agent)
        calls_before = (counter.succeeded, counter.failed) if counter else None
        started = time.perf_counter()
        try:
            return agent(prompt)
        finally:
            seconds = time.perf_counter() - started
            self.agent_call_seconds += seconds
            usage = call_usage(before, read_cumulative_usage(agent), calls_before, counter)
            self.calls.append({**usage, "agent_call_seconds": seconds})

    def __exit__(self, exc_type, exc, traceback):
        llm = _sum_fields(self.calls) if self.calls else dict.fromkeys(LLM_FIELDS)
        llm["source"] = "strands.event_loop_metrics+AfterModelCallEvent" if self.calls else "no agent call"
        _append({
            **self.entry,
            "ended_at": _now(),
            "elapsed_seconds": time.perf_counter() - self._started,
            "agent_call_seconds": self.agent_call_seconds,
            "outcome": "error" if exc_type else "finished",
            "error": f"{exc_type.__name__}: {exc}" if exc_type else None,
            "llm": llm,
            "calls": self.calls,
        })
        return False


# ---------------------------------------------------------------------------
# Training compute from existing experiment records
# ---------------------------------------------------------------------------

def experiment_compute(record):
    status = record.get("status")
    seconds, source = None, None
    if status in ("completed", "failed"):
        if _number(record.get("training_seconds")):
            seconds, source = record["training_seconds"], "trainer.training_seconds"
        elif _number(record.get("runtime_seconds")):
            seconds, source = record["runtime_seconds"], "campaign.runtime_seconds"
    return {
        "experiment_id": record.get("experiment_id"),
        "status": status,
        "architecture": (record.get("configuration") or {}).get("architecture"),
        "device": record.get("device"),
        "training_seconds": seconds,
        "training_seconds_source": source,
        "completed_epochs": record.get("completed_epochs"),
        "peak_gpu_memory_mb": record.get("peak_gpu_memory_mb"),
    }


def _sum_known(values):
    known = [value for value in values if value is not None]
    return (sum(known) if known else None), len(values) - len(known)


def find_campaign_records(campaign_id, version=None, seed=None):
    """Locate a campaign's experiment records without knowing its E-profile."""
    for candidate in ([version] if version else FEEDBACK_VERSIONS):
        records = load_experiments(candidate, campaign_id=campaign_id, seed=seed)
        if records:
            return candidate, records
    return version, []


def estimate_run_cost(dataset, configuration, prices):
    """Conservative cost of a requested run from measured seconds per epoch.

    Uses the slowest measured seconds/epoch among completed runs of the same
    dataset, architecture, and image_size (any campaign), times the requested
    maximum epochs. Nothing is extrapolated from other models; without such a
    measurement, or without a GPU price, the cost is unknown (never zero).
    """
    architecture = configuration.get("architecture") or configuration.get("model_id")
    per_epoch = []
    for version in FEEDBACK_VERSIONS:
        for run in load_experiments(version, dataset=dataset):
            run_config = run.get("configuration") or {}
            if (
                run.get("status") == "completed"
                and run_config.get("architecture") == architecture
                and run_config.get("image_size") == configuration.get("image_size")
                and _number(run.get("training_seconds"))
                and _number(run.get("completed_epochs"), allow_zero=False)
            ):
                per_epoch.append(run["training_seconds"] / run["completed_epochs"])
    estimate = {
        "status": "unknown",
        "estimated_training_seconds": None,
        "estimated_gpu_usd": None,
        "measured_runs": len(per_epoch),
        "basis": None,
    }
    # An adaptive-compute action trains only its allocated epochs, not the horizon.
    epochs = (configuration.get("allocation") or {}).get("epochs", configuration.get("epochs"))
    if not per_epoch or not _number(epochs, allow_zero=False):
        estimate["basis"] = (
            f"no measured completed run of {architecture} at image_size "
            f"{configuration.get('image_size')} on {dataset}"
        )
        return estimate
    seconds = max(per_epoch) * epochs
    estimate["estimated_training_seconds"] = seconds
    estimate["basis"] = (
        f"slowest measured seconds/epoch over {len(per_epoch)} completed {architecture} "
        f"run(s) x {epochs} max epochs; batch size differences ignored"
    )
    gpu_price = _price(prices, "gpu_instance_usd_per_hour")
    if gpu_price is None:
        estimate["basis"] += "; gpu_instance_usd_per_hour not configured"
        return estimate
    estimate["status"] = "estimated"
    estimate["estimated_gpu_usd"] = seconds / 3600.0 * gpu_price
    return estimate


# ---------------------------------------------------------------------------
# Campaign report
# ---------------------------------------------------------------------------

def _bedrock_cost(llm, prices):
    notes = []
    if prices is None:
        return None, ["no price file configured"]
    terms = [
        ("input_tokens", "bedrock_input_usd_per_million_tokens", True),
        ("output_tokens", "bedrock_output_usd_per_million_tokens", True),
        ("cache_read_input_tokens", "bedrock_cache_read_usd_per_million_tokens", False),
        ("cache_write_input_tokens", "bedrock_cache_write_usd_per_million_tokens", False),
    ]
    total = 0.0
    for token_field, price_field, required in terms:
        tokens, price = llm[token_field], prices.get(price_field)
        if tokens is None:
            if required:
                notes.append(f"{token_field} unknown")
            continue
        if tokens and price is None:
            notes.append(f"{price_field} not configured")
            continue
        total += tokens / 1_000_000 * (price or 0.0)
    return (None if notes else total), notes


def campaign_resource_report(campaign_id, prices=None, version=None, seed=None):
    version, records = find_campaign_records(campaign_id, version, seed)
    sessions = load_sessions(campaign_id)

    llm = {"sessions": len(sessions), "sessions_with_unknown_usage": 0}
    for field in LLM_FIELDS:
        total, missing = _sum_known([session["llm"].get(field) for session in sessions])
        # A partial sum is not the campaign total, so it is reported as unknown.
        llm[field] = None if missing else total
        if field in ("input_tokens", "output_tokens"):
            llm["sessions_with_unknown_usage"] = max(llm["sessions_with_unknown_usage"], missing)

    compute = [experiment_compute(record) for record in records]
    training_seconds, missing_runtime = _sum_known([run["training_seconds"] for run in compute])
    active_elapsed, _ = _sum_known([session.get("elapsed_seconds") for session in sessions])
    agent_seconds, _ = _sum_known([session.get("agent_call_seconds") for session in sessions])
    wall_span = None
    if sessions:
        start = datetime.fromisoformat(sessions[0]["started_at"])
        end = max(datetime.fromisoformat(session["ended_at"]) for session in sessions)
        wall_span = (end - start).total_seconds()

    bedrock_usd, bedrock_notes = _bedrock_cost(llm, prices)
    gpu_price = _price(prices, "gpu_instance_usd_per_hour")
    gpu_usd = (
        training_seconds / 3600.0 * gpu_price
        if gpu_price is not None and training_seconds is not None and not missing_runtime
        else None
    )
    instance_usd = (
        active_elapsed / 3600.0 * gpu_price
        if gpu_price is not None and active_elapsed is not None else None
    )
    gpu_notes = []
    if gpu_price is None:
        gpu_notes.append("gpu_instance_usd_per_hour not configured")
    if missing_runtime:
        gpu_notes.append(f"{missing_runtime} experiment(s) without recorded training time")
    return {
        "campaign_id": campaign_id,
        "feedback_version": version,
        "seed": records[0].get("seed") if records else seed,
        "prices": prices,
        "llm": llm,
        "training": {
            "experiments": compute,
            "training_seconds_total": None if missing_runtime else training_seconds,
            "training_seconds_known_subtotal": training_seconds,
            "experiments_missing_training_seconds": missing_runtime,
        },
        "time": {
            "active_elapsed_seconds": active_elapsed,
            "agent_call_seconds": agent_seconds,
            "wall_span_seconds": wall_span,
            "note": "active elapsed sums agent processes; wall span includes paused time",
        },
        "cost_usd": {
            "bedrock": bedrock_usd,
            "gpu_training": gpu_usd,
            "total": (
                bedrock_usd + gpu_usd
                if bedrock_usd is not None and gpu_usd is not None else None
            ),
            "gpu_instance_over_active_elapsed": instance_usd,
            "notes": bedrock_notes + gpu_notes,
        },
    }


def _fmt(value, digits=2):
    if value is None:
        return "unknown"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def format_report(report):
    llm, training, timing, cost = (
        report["llm"], report["training"], report["time"], report["cost_usd"]
    )
    lines = [
        f"Resource report | campaign {report['campaign_id']} "
        f"({report['feedback_version'] or 'no experiments'}, seed {report['seed']})",
        "",
        f"LLM sessions: {llm['sessions']}",
        f"  input tokens:      {_fmt(llm['input_tokens'])}",
        f"  output tokens:     {_fmt(llm['output_tokens'])}",
        f"  model invocations: {_fmt(llm['model_invocations'])} "
        f"(failed attempts: {_fmt(llm['failed_model_calls'])})",
        f"  LLM latency (s):   {_fmt(llm['llm_latency_seconds'])}",
        "",
        "Training:",
    ]
    for run in training["experiments"]:
        lines.append(
            f"  #{run['experiment_id']} {run['architecture']} {run['status']}: "
            f"{_fmt(run['training_seconds'])} s on {run['device'] or 'unknown device'}"
        )
    lines += [
        f"  total training seconds: {_fmt(training['training_seconds_total'])}",
        "",
        f"Active elapsed (s): {_fmt(timing['active_elapsed_seconds'])}   "
        f"wall span (s): {_fmt(timing['wall_span_seconds'])}",
        "",
        f"Cost USD: Bedrock {_fmt(cost['bedrock'], 4)} + GPU training "
        f"{_fmt(cost['gpu_training'], 4)} = {_fmt(cost['total'], 4)}",
        f"  GPU instance over active elapsed: {_fmt(cost['gpu_instance_over_active_elapsed'], 4)}",
    ]
    lines += [f"  note: {note}" for note in cost["notes"]]
    return "\n".join(lines)

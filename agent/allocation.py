"""Adaptive compute allocation and checkpoint continuation (Phase 1D).

Claude decides how much training each search-graph candidate receives:
allocate a first slice of epochs, extend a promising candidate, or abandon one.

Identity versus allocation. A candidate is an immutable SolutionSpec; its
``training.epochs`` is the *horizon*: the length of the learning-rate schedule
and the most epochs the candidate can ever train. Solution IDs are unchanged
from Phase 1A-1C. How many of those epochs have actually been trained is
execution state, recorded per training action in the campaign's experiment
records (``configuration["allocation"]``), never in the spec. Training past the
horizon would change the schedule, so it is a different candidate: propose it.

Three relations are kept distinct:

* ancestry: a search-graph edge; a child never receives its parent's weights;
* initialization: warm-starting a different solution from saved weights
  (still reserved; SolutionSpec.initialization must stay empty);
* continuation: the same candidate's same training run resumes from its own
  saved training state (model, optimizer, scheduler, AMP scaler, RNG, history,
  best-so-far, early stopping), verified by SHA-256 before loading.

Accounting policy. Every training action, an initial allocation or a
continuation, is a separate reservation: it consumes one experiment slot before
any training, whether it later completes, fails, or is interrupted, and passes
the same guards as any launch (search-graph membership, A1, S0-S2, P1 decision
record, HITL gate, budgets). The optional training-time budget counts each
action's wall-clock seconds as measured by the campaign wrapper; a running
action reserves all remaining time and stops at an epoch boundary before
exceeding it. An action interrupted before finalization keeps its reservation
and its runtime is reported as unknown, never zero.

Adaptive compute requires search mode, so it is available only with the
history-visible E2/E3 feedback profiles.
"""

from __future__ import annotations

import math
import statistics
import threading
import uuid

from agent.experiments import _configuration_key, get_budget_status
from agent.search_graph import SearchError, comparability_key
from ml.solution import file_sha256

ACCOUNTING_POLICY = (
    "Every training action (initial allocation or continuation) reserves one experiment slot "
    "before training, including actions that later fail or are interrupted. The training-time "
    "budget counts each action's wall-clock seconds measured by the campaign wrapper "
    "(including data loading); a running action reserves all remaining time and stops at an "
    "epoch boundary before exceeding it. Interrupted actions keep their reservation; their "
    "runtime is unknown, never zero."
)

ALLOCATION_INSTRUCTIONS = """
ADAPTIVE COMPUTE: you decide how much training each candidate receives.
A candidate's epochs value is its horizon: the learning-rate schedule length and the most
epochs it can ever train. allocate_compute trains a proposed candidate for its first N
epochs; extend_candidate continues the same training run from its saved state (weights,
optimizer, scheduler, and random state), so continuing to the horizon is equivalent to
having trained there in one go. To train beyond a horizon, propose a new candidate with
more epochs (it starts fresh). abandon_candidate stops spending on a candidate.
run_campaign_experiment is not used in this mode.
Every allocation or extension is one training action: it uses one experiment slot and
its wall-clock time counts against any training-time budget, even if it fails.
Use inspect_compute (learning curve, epoch history, seconds per epoch, segments),
compute_budget, and compare_continue_vs_explore to weigh continuing a candidate against
exploring another branch. Justify every compute decision with the observation, the
expected benefit, and why it is worth its cost. How to spend the budget is your decision.
"""


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _base(configuration):
    """A run configuration without its allocation: identifies the candidate."""
    return {key: value for key, value in (configuration or {}).items() if key != "allocation"}


class ComputeAllocator:
    def __init__(self, graph, *, max_experiments, max_gpu_seconds=None):
        if graph.version not in ("E2", "E3"):
            raise ValueError("adaptive compute exposes learning curves and history; use E2/E3")
        self.graph = graph
        self.max_experiments = max_experiments
        self.max_gpu_seconds = max_gpu_seconds
        self._launch = None
        self.selection_policy = None
        self._in_flight = set()  # candidates training in this process
        self._guard = threading.Lock()

    def bind_launcher(self, launch, *, selection_policy):
        """Called by make_registered_training_tool with its guarded launch path."""
        self._launch = launch
        self.selection_policy = selection_policy

    # -- reconstruction -------------------------------------------------------------

    def _candidate_config(self, sid):
        configuration = self.graph.configuration_for(sid)
        if configuration is None:
            raise SearchError(
                "CONFIGURATION_UNKNOWN",
                f"{sid} has no recorded run arguments (proposed before adaptive compute); "
                "propose it again with propose_branch")
        dataset_id = configuration["dataset_id"]
        base = {key: value for key, value in configuration.items() if key != "dataset_id"}
        from agent.tools import campaign_solution

        if campaign_solution(dataset_id, base, self.graph.seed)[0] != sid:
            raise SearchError("CONFIGURATION_MISMATCH",
                              f"recorded run arguments do not reproduce {sid}")
        return dataset_id, base

    def _segment(self, record, sid):
        allocation = (record.get("configuration") or {}).get("allocation") or {}
        status = record.get("status")
        if status == "launched":
            status = "running" if sid in self._in_flight else "interrupted"
        trained = record.get("completed_epochs") if record.get("status") == "completed" else None
        seconds = record.get("training_seconds")
        state = record.get("training_state")
        return {
            "experiment_id": record.get("experiment_id"),
            "kind": allocation.get("kind", "unallocated run (no resumable state)"),
            "status": status,
            "allocated_epochs": allocation.get("epochs"),
            "until_epoch": allocation.get("until_epoch"),
            "start_epoch": record.get("start_epoch"),
            "completed_through_epoch": record.get("completed_through_epoch",
                                                  trained if not allocation else None),
            "epochs_trained": trained,
            "best_validation_score_so_far": record.get("validation_score")
            if record.get("status") == "completed" else None,
            "best_epoch": record.get("best_epoch"),
            "runtime_seconds": record.get("runtime_seconds") if record.get("status") in (
                "completed", "failed") else None,
            "training_seconds": seconds if _number(seconds) else None,
            "epoch_seconds": record.get("epoch_seconds"),
            "stopped_for_time": record.get("stopped_for_time"),
            "stopped_early": record.get("stopped_early"),
            "finished": record.get("finished"),
            "finish_reason": record.get("finish_reason"),
            "resumed_from_experiment_id": allocation.get("resume_from_experiment_id"),
            "training_state": state,
            "checkpoint_path": record.get("checkpoint_path"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "error": record.get("error"),
            "_record": record,
        }

    def _segments(self, sid, state=None):
        state = state or self.graph.state()
        found = []
        for record in state["records"]:
            owner = record.get("solution_id") or self.graph._solution_for_record(record)
            if owner == sid:
                found.append(self._segment(record, sid))
        return found, state

    @staticmethod
    def _public(segment):
        return {key: value for key, value in segment.items() if not key.startswith("_")}

    def _state_ok(self, segment):
        state = segment["training_state"]
        try:
            return bool(state) and file_sha256(state["path"]) == state["sha256"]
        except OSError:
            return False

    def _resume_source(self, segments, from_experiment_id=None):
        """The segment to continue, or a SearchError explaining why none can be."""
        completed = [s for s in segments if s["status"] == "completed"]
        if from_experiment_id is not None:
            chosen = next((s for s in completed if s["experiment_id"] == from_experiment_id), None)
            if chosen is None:
                raise SearchError("UNKNOWN_SEGMENT", f"no completed training action "
                                  f"{from_experiment_id} for this candidate")
        elif not completed:
            raise SearchError("NOT_STARTED", "this candidate has no completed training; "
                              "use allocate_compute first")
        else:
            chosen = completed[-1]
        if not chosen["training_state"]:
            raise SearchError("NOT_RESUMABLE", "this run saved only a model checkpoint (no "
                              "optimizer, scheduler, or RNG state), so it cannot be continued "
                              "exactly; propose a new candidate instead")
        if chosen["finished"]:
            raise SearchError("RUN_FINISHED", f"the training run already ended "
                              f"({chosen['finish_reason']}); propose a new candidate with a "
                              "longer horizon to train further")
        if not self._state_ok(chosen):
            intact = [{"experiment_id": s["experiment_id"],
                       "completed_through_epoch": s["completed_through_epoch"]}
                      for s in completed if s is not chosen and s["training_state"]
                      and not s["finished"] and self._state_ok(s)]
            raise SearchError(
                "CHECKPOINT_CORRUPT",
                f"the saved training state of action {chosen['experiment_id']} is missing or "
                "does not match its recorded SHA-256; it was not loaded. You may resume from "
                "an earlier intact state with resume_from_experiment_id.",
                {"recoverable_from": intact})
        return chosen

    # -- views -----------------------------------------------------------------------

    def budget(self, state=None):
        state = state or self.graph.state()
        status = get_budget_status(
            self.graph.version, self.graph.campaign_id, self.graph.dataset_id or "any",
            self.graph.seed, max_experiments=self.max_experiments,
            max_gpu_seconds=self.max_gpu_seconds)
        per_candidate, unknown = {}, 0
        for sid in state["nodes"]:
            segments, _ = self._segments(sid, state)
            if not segments:
                continue
            seconds = [s["runtime_seconds"] for s in segments]
            unknown += seconds.count(None)
            per_candidate[sid] = {
                "training_actions": len(segments),
                "wall_clock_seconds_known": sum(v for v in seconds if v is not None),
                "actions_with_unknown_runtime": seconds.count(None),
            }
        return {
            "training_actions_launched": status["launched_experiments"],
            "max_training_actions": status["max_experiments"],
            "remaining_training_actions": status["remaining_experiments"],
            "training_seconds_used_or_reserved": status["gpu_seconds_used_or_reserved"],
            "max_training_seconds": status["max_gpu_seconds"],
            "remaining_training_seconds": status["remaining_gpu_seconds"],
            "actions_with_unknown_runtime": unknown,
            "per_candidate": per_candidate,
            "accounting_policy": ACCOUNTING_POLICY,
        }

    def candidate(self, sid):
        state = self.graph.state()
        node = state["nodes"].get(sid)
        if node is None:
            raise SearchError("UNKNOWN_SOLUTION", f"{sid} is not a node of this search")
        segments, _ = self._segments(sid, state)
        curve = self._curve(segments)
        rates = [s for seg in segments for s in (seg["epoch_seconds"] or [])]
        try:
            source = self._resume_source(segments)
            continuation = {"can_continue": True, "from_experiment_id": source["experiment_id"],
                            "completed_through_epoch": source["completed_through_epoch"]}
        except SearchError as exc:
            continuation = {"can_continue": False, "reason_code": exc.code, "reason": str(exc),
                            **exc.detail}
        if node["abandoned"]:
            continuation = {"can_continue": False, "reason_code": "CANDIDATE_ABANDONED",
                            "reason": node["abandoned"]["reason"]}
        decisions = [e for e in state["events"] if e.get("solution_id") == sid
                     and e["type"] in ("allocation", "allocation_outcome", "abandon")]
        return {
            "solution_id": sid,
            "status": node["status"],
            "abandoned": node["abandoned"],
            "horizon_epochs": node["horizon_epochs"],
            "epochs_completed": node["epochs_completed"],
            "primary_metric": node["primary_metric"],
            "best_validation_score": node["validation_score"],
            "best_checkpoint_path": node["checkpoint_path"],
            "learning_curve": curve,
            "measured_seconds_per_epoch": {
                "median": statistics.median(rates) if rates else None,
                "max": max(rates) if rates else None, "epochs_measured": len(rates)},
            "segments": [self._public(s) for s in segments],
            "continuation": continuation,
            "decisions": decisions,
        }

    @staticmethod
    def _curve(segments):
        """Epoch history of the latest completed segment (it carries the whole run)."""
        completed = [s for s in segments if s["status"] == "completed"]
        if not completed:
            return []
        record = completed[-1]["_record"]
        values = record.get("validation_metric_curve") or []
        losses = record.get("train_loss") or []
        val_losses = record.get("val_loss") or []
        rates = (record.get("detailed_diagnostics") or {}).get("learning_rate_history") or []
        return [{"epoch": i + 1, "validation_metric": v,
                 "train_loss": losses[i] if i < len(losses) else None,
                 "validation_loss": val_losses[i] if i < len(val_losses) else None,
                 "learning_rate": rates[i] if i < len(rates) else None}
                for i, v in enumerate(values)]

    def compare(self, sid, alternative=None):
        """Measured facts for continue-versus-explore; the decision stays with Claude."""
        state = self.graph.state()
        view = self.candidate(sid)
        curve = [p["validation_metric"] for p in view["learning_curve"]]
        window = min(3, len(curve) - 1) if curve else 0
        gain = (curve[-1] - curve[-1 - window]) / window if window > 0 else None
        remaining = (view["horizon_epochs"] - view["epochs_completed"]
                     if view["continuation"]["can_continue"] else 0)
        per_epoch = view["measured_seconds_per_epoch"]["max"]
        best = None
        if view["primary_metric"] is not None:
            key = comparability_key(state["nodes"][sid]["spec"], view["primary_metric"])
            best = next((g for g in self.graph.incumbents(state["nodes"])
                         if tuple(g["group"].values()) == key), None)
        facts = {
            "candidate": {
                "solution_id": sid, "status": view["status"],
                "epochs_completed": view["epochs_completed"],
                "horizon_epochs": view["horizon_epochs"],
                "epochs_remaining_to_horizon": remaining,
                "can_continue": view["continuation"]["can_continue"],
                "best_validation_score": view["best_validation_score"],
                "last_validation_metric": curve[-1] if curve else None,
                "recent_gain_per_epoch": gain, "gain_window_epochs": window or None,
                "measured_seconds_per_epoch_max": per_epoch,
                "estimated_seconds_to_horizon": (per_epoch * remaining
                                                 if per_epoch is not None else None),
                "naive_linear_extrapolation_to_horizon": (
                    curve[-1] + gain * remaining if gain is not None and remaining else None),
                "extrapolation_note": ("straight-line extension of recent epochs; not a "
                                       "prediction (curves usually flatten and the learning "
                                       "rate keeps decaying)"),
            },
            "incumbent": best,
            "budget": self.budget(state),
        }
        if alternative is not None:
            other = state["nodes"].get(alternative)
            if other is None:
                raise SearchError("UNKNOWN_SOLUTION", f"{alternative} is not a node")
            facts["alternative"] = {
                "solution_id": alternative, "status": other["status"],
                "horizon_epochs": other["horizon_epochs"],
                "best_validation_score": other["validation_score"],
                "abandoned": other["abandoned"],
                "measured_seconds_per_epoch_same_model": self._similar_rate(
                    state, other["spec"]),
            }
        return facts

    def _similar_rate(self, state, spec):
        """Slowest measured seconds/epoch of this campaign's runs with the same model and size."""
        rates = []
        for sid, node in state["nodes"].items():
            if (node["spec"]["model"]["model_id"] == spec["model"]["model_id"]
                    and node["spec"]["inputs"]["image_size"] == spec["inputs"]["image_size"]):
                segments, _ = self._segments(sid, state)
                rates += [s for seg in segments for s in (seg["epoch_seconds"] or [])]
        return max(rates) if rates else None

    def report(self):
        state = self.graph.state()
        return {
            "budget": self.budget(state),
            "candidates": {sid: self.candidate(sid) for sid in sorted(state["nodes"])},
        }

    # -- decisions -------------------------------------------------------------------

    def _decision(self, sid, action, justification, describe):
        missing = [name for name in ("observation", "expected_benefit", "cost_justification")
                   if not _text(justification.get(name))]
        if missing:
            raise SearchError("JUSTIFICATION_REQUIRED",
                              "compute decisions need observation, expected_benefit, and "
                              "cost_justification", {"missing": missing})
        if self.selection_policy == "P1":
            return {"observation": justification["observation"].strip(), "action": describe,
                    "expected_signal": justification["expected_benefit"].strip(),
                    "cost_justification": justification["cost_justification"].strip()}
        return None

    def _time_check(self, state, per_epoch):
        if self.max_gpu_seconds is None:
            return
        remaining = self.budget(state)["remaining_training_seconds"]
        if remaining is not None and remaining <= 0:
            raise SearchError("BUDGET_EXHAUSTED", "the training-time budget is exhausted")
        if per_epoch is not None and remaining is not None and remaining < per_epoch:
            raise SearchError("INSUFFICIENT_TRAINING_TIME",
                              f"{remaining:.1f}s remain but one epoch of this candidate was "
                              f"measured at {per_epoch:.1f}s; no budget was used")

    def _run(self, sid, action, dataset_id, base, allocation, justification, admit, describe):
        if self._launch is None:
            raise RuntimeError("allocator is not bound to the guarded training tool")
        decision_record = self._decision(sid, action, justification, describe)
        with self._guard:
            if sid in self._in_flight:
                raise SearchError("CANDIDATE_RUNNING", f"{sid} is already training")
            self._in_flight.add(sid)
        try:
            decision = self.graph.append_event({
                "type": "allocation", "decision_id": uuid.uuid4().hex, "action": action,
                "solution_id": sid, "allocation": allocation,
                **{key: justification[key].strip() for key in
                   ("observation", "expected_benefit", "cost_justification")},
                "budget_before": {k: v for k, v in self.budget().items() if k != "per_candidate"},
            })
            response = self._launch(dataset_id, base, decision_record, allocation, admit)
            self.graph.append_event({
                "type": "allocation_outcome", "decision_id": decision["decision_id"],
                "solution_id": sid, "status": response.get("status"),
                "experiment_id": response.get("experiment_id"),
                "error_code": response.get("error_code"),
            })
        finally:
            with self._guard:
                self._in_flight.discard(sid)
        view = self.candidate(sid)
        return {**response, "decision_id": decision["decision_id"], "allocation": allocation,
                "candidate": {key: view[key] for key in (
                    "status", "epochs_completed", "horizon_epochs", "best_validation_score",
                    "continuation", "measured_seconds_per_epoch")},
                "budget": {k: v for k, v in self.budget().items() if k != "per_candidate"}}

    def allocate(self, sid, epochs, justification):
        dataset_id, base = self._candidate_config(sid)
        segments, state = self._segments(sid)
        node = state["nodes"].get(sid)
        if node["abandoned"]:
            raise SearchError("CANDIDATE_ABANDONED", f"{sid} was abandoned")
        if any(s["status"] in ("completed", "running") for s in segments):
            raise SearchError("ALREADY_ALLOCATED", "this candidate already trained; use "
                              "extend_candidate to continue its training run")
        horizon = base["epochs"]
        if type(epochs) is not int or not 1 <= epochs <= horizon:
            raise SearchError("INVALID_ALLOCATION", f"epochs must be 1..{horizon} (the horizon)")
        self._time_check(state, None)
        allocation = {"kind": "initial", "epochs": epochs, "until_epoch": epochs,
                      "attempt": len(segments) + 1}
        key = _configuration_key(base)

        def admit(runs):  # under the storage lock: still no completed training for it
            for run in runs:
                if (_configuration_key(_base(run["configuration"])) == key
                        and run["status"] == "completed"):
                    return {"status": "rejected", "error_code": "ALREADY_ALLOCATED",
                            "message": "This candidate was allocated concurrently."}
            return None
        return self._run(sid, "allocate", dataset_id, base, allocation, justification, admit,
                         f"allocate epochs 1-{epochs} of {horizon} to {sid}")

    def extend(self, sid, additional_epochs, justification, resume_from_experiment_id=None):
        dataset_id, base = self._candidate_config(sid)
        segments, state = self._segments(sid)
        if state["nodes"][sid]["abandoned"]:
            raise SearchError("CANDIDATE_ABANDONED", f"{sid} was abandoned")
        if any(s["status"] == "running" for s in segments):
            raise SearchError("CANDIDATE_RUNNING", f"{sid} is already training")
        source = self._resume_source(segments, resume_from_experiment_id)
        horizon = base["epochs"]
        through = source["completed_through_epoch"]
        if type(additional_epochs) is not int or additional_epochs < 1:
            raise SearchError("INVALID_ALLOCATION", "additional_epochs must be a positive integer")
        if through + additional_epochs > horizon:
            raise SearchError(
                "BEYOND_HORIZON",
                f"{through} of {horizon} epochs are done; at most {horizon - through} more fit "
                "this candidate's schedule. To train longer, propose a new candidate with more "
                "epochs (it trains fresh).")
        rates = [s for seg in segments for s in (seg["epoch_seconds"] or [])]
        self._time_check(state, max(rates) if rates else None)
        same_source = [s for s in segments if s["resumed_from_experiment_id"]
                       == source["experiment_id"] and s["until_epoch"] == through + additional_epochs]
        allocation = {
            "kind": "continuation", "epochs": additional_epochs,
            "until_epoch": through + additional_epochs,
            "resume_from_experiment_id": source["experiment_id"],
            "resume_state_path": source["training_state"]["path"],
            "resume_state_sha256": source["training_state"]["sha256"],
            "attempt": len(same_source) + 1,
        }
        key = _configuration_key(base)
        explicit = resume_from_experiment_id is not None

        def admit(runs):  # under the storage lock: the source is still this run's latest state
            mine = [run for run in runs if _configuration_key(_base(run["configuration"])) == key]
            done = [run for run in mine if run["status"] == "completed"]
            if not explicit and done and done[-1]["experiment_id"] != source["experiment_id"]:
                return {"status": "rejected", "error_code": "STALE_CONTINUATION",
                        "message": "The candidate advanced concurrently; inspect it again."}
            return None
        return self._run(sid, "extend", dataset_id, base, allocation, justification, admit,
                         f"continue {sid} from epoch {through} to {through + additional_epochs} "
                         f"of {horizon}")

    def abandon(self, sid, reason):
        if not _text(reason):
            raise SearchError("JUSTIFICATION_REQUIRED", "abandoning needs a reason")
        state = self.graph.state()
        node = state["nodes"].get(sid)
        if node is None:
            raise SearchError("UNKNOWN_SOLUTION", f"{sid} is not a node of this search")
        if node["abandoned"]:
            raise SearchError("ALREADY_ABANDONED", f"{sid} was already abandoned")
        if sid in self._in_flight:
            raise SearchError("CANDIDATE_RUNNING", f"{sid} is training; abandon it afterwards")
        event = self.graph.append_event({"type": "abandon", "solution_id": sid,
                                         "reason": reason.strip()})
        return {"status": "abandoned", "solution_id": sid, "at": event["at"],
                "best_validation_score": node["validation_score"],
                "note": "Its measured results stay in the graph; no further compute is spent.",
                "budget": {k: v for k, v in self.budget(state).items() if k != "per_candidate"}}


def make_allocation_tools(allocator: ComputeAllocator):
    from strands import tool

    def guarded(call):
        try:
            return call()
        except SearchError as exc:
            return exc.to_dict()

    @tool
    def inspect_compute(solution_id: str) -> dict:
        """Show a candidate's training segments, learning curve (validation metric, losses,
        learning rate per epoch), measured seconds per epoch, epochs completed of its
        horizon, whether and from where it can be continued, and past compute decisions."""
        return guarded(lambda: {"status": "success", **allocator.candidate(solution_id)})

    @tool
    def compute_budget() -> dict:
        """Show remaining training actions and training seconds for the campaign, spending
        per candidate, actions whose runtime is unknown, and the accounting policy."""
        return {"status": "success", **allocator.budget()}

    @tool
    def compare_continue_vs_explore(solution_id: str,
                                    alternative_solution_id: str | None = None) -> dict:
        """Measured facts for deciding between continuing solution_id and exploring
        alternative_solution_id (or any other branch): recent validation gain per epoch,
        epochs left to the horizon, measured cost per epoch, the incumbent, and the budget."""
        return guarded(lambda: {"status": "success",
                                **allocator.compare(solution_id, alternative_solution_id)})

    @tool
    def allocate_compute(solution_id: str, epochs: int, observation: str,
                         expected_benefit: str, cost_justification: str) -> dict:
        """Train a proposed candidate for its first `epochs` epochs (1..its horizon).

        Args:
            solution_id: A candidate proposed with propose_branch.
            epochs: Epochs to train now; continue later with extend_candidate.
            observation: The evidence motivating this allocation.
            expected_benefit: What validation result you expect it to show.
            cost_justification: Why this amount of compute is worth spending now.
        """
        return guarded(lambda: allocator.allocate(solution_id, epochs, {
            "observation": observation, "expected_benefit": expected_benefit,
            "cost_justification": cost_justification}))

    @tool
    def extend_candidate(solution_id: str, additional_epochs: int, observation: str,
                         expected_benefit: str, cost_justification: str,
                         resume_from_experiment_id: int | None = None) -> dict:
        """Continue a candidate's training run for more epochs from its saved state.

        Args:
            solution_id: A candidate that already trained with allocate_compute.
            additional_epochs: Further epochs; the total cannot exceed the horizon.
            observation: The learning-curve evidence motivating more compute.
            expected_benefit: The improvement you expect.
            cost_justification: Why this is a better use of budget than exploring.
            resume_from_experiment_id: Only to recover from a corrupt latest state:
                an earlier training action whose state is intact.
        """
        return guarded(lambda: allocator.extend(solution_id, additional_epochs, {
            "observation": observation, "expected_benefit": expected_benefit,
            "cost_justification": cost_justification}, resume_from_experiment_id))

    @tool
    def abandon_candidate(solution_id: str, reason: str) -> dict:
        """Stop spending compute on a candidate. Its results stay visible."""
        return guarded(lambda: allocator.abandon(solution_id, reason))

    return [inspect_compute, compute_budget, compare_continue_vs_explore, allocate_compute,
            extend_candidate, abandon_candidate]

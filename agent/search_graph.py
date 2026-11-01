"""Persistent solution search graph for one campaign (AIDE-inspired).

Nodes are existing immutable solutions (ml/solution.py SolutionSpec ids); a
node's status, validation score, cost, and checkpoint come from the campaign's
experiment records and the solution's run events, never from a copy. The only
new state is an append-only edge log per campaign::

    <RESULTS_DIR>/search/<campaign_id>.jsonl

Each line is one event:

* ``edge``: child solution, 0 parents (a root), 1 parent, or 2 parents (a
  combination), plus intent, observation, hypothesis, expected signal, the
  changes Claude claimed, and the delta the system computed from the specs.
* ``conclude``: Claude ended the search; later launches are refused.

Writes happen under the experiment-storage lock, after re-reading the log, so
concurrent proposals cannot create cycles or duplicate edges. Ancestry is not
initialization: branching from a solution never warm-starts from its weights.
The graph never launches training; the guarded campaign tool does, and in
search mode it only accepts solutions that are nodes of this graph.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone

from agent import experiments
from agent.experiments import get_budget_status, load_campaign_experiments
from ml.solution import load_solution
from ml.solution import propose_solution as record_proposal

SCHEMA = "gotrain.search_graph/1"
INTENTS = ("root", "exploit", "explore", "refine", "revisit", "combine")
HISTORY_MODES = ("E2", "E3")
METRIC_DIRECTION = {"accuracy": "max", "mean_iou": "max"}
SPEC_SECTIONS = ("inputs", "model", "pipeline", "training")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class SearchError(ValueError):
    def __init__(self, code, message, detail=None):
        super().__init__(message)
        self.code = code
        self.detail = detail or {}

    def to_dict(self):
        return {"status": "rejected", "error_code": self.code, "message": str(self), **self.detail}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _log_path(campaign_id):
    if not _SAFE_ID.match(campaign_id or ""):
        raise SearchError("INVALID_CAMPAIGN", f"unsafe campaign id {campaign_id!r}")
    return experiments.RESULTS_DIR / "search" / f"{campaign_id}.jsonl"


def flatten(spec) -> dict:
    """Comparable view of a solution: dotted keys over the reproducible sections."""
    flat = {}

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}", item)
        else:
            flat[prefix] = value
    for section in SPEC_SECTIONS:
        walk(section, spec.get(section, {}))
    flat["custom_components"] = spec.get("custom_components", [])
    return flat


def spec_delta(parent_spec, child_spec) -> dict:
    before, after = flatten(parent_spec), flatten(child_spec)
    return {key: [before.get(key), after.get(key)]
            for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}


def _covers(claim, key):
    return key == claim or key.startswith(claim + ".")


def comparability_key(spec, primary_metric):
    """Results are ranked together only within one dataset, task, metric, split,
    and (for segmentation) evaluation resolution."""
    task = spec["dataset"]["task_type"]
    resolution = spec["inputs"]["image_size"] or tuple(spec["inputs"]["native_resolution"])
    return (spec["dataset"]["dataset_id"], task, primary_metric, "validation",
            str(resolution) if task == "segmentation" else None)


class SearchGraph:
    """Read model reconstructed from persisted data on every call."""

    def __init__(self, campaign_id, version, seed, max_experiments, dataset_id=None):
        self.campaign_id = campaign_id
        self.version = version
        self.seed = seed
        self.max_experiments = max_experiments
        self.dataset_id = dataset_id
        _log_path(campaign_id)

    # -- persistence ---------------------------------------------------------------

    def _events(self):
        path = _log_path(self.campaign_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def _append(self, event):
        path = _log_path(self.campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema": SCHEMA, **event}, sort_keys=True, default=str) + "\n")
            handle.flush()

    # -- reconstruction ------------------------------------------------------------

    @staticmethod
    def _topology(events):
        """Nodes, parents, and children from edge events alone (takes no lock)."""
        edges = [event for event in events if event["type"] == "edge"]
        nodes = {}
        for edge in edges:
            node = nodes.setdefault(edge["child"], {"solution_id": edge["child"], "parents": [],
                                                    "children": [], "edges_in": []})
            node["edges_in"].append(edge["edge_id"])
            for parent in edge["parents"]:
                if parent not in node["parents"]:
                    node["parents"].append(parent)
                nodes.setdefault(parent, {"solution_id": parent, "parents": [], "children": [],
                                          "edges_in": []})
                if edge["child"] not in nodes[parent]["children"]:
                    nodes[parent]["children"].append(edge["child"])
        concluded = next((e for e in reversed(events) if e["type"] == "conclude"), None)
        return nodes, edges, concluded

    def state(self):
        # Never nest the storage lock (flock deadlocks on Linux): read the log under
        # the lock, release it, then read campaign records (which lock themselves).
        with experiments._locked_storage():
            events = self._events()
        records = load_campaign_experiments(self.version, self.campaign_id, self.seed)
        nodes, edges, concluded = self._topology(events)
        runs = {}
        for record in records:
            sid = record.get("solution_id") or self._solution_for_record(record)
            if sid:
                runs[sid] = record
        for sid, node in nodes.items():
            node.update(self._node_result(sid, runs.get(sid)))
        return {"nodes": nodes, "edges": edges, "records": records, "concluded": concluded}

    def _solution_for_record(self, record):
        """Recompute a run's solution id from its configuration (robust to crashes)."""
        from agent.tools import campaign_solution

        try:
            return campaign_solution(record["dataset"], record["configuration"], record["seed"])[0]
        except Exception:
            return None

    def _node_result(self, sid, record):
        spec = load_solution(sid)["spec"]
        result = {"spec": spec, "status": "proposed", "experiment_id": None,
                  "validation_score": None, "primary_metric": None, "training_seconds": None,
                  "checkpoint_path": None, "checkpoint_sha256": None, "error": None}
        if record is None:
            return result
        status = record.get("status")
        result.update({
            "status": {"launched": "running"}.get(status, status),
            "experiment_id": record.get("experiment_id"),
            "primary_metric": record.get("primary_metric"),
            "error": record.get("error"),
        })
        score = record.get("validation_score")
        if status == "completed" and isinstance(score, (int, float)) and math.isfinite(score):
            result["validation_score"] = float(score)
        seconds = record.get("training_seconds", record.get("runtime_seconds"))
        if status in ("completed", "failed") and isinstance(seconds, (int, float)):
            result["training_seconds"] = float(seconds)  # unknown stays None
        for run in reversed(load_solution(sid)["runs"]):
            if run.get("campaign_id") == self.campaign_id and run.get("artifacts"):
                result["checkpoint_path"] = run["artifacts"].get("checkpoint_path")
                result["checkpoint_sha256"] = run["artifacts"].get("checkpoint_sha256")
                break
        return result

    @staticmethod
    def ancestors(nodes, sid):
        seen, stack = set(), list(nodes.get(sid, {}).get("parents", []))
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(nodes.get(current, {}).get("parents", []))
        return seen

    # -- incumbent -----------------------------------------------------------------

    def incumbents(self, nodes):
        """Best measured validation result per comparable group; ties are reported."""
        groups = {}
        for sid, node in nodes.items():
            metric = node["primary_metric"]
            if node["status"] != "completed" or node["validation_score"] is None:
                continue
            if metric not in METRIC_DIRECTION:
                continue  # unknown direction: never ranked
            groups.setdefault(comparability_key(node["spec"], metric), []).append(node)
        result = []
        for key, members in groups.items():
            sign = 1 if METRIC_DIRECTION[key[2]] == "max" else -1
            best = max(sign * member["validation_score"] for member in members)
            leaders = sorted(m["solution_id"] for m in members if sign * m["validation_score"] == best)
            result.append({
                "group": {"dataset_id": key[0], "task_type": key[1], "primary_metric": key[2],
                          "split": key[3], "evaluation_resolution": key[4]},
                "direction": METRIC_DIRECTION[key[2]],
                "incumbent": leaders[0], "tied_with": leaders[1:],
                "validation_score": sign * best, "candidates_compared": len(members),
            })
        return result

    # -- proposals -----------------------------------------------------------------

    def propose(self, child_spec, *, parents, intent, observation, hypothesis, expected_signal,
                claimed_changes=None, changes_from_second_parent=None, context=None):
        """Validate and append one edge atomically; returns the edge (new or existing)."""
        parents = list(parents or [])
        errors = {}
        if intent not in INTENTS:
            errors["intent"] = f"must be one of {list(INTENTS)}"
        for name, text in (("observation", observation), ("hypothesis", hypothesis),
                           ("expected_signal", expected_signal)):
            if not isinstance(text, str) or not text.strip():
                errors[name] = "required nonempty text"
        if (intent == "root") != (not parents):
            errors["parents"] = "a root has no parent; every other intent names a parent"
        if intent == "combine" and len(parents) != 2:
            errors["parents"] = "combine needs exactly two parents"
        if intent != "combine" and len(parents) > 1:
            errors["parents"] = "only combine may name two parents"
        if len(set(parents)) != len(parents):
            errors["parents"] = "parents must be distinct"
        if errors:
            raise SearchError("INVALID_PROPOSAL", "fix the listed fields", {"field_errors": errors})
        if self.dataset_id and child_spec["dataset"]["dataset_id"] != self.dataset_id:
            raise SearchError("DATASET_MISMATCH", f"this search is restricted to {self.dataset_id}")

        with experiments._locked_storage():  # check-then-append is atomic
            nodes, edges, concluded = self._topology(self._events())
            if concluded:
                raise SearchError("SEARCH_CONCLUDED", "the search was concluded; no new branches")
            missing = [p for p in parents if p not in nodes]
            if missing:
                raise SearchError("UNKNOWN_PARENT", f"parents {missing} are not nodes of this search",
                                  {"known_nodes": sorted(nodes)})
            parent_specs = {p: load_solution(p)["spec"] for p in parents}
            for parent, spec in parent_specs.items():
                if spec["dataset"] != child_spec["dataset"]:
                    raise SearchError("INCOMPATIBLE_PARENT", f"{parent} is for another dataset")
            child_sid = record_proposal(child_spec, {
                **(context or {}), "campaign_id": self.campaign_id,
                "parent_solution_id": parents[0] if parents else None,
                "change_description": hypothesis,
            })
            if child_sid in parents:
                raise SearchError("NO_CHANGE", "the proposal is identical to its parent; branch "
                                  "from it with a real change instead of re-proposing it")
            for parent in parents:
                if child_sid in self.ancestors(nodes, parent):
                    raise SearchError("CYCLE", f"{child_sid} is an ancestor of {parent}; propose "
                                      "a new change from it instead of closing a loop")
            deltas = {p: spec_delta(spec, child_spec) for p, spec in parent_specs.items()}
            self._check_claims(intent, parents, deltas, parent_specs, child_spec,
                               claimed_changes, changes_from_second_parent)
            for edge in edges:
                if edge["child"] == child_sid and sorted(edge["parents"]) == sorted(parents):
                    return {**edge, "duplicate": True}
            edge = {
                "type": "edge", "edge_id": uuid.uuid4().hex, "at": _now(),
                "campaign_id": self.campaign_id, "child": child_sid, "parents": parents,
                "intent": intent, "observation": observation.strip(),
                "hypothesis": hypothesis.strip(), "expected_signal": expected_signal.strip(),
                "claimed_changes": claimed_changes or [],
                "changes_from_second_parent": changes_from_second_parent or [],
                "delta": deltas, "reused_existing_node": child_sid in nodes,
                "initialization": "fresh (ancestry is not a warm start)",
                "proposer": (context or {}).get("proposer", "claude"),
            }
            self._append(edge)
            return {**edge, "duplicate": False}

    @staticmethod
    def _check_claims(intent, parents, deltas, parent_specs, child_spec, claimed, from_second):
        if intent == "root":
            return
        base = deltas[parents[0]]
        claimed = list(claimed or [])
        unclaimed = [key for key in base if not any(_covers(c, key) for c in claimed)]
        unchanged = [c for c in claimed if not any(_covers(c, key) for key in base)]
        if unclaimed or unchanged:
            raise SearchError(
                "DELTA_MISMATCH",
                "claimed_changes must name exactly what differs from the first parent",
                {"actual_delta": base, "unclaimed_changes": unclaimed,
                 "claimed_but_unchanged": unchanged})
        if intent == "combine":
            donor = flatten(parent_specs[parents[1]])
            child = flatten(child_spec)
            from_second = list(from_second or [])
            taken = [key for key in base if any(_covers(c, key) for c in from_second)]
            wrong = [key for key in taken if child.get(key) != donor.get(key)]
            if not from_second or not taken or wrong:
                raise SearchError(
                    "COMBINE_NOT_PRESENT",
                    "a combination must actually include the claimed changes from the second "
                    "parent", {"claimed_from_second": from_second, "not_matching_second": wrong,
                               "actual_delta_from_first": base})

    def conclude(self, reason, selected_solution_id=None):
        with experiments._locked_storage():
            nodes, _, _ = self._topology(self._events())
            if selected_solution_id is not None and selected_solution_id not in nodes:
                raise SearchError("UNKNOWN_SOLUTION", f"{selected_solution_id} is not a node")
            event = {"type": "conclude", "at": _now(), "campaign_id": self.campaign_id,
                     "reason": reason, "selected_solution_id": selected_solution_id}
            self._append(event)
            return event

    # -- launch gate ---------------------------------------------------------------

    def check_launch(self, solution_id):
        """Search mode: only graph nodes that have not run yet may be launched."""
        current = self.state()
        if current["concluded"]:
            return {"status": "rejected", "error_code": "SEARCH_CONCLUDED",
                    "message": "The search was concluded; no further runs."}
        node = current["nodes"].get(solution_id)
        if node is None:
            return {"status": "rejected", "error_code": "NOT_IN_SEARCH_GRAPH",
                    "message": "Propose this solution with propose_branch (naming its parent) "
                               "before launching it. No budget was used.",
                    "solution_id": solution_id}
        return None

    # -- views ---------------------------------------------------------------------

    def summary(self, node):
        spec = node["spec"]
        return {
            "solution_id": node["solution_id"], "status": node["status"],
            "primary_metric": node["primary_metric"], "validation_score": node["validation_score"],
            "training_seconds": node["training_seconds"], "parents": node["parents"],
            "children": node["children"], "model": spec["model"]["model_id"],
            "input_view": spec["inputs"]["input_view"],
            "custom_components": [c["component_id"] for c in spec.get("custom_components", [])],
        }

    def frontier(self):
        current = self.state()
        nodes = current["nodes"]
        budget = get_budget_status(self.version, self.campaign_id, self.dataset_id or "any",
                                   self.seed, max_experiments=self.max_experiments)
        return {
            "campaign_id": self.campaign_id,
            "nodes": [self.summary(n) for n in sorted(nodes.values(), key=lambda n: n["solution_id"])],
            "edges": [{k: e[k] for k in ("edge_id", "parents", "child", "intent", "hypothesis")}
                      for e in current["edges"]],
            "ready_to_launch": sorted(s for s, n in nodes.items() if n["status"] == "proposed"),
            "failed": sorted(s for s, n in nodes.items() if n["status"] == "failed"),
            "incumbents": self.incumbents(nodes),
            "budget": {key: budget[key] for key in ("launched_experiments", "max_experiments",
                                                    "remaining_experiments")},
            "concluded": current["concluded"],
        }

    def inspect(self, solution_id):
        current = self.state()
        node = current["nodes"].get(solution_id)
        if node is None:
            raise SearchError("UNKNOWN_SOLUTION", f"{solution_id} is not a node of this search")
        return {
            **self.summary(node),
            "checkpoint_path": node["checkpoint_path"], "checkpoint_sha256": node["checkpoint_sha256"],
            "error": node["error"],
            "incoming_edges": [e for e in current["edges"] if e["child"] == solution_id],
            "outgoing_edges": [{k: e[k] for k in ("edge_id", "child", "intent", "hypothesis")}
                               for e in current["edges"] if solution_id in e["parents"]],
            "specification": {section: node["spec"][section] for section in SPEC_SECTIONS},
        }

    def compare(self, first, second):
        nodes = self.state()["nodes"]
        for sid in (first, second):
            if sid not in nodes:
                raise SearchError("UNKNOWN_SOLUTION", f"{sid} is not a node of this search")
        a, b = nodes[first], nodes[second]
        comparable = (a["primary_metric"] is not None and a["primary_metric"] == b["primary_metric"]
                      and comparability_key(a["spec"], a["primary_metric"])
                      == comparability_key(b["spec"], b["primary_metric"]))
        difference = None
        if comparable and None not in (a["validation_score"], b["validation_score"]):
            difference = b["validation_score"] - a["validation_score"]
        common = (self.ancestors(nodes, first) | {first}) & (self.ancestors(nodes, second) | {second})
        return {
            "first": self.summary(a), "second": self.summary(b),
            "delta_first_to_second": spec_delta(a["spec"], b["spec"]),
            "comparable": comparable,
            "score_difference_second_minus_first": difference,
            "direction": METRIC_DIRECTION.get(a["primary_metric"]) if comparable else None,
            "common_ancestors": sorted(common),
        }

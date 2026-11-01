"""Strands tools for the solution search graph (``--search``).

Search mode exposes history (every candidate, edge, and result in the
campaign), so it is offered only with the E2/E3 feedback profiles; E0/E1
campaigns are refused rather than silently given history. The tools never
train: proposals are validated and recorded, and launching still goes through
run_campaign_experiment, which in search mode accepts only graph nodes.
"""

from __future__ import annotations

from strands import tool

from agent.autonomy import AutonomyPolicy, scope_rejection
from agent.search_graph import HISTORY_MODES, INTENTS, SearchError, SearchGraph
from agent.tools import campaign_solution, check_run_arguments
from ml.components import PipelineError

SEARCH_INSTRUCTIONS = f"""
SEARCH MODE: this campaign is a search over complete candidate solutions.
Use search_frontier to see every candidate, its status, measured validation
results, the incumbent per comparable group, candidates ready to launch, and the
remaining budget. Use inspect_candidate and compare_candidates to study them.
Before launching anything, record it with propose_branch: name the parent
solution (none only for a root), your intent ({", ".join(INTENTS)}), the
observation and hypothesis motivating it, the validation signal you expect, and
claimed_changes listing exactly which settings differ from the parent (dotted
keys such as "training.learning_rate" or "pipeline.optimizer"). You may branch
from any candidate, not only the latest, and revisit earlier branches.
run_campaign_experiment only launches candidates you proposed, each at most
once. Branching never reuses a parent's weights. When the remaining budget does
not justify another run, call conclude_search. Strategy is your decision.
"""


def make_search_tools(*, graph: SearchGraph, autonomy: AutonomyPolicy,
                      required_dataset=None):
    if graph.version not in HISTORY_MODES:
        raise ValueError(
            f"search mode needs a history-visible feedback profile {HISTORY_MODES}; "
            f"{graph.version} campaigns must not see the solution graph"
        )

    @tool
    def search_frontier() -> dict:
        """Show all candidate solutions in this campaign with status, measured validation
        results, parents and children, the best incumbent per comparable group,
        candidates ready to launch, failed candidates, and the remaining budget."""
        return {"status": "success", **graph.frontier()}

    @tool
    def inspect_candidate(solution_id: str) -> dict:
        """Show one candidate: full specification, incoming edges with their rationale and
        delta, children, measured results, checkpoint path and hash."""
        try:
            return {"status": "success", "candidate": graph.inspect(solution_id)}
        except SearchError as exc:
            return exc.to_dict()

    @tool
    def compare_candidates(first_solution_id: str, second_solution_id: str) -> dict:
        """Compare two candidates: exact configuration delta, measured results, whether
        their metrics are comparable, and their common ancestors."""
        try:
            return {"status": "success", **graph.compare(first_solution_id, second_solution_id)}
        except SearchError as exc:
            return exc.to_dict()

    @tool
    def propose_branch(
        intent: str,
        observation: str,
        hypothesis: str,
        expected_signal: str,
        dataset_id: str,
        model_id: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        parent_solution_id: str | None = None,
        claimed_changes: list[str] | None = None,
        weight_decay: float = 0.0001,
        pretrained: bool = True,
        image_size: int | None = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.0,
        pipeline: dict | None = None,
        second_parent_id: str | None = None,
        changes_from_second_parent: list[str] | None = None,
    ) -> dict:
        """Record a candidate solution in the search graph without training or using budget.

        Args:
            intent: root, exploit, explore, refine, revisit, or combine.
            observation: What the evidence so far shows.
            hypothesis: Why this change should help.
            expected_signal: The validation result that would support the hypothesis.
            parent_solution_id: The candidate this one changes (omit only for a root).
            claimed_changes: Dotted keys that differ from the parent, e.g.
                "training.learning_rate" or "pipeline.loss"; must match exactly.
            pipeline: Optional composed components (scope S1 or S2).
            second_parent_id: For combine: the candidate contributing other ideas.
            changes_from_second_parent: For combine: keys taken from the second parent.
        """
        if required_dataset is not None and dataset_id != required_dataset:
            return {"status": "rejected", "error_code": "DATASET_NOT_ALLOWED",
                    "message": f"This campaign is restricted to {required_dataset}."}
        if pipeline is not None and not autonomy.composes_pipelines:
            return {"status": "rejected", "error_code": "SCOPE_NOT_PERMITTED",
                    "message": "Composing pipeline components requires scope S1."}
        out_of_scope = scope_rejection(autonomy, pipeline)
        if out_of_scope:
            return out_of_scope
        configuration = {
            "architecture": model_id, "learning_rate": learning_rate, "epochs": epochs,
            "batch_size": batch_size, "weight_decay": weight_decay, "pretrained": pretrained,
            "image_size": image_size, "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        }
        if pipeline is not None:
            configuration["pipeline"] = pipeline
        try:
            check_run_arguments(dataset_id, model_id, epochs, batch_size, learning_rate,
                                weight_decay, image_size, early_stopping_patience,
                                early_stopping_min_delta)
            _, spec = campaign_solution(dataset_id, configuration, graph.seed)
        except PipelineError as exc:
            return {"status": "rejected", "error_code": "INVALID_PIPELINE", "errors": exc.errors}
        except Exception as exc:
            return {"status": "rejected", "error_code": "INVALID_MODEL_CONFIGURATION",
                    "message": str(exc)}
        parents = [p for p in (parent_solution_id, second_parent_id) if p]
        try:
            edge = graph.propose(
                spec, parents=parents, intent=intent, observation=observation,
                hypothesis=hypothesis, expected_signal=expected_signal,
                claimed_changes=claimed_changes,
                changes_from_second_parent=changes_from_second_parent,
                context={"feedback_version": graph.version, "autonomy": autonomy.to_dict(),
                         "proposer": "claude"},
                configuration={"dataset_id": dataset_id, **configuration},
            )
        except SearchError as exc:
            return exc.to_dict()
        return {
            "status": "proposed", "solution_id": edge["child"], "edge_id": edge["edge_id"],
            "parents": edge["parents"], "delta": edge["delta"],
            "reused_existing_node": edge["reused_existing_node"],
            "duplicate_edge": edge["duplicate"],
            "message": "Recorded; launch it with run_campaign_experiment using the same arguments.",
        }

    @tool
    def conclude_search(reason: str, selected_solution_id: str | None = None) -> dict:
        """End the search (for example when the remaining budget does not justify another
        run). No further candidates can be proposed or launched afterwards."""
        try:
            return {"status": "concluded", **graph.conclude(reason, selected_solution_id),
                    "incumbents": graph.incumbents(graph.state()["nodes"])}
        except SearchError as exc:
            return exc.to_dict()

    return [search_frontier, inspect_candidate, compare_candidates, propose_branch,
            conclude_search]

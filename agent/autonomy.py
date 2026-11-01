"""Autonomy controls: who decides (authority) and what may change (scope).

Authority:
- A0: deterministic or constrained system decisions (validation, budgets,
  forced comparison runs). Never granted to Claude.
- A1: Claude proposes a solution with ``propose_solution``; the system
  validates and records it, and only proposed solutions may be launched.
- A2: Claude chooses and launches directly within hard safety boundaries
  (the historical behavior).

Scope:
- S0: existing configuration knobs (model, epochs, batch size, learning rate,
  weight decay, pretrained, image size, early stopping). Historical default.
- S1: S0 plus composing supported pipeline components (ml/components.py).
- S2: S1 plus generated code: Claude writes custom loss functions that are
  validated and executed only in the isolated Linux sandbox (ml/sandbox.py).
  Must be enabled explicitly; S0 and S1 reject generated components before
  any validation or budget use.

Both axes are independent of the P0/P1 selection policy and the HITL mode,
which keep their existing behavior. Every hard boundary (model eligibility,
budgets, duplicate detection, validation-only data, sealed test split) is
enforced by deterministic code regardless of authority or scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

AUTHORITY_LEVELS = {
    "A0": "deterministic/constrained system decisions",
    "A1": "Claude proposes; the system validates and gates launches",
    "A2": "Claude chooses and executes within hard safety boundaries",
}
SCOPE_LEVELS = {
    "S0": "existing configuration knobs",
    "S1": "composing supported pipeline components",
    "S2": "generated custom losses, executed only in the isolated sandbox",
}
CLAUDE_AUTHORITIES = ("A1", "A2")
AVAILABLE_SCOPES = ("S0", "S1", "S2")


@dataclass(frozen=True)
class AutonomyPolicy:
    authority: str = "A2"
    scope: str = "S0"

    def __post_init__(self):
        if self.authority not in AUTHORITY_LEVELS:
            raise ValueError(f"Unknown authority {self.authority!r}.")
        if self.authority not in CLAUDE_AUTHORITIES:
            raise ValueError("A0 is reserved for system decisions; Claude campaigns use A1 or A2.")
        if self.scope not in SCOPE_LEVELS:
            raise ValueError(f"Unknown scope {self.scope!r}.")
        if self.scope not in AVAILABLE_SCOPES:
            raise ValueError(f"Scope {self.scope!r} is not available.")

    @property
    def composes_pipelines(self) -> bool:
        return self.scope in ("S1", "S2")

    @property
    def generates_code(self) -> bool:
        return self.scope == "S2"

    @property
    def requires_proposal(self) -> bool:
        return self.authority == "A1"

    def to_dict(self) -> dict:
        return asdict(self)

    def check_launch(self, solution_id: str, campaign_id: str) -> dict | None:
        """Authority gate before reservation; None means the launch may proceed."""
        if not self.requires_proposal:
            return None
        from ml.solution import load_solution, solution_exists

        proposed = solution_exists(solution_id) and any(
            event.get("campaign_id") == campaign_id
            for event in load_solution(solution_id)["proposals"]
        )
        if proposed:
            return None
        return {
            "status": "rejected",
            "error_code": "PROPOSAL_REQUIRED",
            "message": (
                "Authority A1: call propose_solution with this exact configuration first; "
                "only proposed, validated solutions may be launched. No budget was used."
            ),
            "solution_id": solution_id,
        }

    def instructions(self) -> str:
        if self == AutonomyPolicy():
            return ""
        lines = [f"\nAUTONOMY: authority {self.authority} ({AUTHORITY_LEVELS[self.authority]}), "
                 f"scope {self.scope} ({SCOPE_LEVELS[self.scope]})."]
        if self.composes_pipelines:
            lines.append(
                "You may compose supported pipeline components (preprocessing view and "
                "normalization, augmentations, loss, optimizer, scheduler, dropout) through the "
                "`pipeline` argument. Call list_pipeline_components to see what exists and what "
                "is compatible; unsupported combinations are rejected with explicit errors and "
                "use no budget. Use parent_solution_id and change_description to record how a "
                "solution relates to an earlier one, and inspect_solutions to review solutions, "
                "checkpoints, and validation evidence."
            )
        if self.generates_code:
            lines.append(
                "Scope S2: when no supported loss fits your diagnosis, you may write a custom "
                "loss. Call describe_custom_loss_contract, then submit_custom_loss with the "
                "Python source; it is validated on synthetic data in an isolated sandbox and "
                "the structured results tell you what to fix. Only accepted components can be "
                "used, via pipeline loss={'name': 'custom', 'component_id': ...}. Record in "
                "change_description what limitation the loss addresses, and compare its "
                "validation results against your earlier solutions. A custom loss changes only "
                "the training signal; metrics, splits, and checkpoint selection are unchanged."
            )
        if self.requires_proposal:
            lines.append(
                "Before each launch, call propose_solution with the exact configuration; "
                "run_campaign_experiment only accepts proposed solutions."
            )
        return "\n".join(lines) + "\n"


def uses_generated_code(pipeline) -> bool:
    """True when a pipeline references a generated (scope S2) component."""
    if not isinstance(pipeline, dict):
        return False
    loss = pipeline.get("loss")
    name = loss if isinstance(loss, str) else loss.get("name") if isinstance(loss, dict) else None
    return name == "custom" or bool(pipeline.get("custom_components"))


def scope_rejection(policy: AutonomyPolicy, pipeline) -> dict | None:
    if uses_generated_code(pipeline) and not policy.generates_code:
        return {"status": "rejected", "error_code": "SCOPE_NOT_PERMITTED",
                "message": "Generated components require scope S2. No budget was used."}
    return None

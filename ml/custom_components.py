"""Generated (custom) loss components: contract, validation, artifacts, execution.

Contract ``gotrain.custom_loss/1``. Claude submits Python source defining::

    import torch
    import torch.nn.functional as F

    def compute_loss(logits, targets, *, task, class_weights=None, ignore_index=None, **params):
        # logits:  float32 [N, C] (classification) or [N, C, H, W] (segmentation)
        # targets: int64   [N]                     or [N, H, W]
        # task:    "classification" or "segmentation"
        # class_weights: float32 [C] (mean 1) or None
        # ignore_index:  int or None; those targets must not affect the loss
        # params: the JSON scalars given at submission
        # returns a 0-dim tensor that is differentiable with respect to logits
        ...

Execution boundary: the source is executed only inside the isolated sandbox
(ml/sandbox.py). The trainer never imports it; it sends logits and labels to
the sandbox and applies the returned gradient. The host generates every
validation input and checks every reply itself, so a component cannot forge
its own validation. The static checks below are a convenience for early,
readable errors; they are not the security boundary.

A generated loss only produces the training signal. Evaluation metrics, data
splits, checkpoint selection and verification, resource accounting, and the
trainer's safety controls stay in trusted code and never call it (the reported
validation *loss* value is the only number it influences).

Artifacts (immutable, content addressed):

    <components_dir>/<component_id>/source.py         exact submitted code
    <components_dir>/<component_id>/component.json    contract, tasks, params, hashes
    <components_dir>/<component_id>/validations.jsonl every validation attempt

A component is usable only if its files still hash to its ID and its latest
validation was accepted by a verified sandbox. Because validations.jsonl is an
ordinary local file, training never trusts it alone: the training sandbox
re-runs the full validation suite before the first batch.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

CONTRACT = "gotrain.custom_loss/1"
COMPONENT_ID = re.compile(r"^cl_[0-9a-f]{20}$")
MAX_SOURCE_CHARS = 20_000
MAX_AST_NODES = 4_000
ALLOWED_IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "math", "typing"}
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "memoryview", "exit", "quit", "help",
}
FORBIDDEN_ATTRIBUTES = {
    "load", "save", "hub", "utils", "distributed", "multiprocessing", "jit", "compile", "ops",
    "_C", "backends", "library", "cuda", "system", "popen", "spawn", "fork", "from_file",
}
TASKS = ("classification", "segmentation")


class ComponentError(ValueError):
    """Structured, revisable error (code + list of findings)."""

    def __init__(self, code, message, findings=None):
        super().__init__(message)
        self.code = code
        self.findings = findings or []

    def to_dict(self):
        return {"status": "rejected", "error_code": self.code, "message": str(self),
                "findings": self.findings}


def components_dir() -> Path:
    return Path(os.environ.get("GOTRAIN_COMPONENTS_DIR", "artifacts/custom_components"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def component_id(source: str, params: dict, tasks) -> str:
    identity = {"contract": CONTRACT, "kind": "loss", "source": source,
                "params": params, "tasks": sorted(tasks)}
    return "cl_" + _sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")))[:20]


# ---------------------------------------------------------------------------
# Static checks (supplementary; the sandbox is the boundary)
# ---------------------------------------------------------------------------

def static_findings(source: str, params: dict, tasks) -> list[dict]:
    findings = []

    def finding(message, node=None):
        findings.append({"stage": "static", "line": getattr(node, "lineno", None),
                         "message": message})

    if not isinstance(source, str) or not source.strip():
        return [{"stage": "static", "line": None, "message": "source must be nonempty Python"}]
    if len(source) > MAX_SOURCE_CHARS:
        finding(f"source exceeds {MAX_SOURCE_CHARS} characters")
    if not tasks or any(task not in TASKS for task in tasks):
        finding(f"tasks must be a nonempty subset of {list(TASKS)}")
    if not isinstance(params, dict) or any(
        not isinstance(key, str) or not key.isidentifier()
        or not (isinstance(value, (int, float, str, bool)) or value is None)
        or (isinstance(value, float) and not math.isfinite(value))
        for key, value in (params or {}).items()
    ):
        finding("params must map identifier names to finite JSON scalars")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return findings + [{"stage": "static", "line": exc.lineno, "message": f"SyntaxError: {exc.msg}"}]
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        finding(f"source is too complex (more than {MAX_AST_NODES} syntax nodes)")
    function = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "compute_loss":
                function = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            pass
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and all(
            isinstance(child, (ast.Constant, ast.Name, ast.Store, ast.Load, ast.Tuple,
                               ast.List, ast.UnaryOp, ast.USub, ast.Attribute))
            for child in ast.walk(node.value if node.value is not None else ast.Constant(None))
        ):
            pass
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            pass  # docstring
        else:
            finding("only imports, constant assignments, and function definitions are allowed "
                    "at module level", node)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    finding(f"import of {alias.name!r} is not allowed "
                            f"(allowed: {sorted(ALLOWED_IMPORTS)})", node)
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS or node.level:
                finding(f"import from {node.module!r} is not allowed "
                        f"(allowed: {sorted(ALLOWED_IMPORTS)})", node)
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("__") or \
                        alias.name in FORBIDDEN_ATTRIBUTES:
                    finding(f"importing {alias.name!r} from {node.module!r} is not allowed", node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in FORBIDDEN_CALLS:
            finding(f"call to {node.func.id}() is not allowed", node)
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS | {"__builtins__"}:
            finding(f"use of {node.id!r} is not allowed", node)
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES
        ):
            finding(f"attribute {node.attr!r} is not allowed", node)
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await,
                             ast.Yield, ast.YieldFrom, ast.ClassDef)):
            finding(f"{type(node).__name__} is not allowed", node)
    if function is None:
        finding("define a module-level function compute_loss(logits, targets, *, task, "
                "class_weights=None, ignore_index=None, ...)")
    else:
        positional = [arg.arg for arg in function.args.args]
        keyword = {arg.arg for arg in function.args.kwonlyargs}
        if positional[:2] != ["logits", "targets"] or len(positional) != 2:
            finding("compute_loss must take exactly (logits, targets) positionally", function)
        missing = {"task", "class_weights", "ignore_index"} - keyword
        if missing:
            finding(f"compute_loss needs keyword-only arguments {sorted(missing)}", function)
        undeclared = set(params or {}) - keyword
        if undeclared and function.args.kwarg is None:
            finding(f"params {sorted(undeclared)} are not keyword-only arguments of compute_loss",
                    function)
    return findings


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------

def _paths(cid):
    if not isinstance(cid, str) or not COMPONENT_ID.match(cid):
        raise ComponentError("INVALID_COMPONENT_ID", f"invalid component id {cid!r}")
    directory = components_dir() / cid
    return directory / "source.py", directory / "component.json", directory / "validations.jsonl"


def _atomic_write(path: Path, text: str):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _store(source, params, tasks, metadata):
    cid = component_id(source, params, tasks)
    source_path, manifest_path, _ = _paths(cid)
    if not manifest_path.exists():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(source_path, source)
        manifest = {
            "component_id": cid, "contract": CONTRACT, "kind": "loss", "tasks": sorted(tasks),
            "params": params, "source_sha256": _sha256(source), "created_at": _now(),
            "name": metadata.get("name"), "description": metadata.get("description"),
            "campaign_id": metadata.get("campaign_id"), "proposer": metadata.get("proposer"),
        }
        _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_component(cid)
    return cid


def verify_component(cid) -> dict:
    """Re-hash the stored files; raises COMPONENT_TAMPERED if anything changed."""
    source_path, manifest_path, _ = _paths(cid)
    if not manifest_path.is_file() or not source_path.is_file():
        raise ComponentError("UNKNOWN_COMPONENT", f"no custom component {cid!r}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ComponentError("COMPONENT_TAMPERED", f"{cid} has an unreadable manifest",
                             [{"stage": "integrity", "message": str(exc)}]) from exc
    source = source_path.read_text(encoding="utf-8")
    problems = []
    if _sha256(source) != manifest.get("source_sha256"):
        problems.append("source.py no longer matches its recorded SHA-256")
    if component_id(source, manifest.get("params", {}), manifest.get("tasks", [])) != cid or \
            manifest.get("component_id") != cid or manifest.get("contract") != CONTRACT:
        problems.append("component.json or source.py no longer matches the component id")
    if problems:
        raise ComponentError("COMPONENT_TAMPERED", f"{cid} failed integrity verification",
                             [{"stage": "integrity", "message": p} for p in problems])
    return {**manifest, "source": source}


def _record_validation(cid, report):
    _, _, validations = _paths(cid)
    with validations.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), **report}, sort_keys=True, default=str) + "\n")


def load_component(cid) -> dict:
    component = verify_component(cid)
    _, _, validations = _paths(cid)
    records = []
    if validations.is_file():
        records = [json.loads(line) for line in validations.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    latest = records[-1] if records else None
    component["validations"] = records
    component["status"] = latest["status"] if latest else "unvalidated"
    component["accepted"] = bool(
        latest and latest["status"] == "accepted"
        and latest.get("source_sha256") == component["source_sha256"]
    )
    return component


def component_errors(cid, task) -> list[str]:
    """Why this component cannot be used for ``task`` (empty list means it can)."""
    try:
        component = load_component(cid)
    except ComponentError as exc:
        return [f"{exc.code}: {exc}"] + [f["message"] for f in exc.findings]
    errors = []
    if not component["accepted"]:
        errors.append(f"custom component {cid} is {component['status']}; only components "
                      "accepted by the isolated validator can be used")
    if task not in component["tasks"]:
        errors.append(f"custom component {cid} supports {component['tasks']}, not {task}")
    return errors


def describe(cid, include_source=False) -> dict:
    component = load_component(cid)
    latest = component["validations"][-1] if component["validations"] else {}
    summary = {key: component[key] for key in ("component_id", "contract", "tasks", "params",
                                               "source_sha256", "name", "description", "status",
                                               "accepted", "created_at")}
    summary["latest_validation"] = {key: latest.get(key) for key in
                                    ("status", "error_code", "findings", "checks", "seconds")}
    if include_source:
        summary["source"] = component["source"]
    return summary


def campaign_components(campaign_id) -> list[str]:
    root = components_dir()
    found = []
    for manifest in root.glob("cl_*/component.json") if root.is_dir() else ():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("campaign_id") == campaign_id:
            found.append((data["created_at"], data["component_id"]))
    return [cid for _, cid in sorted(found)]


def archive_component(cid, destination: Path) -> dict:
    """Copy the exact verified source next to a checkpoint."""
    component = verify_component(cid)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{cid}.py"
    target.write_text(component["source"], encoding="utf-8")
    (destination / f"{cid}.json").write_text(
        json.dumps({k: v for k, v in component.items() if k != "source"}, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")
    return {"component_id": cid, "kind": "loss", "contract": CONTRACT,
            "source_sha256": component["source_sha256"], "archived_source": str(target)}


# ---------------------------------------------------------------------------
# Remote evaluation and host-side validation suite
# ---------------------------------------------------------------------------

def _session_files(component):
    return {"component.py": component["source"], "params.json": json.dumps(component["params"])}


def remote_loss(session, logits, targets, *, task, weights=None, ignore_index=None,
                need_grad=True, seed=0):
    """Evaluate the sandboxed loss; returns (loss float, gradient ndarray or None)."""
    import numpy as np

    from .sandbox import SandboxError

    logits_np = np.ascontiguousarray(logits.detach().to("cpu").float().numpy())
    targets_np = np.ascontiguousarray(targets.detach().to("cpu").long().numpy())
    buffers = [logits_np.tobytes(), targets_np.tobytes()]
    inputs = [{"shape": list(logits_np.shape)}, {"shape": list(targets_np.shape)}]
    if weights is not None:
        weights_np = np.ascontiguousarray(weights.detach().to("cpu").float().numpy())
        buffers.append(weights_np.tobytes())
        inputs.append({"shape": list(weights_np.shape)})
    reply, out = session.request(
        {"op": "loss", "task": task, "inputs": inputs, "ignore_index": ignore_index,
         "need_grad": need_grad, "seed": seed},
        buffers, expect=[logits_np.nbytes] if need_grad else [],
    )
    if reply.get("status") != "ok":
        raise SandboxError(reply.get("error_type", "COMPONENT_EXECUTION"),
                           reply.get("message", "component failed"),
                           {"exception": reply.get("exception"),
                            "traceback": reply.get("traceback")})
    value = reply.get("loss")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SandboxError("COMPONENT_NONFINITE", f"loss is not a finite number: {value!r}")
    grad = None
    if need_grad:
        grad = np.frombuffer(out[0], dtype=np.float32).reshape(logits_np.shape).copy()
        if not np.isfinite(grad).all():
            raise SandboxError("COMPONENT_NONFINITE", "gradient contains NaN or infinity")
    return float(value), grad


def _cases(task):
    """Synthetic cases (no dataset access): (name, N, C, spatial, weights, ignore)."""
    if task == "classification":
        return [("basic", 8, 5, None, False, None), ("weighted", 8, 5, None, True, None),
                ("ignored", 8, 5, None, False, -100), ("single", 1, 2, None, False, None)]
    return [("basic", 2, 4, (16, 16), False, None), ("weighted", 2, 4, (16, 16), True, None),
            ("ignored", 2, 4, (16, 16), False, 255), ("single", 1, 7, (8, 8), False, None)]


def validation_suite(session, tasks, per_call_limit=10.0) -> list[dict]:
    """Every check is computed on the host from raw replies."""
    import numpy as np
    import torch

    from .sandbox import SandboxError

    checks = []

    def check(task, case, name, passed, detail=""):
        checks.append({"task": task, "case": case, "check": name, "passed": bool(passed),
                       "detail": str(detail)[:500]})

    generator = torch.Generator().manual_seed(1234)
    for task in tasks:
        for case, n, c, spatial, weighted, ignore in _cases(task):
            shape = (n, c, *spatial) if spatial else (n, c)
            target_shape = (n, *spatial) if spatial else (n,)
            logits = torch.randn(*shape, generator=generator)
            targets = torch.randint(0, c, target_shape, generator=generator)
            if ignore is not None:
                targets.view(-1)[::3] = ignore
            weights = (torch.rand(c, generator=generator) + 0.5) if weighted else None
            if weights is not None:
                weights = weights / weights.mean()
            kwargs = {"task": task, "weights": weights, "ignore_index": ignore}
            try:
                started = time.perf_counter()
                value, grad = remote_loss(session, logits, targets, **kwargs)
                seconds = time.perf_counter() - started
            except SandboxError as exc:
                check(task, case, "executes", False, f"{exc.error_type}: {exc} "
                      f"{exc.detail.get('traceback') or ''}")
                if exc.error_type in ("SANDBOX_TIMEOUT", "SANDBOX_CRASHED", "SANDBOX_PROTOCOL"):
                    return checks
                continue
            check(task, case, "executes", True, f"loss={value:.6g}")
            check(task, case, "finite_loss_and_gradient", True)
            check(task, case, "gradient_shape", grad.shape == tuple(shape), grad.shape)
            active = np.ones(target_shape, dtype=bool) if ignore is None else (
                targets.numpy() != ignore)
            moved = np.moveaxis(grad, 1, -1)  # [..., C]
            check(task, case, "gradient_flows", np.abs(moved[active]).sum() > 0,
                  f"|grad| over counted targets = {np.abs(moved[active]).sum():.3g}")
            if ignore is not None:
                leaked = np.abs(moved[~active]).max() if (~active).any() else 0.0
                check(task, case, "ignored_targets_have_zero_gradient", leaked <= 1e-8,
                      f"max |grad| at ignored targets = {leaked:.3g}")
            check(task, case, "time_per_call", seconds <= per_call_limit, f"{seconds:.3f}s")
            try:
                again, grad_again = remote_loss(session, logits, targets, **kwargs)
                check(task, case, "deterministic", again == value and np.array_equal(
                    grad, grad_again), f"{value!r} vs {again!r}")
                if case == "basic":
                    perfect = torch.nn.functional.one_hot(
                        targets.clamp(min=0), c).movedim(-1, 1).float() * 12.0 - 6.0
                    good, _ = remote_loss(session, perfect, targets, need_grad=False, **kwargs)
                    check(task, case, "prefers_correct_predictions", good < value,
                          f"perfect={good:.6g} random={value:.6g}")
            except SandboxError as exc:
                check(task, case, "repeatable", False, f"{exc.error_type}: {exc}")
                if exc.error_type in ("SANDBOX_TIMEOUT", "SANDBOX_CRASHED", "SANDBOX_PROTOCOL"):
                    return checks
    return checks


def _run_validation(component, limits=None):
    from .sandbox import IsolationUnavailable, SandboxError, SandboxSession, require_runtime

    started = time.perf_counter()
    report = {"source_sha256": component["source_sha256"], "contract": CONTRACT}
    try:
        runtime = require_runtime(limits)
    except IsolationUnavailable as exc:
        return {**report, "status": "isolation_unavailable", "error_code": "ISOLATION_UNAVAILABLE",
                "message": f"Generated code was not executed: {exc}",
                "seconds": time.perf_counter() - started}
    report["runtime"] = {"backend": runtime.backend, "limits": runtime.limits}
    try:
        with SandboxSession(_session_files(component), limits) as session:
            reply, _ = session.request({"op": "load"}, timeout=session.limits.startup_timeout,
                                       expect=[])
            if reply.get("status") != "ok":
                return {**report, "status": "rejected", "error_code": reply.get("error_type"),
                        "findings": [{"stage": "load", "message": reply.get("message"),
                                      "traceback": reply.get("traceback")}],
                        "seconds": time.perf_counter() - started}
            checks = validation_suite(session, component["tasks"])
    except SandboxError as exc:
        return {**report, "status": "rejected", "error_code": exc.error_type,
                "findings": [{"stage": "sandbox", "message": str(exc), **exc.detail}],
                "seconds": time.perf_counter() - started}
    failed = [c for c in checks if not c["passed"]]
    return {**report, "status": "rejected" if failed or not checks else "accepted",
            "error_code": "VALIDATION_FAILED" if failed else None,
            "findings": [{"stage": "validation", **c} for c in failed], "checks": checks,
            "seconds": time.perf_counter() - started}


def submit_component(source, *, tasks, params=None, name=None, description=None,
                     campaign_id=None, proposer="claude", limits=None) -> dict:
    """Archive, statically check, and validate in the sandbox. Never executes on the host."""
    params = dict(params or {})
    tasks = sorted(set(tasks or []))
    findings = static_findings(source, params, tasks)
    if any("tasks must" in f["message"] or "params must" in f["message"] for f in findings) or \
            not isinstance(source, str):
        return {"status": "rejected", "error_code": "INVALID_SUBMISSION", "findings": findings}
    cid = _store(source, params, tasks, {"name": name, "description": description,
                                         "campaign_id": campaign_id, "proposer": proposer})
    component = verify_component(cid)
    if findings:
        report = {"status": "rejected", "error_code": "STATIC_CHECK_FAILED", "findings": findings,
                  "source_sha256": component["source_sha256"], "contract": CONTRACT}
    else:
        report = _run_validation(component, limits)
    _record_validation(cid, report)
    result = {"component_id": cid, **{k: report.get(k) for k in (
        "status", "error_code", "message", "findings", "checks", "seconds")}}
    if report["status"] == "accepted":
        result["use_with"] = {"pipeline": {"loss": {"name": "custom", "component_id": cid}}}
    return result


def contract_description() -> dict:
    """What Claude needs to write a component, as returned by the agent tools."""
    return {
        "contract": CONTRACT,
        "signature": ("def compute_loss(logits, targets, *, task, class_weights=None, "
                      "ignore_index=None, **params) -> 0-dim tensor"),
        "arguments": {
            "logits": "float32 [N, C] (classification) or [N, C, H, W] (segmentation)",
            "targets": "int64 [N] or [N, H, W]",
            "task": "'classification' or 'segmentation'",
            "class_weights": "float32 [C] with mean 1, or None",
            "ignore_index": "int or None; ignored targets must not affect the loss or gradient",
            "params": "the JSON scalars you submit, passed as keyword arguments",
        },
        "allowed_imports": sorted(ALLOWED_IMPORTS),
        "module_level": "imports, constant assignments, and function definitions only",
        "validation_checks": [
            "executes on synthetic inputs (basic, class-weighted, ignored targets, batch of one)",
            "finite loss and gradient", "gradient shape equals logits shape",
            "gradient flows to counted targets", "zero gradient at ignored targets",
            "deterministic for identical inputs", "lower loss for correct than random predictions",
            "per-call time limit",
        ],
        "execution": ("Runs only in an isolated Linux sandbox (no network, files, shell, or "
                      "credentials); the trainer applies the gradient it returns. Rejected "
                      "components can never be used for training. Metrics, data splits, and "
                      "checkpoint selection are computed by trusted code, not by the loss."),
    }


# ---------------------------------------------------------------------------
# Training-time loss (host side; the component runs only in the sandbox)
# ---------------------------------------------------------------------------

def sandboxed_loss(cid, task, weight=None, ignore_index=None, seed=0, limits=None):
    """A torch loss module backed by a sandboxed, re-validated component."""
    import torch

    from .sandbox import SandboxError, SandboxSession, require_runtime

    errors = component_errors(cid, task)
    if errors:
        raise ComponentError("COMPONENT_NOT_USABLE", "; ".join(errors))
    require_runtime(limits)
    component = verify_component(cid)

    class _RemoteLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, targets, owner):
            need_grad = bool(ctx.needs_input_grad[0])
            value, grad = owner.evaluate(logits, targets, need_grad)
            if need_grad:
                ctx.save_for_backward(torch.from_numpy(grad).to(logits.device, logits.dtype))
            return torch.tensor(value, dtype=torch.float32, device=logits.device)

        @staticmethod
        def backward(ctx, grad_output):
            (grad,) = ctx.saved_tensors
            return grad_output.to(grad.dtype) * grad, None, None

    class SandboxedLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.component_id = cid
            self.source_sha256 = component["source_sha256"]
            self.register_buffer("weight", weight)
            self.ignore_index = ignore_index
            self.calls = 0
            self.session = SandboxSession(_session_files(component), limits)
            try:
                reply, _ = self.session.request({"op": "load"},
                                                timeout=self.session.limits.startup_timeout,
                                                expect=[])
                if reply.get("status") != "ok":
                    raise ComponentError("COMPONENT_LOAD", reply.get("message", "load failed"))
                failed = [c for c in validation_suite(self.session, [task]) if not c["passed"]]
                if failed:
                    raise ComponentError("REVALIDATION_FAILED",
                                         f"{cid} failed re-validation in the training sandbox",
                                         failed)
            except BaseException:
                self.session.close()
                raise

        def evaluate(self, logits, targets, need_grad):
            self.calls += 1
            try:
                return remote_loss(self.session, logits, targets, task=task, weights=self.weight,
                                   ignore_index=self.ignore_index, need_grad=need_grad,
                                   seed=seed + self.calls)
            except SandboxError as exc:
                raise ComponentError(exc.error_type, f"custom loss {cid} failed during training: "
                                     f"{exc}", [exc.detail]) from exc

        def forward(self, logits, targets):
            return _RemoteLoss.apply(logits, targets, self)

        def close(self):
            self.session.close()

    return SandboxedLoss()


def main(argv=None) -> int:
    """python -m ml.custom_components {submit,show,list}: human inspection and submission."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ml.custom_components")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="validate a loss source file in the sandbox")
    submit.add_argument("source_file")
    submit.add_argument("--tasks", nargs="+", required=True, choices=TASKS)
    submit.add_argument("--params", type=json.loads, default={})
    submit.add_argument("--name")
    submit.add_argument("--description")
    show = commands.add_parser("show", help="print a component, its source and validations")
    show.add_argument("component_id")
    commands.add_parser("list", help="list stored components")
    args = parser.parse_args(argv)
    if args.command == "submit":
        result = submit_component(Path(args.source_file).read_text(encoding="utf-8"),
                                  tasks=args.tasks, params=args.params, name=args.name,
                                  description=args.description, proposer="human-cli")
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
        print(f"checks passed: {sum(c['passed'] for c in result.get('checks') or [])}/"
              f"{len(result.get('checks') or [])}")
        return 0 if result["status"] == "accepted" else 1
    try:
        if args.command == "show":
            print(json.dumps(describe(args.component_id, include_source=True), indent=2))
        else:
            root = components_dir()
            for manifest in sorted(root.glob("cl_*/component.json")) if root.is_dir() else ():
                info = describe(manifest.parent.name)
                print(f"{info['component_id']}  {info['status']:22s} {info['tasks']}  "
                      f"{info['name'] or ''}")
    except ComponentError as exc:
        print(json.dumps(exc.to_dict(), indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

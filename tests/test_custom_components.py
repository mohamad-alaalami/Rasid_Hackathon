"""Phase 1B tests that run on any host (no sandbox execution needed).

Covers the contract's static checks, content-addressed artifacts and tamper
detection, refusal when no verified isolation runtime exists, pipeline and
SolutionSpec integration, S2 scope authorization, and HITL gating of generated
code. Real sandbox execution is tested in test_sandbox_linux.py.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import strands_stub
    from generated_losses import POLY1, STATIC_VIOLATIONS, TVERSKY
except ImportError:  # run as tests.<module> from the repo root
    from tests import strands_stub
    from tests.generated_losses import POLY1, STATIC_VIOLATIONS, TVERSKY

strands_stub.install()

from agent import tools  # noqa: E402
from agent.autonomy import AutonomyPolicy, uses_generated_code  # noqa: E402
from agent.pipeline_tools import make_pipeline_tools  # noqa: E402
from ml import custom_components as cc  # noqa: E402
from ml import sandbox  # noqa: E402
from ml.components import CATALOG, PipelineError, describe_configuration_space  # noqa: E402
from ml.config import TrainingConfig  # noqa: E402
from ml.solution import build_solution_spec, propose_solution  # noqa: E402


def accept_for_test(cid):
    """Test fixture only: mark a stored component accepted without a sandbox."""
    component = cc.verify_component(cid)
    cc._record_validation(cid, {"status": "accepted", "source_sha256": component["source_sha256"],
                                "checks": [], "fixture": True})


def tool_names(tool_list):
    return [getattr(t, "tool_name", None) or t.__name__ for t in tool_list]


class Base(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.patches = [
            patch.dict(os.environ, {"GOTRAIN_COMPONENTS_DIR": str(root / "components"),
                                    "GOTRAIN_SOLUTIONS_DIR": str(root / "solutions"),
                                    "GOTRAIN_SANDBOX_DISABLE": "1", "GOTRAIN_STORE": ""}),
            patch("agent.experiments.RESULTS_DIR", root / "campaigns"),
        ]
        for item in self.patches:
            item.start()
        sandbox._CACHE.clear()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        sandbox._CACHE.clear()
        self.temporary.cleanup()


class StaticCheckTests(Base):
    def test_valid_sources_pass_static_checks(self):
        self.assertEqual(cc.static_findings(POLY1, {"epsilon": 1.0}, ["classification"]), [])
        self.assertEqual(cc.static_findings(TVERSKY, {"alpha": 0.3}, ["segmentation"]), [])

    def test_violations_are_reported_with_lines(self):
        findings = cc.static_findings(STATIC_VIOLATIONS, {"gamma": 2}, ["classification"])
        text = " | ".join(f"L{f['line']}: {f['message']}" for f in findings)
        for fragment in ("import of 'os'", "import of 'subprocess'", "module level",
                         "ClassDef is not allowed", "call to eval()",
                         "exactly (logits, targets)", "keyword-only arguments",
                         "params ['gamma'] are not keyword-only"):
            self.assertIn(fragment, text)
        self.assertTrue(all(f["line"] for f in findings if "import" in f["message"]))
        dunder = POLY1.replace("return loss.sum()", "return loss.__class__.__bases__ or loss.sum()")
        self.assertIn("attribute '__class__'", " ".join(
            f["message"] for f in cc.static_findings(dunder, {}, ["classification"])))
        sneaky = POLY1.replace("import torch\n", "import torch\nfrom torch import hub\n")
        self.assertIn("importing 'hub'", " ".join(
            f["message"] for f in cc.static_findings(sneaky, {}, ["classification"])))

    def test_submission_shape_errors(self):
        self.assertEqual(cc.submit_component(POLY1, tasks=["detection"])["error_code"],
                         "INVALID_SUBMISSION")
        self.assertEqual(cc.submit_component(POLY1, tasks=["classification"],
                                             params={"epsilon": float("nan")})["error_code"],
                         "INVALID_SUBMISSION")
        syntax = cc.submit_component("def compute_loss(:", tasks=["classification"])
        self.assertEqual(syntax["error_code"], "STATIC_CHECK_FAILED")
        self.assertFalse(cc.load_component(syntax["component_id"])["accepted"])


class ArtifactTests(Base):
    def test_identity_versioning_and_idempotent_storage(self):
        first = cc.submit_component(POLY1, tasks=["classification"], params={"epsilon": 1.0})
        again = cc.submit_component(POLY1, tasks=["classification"], params={"epsilon": 1.0})
        other = cc.submit_component(POLY1, tasks=["classification"], params={"epsilon": 2.0})
        both = cc.submit_component(POLY1, tasks=["segmentation", "classification"],
                                   params={"epsilon": 1.0})
        self.assertEqual(first["component_id"], again["component_id"])
        self.assertEqual(len({first["component_id"], other["component_id"],
                              both["component_id"]}), 3)
        component = cc.load_component(first["component_id"])
        self.assertEqual(component["source"], POLY1)  # exact code archived
        self.assertEqual(len(component["validations"]), 2)  # each attempt recorded

    def test_tampering_is_detected(self):
        cid = cc.submit_component(TVERSKY, tasks=["segmentation"])["component_id"]
        accept_for_test(cid)
        self.assertEqual(cc.component_errors(cid, "segmentation"), [])
        folder = Path(os.environ["GOTRAIN_COMPONENTS_DIR"]) / cid
        original = (folder / "source.py").read_text()
        (folder / "source.py").write_text(original.replace("0.5", "0.9"))
        with self.assertRaises(cc.ComponentError) as raised:
            cc.verify_component(cid)
        self.assertEqual(raised.exception.code, "COMPONENT_TAMPERED")
        self.assertTrue(any("COMPONENT_TAMPERED" in e for e in cc.component_errors(cid, "segmentation")))
        (folder / "source.py").write_text(original)
        manifest = json.loads((folder / "component.json").read_text())
        manifest["params"] = {"alpha": 0.9}
        (folder / "component.json").write_text(json.dumps(manifest))
        with self.assertRaises(cc.ComponentError):
            cc.verify_component(cid)
        (folder / "component.json").write_text("{not json")
        with self.assertRaises(cc.ComponentError) as raised:
            cc.verify_component(cid)
        self.assertEqual(raised.exception.code, "COMPONENT_TAMPERED")

    def test_unknown_and_malformed_ids(self):
        self.assertIn("UNKNOWN_COMPONENT", cc.component_errors("cl_" + "0" * 20, "segmentation")[0])
        for bad in ("../../etc/passwd", None, 7):
            with self.assertRaises(cc.ComponentError):
                cc.verify_component(bad)

    def test_archive_copies_the_exact_verified_code(self):
        cid = cc.submit_component(POLY1, tasks=["classification"])["component_id"]
        record = cc.archive_component(cid, Path(self.temporary.name) / "ckpt")
        self.assertEqual(Path(record["archived_source"]).read_text(), POLY1)
        self.assertEqual(record["source_sha256"], cc.verify_component(cid)["source_sha256"])


class RefusalTests(Base):
    def test_no_verified_runtime_means_no_execution(self):
        with patch.object(sandbox, "SandboxSession", side_effect=AssertionError("must not run")):
            result = cc.submit_component(POLY1, tasks=["classification"], params={"epsilon": 1.0})
        self.assertEqual(result["status"], "isolation_unavailable")
        self.assertIn("not executed", result["message"])
        cid = result["component_id"]
        self.assertFalse(cc.load_component(cid)["accepted"])
        self.assertIn("isolation_unavailable", cc.component_errors(cid, "classification")[0])
        self.assertNotIn("gotrain_custom_component", sys.modules)

    def test_rejected_component_can_never_train(self):
        cid = cc.submit_component(POLY1, tasks=["classification"])["component_id"]
        config = TrainingConfig.from_mapping({"model": "resnet18_rgb", "pipeline": {
            "loss": {"name": "custom", "component_id": cid}}})
        with self.assertRaises(PipelineError) as raised:
            config.validate_for_dataset("m-eurosat")
        self.assertIn("only components accepted", " ".join(raised.exception.errors))
        with self.assertRaises(cc.ComponentError):
            cc.sandboxed_loss(cid, "classification")

    def test_accepted_component_still_needs_the_runtime_to_execute(self):
        cid = cc.submit_component(POLY1, tasks=["classification"])["component_id"]
        accept_for_test(cid)
        with self.assertRaises(sandbox.IsolationUnavailable):
            cc.sandboxed_loss(cid, "classification")

    def test_runtime_refuses_when_disabled(self):
        with patch.object(sandbox, "SandboxSession", side_effect=AssertionError("must not run")):
            report = sandbox.check_runtime(refresh=True)
        self.assertFalse(report.available)
        self.assertIn("GOTRAIN_SANDBOX_DISABLE", report.reason)

    def test_probe_evaluation_requires_every_guarantee(self):
        good = {"results": {name: {"blocked": True} for name in sandbox.REQUIRED_BLOCKS},
                "effective_capabilities": 0, "seccomp": {"architecture": "x86_64"},
                "uid_map": ["0", "65534", "1"], "environment": ["PATH"], "torch_ok": True}
        self.assertEqual(sandbox._probe_failures(good), [])
        for change, expected in (
            ({"uid_map": ["0", "0", "1"]}, "uid_maps_to_host_root"),
            ({"effective_capabilities": 0x200000}, "capabilities"),
            ({"seccomp": None}, "seccomp_not_installed"),
            ({"environment": ["AWS_SECRET_ACCESS_KEY"]}, "AWS_SECRET_ACCESS_KEY"),
            ({"results": {**good["results"], "fork": {"blocked": False}}}, "fork"),
            ({"results": {k: v for k, v in good["results"].items() if k != "dns"}},
             "missing_probe:dns"),
        ):
            with self.subTest(expected):
                failures = sandbox._probe_failures({**good, **change})
                self.assertTrue(any(expected in failure for failure in failures), failures)


class IntegrationTests(Base):
    def test_catalog_exposes_the_custom_loss(self):
        self.assertIn(("loss", "custom"), CATALOG)
        self.assertIn("custom_loss", describe_configuration_space())

    def test_pipeline_compatibility_for_custom_losses(self):
        segmentation_only = cc.submit_component(TVERSKY, tasks=["segmentation"])["component_id"]
        accept_for_test(segmentation_only)
        ok = TrainingConfig.from_mapping({"model": "unet_r34", "pipeline": {
            "loss": {"name": "custom", "component_id": segmentation_only,
                     "class_weighting": "inverse_frequency"}}})
        self.assertEqual(ok.validate_for_dataset("m-cashew-plant"), "unet_r34")
        wrong_task = TrainingConfig.from_mapping({"model": "resnet18_rgb", "pipeline": {
            "loss": {"name": "custom", "component_id": segmentation_only}}})
        with self.assertRaises(PipelineError) as raised:
            wrong_task.validate_for_dataset("m-eurosat")
        self.assertIn("not classification", " ".join(raised.exception.errors))
        for bad in ("not-an-id", None):
            with self.assertRaises(PipelineError):
                TrainingConfig.from_mapping({"model": "resnet18_rgb", "pipeline": {
                    "loss": {"name": "custom", "component_id": bad}}})

    def test_solution_spec_binds_the_exact_code(self):
        cid = cc.submit_component(POLY1, tasks=["classification"])["component_id"]
        accept_for_test(cid)
        cfg = TrainingConfig.from_mapping({"model": "resnet18_rgb", "pipeline": {
            "loss": {"name": "custom", "component_id": cid}}})
        spec = build_solution_spec("m-eurosat", cfg)
        self.assertEqual(spec["custom_components"], [{
            "component_id": cid, "kind": "loss", "contract": cc.CONTRACT,
            "source_sha256": cc.verify_component(cid)["source_sha256"]}])
        baseline = build_solution_spec("m-eurosat", TrainingConfig.from_mapping(
            {"model": "resnet18_rgb"}))
        self.assertEqual(baseline["custom_components"], [])  # Phase 1A identity unchanged
        self.assertTrue(propose_solution(spec).startswith("sol_"))
        forged = {**spec, "custom_components": [{**spec["custom_components"][0],
                                                 "source_sha256": "0" * 64}]}
        with self.assertRaisesRegex(ValueError, "not an accepted"):
            propose_solution(forged)
        cc._record_validation(cid, {"status": "rejected", "source_sha256": "x"})
        with self.assertRaisesRegex(ValueError, "not an accepted"):
            propose_solution(spec)


class ScopeTests(Base):
    def tool(self, scope):
        return tools.make_registered_training_tool(
            version="E2", campaign_id="c", seed=42, max_experiments=3,
            required_dataset="m-eurosat", training_fn=lambda **kw: {"status": "failed"},
            autonomy=AutonomyPolicy(scope=scope))

    def args(self, cid):
        return {"dataset_id": "m-eurosat", "model_id": "resnet18_rgb", "epochs": 1,
                "batch_size": 2, "learning_rate": 0.01,
                "pipeline": {"loss": {"name": "custom", "component_id": cid}}}

    def test_generated_code_detection(self):
        self.assertTrue(uses_generated_code({"loss": {"name": "custom"}}))
        self.assertTrue(uses_generated_code({"loss": "custom"}))
        self.assertTrue(uses_generated_code({"custom_components": [{"x": 1}]}))
        self.assertFalse(uses_generated_code({"loss": {"name": "focal"}}))
        self.assertFalse(uses_generated_code(None))

    def test_s1_rejects_generated_components_and_s2_accepts_them(self):
        cid = cc.submit_component(POLY1, tasks=["classification"])["component_id"]
        accept_for_test(cid)
        with patch.object(tools, "model_runtime_status", lambda m: {"eligible": True, "reason": None}):
            refused = self.tool("S1")(**self.args(cid))
            self.assertEqual(refused["error_code"], "SCOPE_NOT_PERMITTED")
            s1 = make_pipeline_tools(version="E2", campaign_id="c", seed=42,
                                     autonomy=AutonomyPolicy(scope="S1"))
            self.assertNotIn("submit_custom_loss", tool_names(s1))
            proposal = {t.tool_name if hasattr(t, "tool_name") else t.__name__: t
                        for t in s1}["propose_solution"](**self.args(cid))
            self.assertEqual(proposal["error_code"], "SCOPE_NOT_PERMITTED")
            s2_names = tool_names(make_pipeline_tools(
                version="E2", campaign_id="c", seed=42, autonomy=AutonomyPolicy(scope="S2")))
            for name in ("describe_custom_loss_contract", "submit_custom_loss",
                         "inspect_custom_components", "list_pipeline_components"):
                self.assertIn(name, s2_names)
            self.assertEqual(make_pipeline_tools(version="E2", campaign_id="c", seed=42,
                                                 autonomy=AutonomyPolicy()), [])
            self.assertIn("Scope S2", AutonomyPolicy(scope="S2").instructions())
            self.assertNotIn("custom loss", AutonomyPolicy(scope="S1").instructions())

    def test_s2_with_a1_still_requires_a_proposal(self):
        cid = cc.submit_component(POLY1, tasks=["classification"])["component_id"]
        accept_for_test(cid)
        run = tools.make_registered_training_tool(
            version="E2", campaign_id="c", seed=42, max_experiments=3,
            required_dataset="m-eurosat", training_fn=lambda **kw: {"status": "failed"},
            autonomy=AutonomyPolicy(authority="A1", scope="S2"))
        with patch.object(tools, "model_runtime_status", lambda m: {"eligible": True, "reason": None}):
            self.assertEqual(run(**self.args(cid))["error_code"], "PROPOSAL_REQUIRED")

    def test_s2_tools_submit_and_inspect(self):
        s2 = dict(zip(tool_names(tools_list := make_pipeline_tools(
            version="E0", campaign_id="camp", seed=42, autonomy=AutonomyPolicy(scope="S2"))),
            tools_list))
        contract = s2["describe_custom_loss_contract"]()
        self.assertIn("compute_loss", contract["signature"])
        result = s2["submit_custom_loss"]("poly1", "PolyLoss-1", POLY1, ["classification"],
                                          {"epsilon": 1.0})
        self.assertEqual(result["status"], "isolation_unavailable")
        listed = s2["inspect_custom_components"]()["components"]
        self.assertEqual([c["component_id"] for c in listed], [result["component_id"]])
        detail = s2["inspect_custom_components"](result["component_id"])["component"]
        self.assertEqual(detail["source"], POLY1)
        self.assertEqual(s2["inspect_custom_components"]("cl_" + "1" * 20)["error_code"],
                         "NOT_VISIBLE")


class HitlCodeChangeTests(Base):
    # agent.hitl is imported lazily: strands_stub.install() replaces the stub's
    # InterruptException on each call, so importing hitl at collection time would
    # bind a class that test_hitl.py no longer catches.
    def controller(self, mode="critical"):
        from agent import hitl

        settings = {"feedback": "E2", "policy": "P0", "seed": 42, "max_experiments": 5,
                    "dataset": "m-eurosat", "model_id": "m", "region": "r", "hitl_mode": mode,
                    "failure_threshold": 2, "cost_threshold_usd": None,
                    "unknown_cost_policy": "ask", "prices": None}
        hitl.create_state("hc", settings)
        return hitl.HitlController("hc", settings)

    def approve(self, trigger, configuration):
        from agent import hitl

        def change(state):
            state["requests"].append({"id": len(state["requests"]) + 1, "trigger": trigger,
                                      "status": "delivered",
                                      "approval": {"configuration": configuration,
                                                   "consumed_by_experiment": None}})
        hitl._update_state("hc", change)

    def test_generated_code_requires_an_approval_naming_the_component(self):
        cid = cc.submit_component(POLY1, tasks=["classification"])["component_id"]
        accept_for_test(cid)
        gate = self.controller()
        custom = {"architecture": "resnet18_rgb", "epochs": 1,
                  "pipeline": {"loss": {"name": "custom", "component_id": cid}}}
        self.assertEqual(gate.check("m-eurosat", {"architecture": "resnet18_rgb"}), (None, None))
        _, blocked = gate.check("m-eurosat", custom)
        self.assertEqual(blocked["error_code"], "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(blocked["triggers"], ["code_change"])
        self.assertEqual(blocked["trigger_details"]["code_change"]["generated_components"], [cid])
        self.approve("insufficient_evidence", None)  # delegation for another reason
        self.approve("cost_threshold", {"model_id": "resnet18_rgb"})  # does not name the code
        self.assertEqual(gate.check("m-eurosat", custom)[1]["error_code"],
                         "HUMAN_SELECTION_MISMATCH")
        self.approve("code_change", {"model_id": "resnet18_rgb", "pipeline": custom["pipeline"]})
        approval, blocked = gate.check("m-eurosat", custom)
        self.assertIsNone(blocked)
        self.assertEqual(approval, 3)

    def test_consultation_shows_the_exact_source(self):
        from agent import hitl

        cid = cc.submit_component(TVERSKY, tasks=["segmentation"], name="tversky")["component_id"]
        self.controller()
        state = hitl.load_state("hc")
        request = {"id": 1, "trigger": "code_change", "situation": "small segments missed",
                   "recommended_option": "a", "rationale": "recall",
                   "options": [{"id": "a", "title": "Tversky loss", "tradeoffs": "new code",
                                "configuration": {"model_id": "unet_r34", "pipeline": {
                                    "loss": {"name": "custom", "component_id": cid}}}}]}
        text = hitl.format_request(state, request)
        self.assertIn(f"generated component {cid} (tversky)", text)
        self.assertIn("| def compute_loss(logits, targets, *, task", text)
        self.assertIn("status isolation_unavailable", text)


if __name__ == "__main__":
    unittest.main()

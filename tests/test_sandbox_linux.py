"""Phase 1B tests that execute generated code in the real isolation sandbox.

Skipped unless the host is Linux and ml.sandbox verifies the runtime (user,
mount, network, PID namespaces via util-linux, plus seccomp). No dataset,
network, or test split is used; training runs on CPU with tiny stand-in models
(one optional test uses CUDA when it is available).
"""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import strands_stub
    import generated_losses as gl
except ImportError:  # run as tests.<module> from the repo root
    from tests import strands_stub
    from tests import generated_losses as gl

strands_stub.install()

import torch  # noqa: E402

from ml import custom_components as cc  # noqa: E402
from ml import sandbox  # noqa: E402

RUNTIME = sandbox.check_runtime() if sys.platform == "linux" else None
AVAILABLE = bool(RUNTIME and RUNTIME.available)
REASON = (RUNTIME.reason if RUNTIME else "not Linux") or "runtime unavailable"
HOSTILE_ATTEMPTS = {
    "read_repo", "read_canary", "list_home", "read_aws_credentials", "write_repo", "write_work",
    "write_python_libs", "write_core_pattern", "connect_metadata", "connect_internet", "shell",
    "exec_python", "aws_environment", "chroot_escape", "os_system", "fork",
    "raw_unshare_userns", "raw_mount", "see_other_processes",
}


def _bound_names_under_home():
    """Directory names under $HOME that the sandbox must expose (a venv living there)."""
    home = Path.home().resolve()
    names = set()
    for path in (Path(sys.prefix).resolve(), Path(torch.__file__).resolve()):
        if home in path.parents:
            names.add(path.relative_to(home).parts[0])
    return names


@unittest.skipUnless(AVAILABLE, f"isolation runtime unavailable: {REASON}")
class ProbeTests(unittest.TestCase):
    def test_probe_blocked_every_escape(self):
        results = RUNTIME.probe["results"]
        self.assertTrue(all(outcome["blocked"] for outcome in results.values()), results)
        self.assertLessEqual(set(sandbox.REQUIRED_BLOCKS), set(results))
        self.assertFalse([k for k in RUNTIME.probe["environment"] if k.startswith("AWS")])
        self.assertLessEqual(set(RUNTIME.probe["root_entries"]),
                             {".gotrain", ".old", "dev", "etc", "home", "lib", "lib32", "lib64",
                              "proc", "tmp", "usr", "work", "opt", "root", "var", "srv"})
        self.assertNotIn("bin", RUNTIME.probe["root_entries"])
        self.assertTrue(RUNTIME.probe["torch_ok"])
        self.assertEqual(RUNTIME.probe["effective_capabilities"], 0)
        self.assertNotEqual(RUNTIME.probe["uid_map"][1], "0")
        self.assertIn("execve", RUNTIME.probe["seccomp"]["denied"])


@unittest.skipUnless(AVAILABLE, f"isolation runtime unavailable: {REASON}")
class HostileCodeTests(unittest.TestCase):
    """Bypass the static checks entirely: the sandbox alone must contain the code."""

    def run_component(self, source, limits=None):
        with sandbox.SandboxSession({"component.py": source, "params.json": "{}"},
                                    limits) as session:
            reply, _ = session.request({"op": "load"}, timeout=60, expect=[])
            self.assertEqual(reply["status"], "ok", reply)
            return cc.remote_loss(session, torch.randn(2, 3), torch.tensor([0, 1]),
                                  task="classification")

    def test_filesystem_network_process_and_credential_escapes_fail(self):
        repo = Path(__file__).resolve().parents[1]
        torch_dir = Path(torch.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as canary_dir:
            canary = Path(canary_dir) / "canary.txt"
            canary.write_text("secret")
            source = gl.HOSTILE_TEMPLATE.replace(
                "RESULTS = {}", "RESULTS = {}\n" + "\n".join([
                    f"REPO_FILE = {str(repo / 'README.md')!r}",
                    f"REPO_DIR = {str(repo)!r}",
                    f"CANARY = {str(canary)!r}",
                    f"HOME = {str(Path.home())!r}",
                    f"TORCH_DIR = {str(torch_dir)!r}",
                ]))
            with self.assertRaises(sandbox.SandboxError) as raised:
                self.run_component(source)
        message = str(raised.exception)
        results = json.loads(message[message.index("RESULTS=") + 8:])
        self.assertEqual(set(results), HOSTILE_ATTEMPTS)
        escaped = {name: value for name, value in results.items() if value.startswith("ESCAPED")}
        # A Python environment that lives under $HOME must be bound at its own path, so
        # listing $HOME can reveal that one directory name (and nothing else).
        listing = escaped.pop("list_home", None)
        if listing is not None:
            names = set(ast.literal_eval(listing[len("ESCAPED: "):]))
            self.assertLessEqual(names, _bound_names_under_home(), listing)
        self.assertEqual(escaped, {})
        self.assertFalse((repo / "ESCAPE.txt").exists())
        self.assertFalse((torch_dir / "ESCAPE.txt").exists())

    def test_timeout_kills_the_sandbox(self):
        limits = sandbox.Limits(request_timeout=3)
        with sandbox.SandboxSession({"component.py": gl.INFINITE_LOOP, "params.json": "{}"},
                                    limits) as session:
            session.request({"op": "load"}, timeout=60, expect=[])
            with self.assertRaises(sandbox.SandboxError) as raised:
                cc.remote_loss(session, torch.randn(2, 3), torch.tensor([0, 1]),
                               task="classification")
            self.assertEqual(raised.exception.error_type, "SANDBOX_TIMEOUT")
            self.assertIsNotNone(session.process.poll())  # whole tree killed

    def test_memory_bomb_is_contained(self):
        bomb = gl.NO_GRADIENT.replace("return torch.tensor(1.0)",
                                      "return torch.tensor(float(len(bytearray(64 << 30))))")
        with self.assertRaises(sandbox.SandboxError) as raised:
            self.run_component(bomb)
        self.assertEqual(raised.exception.error_type, "SANDBOX_MEMORY_LIMIT")

    def test_crash_and_protocol_injection_are_detected(self):
        crash = gl.NO_GRADIENT.replace("return torch.tensor(1.0)", "import os\n    os._exit(3)")
        with self.assertRaises(sandbox.SandboxError) as raised:
            self.run_component(crash)
        self.assertEqual(raised.exception.error_type, "SANDBOX_CRASHED")
        inject = gl.NO_GRADIENT.replace(
            "return torch.tensor(1.0)",
            "import os\n    for fd in range(3, 12):\n        try:\n"
            "            os.write(fd, b'\\x00\\x00\\x00\\x10{\"status\": \"ok\"}')\n"
            "        except OSError:\n            pass\n    return torch.tensor(1.0)")
        with self.assertRaises(sandbox.SandboxError) as raised:
            self.run_component(inject)
        self.assertIn(raised.exception.error_type,
                      {"SANDBOX_PROTOCOL", "COMPONENT_EXECUTION", "SANDBOX_CRASHED"})

    def test_second_component_cannot_be_loaded_into_a_session(self):
        with sandbox.SandboxSession({"component.py": gl.POLY1, "params.json": "{}"}) as session:
            self.assertEqual(session.request({"op": "load"}, timeout=60, expect=[])[0]["status"],
                             "ok")
            self.assertEqual(session.request({"op": "load"}, timeout=60, expect=[])[0]["status"],
                             "error")


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patches = [
            patch.dict(os.environ, {"GOTRAIN_COMPONENTS_DIR": str(self.root / "components"),
                                    "GOTRAIN_SOLUTIONS_DIR": str(self.root / "solutions"),
                                    "GOTRAIN_STORE": ""}),
            patch("agent.experiments.RESULTS_DIR", self.root / "campaigns"),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()


@unittest.skipUnless(AVAILABLE, f"isolation runtime unavailable: {REASON}")
class ValidationTests(_StoreCase):
    def test_two_generated_losses_are_accepted(self):
        poly = cc.submit_component(gl.POLY1, tasks=["classification", "segmentation"],
                                   params={"epsilon": 1.0}, name="poly1")
        tversky = cc.submit_component(gl.TVERSKY, tasks=["segmentation"],
                                      params={"alpha": 0.3, "beta": 0.7, "mix": 0.5})
        for result in (poly, tversky):
            self.assertEqual(result["status"], "accepted", result.get("findings"))
            self.assertTrue(all(check["passed"] for check in result["checks"]))
            self.assertEqual(result["use_with"]["pipeline"]["loss"]["name"], "custom")
        names = {check["check"] for check in poly["checks"]}
        self.assertLessEqual({"deterministic", "gradient_flows", "prefers_correct_predictions",
                              "ignored_targets_have_zero_gradient"}, names)
        self.assertNotIn("gotrain_custom_component", sys.modules)  # never imported on host

    def test_invalid_losses_are_rejected_with_actionable_findings(self):
        for source, task, expected in (
            (gl.NO_GRADIENT, "classification", "executes"),
            (gl.IGNORED_PIXELS_LEAK, "segmentation", "ignored_targets_have_zero_gradient"),
            (gl.NONFINITE, "classification", "executes"),
            (gl.PREFERS_WRONG, "classification", "prefers_correct_predictions"),
        ):
            with self.subTest(expected):
                result = cc.submit_component(source, tasks=[task])
                self.assertEqual(result["status"], "rejected")
                failed = {f.get("check") for f in result["findings"]}
                self.assertIn(expected, failed)
                self.assertFalse(cc.load_component(result["component_id"])["accepted"])

    def test_infinite_loop_is_rejected_by_timeout(self):
        result = cc.submit_component(gl.INFINITE_LOOP, tasks=["classification"],
                                     limits=sandbox.Limits(request_timeout=3))
        self.assertEqual(result["status"], "rejected")
        self.assertIn("SANDBOX_TIMEOUT", result["findings"][0]["detail"])


@unittest.skipUnless(AVAILABLE, f"isolation runtime unavailable: {REASON}")
class TrainingTests(_StoreCase):
    def setUp(self):
        super().setUp()
        try:
            from test_solutions_and_autonomy import tiny_model
            from test_geobench_datasets import FakeSplit, FakeTask
        except ImportError:
            from tests.test_solutions_and_autonomy import tiny_model
            from tests.test_geobench_datasets import FakeSplit, FakeTask
        self.task_class, self.split_class = FakeTask, FakeSplit
        self.extra = [patch("ml.models.create_model", tiny_model),
                      patch("ml.dataset.find_task",
                            lambda dataset_id, benchmark_dir=None: FakeTask(dataset_id))]
        for item in self.extra:
            item.start()

    def tearDown(self):
        for item in self.extra:
            item.stop()
        super().tearDown()

    def train(self, dataset_id, model, pipeline=None, device="cpu"):
        from ml.trainer import train_model

        return train_model(dataset_id, {
            "model": model, "epochs": 2, "batch_size": 2, "num_workers": 0, "device": device,
            "learning_rate": 0.01, "output_dir": str(self.root / "experiments"),
            "pipeline": pipeline})

    def test_both_generated_losses_train_and_are_archived(self):
        from ml.solution import load_solution

        cases = (
            ("m-eurosat", "resnet18_rgb", gl.POLY1, ["classification", "segmentation"],
             {"epsilon": 1.0}),
            ("m-cashew-plant", "unet_r34", gl.TVERSKY, ["segmentation"],
             {"alpha": 0.3, "beta": 0.7, "mix": 0.5}),
        )
        solution_ids = set()
        for dataset_id, model, source, tasks, params in cases:
            with self.subTest(dataset_id):
                cid = cc.submit_component(source, tasks=tasks, params=params)["component_id"]
                baseline = self.train(dataset_id, model)
                custom = self.train(dataset_id, model, {
                    "loss": {"name": "custom", "component_id": cid,
                             "class_weighting": "inverse_frequency"}})
                self.assertEqual(custom["status"], "completed", custom.get("message"))
                self.assertNotEqual(custom["solution_id"], baseline["solution_id"])
                solution_ids.add(custom["solution_id"])
                checkpoint = torch.load(custom["checkpoint_path"], weights_only=True)
                archived = checkpoint["custom_components"][0]
                self.assertEqual(Path(archived["archived_source"]).read_text(), source)
                self.assertEqual(archived["source_sha256"],
                                 cc.verify_component(cid)["source_sha256"])
                self.assertEqual(custom["custom_components"][0]["component_id"], cid)
                spec = load_solution(custom["solution_id"])["spec"]
                self.assertEqual(spec["custom_components"][0]["component_id"], cid)
                run = load_solution(custom["solution_id"])["runs"][-1]
                self.assertEqual(run["artifacts"]["custom_components"][0]["component_id"], cid)
                base_weights = torch.load(baseline["checkpoint_path"],
                                          weights_only=True)["model_state_dict"]
                self.assertFalse(all(torch.equal(checkpoint["model_state_dict"][k], v)
                                     for k, v in base_weights.items()),
                                 "the generated loss did not change training")
                self.assertIn(checkpoint["primary_metric"], ("accuracy", "mean_iou"))
                self.assertEqual(torch.load(baseline["checkpoint_path"],
                                            weights_only=True)["custom_components"], [])
        self.assertEqual(len(solution_ids), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_generated_loss_trains_on_cuda(self):
        cid = cc.submit_component(gl.POLY1, tasks=["classification"],
                                  params={"epsilon": 1.0})["component_id"]
        result = self.train("m-eurosat", "resnet18_rgb",
                            {"loss": {"name": "custom", "component_id": cid}}, device="cuda")
        self.assertEqual(result["status"], "completed", result.get("message"))
        self.assertTrue(result["device"].startswith("cuda"))

    def test_checkpoint_reloads_without_the_sandbox(self):
        from ml.test_evaluation import evaluate_checkpoint_on_test

        cid = cc.submit_component(gl.POLY1, tasks=["classification"],
                                  params={"epsilon": 1.0})["component_id"]
        result = self.train("m-eurosat", "resnet18_rgb",
                            {"loss": {"name": "custom", "component_id": cid}})
        self.assertEqual(result["status"], "completed", result.get("message"))
        split_class = self.split_class

        class EvaluatorTask(self.task_class):
            def get_dataset(self, split, band_names=None):
                if split == "test":
                    return split_class(self.dataset_id, 2, seed=9)
                return super().get_dataset(split, band_names)
        with patch("ml.dataset.find_task", lambda d, b=None: EvaluatorTask(d)), \
                patch.dict(os.environ, {"GOTRAIN_SANDBOX_DISABLE": "1"}):
            evaluation = evaluate_checkpoint_on_test(result["checkpoint_path"], num_workers=0,
                                                     device="cpu")
        self.assertEqual(evaluation["num_samples"], 2)

    def test_tampered_or_rejected_components_never_train(self):
        cid = cc.submit_component(gl.POLY1, tasks=["classification"])["component_id"]
        source = Path(os.environ["GOTRAIN_COMPONENTS_DIR"]) / cid / "source.py"
        source.write_text(source.read_text().replace("1.0", "5.0"))
        tampered = self.train("m-eurosat", "resnet18_rgb",
                              {"loss": {"name": "custom", "component_id": cid}})
        self.assertEqual(tampered["status"], "failed")
        self.assertIn("COMPONENT_TAMPERED", tampered["message"])
        rejected = cc.submit_component(gl.PREFERS_WRONG, tasks=["classification"])
        refused = self.train("m-eurosat", "resnet18_rgb", {
            "loss": {"name": "custom", "component_id": rejected["component_id"]}})
        self.assertEqual(refused["status"], "failed")
        self.assertIn("only components accepted", refused["message"])
        self.assertFalse((self.root / "experiments").exists())  # nothing trained

    def test_forged_acceptance_is_caught_by_training_revalidation(self):
        rejected = cc.submit_component(gl.PREFERS_WRONG, tasks=["classification"])
        cid = rejected["component_id"]
        cc._record_validation(cid, {"status": "accepted", "source_sha256":
                                    cc.verify_component(cid)["source_sha256"]})
        forged = self.train("m-eurosat", "resnet18_rgb",
                            {"loss": {"name": "custom", "component_id": cid}})
        self.assertEqual(forged["status"], "failed")
        self.assertIn("re-validation", forged["message"])

    def test_s2_agent_flow_submit_then_train(self):
        from agent import tools
        from agent.autonomy import AutonomyPolicy
        from agent.experiments import load_campaign_experiments
        from agent.pipeline_tools import make_pipeline_tools
        from ml.trainer import train_model

        policy = AutonomyPolicy(scope="S2")
        s2 = {getattr(t, "tool_name", None) or t.__name__: t for t in make_pipeline_tools(
            version="E2", campaign_id="s2", seed=42, autonomy=policy)}
        first = s2["submit_custom_loss"]("poly1", "Under-confident predictions",
                                         gl.PREFERS_WRONG, ["classification"])
        self.assertEqual(first["status"], "rejected")  # Claude revises after reading findings
        self.assertIn("prefers_correct_predictions",
                      {f.get("check") for f in first["findings"]})
        submitted = s2["submit_custom_loss"]("poly1", "Under-confident predictions",
                                             gl.POLY1, ["classification"], {"epsilon": 1.0})
        self.assertEqual(submitted["status"], "accepted", submitted.get("findings"))

        def train_in_temp(dataset_id, config, **context):
            return train_model(dataset_id, {**config, "output_dir": str(self.root / "runs"),
                                            "device": "cpu", "num_workers": 0}, **context)
        with patch.object(tools, "model_runtime_status", lambda m: {"eligible": True, "reason": None}):
            run = tools.make_registered_training_tool(
                version="E2", campaign_id="s2", seed=42, max_experiments=2,
                required_dataset="m-eurosat", training_fn=train_in_temp, autonomy=policy)
            response = run(dataset_id="m-eurosat", model_id="resnet18_rgb", epochs=1,
                           batch_size=2, learning_rate=0.01,
                           pipeline=submitted["use_with"]["pipeline"],
                           change_description="PolyLoss-1 for under-confident classes")
        self.assertEqual(response["status"], "completed", response)
        record = load_campaign_experiments("E2", "s2", 42)[0]
        self.assertEqual(record["configuration"]["pipeline"]["loss"]["component_id"],
                         submitted["component_id"])
        inspected = s2["inspect_solutions"](record["solution_id"])["solution"]
        self.assertEqual(inspected["custom_components"][0]["component_id"],
                         submitted["component_id"])


if __name__ == "__main__":
    unittest.main()

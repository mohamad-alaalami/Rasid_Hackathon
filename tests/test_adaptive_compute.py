"""Phase 1D: adaptive compute allocation and checkpoint continuation.

Real ml.train_model on CPU with tiny stand-in models and a fake GEO-Bench task;
the real guarded campaign tool, search graph, and allocator. No GPU, Bedrock,
network, or test split. The SDK-level class drives agent.main --search
--adaptive-compute through the real Strands agent loop with a scripted model;
it is not a Bedrock campaign.
"""

import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import strands_stub
    from test_geobench_datasets import FakeTask
    from test_solutions_and_autonomy import tiny_model
except ImportError:  # run as tests.<module> from the repo root
    from tests import strands_stub
    from tests.test_geobench_datasets import FakeTask
    from tests.test_solutions_and_autonomy import tiny_model

strands_stub.install()

import torch  # noqa: E402

from agent import main as campaign_main  # noqa: E402
from agent import tools  # noqa: E402
from agent.allocation import ComputeAllocator, make_allocation_tools  # noqa: E402
from agent.autonomy import AutonomyPolicy  # noqa: E402
from agent.experiments import get_budget_status, reserve_training  # noqa: E402
from agent.policies import load_decisions  # noqa: E402
from agent.search_graph import SearchGraph  # noqa: E402
from agent.search_tools import make_search_tools  # noqa: E402
from ml.solution import file_sha256  # noqa: E402
from ml.trainer import STATE_FILENAME, train_model  # noqa: E402
from research.resource_accounting import estimate_run_cost  # noqa: E402

DATASET = "m-eurosat"


def name_of(tool_object):
    return getattr(tool_object, "tool_name", None) or tool_object.__name__


def corrupt(path):
    data = bytearray(Path(path).read_bytes())
    data[len(data) // 2] ^= 0xFF
    Path(path).write_bytes(bytes(data))


class Isolated(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patches = [
            patch("ml.models.create_model", tiny_model),
            patch("ml.dataset.find_task", lambda dataset_id, benchmark_dir=None: FakeTask(dataset_id)),
            patch.dict(os.environ, {"GOTRAIN_SOLUTIONS_DIR": str(self.root / "solutions"),
                                    "GOTRAIN_STORE": ""}),
            patch("agent.experiments.RESULTS_DIR", self.root / "campaigns"),
            patch.object(tools, "model_runtime_status", lambda m: {"eligible": True, "reason": None}),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()


# ---------------------------------------------------------------------------
# Trainer: exact continuation
# ---------------------------------------------------------------------------

class TrainerContinuationTests(Isolated):
    def config(self, **overrides):
        return {"model": "resnet18_rgb", "epochs": 5, "batch_size": 2, "learning_rate": 0.01,
                "num_workers": 0, "device": "cpu", "output_dir": str(self.root / "runs"),
                **overrides}

    def test_two_plus_three_equals_uninterrupted_five(self):
        for scheduler in ("cosine", "warmup_cosine", "step"):
            with self.subTest(scheduler=scheduler):
                cfg = self.config(pipeline={"scheduler": scheduler})
                full = train_model(DATASET, cfg, save_training_state=True)
                first = train_model(DATASET, cfg, until_epoch=2, save_training_state=True)
                self.assertEqual((first["completed_epochs"], first["finished"]), (2, False))
                second = train_model(DATASET, cfg, until_epoch=5, save_training_state=True,
                                     resume_from=first["training_state"])
                self.assertEqual(second["status"], "completed", second.get("message"))
                self.assertEqual((second["start_epoch"], second["completed_epochs"],
                                  second["completed_through_epoch"]), (3, 3, 5))
                self.assertTrue(second["finished"])
                self.assertEqual(second["solution_id"], full["solution_id"])
                full_history = json.loads(Path(full["metrics_path"]).read_text())
                resumed_history = json.loads(Path(second["metrics_path"]).read_text())
                # Bitwise-identical on CPU with num_workers=0: losses, metrics, LR schedule.
                self.assertEqual(full_history, resumed_history)
                a = torch.load(full["training_state"]["path"], weights_only=True)
                b = torch.load(second["training_state"]["path"], weights_only=True)
                for key, tensor in a["model_state_dict"].items():
                    self.assertTrue(torch.equal(tensor, b["model_state_dict"][key]), key)
                for index, slot in a["optimizer_state_dict"]["state"].items():
                    for name, value in slot.items():
                        other = b["optimizer_state_dict"]["state"][index][name]
                        self.assertTrue(torch.equal(torch.as_tensor(value),
                                                    torch.as_tensor(other)), name)
                self.assertEqual(a["scheduler_state_dict"]["last_epoch"], 5)
                self.assertEqual(a["scheduler_state_dict"], b["scheduler_state_dict"])
                self.assertEqual(full["best_val_score"], second["best_val_score"])

    def test_restored_state_drives_the_learning_rate(self):
        cfg = self.config()
        first = train_model(DATASET, cfg, until_epoch=2, save_training_state=True)
        state = torch.load(first["training_state"]["path"], weights_only=True)
        self.assertEqual(state["scheduler_state_dict"]["last_epoch"], 2)
        self.assertEqual(len(state["optimizer_state_dict"]["state"]) > 0, True)
        second = train_model(DATASET, cfg, until_epoch=3, save_training_state=True,
                             resume_from=first["training_state"])
        rates = [e["learning_rate"] for e in json.loads(Path(second["metrics_path"]).read_text())]
        # Cosine over the 5-epoch horizon, continued (not restarted) at epoch 3.
        self.assertLess(rates[2], rates[1])
        self.assertNotAlmostEqual(rates[2], 0.01 * 0.5 * (1 + __import__("math").cos(
            __import__("math").pi / 5)))

    def test_continuation_never_rewrites_earlier_artifacts(self):
        cfg = self.config()
        first = train_model(DATASET, cfg, until_epoch=2, save_training_state=True)
        before = {path: file_sha256(path) for path in Path(first["checkpoint_path"]).parent.iterdir()
                  if path.is_file()}
        second = train_model(DATASET, cfg, until_epoch=4, save_training_state=True,
                             resume_from=first["training_state"])
        self.assertNotEqual(Path(second["metrics_path"]).parent, Path(first["metrics_path"]).parent)
        after = {path: file_sha256(path) for path in before}
        self.assertEqual(before, after)
        if not second["new_best_this_segment"]:
            self.assertEqual(second["checkpoint_path"], first["checkpoint_path"])
        self.assertEqual(second["checkpoint_integrity"], "verified")

    def test_corrupt_or_mismatched_state_is_refused_before_training(self):
        cfg = self.config()
        first = train_model(DATASET, cfg, until_epoch=2, save_training_state=True)
        runs = set((self.root / "runs").iterdir())
        state = dict(first["training_state"])
        cases = {
            "modified": (cfg, 4, {**state, "sha256": "0" * 64}),
            "other_candidate": (self.config(learning_rate=0.02), 4, state),
            "not_beyond_done": (cfg, 2, state),
            "missing": (cfg, 4, {**state, "path": str(self.root / "nope.pt")}),
        }
        for name, (config, until, source) in cases.items():
            with self.subTest(name):
                result = train_model(DATASET, config, until_epoch=until, resume_from=source,
                                     save_training_state=True)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error_type"], "RESUME_STATE_INVALID", result["message"])
        self.assertEqual(set((self.root / "runs").iterdir()), runs)  # nothing was trained
        corrupt(state["path"])
        result = train_model(DATASET, cfg, until_epoch=4, resume_from=state)
        self.assertEqual(result["error_type"], "RESUME_STATE_INVALID")
        self.assertIn("corrupted", result["message"])

    def test_finished_run_cannot_be_continued(self):
        cfg = self.config(epochs=2)
        done = train_model(DATASET, cfg, save_training_state=True)
        self.assertTrue(done["finished"])
        again = train_model(DATASET, self.config(epochs=2), until_epoch=2,
                            resume_from=done["training_state"])
        self.assertEqual(again["error_type"], "RESUME_STATE_INVALID")

    def test_time_allocation_stops_at_an_epoch_boundary_and_can_continue(self):
        cfg = self.config()
        result = train_model(DATASET, cfg, save_training_state=True, max_runtime_seconds=1e-6)
        self.assertEqual(result["completed_epochs"], 1)  # the first epoch cannot be predicted
        self.assertTrue(result["stopped_for_time"])
        self.assertFalse(result["finished"])
        resumed = train_model(DATASET, cfg, until_epoch=3, resume_from=result["training_state"])
        self.assertEqual((resumed["status"], resumed["completed_through_epoch"]), ("completed", 3))

    def test_default_call_is_unchanged(self):
        result = train_model(DATASET, self.config(epochs=2))
        self.assertIsNone(result["training_state"])
        self.assertFalse((Path(result["metrics_path"]).parent / STATE_FILENAME).exists())
        self.assertEqual((result["requested_epochs"], result["completed_epochs"]), (2, 2))
        checkpoint = torch.load(result["checkpoint_path"], weights_only=False)
        self.assertEqual(sorted(checkpoint), sorted([
            "model_state_dict", "model_name", "dataset_id", "task_type", "num_classes",
            "class_names", "preprocessing", "input_padding", "solution_id", "custom_components",
            "pipeline", "class_counts", "class_weights", "pretraining", "config", "seed",
            "best_epoch", "primary_metric", "best_val_score"]))
        self.assertEqual(result["checkpoint_sha256"], file_sha256(result["checkpoint_path"]))


# ---------------------------------------------------------------------------
# Campaign: allocator, guards, budgets
# ---------------------------------------------------------------------------

class AllocationCase(Isolated):
    max_experiments = 8

    def setUp(self):
        super().setUp()
        self.fail_resumes = 0

    def setup_campaign(self, campaign="adaptive", version="E2", launch_gate=None, policy="P0",
                       max_gpu_seconds=None, max_experiments=None):
        max_experiments = max_experiments or self.max_experiments
        self.graph = SearchGraph(campaign, version, 42, max_experiments, DATASET)
        autonomy = AutonomyPolicy(scope="S1")
        self.search = {name_of(t): t for t in make_search_tools(
            graph=self.graph, autonomy=autonomy, required_dataset=DATASET)}
        self.allocator = ComputeAllocator(self.graph, max_experiments=max_experiments,
                                          max_gpu_seconds=max_gpu_seconds)

        def trainer(dataset_id, config, **context):
            if context.get("resume_from") and self.fail_resumes:
                self.fail_resumes -= 1
                return {"status": "failed", "message": "simulated crash during continuation",
                        "training_seconds": 0.2, "error_type": "TRAINING_ERROR"}
            return train_model(dataset_id, {**config, "output_dir": str(self.root / "runs")},
                               **context)
        self.run_tool = tools.make_registered_training_tool(
            version=version, campaign_id=campaign, seed=42, max_experiments=max_experiments,
            required_dataset=DATASET, training_fn=trainer, autonomy=autonomy,
            search_graph=self.graph, launch_gate=launch_gate, selection_policy=policy,
            max_gpu_seconds=max_gpu_seconds, allocator=self.allocator)
        self.tools = {name_of(t): t for t in make_allocation_tools(self.allocator)}
        self.campaign = campaign
        self.version = version

    def propose(self, intent="root", parent=None, claimed=None, lr=0.01, epochs=5):
        return self.search["propose_branch"](
            intent=intent, observation="evidence", hypothesis="should help",
            expected_signal="higher accuracy", parent_solution_id=parent,
            claimed_changes=claimed, dataset_id=DATASET, model_id="resnet18_rgb",
            epochs=epochs, batch_size=2, learning_rate=lr)["solution_id"]

    @staticmethod
    def why(**overrides):
        return {"observation": "validation accuracy still rising",
                "expected_benefit": "higher validation accuracy",
                "cost_justification": "cheap relative to the remaining budget", **overrides}

    def allocate(self, sid, epochs, **why):
        return self.tools["allocate_compute"](solution_id=sid, epochs=epochs, **self.why(**why))

    def extend(self, sid, epochs, **kwargs):
        return self.tools["extend_candidate"](solution_id=sid, additional_epochs=epochs,
                                              **self.why(), **kwargs)

    def budget(self):
        return get_budget_status(self.version, self.campaign, DATASET, 42,
                                 max_experiments=self.max_experiments)


class AdaptiveCampaignTests(AllocationCase):
    def test_explore_cheaply_continue_one_abandon_another(self):
        self.setup_campaign()
        a = self.propose()
        b = self.propose("explore", a, ["training.learning_rate"], lr=0.03)
        self.assertEqual(self.allocate(a, 2)["status"], "completed")
        self.assertEqual(self.allocate(b, 1)["status"], "completed")
        compared = self.tools["compare_continue_vs_explore"](solution_id=a,
                                                             alternative_solution_id=b)
        self.assertEqual(compared["candidate"]["epochs_remaining_to_horizon"], 3)
        self.assertIsNotNone(compared["candidate"]["measured_seconds_per_epoch_max"])
        extended = self.extend(a, 3)
        self.assertEqual(extended["status"], "completed", extended)
        self.assertEqual(extended["candidate"]["epochs_completed"], 5)
        self.assertFalse(extended["candidate"]["continuation"]["can_continue"])
        abandoned = self.tools["abandon_candidate"](solution_id=b, reason="lower accuracy")
        self.assertEqual(abandoned["status"], "abandoned")
        self.assertEqual(self.extend(b, 1)["error_code"], "CANDIDATE_ABANDONED")

        view = self.tools["inspect_compute"](solution_id=a)
        self.assertEqual([s["kind"] for s in view["segments"]], ["initial", "continuation"])
        self.assertEqual([p["epoch"] for p in view["learning_curve"]], [1, 2, 3, 4, 5])
        self.assertEqual(view["segments"][1]["resumed_from_experiment_id"],
                         view["segments"][0]["experiment_id"])
        # Identity unchanged by allocation: one node, two training actions.
        state = self.graph.state()
        self.assertEqual(len(state["nodes"]), 2)
        node = state["nodes"][a]
        self.assertEqual((node["segments"], node["epochs_completed"], node["horizon_epochs"]),
                         (2, 5, 5))
        self.assertEqual(node["checkpoint_sha256"], file_sha256(node["checkpoint_path"]))
        # Incumbent: best measured validation score over completed candidates.
        [group] = self.graph.incumbents(state["nodes"])
        scores = {sid: n["validation_score"] for sid, n in state["nodes"].items()}
        self.assertEqual(group["validation_score"], max(scores.values()))
        # Accounting: three training actions, each one experiment slot.
        self.assertEqual(self.budget()["launched_experiments"], 3)
        report = self.allocator.budget()
        self.assertEqual(report["per_candidate"][a]["training_actions"], 2)
        self.assertEqual(report["actions_with_unknown_runtime"], 0)
        # Decisions are logged with their justification and outcome.
        events = [e for e in self.graph.state()["events"] if e["type"] != "edge"]
        self.assertEqual([e["type"] for e in events],
                         ["allocation", "allocation_outcome"] * 3 + ["abandon"])
        self.assertTrue(all(e["cost_justification"] for e in events if e["type"] == "allocation"))

    def test_plain_launch_and_invalid_allocations_use_no_budget(self):
        self.setup_campaign()
        a = self.propose()
        refused = self.run_tool(dataset_id=DATASET, model_id="resnet18_rgb", epochs=5,
                                batch_size=2, learning_rate=0.01)
        self.assertEqual(refused["error_code"], "USE_COMPUTE_ALLOCATION")
        self.assertEqual(self.extend(a, 1)["error_code"], "NOT_STARTED")
        self.assertEqual(self.allocate(a, 6)["error_code"], "INVALID_ALLOCATION")
        self.assertEqual(self.allocate(a, 2, observation=" ")["error_code"],
                         "JUSTIFICATION_REQUIRED")
        unknown = self.tools["allocate_compute"](solution_id="sol_" + "0" * 20, epochs=1,
                                                 **self.why())
        self.assertEqual(unknown["status"], "rejected")
        self.assertEqual(self.budget()["launched_experiments"], 0)
        self.assertEqual(self.allocate(a, 2)["status"], "completed")
        self.assertEqual(self.allocate(a, 2)["error_code"], "ALREADY_ALLOCATED")
        self.assertEqual(self.extend(a, 4)["error_code"], "BEYOND_HORIZON")
        self.assertEqual(self.budget()["launched_experiments"], 1)

    def test_failed_and_interrupted_continuations_are_counted_and_recoverable(self):
        self.setup_campaign()
        a = self.propose()
        first = self.allocate(a, 2)
        source_state = self.tools["inspect_compute"](solution_id=a)["segments"][0]["training_state"]
        before = file_sha256(source_state["path"])
        self.fail_resumes = 1
        failed = self.extend(a, 2)
        self.assertEqual(failed["status"], "failed")
        node = self.graph.state()["nodes"][a]
        self.assertEqual(node["status"], "completed")  # the measured result survives
        self.assertEqual(node["validation_score"], first["candidate"]["best_validation_score"])
        self.assertEqual(file_sha256(source_state["path"]), before)
        # An interrupted action: reserved, never finalized (simulated crash).
        configuration = dict(self.graph.configuration_for(a))
        configuration.pop("dataset_id")
        crashed = reserve_training(self.version, self.campaign, DATASET, 42, {
            **configuration, "allocation": {"kind": "continuation", "epochs": 1,
                                            "until_epoch": 3, "attempt": 99}},
            max_experiments=self.max_experiments)
        self.assertEqual(crashed["status"], "reserved")
        segments = self.tools["inspect_compute"](solution_id=a)["segments"]
        self.assertEqual([s["status"] for s in segments], ["completed", "failed", "interrupted"])
        self.assertIsNone(segments[2]["runtime_seconds"])  # unknown, not zero
        self.assertEqual(self.allocator.budget()["actions_with_unknown_runtime"], 1)
        # Recovery: retry from the last intact state; every attempt used a slot.
        retried = self.extend(a, 2)
        self.assertEqual(retried["status"], "completed", retried)
        self.assertEqual(retried["allocation"]["attempt"], 2)
        self.assertEqual(retried["candidate"]["epochs_completed"], 4)
        self.assertEqual(self.budget()["launched_experiments"], 4)

    def test_corrupt_latest_state_is_detected_and_recovered_from_earlier(self):
        self.setup_campaign()
        a = self.propose()
        self.allocate(a, 1)
        self.extend(a, 2)
        segments = self.tools["inspect_compute"](solution_id=a)["segments"]
        best_path, best_sha = (self.graph.state()["nodes"][a][k]
                               for k in ("checkpoint_path", "checkpoint_sha256"))
        corrupt(segments[1]["training_state"]["path"])
        refused = self.extend(a, 1)
        self.assertEqual(refused["error_code"], "CHECKPOINT_CORRUPT")
        self.assertEqual(refused["recoverable_from"],
                         [{"experiment_id": segments[0]["experiment_id"],
                           "completed_through_epoch": 1}])
        self.assertEqual(self.budget()["launched_experiments"], 2)  # refused before reservation
        recovered = self.extend(a, 3, resume_from_experiment_id=segments[0]["experiment_id"])
        self.assertEqual(recovered["status"], "completed", recovered)
        self.assertEqual(recovered["candidate"]["epochs_completed"], 4)
        self.assertEqual(file_sha256(best_path), best_sha)  # best checkpoint untouched

    def test_training_time_budget(self):
        self.setup_campaign(max_gpu_seconds=3600.0)
        a = self.propose()
        self.allocate(a, 1)
        budget = self.tools["compute_budget"]()
        used = budget["training_seconds_used_or_reserved"]
        self.assertGreater(used, 0)
        self.assertEqual(budget["remaining_training_seconds"], 3600.0 - used)
        record = self.allocator.candidate(a)["segments"][0]
        self.assertAlmostEqual(used, record["runtime_seconds"])
        # Remaining time below one measured epoch: refused before any reservation.
        per_epoch = self.allocator.candidate(a)["measured_seconds_per_epoch"]["max"]
        with patch.object(self.allocator, "budget",
                          lambda state=None: {"remaining_training_seconds": per_epoch / 2}):
            self.assertEqual(self.extend(a, 1)["error_code"], "INSUFFICIENT_TRAINING_TIME")
        self.assertEqual(self.budget()["launched_experiments"], 1)

    def test_concurrent_extensions_of_one_candidate(self):
        self.setup_campaign()
        a = self.propose()
        self.allocate(a, 1)
        results, barrier = [], threading.Barrier(2)

        def worker(epochs):
            barrier.wait()
            results.append(self.extend(a, epochs))
        threads = [threading.Thread(target=worker, args=(n,)) for n in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        statuses = sorted(r["status"] for r in results)
        self.assertEqual(statuses.count("completed"), 1, results)
        rejected = next(r for r in results if r["status"] == "rejected")
        self.assertIn(rejected["error_code"], ("CANDIDATE_RUNNING", "STALE_CONTINUATION"))
        self.assertEqual(self.budget()["launched_experiments"], 2)

    def test_hitl_gates_continuations(self):
        class Gate:
            calls = []

            def check(self, dataset_id, configuration):
                self.calls.append(configuration.get("allocation", {}).get("kind"))
                if configuration["allocation"]["kind"] == "continuation":
                    return None, {"status": "rejected", "error_code": "HUMAN_APPROVAL_REQUIRED"}
                return None, None

            def record_launch(self, approval, response):
                pass
        gate = Gate()
        self.setup_campaign(launch_gate=gate)
        a = self.propose()
        self.assertEqual(self.allocate(a, 1)["status"], "completed")
        self.assertEqual(self.extend(a, 1)["error_code"], "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(gate.calls, ["initial", "continuation"])
        self.assertEqual(self.budget()["launched_experiments"], 1)

    def test_p1_decision_record_comes_from_the_justification(self):
        self.setup_campaign(policy="P1")
        a = self.propose()
        self.assertEqual(self.allocate(a, 1)["status"], "completed")
        [entry, *_] = [d for d in load_decisions(self.campaign, 42)
                       if d["outcome"].get("status") == "completed"]
        self.assertEqual(entry["decision_record"]["cost_justification"],
                         "cheap relative to the remaining budget")
        self.assertIn("allocate epochs 1-1 of 5", entry["decision_record"]["action"])

    def test_cost_estimate_uses_allocated_epochs(self):
        self.setup_campaign()
        a = self.propose()
        self.allocate(a, 2)
        prices = {"gpu_instance_usd_per_hour": 3600.0}
        configuration = {"architecture": "resnet18_rgb", "image_size": None, "epochs": 5}
        horizon = estimate_run_cost(DATASET, configuration, prices)
        slice_ = estimate_run_cost(DATASET, {**configuration, "allocation": {"epochs": 1}}, prices)
        self.assertAlmostEqual(horizon["estimated_training_seconds"],
                               5 * slice_["estimated_training_seconds"])


class IsolationAndCompatibilityTests(AllocationCase):
    def test_history_blind_profiles_are_refused(self):
        for version in ("E0", "E1"):
            graph = SearchGraph("blind", version, 42, 3, DATASET)
            with self.assertRaises(ValueError):
                ComputeAllocator(graph, max_experiments=3)
        args = campaign_main.build_parser().parse_args(["--adaptive-compute", "--feedback", "E2"])
        with self.assertRaises(SystemExit):
            campaign_main.allocator_for(args, None)

    def test_earlier_campaigns_and_model_only_checkpoints(self):
        # A Phase 1C campaign: plain search mode, no allocator, one full run per node.
        graph = SearchGraph("legacy", "E2", 42, 3, DATASET)
        autonomy = AutonomyPolicy(scope="S1")
        search = {name_of(t): t for t in make_search_tools(graph=graph, autonomy=autonomy,
                                                           required_dataset=DATASET)}
        run = tools.make_registered_training_tool(
            version="E2", campaign_id="legacy", seed=42, max_experiments=3,
            required_dataset=DATASET, autonomy=autonomy, search_graph=graph,
            training_fn=lambda dataset_id, config, **c: train_model(
                dataset_id, {**config, "output_dir": str(self.root / "runs")}, **c))
        args = {"dataset_id": DATASET, "model_id": "resnet18_rgb", "epochs": 2,
                "batch_size": 2, "learning_rate": 0.01}
        sid = search["propose_branch"](intent="root", observation="o", hypothesis="h",
                                       expected_signal="s", **args)["solution_id"]
        self.assertEqual(run(**args)["status"], "completed")
        node = graph.state()["nodes"][sid]
        self.assertEqual((node["status"], node["segments"], node["epochs_completed"]),
                         ("completed", 1, 2))
        self.assertEqual(node["checkpoint_sha256"], file_sha256(node["checkpoint_path"]))
        # The same campaign viewed by the allocator: visible, measured, not resumable.
        allocator = ComputeAllocator(graph, max_experiments=3)
        view = allocator.candidate(sid)
        self.assertEqual(view["continuation"]["reason_code"], "NOT_RESUMABLE")
        self.assertEqual(view["segments"][0]["completed_through_epoch"], 2)
        self.assertIsNone(view["segments"][0]["training_state"])
        # The model-only checkpoint keeps its format and still loads.
        checkpoint = torch.load(node["checkpoint_path"], weights_only=False)
        self.assertIn("model_state_dict", checkpoint)
        self.assertNotIn("optimizer_state_dict", checkpoint)


@unittest.skipUnless(strands_stub.real_strands_available(), "strands-agents is not installed")
class StrandsAdaptiveCampaignTests(Isolated):
    """agent.main --search --adaptive-compute through the real Strands loop, scripted model."""

    def test_scripted_agent_continues_one_candidate_and_abandons_another(self):
        from strands import Agent

        try:
            from test_strands_integration import ScriptedModel
        except ImportError:
            from tests.test_strands_integration import ScriptedModel
        real_factory = tools.make_registered_training_tool

        def factory(**kwargs):
            def trainer(dataset_id, config, **context):
                return train_model(dataset_id, {**config, "output_dir": str(self.root / "runs")},
                                   **context)
            return real_factory(training_fn=trainer, **kwargs)
        run = {"dataset_id": DATASET, "model_id": "resnet18_rgb", "epochs": 3,
               "batch_size": 2, "learning_rate": 0.01}
        why = {"observation": "o", "expected_benefit": "b", "cost_justification": "c"}
        script, seen = [
            ("tool", ("propose_branch", {"intent": "root", "observation": "start",
                                         "hypothesis": "baseline", "expected_signal": "any",
                                         **run})),
            ("tool", ("propose_branch", {"intent": "explore", "observation": "alt",
                                         "hypothesis": "higher lr", "expected_signal": "acc",
                                         "parent_solution_id": "__0__",
                                         "claimed_changes": ["training.learning_rate"],
                                         **{**run, "learning_rate": 0.05}})),
            ("tool", ("allocate_compute", {"solution_id": "__0__", "epochs": 1, **why})),
            ("tool", ("allocate_compute", {"solution_id": "__1__", "epochs": 1, **why})),
            ("tool", ("compare_continue_vs_explore", {"solution_id": "__0__",
                                                      "alternative_solution_id": "__1__"})),
            ("tool", ("extend_candidate", {"solution_id": "__0__", "additional_epochs": 2,
                                           **why})),
            ("tool", ("abandon_candidate", {"solution_id": "__1__", "reason": "worse curve"})),
            ("tool", ("conclude_search", {"reason": "horizon reached"})),
            ("text", "done"),
        ], []

        class IdFillingModel(ScriptedModel):
            async def stream(inner, messages, tool_specs=None, system_prompt=None, **kwargs):
                import re

                found = []
                for sid in re.findall(r"sol_[0-9a-f]{20}", json.dumps(messages, default=str)):
                    if sid not in found:
                        found.append(sid)
                if script and script[0][0] == "tool":
                    arguments = script[0][1][1]
                    for key, value in list(arguments.items()):
                        if isinstance(value, str) and value.startswith("__") and value.endswith("__"):
                            arguments[key] = found[int(value[2:-2])]
                async for event in ScriptedModel.stream(inner, messages, tool_specs,
                                                        system_prompt, **kwargs):
                    yield event
        with patch.object(campaign_main, "_bedrock_model",
                          lambda *a: IdFillingModel(script, seen)), \
                patch.object(campaign_main, "make_registered_training_tool", factory), \
                patch.object(campaign_main, "Agent",
                             lambda **kw: Agent(callback_handler=None, **kw)):
            out = io.StringIO()
            with redirect_stdout(out):
                campaign_main.run_campaign(campaign_main.build_parser().parse_args(
                    ["--campaign-id", "sdk-adaptive", "--dataset", DATASET, "--feedback", "E2",
                     "--search", "--adaptive-compute", "--max-experiments", "4"]))
        output = out.getvalue()
        report = json.loads(output[output.index("{", output.index("FINAL PERSISTED")):])
        self.assertEqual(len(report["experiments"]), 3)  # 2 allocations + 1 extension
        compute = report["compute_allocation"]
        by_status = {sid: c for sid, c in compute["candidates"].items()}
        continued = [c for c in by_status.values() if c["epochs_completed"] == 3]
        abandoned = [c for c in by_status.values() if c["abandoned"]]
        self.assertEqual(len(continued), 1)
        self.assertEqual(len(abandoned), 1)
        self.assertEqual(abandoned[0]["epochs_completed"], 1)
        self.assertEqual(compute["budget"]["training_actions_launched"], 3)
        self.assertIsNotNone(report["search_graph"]["concluded"])


if __name__ == "__main__":
    unittest.main()

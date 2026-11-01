"""End-to-end trainer wiring, solution records, reproducibility, and autonomy tools.

ml.train_model runs for real on CPU with a fake GEO-Bench task and a tiny
stand-in network (torchvision is not needed); no GPU, network, or test split.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

try:
    import strands_stub
    from test_geobench_datasets import FakeTask
except ImportError:  # run as tests.<module> from the repo root
    from tests import strands_stub
    from tests.test_geobench_datasets import FakeTask

strands_stub.install()  # no-op when strands-agents is installed

from agent import tools
from agent.autonomy import AutonomyPolicy
from agent.experiments import get_budget_status, load_campaign_experiments
from agent.main import build_parser
from agent.pipeline_tools import make_pipeline_tools
from ml import solution as solutions
from ml.config import TrainingConfig
from ml.trainer import train_model


class TinyClassifier(torch.nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4, classes)

    def forward(self, images):
        return self.fc(self.conv(images).mean(dim=(2, 3)))


class TinySegmenter(torch.nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.head = torch.nn.Conv2d(channels, classes, 1)

    def forward(self, images):
        return self.head(images)


def tiny_model(name, task_type, num_classes, input_channels, pretrained=True,
               pretrained_checkpoint=None, rgb_positions=(0, 1, 2)):
    cls = TinyClassifier if task_type == "classification" else TinySegmenter
    return cls(input_channels, num_classes)


class TrainerTestCase(unittest.TestCase):
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

    def train(self, dataset_id="m-eurosat", model="resnet18_rgb", context=None, **settings):
        config = {"model": model, "epochs": 2, "batch_size": 2, "num_workers": 0,
                  "device": "cpu", "learning_rate": 0.01,
                  "output_dir": str(self.root / "experiments"), **settings}
        return train_model(dataset_id, config, solution_context=context)

    @staticmethod
    def weights(result):
        return torch.load(result["checkpoint_path"], weights_only=True)["model_state_dict"]


class TrainerWiringTests(TrainerTestCase):
    def test_historical_defaults_are_recorded_and_reproducible(self):
        first = self.train()
        second = self.train()
        self.assertEqual(first["status"], "completed", first.get("message"))
        self.assertEqual(first["solution_id"], second["solution_id"])
        for key, value in self.weights(first).items():
            self.assertTrue(torch.equal(value, self.weights(second)[key]), key)
        checkpoint = torch.load(first["checkpoint_path"], weights_only=True)
        self.assertEqual(checkpoint["pipeline"]["optimizer"]["name"], "adamw")
        self.assertEqual(checkpoint["solution_id"], first["solution_id"])
        record = solutions.load_solution(first["solution_id"])
        self.assertEqual(len(record["proposals"]), 2)
        self.assertEqual([run["status"] for run in record["runs"]], ["completed", "completed"])
        run = record["runs"][0]
        self.assertEqual(run["artifacts"]["checkpoint_sha256"],
                         solutions.file_sha256(first["checkpoint_path"]))
        self.assertEqual(run["validation"]["split"], "validation")
        self.assertIn("packages", run["environment"])
        self.assertEqual(run["dataset_fingerprint"]["status"], "unavailable")

    def test_each_component_choice_changes_training(self):
        baseline = self.weights(self.train())
        for name, pipeline in (
            ("optimizer", {"optimizer": {"name": "sgd", "momentum": 0.9}}),
            ("scheduler", {"scheduler": "constant"}),
            ("loss", {"loss": {"name": "focal", "gamma": 2.0}}),
            ("label_smoothing", {"loss": {"name": "cross_entropy", "label_smoothing": 0.2}}),
            ("augmentation", {"augmentations": [{"name": "rot90", "p": 1.0}]}),
            ("dropout", {"dropout": 0.5}),
            ("normalization", {"normalization": "train_channel_stats"}),
        ):
            with self.subTest(name):
                result = self.train(pipeline=pipeline)
                self.assertEqual(result["status"], "completed", result.get("message"))
                changed = self.weights(result)
                self.assertFalse(all(torch.equal(changed[k], baseline[k]) for k in baseline),
                                 f"{name} had no effect on training")

    def test_segmentation_with_class_weighted_dice(self):
        result = self.train("m-cashew-plant", "unet_r34", pipeline={
            "loss": {"name": "dice_cross_entropy", "class_weighting": "inverse_frequency"},
            "augmentations": [{"name": "rot90"}, {"name": "random_resized_crop"}],
        })
        self.assertEqual(result["status"], "completed", result.get("message"))
        checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
        self.assertEqual(len(checkpoint["class_weights"]), 7)
        self.assertEqual(checkpoint["pipeline"]["loss"]["name"], "dice_cross_entropy")

    def test_incompatible_pipeline_fails_before_any_training(self):
        result = self.train(pipeline={"loss": {"name": "dice_cross_entropy"}})
        self.assertEqual(result["status"], "failed")
        self.assertIn("does not support classification", result["message"])
        self.assertFalse((self.root / "experiments").exists())


class SolutionStoreTests(TrainerTestCase):
    def spec(self, **settings):
        cfg = TrainingConfig.from_mapping({"model": "resnet18_rgb", **settings})
        return solutions.build_solution_spec("m-eurosat", cfg)

    def test_identity_is_content_addressed_and_excludes_lineage(self):
        spec = self.spec()
        self.assertEqual(solutions.solution_id(spec), solutions.solution_id(self.spec()))
        self.assertNotEqual(solutions.solution_id(spec),
                            solutions.solution_id(self.spec(learning_rate=0.001)))
        parent = solutions.propose_solution(spec)
        child_spec = self.spec(pipeline={"optimizer": "sgd"})
        child = solutions.propose_solution(child_spec, {"parent_solution_id": parent,
                                                       "change_description": "try SGD"})
        again = solutions.propose_solution(child_spec, {"parent_solution_id": None})
        self.assertEqual(child, again)
        proposals = solutions.load_solution(child)["proposals"]
        self.assertEqual([p["parent_solution_id"] for p in proposals], [parent, None])
        self.assertEqual(proposals[0]["change_description"], "try SGD")

    def test_spec_contents_and_round_trip(self):
        spec = self.spec(pipeline={"input_view": "rgb", "dropout": 0.1})
        self.assertEqual(spec["schema"], "gotrain.solution/1")
        self.assertEqual(spec["inputs"]["band_names"], ["B04", "B03", "B02"])
        self.assertEqual(spec["model"]["checkpoint_id"], "ResNet18_Weights.IMAGENET1K_V1")
        self.assertEqual(spec["pipeline"]["dropout"], 0.1)
        self.assertEqual(spec["custom_components"], [])
        sid = solutions.propose_solution(spec)
        loaded = solutions.load_solution(sid)
        self.assertEqual(loaded["spec"], json.loads(json.dumps(spec)))

    def test_immutability_and_reserved_extensions_are_enforced(self):
        spec = self.spec()
        sid = solutions.propose_solution(spec)
        stored = Path(os.environ["GOTRAIN_SOLUTIONS_DIR"]) / sid / "spec.json"
        tampered = json.loads(stored.read_text())
        tampered["training"]["epochs"] = 99
        stored.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(ValueError, "immutable"):
            solutions.propose_solution(spec)
        with self.assertRaisesRegex(ValueError, "Unknown parent"):
            solutions.propose_solution(self.spec(seed=1), {"parent_solution_id": "sol_missing"})
        with self.assertRaisesRegex(ValueError, "S2"):
            solutions.propose_solution({**spec, "custom_components": [{"file": "x.py"}]})
        warm = {**spec, "initialization": {"warm_start_from_solution": sid,
                                           "warm_start_checkpoint": None}}
        with self.assertRaisesRegex(ValueError, "Warm starts"):
            solutions.propose_solution(warm)

    def test_dataset_fingerprint_from_local_files(self):
        directory = self.root / "m-test"
        directory.mkdir()
        (directory / "task_specs.pkl").write_bytes(b"spec")
        (directory / "train.hdf5").write_bytes(b"x" * 10)
        task = type("Task", (), {"get_dataset_dir": lambda self: directory})()
        first = solutions.dataset_fingerprint(task)
        (directory / "train.hdf5").write_bytes(b"x" * 11)
        self.assertEqual(first["file_count"], 2)
        self.assertNotEqual(first["sha256"], solutions.dataset_fingerprint(task)["sha256"])


def tool_params(function):
    spec = getattr(function, "tool_spec", None)
    if spec:
        return list(spec["inputSchema"]["json"]["properties"])
    import inspect

    return list(inspect.signature(function).parameters)


class AgentAutonomyTests(TrainerTestCase):
    HISTORICAL = ["dataset_id", "model_id", "epochs", "batch_size", "learning_rate",
                  "weight_decay", "pretrained", "image_size", "early_stopping",
                  "early_stopping_patience", "early_stopping_min_delta"]

    def tool(self, autonomy=None, version="E2", campaign="c", policy="P0"):
        output_dir = str(self.root / "experiments")

        def train_in_temp(dataset_id, config, **context):
            return train_model(dataset_id, {**config, "output_dir": output_dir}, **context)

        return tools.make_registered_training_tool(
            version=version, campaign_id=campaign, seed=42, max_experiments=4,
            required_dataset="m-eurosat", training_fn=train_in_temp,
            selection_policy=policy, autonomy=autonomy,
        )

    def args(self, lr=0.01, **extra):
        return {"dataset_id": "m-eurosat", "model_id": "resnet18_rgb", "epochs": 2,
                "batch_size": 2, "learning_rate": lr, **extra}

    def test_default_policy_keeps_the_historical_tool(self):
        self.assertEqual(tool_params(self.tool()), self.HISTORICAL)
        self.assertEqual(AutonomyPolicy().instructions(), "")
        self.assertEqual(make_pipeline_tools(version="E2", campaign_id="c", seed=42,
                                             autonomy=AutonomyPolicy()), [])
        parsed = build_parser().parse_args([])
        self.assertEqual((parsed.authority, parsed.scope), ("A2", "S0"))
        self.assertIn("pipeline", tool_params(self.tool(AutonomyPolicy(scope="S1"))))
        self.assertIn("decision_record", tool_params(self.tool(AutonomyPolicy(scope="S1"),
                                                               policy="P1")))

    def test_reserved_levels_are_refused(self):
        with self.assertRaisesRegex(ValueError, "A0"):
            AutonomyPolicy(authority="A0")
        with self.assertRaisesRegex(ValueError, "S2"):
            AutonomyPolicy(scope="S2")

    def test_s1_rejects_unsupported_combinations_before_budget(self):
        run = self.tool(AutonomyPolicy(scope="S1"))
        response = run(**self.args(pipeline={"loss": {"name": "dice_cross_entropy"},
                                             "optimizer": {"name": "rmsprop"}}))
        self.assertEqual(response["error_code"], "INVALID_PIPELINE")
        self.assertEqual(len(response["errors"]), 1)  # rmsprop rejected during resolution
        self.assertIn("unknown optimizer 'rmsprop'", response["errors"][0])
        response = run(**self.args(pipeline={"loss": {"name": "dice_cross_entropy"}}))
        self.assertIn("does not support classification", response["errors"][0])
        self.assertEqual(get_budget_status("E2", "c", "m-eurosat", 42)["launched_experiments"], 0)

    def test_s1_composed_run_records_lineage_and_is_inspectable(self):
        policy = AutonomyPolicy(scope="S1")
        run = self.tool(policy)
        first = run(**self.args())
        self.assertEqual(first["status"], "completed")
        parent = load_campaign_experiments("E2", "c", 42)[0]["solution_id"]
        second = run(**self.args(pipeline={"optimizer": "sgd", "dropout": 0.2},
                                 parent_solution_id=parent,
                                 change_description="SGD with head dropout"))
        self.assertEqual(second["status"], "completed")
        records = load_campaign_experiments("E2", "c", 42)
        self.assertEqual(records[1]["configuration"]["pipeline"]["optimizer"], "sgd")
        child = solutions.load_solution(records[1]["solution_id"])
        proposal = child["proposals"][0]
        self.assertEqual((proposal["parent_solution_id"], proposal["campaign_id"],
                          proposal["autonomy"]), (parent, "c", {"authority": "A2", "scope": "S1"}))
        self.assertEqual(child["runs"][0]["campaign_id"], "c")

        pipeline_tools = {t.__name__ if not hasattr(t, "tool_name") else t.tool_name: t
                          for t in make_pipeline_tools(version="E2", campaign_id="c", seed=42,
                                                       autonomy=policy)}
        listed = pipeline_tools["inspect_solutions"]()["solutions"]
        self.assertEqual([s["solution_id"] for s in listed], [parent, child["solution_id"]])
        self.assertEqual(listed[1]["change_description"], "SGD with head dropout")
        self.assertTrue(listed[1]["runs"][0]["checkpoint_path"].endswith("best.pt"))
        space = pipeline_tools["list_pipeline_components"]("m-eurosat")
        self.assertIn("resnet18_s2", space["dataset"]["models"])
        self.assertEqual(space["dataset"]["models"]["resnet18_s2"]["normalizations"],
                         ["dataset_default"])

    def test_e0_sees_only_the_latest_solution(self):
        run = self.tool(AutonomyPolicy(scope="S1"), version="E0", campaign="e0")
        run(**self.args())
        run(**self.args(lr=0.02))
        inspect = {getattr(t, "tool_name", None) or t.__name__: t for t in make_pipeline_tools(
            version="E0", campaign_id="e0", seed=42, autonomy=AutonomyPolicy(scope="S1"))}
        visible = inspect["inspect_solutions"]()["solutions"]
        self.assertEqual(len(visible), 1)
        latest = load_campaign_experiments("E0", "e0", 42)[-1]["solution_id"]
        self.assertEqual(visible[0]["solution_id"], latest)
        first = load_campaign_experiments("E0", "e0", 42)[0]["solution_id"]
        self.assertEqual(inspect["inspect_solutions"](first)["error_code"], "NOT_VISIBLE")

    def test_a1_requires_a_validated_proposal(self):
        policy = AutonomyPolicy(authority="A1", scope="S0")
        run = self.tool(policy, campaign="a1")
        blocked = run(**self.args())
        self.assertEqual(blocked["error_code"], "PROPOSAL_REQUIRED")
        self.assertEqual(get_budget_status("E2", "a1", "m-eurosat", 42)["launched_experiments"], 0)
        propose = {getattr(t, "tool_name", None) or t.__name__: t for t in make_pipeline_tools(
            version="E2", campaign_id="a1", seed=42, autonomy=policy)}["propose_solution"]
        refused = propose(**self.args(pipeline={"optimizer": "sgd"}))
        self.assertEqual(refused["error_code"], "SCOPE_NOT_PERMITTED")
        proposed = propose(**self.args())
        self.assertEqual(proposed["status"], "proposed")
        launched = run(**self.args())
        self.assertEqual(launched["status"], "completed")
        record = load_campaign_experiments("E2", "a1", 42)[0]
        self.assertEqual(record["solution_id"], proposed["solution_id"])


if __name__ == "__main__":
    unittest.main()

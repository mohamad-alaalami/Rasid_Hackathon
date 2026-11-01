import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import strands  # noqa: F401
except ImportError:  # Local machines without Strands: tools become plain functions.
    stub = types.ModuleType("strands")
    stub.tool = lambda function: function
    stub.Agent = object
    stub.models = types.ModuleType("strands.models")
    stub.models.BedrockModel = object
    sys.modules["strands"] = stub
    sys.modules["strands.models"] = stub.models

from agent import tools
from agent.experiments import FEEDBACK_VERSIONS, get_budget_status, load_campaign_experiments
from agent.main import build_parser
from agent.policies import (
    DECISION_FIELDS,
    build_decision_trace,
    build_system_prompt,
    load_decisions,
    validate_decision_record,
)
from agent.policy_comparison import aggregate_policy_comparison, write_policy_reports
from agent.prompts import SYSTEM_PROMPT

DECISION = {
    "observation": "No experiment has run yet; there is no evidence about convergence.",
    "action": "Run the eligible ImageNet ResNet18 at native resolution as a reference.",
    "expected_signal": "A finite validation accuracy above chance; near-chance would challenge it.",
    "cost_justification": "Smallest eligible model; runtime has not been measured yet.",
}


class FakeTrainer:
    """Stands in for ml.train_model; counts every GPU launch."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.calls = []

    def __call__(self, *, dataset_id, config):
        self.calls.append((dataset_id, config))
        score = {0.0003: 0.71, 0.001: 0.72}[config["learning_rate"]]
        metrics = self.directory / f"metrics_{len(self.calls)}.json"
        metrics.write_text(json.dumps([
            {"train_loss": 0.9, "val_loss": 0.8, "val_accuracy": score - 0.05, "learning_rate": 0.001},
            {"train_loss": 0.5, "val_loss": 0.6, "val_accuracy": score, "learning_rate": 0.0005},
        ]), encoding="utf-8")
        return {
            "status": "completed",
            "experiment_id": f"ml-{len(self.calls)}",
            "best_val_score": score,
            "primary_metric": "accuracy",
            "metrics_path": str(metrics),
            "checkpoint_path": f"experiments/ml-{len(self.calls)}/best.pt",
            "native_resolution": [320, 320],
            "effective_resolution": [320, 320],
            "training_seconds": 12.5,
            "peak_gpu_memory_mb": 900.0,
            "device": "cuda",
        }


def run_args(model="resnet18_rgb", learning_rate=0.0003, epochs=2, batch_size=16):
    return {
        "dataset_id": "m-pv4ger",
        "model_id": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


class SelectionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.patches = [
            patch("agent.experiments.RESULTS_DIR", root / "campaigns"),
            # Local machines lack torchvision; eligibility is not under test here.
            patch.object(tools, "model_runtime_status", lambda model_id: {"eligible": True, "reason": None}),
        ]
        for item in self.patches:
            item.start()
        self.trainer = FakeTrainer(root)

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def tool(self, policy, campaign="campaign-p", version="E2", max_experiments=3, attempts=None):
        return tools.make_registered_training_tool(
            version=version,
            campaign_id=campaign,
            seed=42,
            max_experiments=max_experiments,
            required_dataset="m-pv4ger",
            training_fn=self.trainer,
            attempt_log=attempts,
            selection_policy=policy,
        )

    def budget(self, campaign, version="E2"):
        return get_budget_status(version, campaign, "m-pv4ger", 42, max_experiments=3)

    def test_p0_is_default_and_keeps_existing_prompt(self):
        self.assertEqual(build_parser().parse_args([]).policy, "P0")
        self.assertIs(build_system_prompt("P0"), SYSTEM_PROMPT)
        self.assertIn("decision_record", build_system_prompt("P1"))
        default_tool = tools.make_registered_training_tool(
            version="E0", campaign_id="default", seed=42, max_experiments=1,
            training_fn=self.trainer,
        )
        self.assertEqual(default_tool(**run_args())["status"], "completed")
        self.assertEqual(load_decisions("default")[0]["selection_policy"], "P0")

    def test_p0_launches_training_without_decision_record(self):
        response = self.tool("P0")(**run_args())
        self.assertEqual(response["status"], "completed")
        self.assertEqual(len(self.trainer.calls), 1)
        trace = build_decision_trace("E2", "campaign-p", 42)
        self.assertIsNone(trace[0]["decision_record"])
        self.assertEqual(trace[0]["selection_policy"], "P0")

    def test_p1_rejects_malformed_records_without_training_or_budget(self):
        run = self.tool("P1", max_experiments=1)
        malformed = [
            None,
            "not json",
            {key: value for key, value in DECISION.items() if key != "expected_signal"},
            {**DECISION, "cost_justification": "   "},
            {**DECISION, "confidence": "95%"},
        ]
        for record in malformed:
            response = run(decision_record=record, **run_args())
            self.assertEqual(response["status"], "rejected")
            self.assertEqual(response["error_code"], "INVALID_DECISION_RECORD")
            self.assertEqual(response["required_fields"], list(DECISION_FIELDS))
            self.assertTrue(response["field_errors"])
        self.assertEqual(self.trainer.calls, [])
        self.assertEqual(load_campaign_experiments("E2", "campaign-p", 42), [])
        self.assertEqual(self.budget("campaign-p")["launched_experiments"], 0)
        # The single-experiment budget is still available after correction.
        self.assertEqual(run(decision_record=DECISION, **run_args())["status"], "completed")
        rejected = [step for step in build_decision_trace("E2", "campaign-p", 42)
                    if step["outcome"]["status"] == "rejected"]
        self.assertEqual(len(rejected), len(malformed))

    def test_p1_records_decision_separately_from_measured_result(self):
        run = self.tool("P1")
        self.assertEqual(run(decision_record=DECISION, **run_args())["status"], "completed")
        response = run(decision_record=json.dumps(DECISION), **run_args(learning_rate=0.001))
        self.assertEqual(response["status"], "completed")
        trace = build_decision_trace("E2", "campaign-p", 42)
        second = trace[1]
        self.assertEqual(second["selection_policy"], "P1")
        self.assertEqual(second["campaign_id"], "campaign-p")
        self.assertEqual(second["decision_record"], DECISION)
        self.assertEqual(second["context_at_decision"]["previous_experiment_id"], 1)
        self.assertEqual(second["context_at_decision"]["incumbent_experiment_id"], 1)
        self.assertEqual(second["measured"]["validation_score"], 0.72)
        self.assertEqual(second["measured"]["training_seconds"], 12.5)
        self.assertAlmostEqual(second["measured_delta_vs_incumbent"], 0.01)
        self.assertEqual(second["configuration"]["learning_rate"], 0.001)

    def test_decision_records_do_not_reach_agent_observations(self):
        run = self.tool("P1", version="E3")
        response = run(decision_record=DECISION, **run_args())
        self.assertNotIn("decision_record", json.dumps(response))
        for record in load_campaign_experiments("E3", "campaign-p", 42):
            self.assertNotIn("decision_record", record)
            self.assertNotIn("selection_policy", record)

    def test_policies_share_registry_budget_and_evaluator(self):
        p0, p1 = self.tool("P0", "c0", max_experiments=1), self.tool("P1", "c1", max_experiments=1)
        incompatible = run_args(model="unet_r34")
        too_large = run_args(batch_size=64)
        for args in (incompatible, too_large):
            self.assertEqual(p0(**args)["error_code"], "INVALID_MODEL_CONFIGURATION")
            self.assertEqual(p1(decision_record=DECISION, **args)["error_code"],
                             "INVALID_MODEL_CONFIGURATION")
        first_p0, first_p1 = p0(**run_args()), p1(decision_record=DECISION, **run_args())
        self.assertEqual(set(first_p0), set(first_p1))
        self.assertEqual(set(first_p0["feedback"]), set(first_p1["feedback"]))
        self.assertEqual(p0(**run_args(learning_rate=0.001))["error_code"], "BUDGET_EXHAUSTED")
        self.assertEqual(
            p1(decision_record=DECISION, **run_args(learning_rate=0.001))["error_code"],
            "BUDGET_EXHAUSTED",
        )
        self.assertEqual(len(self.trainer.calls), 2)
        for _, config in self.trainer.calls:
            self.assertEqual(config["model"], "resnet18_rgb")
            self.assertEqual(config["seed"], 42)

    def test_separate_campaigns_do_not_share_history(self):
        self.assertEqual(self.tool("P0", "c0")(**run_args())["status"], "completed")
        self.assertEqual(
            self.tool("P1", "c1")(decision_record=DECISION, **run_args())["status"], "completed"
        )
        self.assertEqual(len(load_campaign_experiments("E2", "c0", 42)), 1)
        self.assertEqual(len(load_campaign_experiments("E2", "c1", 42)), 1)
        self.assertEqual([d["selection_policy"] for d in load_decisions("c0")], ["P0"])
        self.assertEqual([d["selection_policy"] for d in load_decisions("c1")], ["P1"])

    def test_policy_cannot_change_inside_a_campaign(self):
        self.assertEqual(self.tool("P0", "shared")(**run_args())["status"], "completed")
        response = self.tool("P1", "shared")(decision_record=DECISION, **run_args(learning_rate=0.001))
        self.assertEqual(response["error_code"], "POLICY_MISMATCH")
        self.assertEqual(len(self.trainer.calls), 1)

    def test_every_feedback_profile_combines_with_both_policies(self):
        for version in FEEDBACK_VERSIONS:
            feedback = {}
            for policy in ("P0", "P1"):
                run = self.tool(policy, f"{version}-{policy}", version=version)
                extra = {"decision_record": DECISION} if policy == "P1" else {}
                response = run(**extra, **run_args())
                self.assertEqual(response["status"], "completed", (version, policy))
                feedback[policy] = response["feedback"]
            # Same configuration, same profile: P1 receives exactly P0's observations.
            self.assertEqual(feedback["P0"], feedback["P1"], version)

    def test_early_stopping_reaches_trainer_under_both_policies(self):
        options = {"early_stopping": True, "early_stopping_patience": 2,
                   "early_stopping_min_delta": 0.001}
        self.tool("P0", "c0")(**run_args(), **options)
        self.tool("P1", "c1")(decision_record=DECISION, **run_args(), **options)
        for _, config in self.trainer.calls:
            self.assertEqual({key: config[key] for key in options}, options)
        bad = self.tool("P1", "c2")(decision_record=DECISION, **run_args(),
                                    early_stopping=True, early_stopping_patience=0)
        self.assertEqual(bad["error_code"], "INVALID_MODEL_CONFIGURATION")
        self.assertEqual(len(self.trainer.calls), 2)

    def test_validator_normalizes_whitespace(self):
        record, error = validate_decision_record({key: f"  {value} " for key, value in DECISION.items()})
        self.assertIsNone(error)
        self.assertEqual(record, DECISION)


def trial(policy, seed=42, improvement=0.05, feedback="E2", campaign=None, rejected=0):
    return {
        "selection_policy": policy,
        "seed": seed,
        "campaign_id": campaign or f"c-{seed}-{policy}",
        "feedback_version": feedback,
        "dataset": "m-pv4ger",
        "max_experiments": 3,
        "initial_configuration": {"architecture": "resnet18_rgb", "epochs": 2},
        "best_validation_score": 0.7 + improvement,
        "validation_improvement": improvement,
        "total_gpu_training_seconds": 30.0,
        "agent_overhead_seconds": 4.0,
        "successful_runs": 3,
        "failed_runs": 0,
        "total_experiments": 3,
        "duplicate_attempts": 0,
        "invalid_attempts": rejected,
        "rejected_decision_records": rejected,
        "decision_trace": [],
    }


class PolicyComparisonTests(unittest.TestCase):
    metadata = {"seeds": [42], "dataset": "m-pv4ger", "feedback_version": "E2", "max_experiments": 3}

    def test_descriptive_summary_never_claims_a_winner(self):
        report = aggregate_policy_comparison(
            [trial("P0", improvement=0.02), trial("P1", improvement=0.04, rejected=1)], self.metadata
        )
        self.assertEqual(report["descriptive_leader"], "P1")
        self.assertFalse(report["winner_claimed"])
        self.assertIn("1 matched seed", report["statistical_note"])
        self.assertEqual(report["policies"]["P1"]["rejected_decision_records"], 1)
        self.assertEqual(report["policies"]["P1"]["invalid_configuration_attempts"], 0)

    def test_unmatched_conditions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "feedback_version"):
            aggregate_policy_comparison([trial("P0"), trial("P1", feedback="E3")], self.metadata)
        with self.assertRaisesRegex(ValueError, "campaign ID"):
            aggregate_policy_comparison(
                [trial("P0", campaign="same"), trial("P1", campaign="same")], self.metadata
            )
        with self.assertRaisesRegex(ValueError, "one P0 and one P1"):
            aggregate_policy_comparison([trial("P0")], self.metadata)

    def test_reports_are_written(self):
        report = aggregate_policy_comparison([trial("P0"), trial("P1")], self.metadata)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_policy_reports(report, temporary)
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            self.assertIsNone(report["descriptive_leader"])


if __name__ == "__main__":
    unittest.main()

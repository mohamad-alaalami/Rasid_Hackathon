"""Mocked resource-accounting tests (no Strands, Bedrock, or GPU needed).

SDK-level checks against real Strands objects live in test_strands_integration.py.
"""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.experiments import finalize_training, reserve_training
from research.resource_accounting import (
    CampaignSession,
    ModelCallCounter,
    campaign_resource_report,
    estimate_run_cost,
    format_report,
    load_prices,
    load_sessions,
)

PRICES = {
    "bedrock_input_usd_per_million_tokens": 3.0,
    "bedrock_output_usd_per_million_tokens": 15.0,
    "gpu_instance_usd_per_hour": 1.0,
}


class FakeHooks:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, event_type, callback):
        self.callbacks.append(callback)


class FakeAgent:
    """Mimics the Strands 1.63 contract: cumulative metrics on the Agent and
    one AfterModelCallEvent per model request attempt."""

    def __init__(self, per_model_call=(100, 10, 250), model_calls=2, failed=0,
                 metrics=True, hooks=True, report_usage=True):
        self.per_model_call = per_model_call
        self.model_calls, self.failed, self.report_usage = model_calls, failed, report_usage
        self.hooks = FakeHooks() if hooks else None
        self.event_loop_metrics = types.SimpleNamespace(
            accumulated_usage={"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            accumulated_metrics={"latencyMs": 0},
        ) if metrics else None

    def _fire(self, exception=None):
        for callback in (self.hooks.callbacks if self.hooks else []):
            callback(types.SimpleNamespace(exception=exception))

    def __call__(self, prompt):
        for _ in range(self.failed):
            self._fire(exception=RuntimeError("ThrottlingException"))
        for _ in range(self.model_calls):
            if self.event_loop_metrics is not None and self.report_usage:
                usage = self.event_loop_metrics.accumulated_usage
                usage["inputTokens"] += self.per_model_call[0]
                usage["outputTokens"] += self.per_model_call[1]
                usage["totalTokens"] += sum(self.per_model_call[:2])
                self.event_loop_metrics.accumulated_metrics["latencyMs"] += self.per_model_call[2]
            self._fire()
        return "done"


def fake_attach(agent):
    if getattr(agent, "hooks", None) is None:
        return None
    counter = ModelCallCounter()
    agent.hooks.add_callback("AfterModelCallEvent", counter)
    return counter


def run(campaign, configuration, *, seconds=None, status="completed", score=0.8,
        version="E2", dataset="m-pv4ger", epochs=2):
    reservation = reserve_training(version, campaign, dataset, 42, configuration, max_experiments=5)
    result = {"status": status, "evaluation_split": "validation", "runtime_seconds": 99.0}
    if status == "completed":
        result.update(validation_score=score, completed_epochs=epochs, device="cuda")
    if seconds is not None:
        result["training_seconds"] = seconds
    return finalize_training(version, campaign, dataset, 42, reservation["experiment_id"], result)


def config(model="resnet18_rgb", lr=0.0003, epochs=2):
    return {"architecture": model, "learning_rate": lr, "epochs": epochs,
            "batch_size": 16, "image_size": None}


class ResourceAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.patches = [
            patch("agent.experiments.RESULTS_DIR", Path(self.temporary.name)),
            patch("research.resource_accounting.accounting.attach_model_call_counter", fake_attach),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def session(self, campaign, *agents_and_prompts):
        with CampaignSession(campaign, model_id="m") as session:
            for agent, prompt in agents_and_prompts:
                session.call(agent, prompt)
        return session

    def test_repeated_calls_on_one_cumulative_agent_are_not_double_counted(self):
        agent = FakeAgent()
        session = self.session("c1", (agent, "first"), (agent, "interrupt response"))
        self.assertEqual([call["input_tokens"] for call in session.calls], [200, 200])
        self.assertEqual([call["model_invocations"] for call in session.calls], [2, 2])
        llm = campaign_resource_report("c1")["llm"]
        self.assertEqual(llm["input_tokens"], 400)
        self.assertEqual(llm["output_tokens"], 40)
        self.assertEqual(llm["model_invocations"], 4)
        self.assertAlmostEqual(llm["llm_latency_seconds"], 1.0)

    def test_resumed_agent_session_adds_only_its_own_calls(self):
        self.session("c2", (FakeAgent(), "start"))
        self.session("c2", (FakeAgent(model_calls=3), "resume"))  # new process, fresh counters
        self.assertEqual(len(load_sessions("c2")), 2)
        llm = campaign_resource_report("c2")["llm"]
        self.assertEqual(llm["model_invocations"], 5)
        self.assertEqual(llm["input_tokens"], 500)

    def test_failed_model_attempts_are_counted_separately(self):
        self.session("c3", (FakeAgent(model_calls=1, failed=2), "x"))
        llm = campaign_resource_report("c3")["llm"]
        self.assertEqual(llm["model_invocations"], 1)
        self.assertEqual(llm["failed_model_calls"], 2)
        self.assertEqual(llm["input_tokens"], 100)

    def test_missing_metadata_is_unknown_not_zero(self):
        self.session("c4", (FakeAgent(metrics=False), "x"))
        llm = campaign_resource_report("c4", PRICES)["llm"]
        self.assertIsNone(llm["input_tokens"])
        self.assertIsNone(llm["llm_latency_seconds"])
        self.assertEqual(llm["model_invocations"], 2)

        self.session("c5", (FakeAgent(hooks=False), "x"))
        self.assertIsNone(campaign_resource_report("c5")["llm"]["model_invocations"])

    def test_zero_tokens_after_model_calls_is_unknown_but_no_call_is_zero(self):
        self.session("c6", (FakeAgent(report_usage=False), "x"))
        llm = campaign_resource_report("c6")["llm"]
        self.assertIsNone(llm["input_tokens"])
        self.assertIsNone(llm["llm_latency_seconds"])
        self.session("c7", (FakeAgent(model_calls=0), "rejected before any model call"))
        llm = campaign_resource_report("c7")["llm"]
        self.assertEqual(llm["input_tokens"], 0)
        self.assertEqual(llm["model_invocations"], 0)

    def test_one_unknown_call_makes_campaign_totals_unknown(self):
        self.session("c8", (FakeAgent(), "x"))
        self.session("c8", (FakeAgent(metrics=False), "y"))
        report = campaign_resource_report("c8", PRICES)
        self.assertIsNone(report["llm"]["input_tokens"])
        self.assertEqual(report["llm"]["sessions_with_unknown_usage"], 1)
        self.assertIsNone(report["cost_usd"]["bedrock"])
        self.assertIn("input_tokens unknown", report["cost_usd"]["notes"])

    def test_session_is_recorded_when_the_agent_raises(self):
        def failing(prompt):
            raise RuntimeError("throttled")
        with self.assertRaises(RuntimeError):
            self.session("c9", (failing, "x"))
        [entry] = load_sessions("c9")
        self.assertEqual(entry["outcome"], "error")
        self.assertIsNone(entry["llm"]["input_tokens"])

    def test_training_seconds_come_from_records_not_llm_time(self):
        run("c10", config(lr=0.001), seconds=120.0)
        run("c10", config(lr=0.002), status="failed")  # no trainer time: wrapper runtime
        reserve_training("E2", "c10", "m-pv4ger", 42, config(lr=0.003), max_experiments=5)
        self.session("c10", (FakeAgent(), "x"))
        report = campaign_resource_report("c10", PRICES)
        runs = report["training"]["experiments"]
        self.assertEqual(runs[0]["training_seconds_source"], "trainer.training_seconds")
        self.assertEqual(runs[1]["training_seconds_source"], "campaign.runtime_seconds")
        self.assertIsNone(runs[2]["training_seconds"])  # launched, never finalized
        self.assertIsNone(report["training"]["training_seconds_total"])
        self.assertEqual(report["training"]["training_seconds_known_subtotal"], 219.0)
        self.assertAlmostEqual(report["llm"]["llm_latency_seconds"], 0.5)
        self.assertIsNone(report["cost_usd"]["gpu_training"])

    def test_costs_use_only_configured_prices(self):
        run("c11", config(), seconds=3600.0)
        self.session("c11", (FakeAgent(per_model_call=(500_000, 50_000, 100)), "x"))
        priced = campaign_resource_report("c11", PRICES)["cost_usd"]
        self.assertAlmostEqual(priced["bedrock"], 3.0 + 1.5)
        self.assertAlmostEqual(priced["gpu_training"], 1.0)
        self.assertAlmostEqual(priced["total"], 5.5)
        unpriced = campaign_resource_report("c11", None)
        self.assertIsNone(unpriced["cost_usd"]["total"])
        self.assertIn("no price file configured", unpriced["cost_usd"]["notes"])
        self.assertIn("unknown", format_report(unpriced))

    def test_price_file_validation(self):
        path = Path(self.temporary.name) / "prices.json"
        path.write_text(json.dumps({"gpu_instance_usd_per_hour": None, "source": "x"}))
        self.assertIsNone(load_prices(path)["gpu_instance_usd_per_hour"])
        path.write_text(json.dumps({"gpu_usd": 1.0}))
        with self.assertRaisesRegex(ValueError, "Unknown price fields"):
            load_prices(path)
        path.write_text(json.dumps({"gpu_instance_usd_per_hour": -1}))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            load_prices(path)
        example = Path(__file__).resolve().parents[1] / "configs" / "prices.example.json"
        self.assertTrue(all(load_prices(example)[k] is None for k in PRICES))

    def test_run_cost_estimate_is_conservative_and_measured_only(self):
        unknown = estimate_run_cost("m-pv4ger", config(epochs=4), PRICES)
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(unknown["estimated_gpu_usd"])
        run("c12", config(epochs=2), seconds=3600.0, epochs=2)        # 1800 s/epoch
        run("c12", config(lr=0.01, epochs=2), seconds=1800.0, epochs=2)  # 900 s/epoch
        known = estimate_run_cost("m-pv4ger", config(lr=0.02, epochs=4), PRICES)
        self.assertEqual(known["status"], "estimated")
        self.assertEqual(known["measured_runs"], 2)
        self.assertEqual(known["estimated_training_seconds"], 7200.0)  # slowest, not mean
        self.assertEqual(known["estimated_gpu_usd"], 2.0)
        no_price = estimate_run_cost("m-pv4ger", config(lr=0.02, epochs=4), None)
        self.assertEqual(no_price["status"], "unknown")
        self.assertEqual(no_price["estimated_training_seconds"], 7200.0)
        other = estimate_run_cost("m-pv4ger", config(model="convnext_tiny_rgb"), PRICES)
        self.assertEqual(other["status"], "unknown")


if __name__ == "__main__":
    unittest.main()

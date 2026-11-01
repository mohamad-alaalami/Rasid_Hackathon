"""SDK-level tests: the real Strands Agent, event loop, FileSessionManager,
interrupts, hooks, and metrics, with a scripted model instead of Bedrock and a
fake trainer instead of the GPU. Skipped when strands-agents is not installed.
No network, Bedrock, or GPU calls are made.
"""

import builtins
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import strands_stub
except ImportError:
    from tests import strands_stub

REAL_STRANDS = strands_stub.real_strands_available()
if REAL_STRANDS:
    from strands import Agent, ModelRetryStrategy
    from strands.models import Model
    from strands.types.exceptions import ModelThrottledException

    from agent import hitl, main as campaign_main, tools
    from research.resource_accounting import CampaignSession, campaign_resource_report

    class ScriptedModel(Model):
        """Streams Bedrock-format events from a shared script; records requests."""

        def __init__(self, script, seen):
            self.script, self.seen = script, seen

        def update_config(self, **config):
            pass

        def get_config(self):
            return {}

        async def structured_output(self, *args, **kwargs):
            raise NotImplementedError

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            if not self.script:
                raise AssertionError("Unexpected model call")
            kind, payload = self.script.pop(0)
            self.seen.append(json.dumps(messages, default=str))
            if kind == "throttle":
                raise ModelThrottledException("ThrottlingException")
            yield {"messageStart": {"role": "assistant"}}
            if kind == "tool":
                name, arguments = payload
                yield {"contentBlockStart": {"start": {"toolUse": {
                    "toolUseId": f"tu-{len(self.seen)}", "name": name}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(arguments)}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                yield {"contentBlockDelta": {"delta": {"text": payload}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}
            yield {"metadata": {"usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
                                "metrics": {"latencyMs": 200}}}


OPTION_A = {"model_id": "resnet18_rgb", "epochs": 2, "batch_size": 16, "learning_rate": 0.001}
CONSULT = {
    "trigger": "before_experiment",
    "situation": "No run yet; evidence is insufficient to prefer a learning rate.",
    "options": [
        {"id": "A", "title": "Baseline lr 1e-3", "tradeoffs": "Fast; may be unstable.",
         "configuration": OPTION_A},
        {"id": "B", "title": "Stop", "tradeoffs": "Saves budget."},
    ],
    "recommended_option": "A",
    "rationale": "Cheapest informative first run.",
}


def run_args(lr=0.0003):
    return {"dataset_id": "m-pv4ger", "model_id": "resnet18_rgb", "epochs": 2,
            "batch_size": 16, "learning_rate": lr}


class FakeTrainer:
    def __init__(self, directory):
        self.directory, self.calls = Path(directory), []

    def __call__(self, *, dataset_id, config):
        self.calls.append(config)
        metrics = self.directory / f"metrics_{len(self.calls)}.json"
        metrics.write_text(json.dumps([{"train_loss": 0.5, "val_loss": 0.6,
                                        "val_accuracy": 0.8, "learning_rate": 0.001}]))
        return {"status": "completed", "experiment_id": f"ml-{len(self.calls)}",
                "best_val_score": 0.8, "primary_metric": "accuracy",
                "metrics_path": str(metrics), "checkpoint_path": "best.pt",
                "native_resolution": [320, 320], "effective_resolution": [320, 320],
                "training_seconds": 10.0, "device": "cuda", "completed_epochs": 2}


@unittest.skipUnless(REAL_STRANDS, "strands-agents is not installed")
class StrandsCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trainer = FakeTrainer(self.root)
        self.script, self.seen = [], []
        self.agent_kwargs = []
        real_factory = tools.make_registered_training_tool

        def factory(**kwargs):
            return real_factory(training_fn=self.trainer, **kwargs)

        def agent_class(**kwargs):
            self.agent_kwargs.append(sorted(kwargs))
            return Agent(callback_handler=None, **kwargs)

        self.patches = [
            patch("agent.experiments.RESULTS_DIR", self.root / "campaigns"),
            patch.object(tools, "model_runtime_status", lambda model_id: {"eligible": True, "reason": None}),
            patch.dict(os.environ, {"GOTRAIN_STORE": ""}),
            patch.object(campaign_main, "_bedrock_model",
                         lambda *a: ScriptedModel(self.script, self.seen)),
            patch.object(campaign_main, "make_registered_training_tool", factory),
            patch.object(campaign_main, "Agent", agent_class),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def run_main(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            campaign_main.run_campaign(campaign_main.build_parser().parse_args(argv))
        return out.getvalue()

    def report(self, output):
        return json.loads(output[output.index("{", output.index("FINAL PERSISTED")):])

    def test_off_mode_uses_an_unchanged_agent(self):
        self.script += [("tool", ("run_campaign_experiment", run_args())), ("text", "done")]
        report = self.report(self.run_main(["--campaign-id", "off1", "--dataset", "m-pv4ger"]))
        self.assertEqual(self.agent_kwargs, [["model", "system_prompt", "tools"]])
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertFalse((self.root / "campaigns" / "hitl").exists())
        self.assertEqual(report["resources"]["llm"]["model_invocations"], 2)
        self.assertEqual(report["resources"]["llm"]["input_tokens"], 200)

    def test_pause_answer_resume_exactly_once_with_restored_conversation(self):
        self.script += [
            ("tool", ("run_campaign_experiment", run_args())),       # gate: approval required
            ("tool", ("request_human_consultation", CONSULT)),        # Strands interrupt
        ]
        output = self.run_main(["--campaign-id", "h1", "--dataset", "m-pv4ger", "--feedback", "E1",
                                "--hitl", "always", "--max-experiments", "3"])
        self.assertIn("CAMPAIGN PAUSED", output)
        self.assertEqual(self.script, [])
        self.assertEqual(self.trainer.calls, [])
        request = hitl.pending_request(hitl.load_state("h1"))
        self.assertTrue(request["interrupt_id"].startswith("v1:tool_call:"))

        with self.assertRaisesRegex(SystemExit, "still awaits"):
            self.run_main(["--resume", "--campaign-id", "h1"])
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):  # stale/unknown request id
                hitl.main(["respond", "--campaign-id", "h1", "--request-id", "2", "--choice", "A"])
            hitl.main(["respond", "--campaign-id", "h1", "--request-id", "1", "--choice", "A"])

        self.script += [
            ("tool", ("run_campaign_experiment", run_args(lr=0.001))),
            ("text", "Campaign finished."),
        ]
        report = self.report(self.run_main(["--resume", "--campaign-id", "h1"]))
        restored_request = self.seen[2]
        self.assertIn("Run a real autonomous GoTrain campaign", restored_request)
        self.assertIn('\\"human_choice\\": \\"A\\"', restored_request)
        self.assertEqual(report["feedback_version"], "E1")
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertEqual(self.trainer.calls[0]["learning_rate"], 0.001)
        state = hitl.load_state("h1")
        self.assertEqual(state["requests"][0]["status"], "delivered")
        self.assertEqual(state["requests"][0]["approval"]["consumed_by_experiment"], 1)
        llm = report["resources"]["llm"]
        self.assertEqual(llm["sessions"], 2)
        self.assertEqual(llm["model_invocations"], 4)
        self.assertEqual(llm["input_tokens"], 400)
        self.assertAlmostEqual(llm["llm_latency_seconds"], 0.8)

        with self.assertRaisesRegex(SystemExit, "Nothing to resume"):
            self.run_main(["--resume", "--campaign-id", "h1"])
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertEqual(len(self.seen), 4)  # no further model call

    def test_failed_restoration_reports_error_and_keeps_state(self):
        self.script += [("tool", ("request_human_consultation", CONSULT))]
        self.run_main(["--campaign-id", "h2", "--dataset", "m-pv4ger", "--hitl", "always"])
        with redirect_stdout(io.StringIO()):
            hitl.main(["respond", "--campaign-id", "h2", "--request-id", "1", "--choice", "A"])
        session_dir = hitl.session_storage_dir() / "session_h2"
        shutil.rmtree(session_dir / "agents")  # conversation lost or corrupted
        with self.assertRaisesRegex(SystemExit, "not delivered, so no training was launched"):
            self.run_main(["--resume", "--campaign-id", "h2"])
        self.assertEqual(hitl.awaiting_delivery(hitl.load_state("h2"))["id"], 1)
        shutil.rmtree(session_dir)
        with self.assertRaisesRegex(SystemExit, "No saved Strands session"):
            self.run_main(["--resume", "--campaign-id", "h2"])
        self.assertEqual(self.trainer.calls, [])
        self.assertEqual(len(self.seen), 1)

    def test_interactive_mode_answers_between_agent_calls(self):
        self.script += [
            ("tool", ("request_human_consultation", CONSULT)),
            ("tool", ("run_campaign_experiment", run_args(lr=0.001))),
            ("text", "done"),
        ]
        answers = iter(["A", ""])
        with patch.object(builtins, "input", lambda prompt="": next(answers)):
            report = self.report(self.run_main([
                "--campaign-id", "h3", "--dataset", "m-pv4ger", "--hitl", "always",
                "--hitl-interactive",
            ]))
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertEqual(hitl.load_state("h3")["requests"][0]["response"]["via"], "terminal")
        self.assertEqual(report["resources"]["llm"]["sessions"], 1)
        self.assertEqual(report["resources"]["llm"]["model_invocations"], 3)


@unittest.skipUnless(REAL_STRANDS, "strands-agents is not installed")
class StrandsAccountingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.patch = patch("agent.experiments.RESULTS_DIR", Path(self.temporary.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def test_retries_multiple_calls_and_cumulative_metrics(self):
        script, seen = [("throttle", None), ("text", "one"), ("text", "two")], []
        agent = Agent(model=ScriptedModel(script, seen), callback_handler=None,
                      retry_strategy=ModelRetryStrategy(max_attempts=3, initial_delay=0, max_delay=0))
        with CampaignSession("acct", model_id="m") as session:
            session.call(agent, "first")   # one throttled attempt + one response
            session.call(agent, "second")  # cumulative counters on the same Agent
        self.assertEqual([call["model_invocations"] for call in session.calls], [1, 1])
        self.assertEqual([call["failed_model_calls"] for call in session.calls], [1, 0])
        self.assertEqual([call["input_tokens"] for call in session.calls], [100, 100])
        # Strands' own cumulative counter covers both calls; deltas do not re-add it.
        self.assertEqual(agent.event_loop_metrics.accumulated_usage["inputTokens"], 200)
        llm = campaign_resource_report("acct")["llm"]
        self.assertEqual(llm["input_tokens"], 200)
        self.assertEqual(llm["failed_model_calls"], 1)
        self.assertAlmostEqual(llm["llm_latency_seconds"], 0.4)


if __name__ == "__main__":
    unittest.main()

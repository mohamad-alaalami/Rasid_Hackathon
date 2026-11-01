"""Mocked HITL tests: gate, consultation lifecycle, answers, resume guards.

A FakeToolContext follows the documented Strands contract for
``ToolContext.interrupt``: raise InterruptException until a response exists,
then return it. Real-SDK flows are in test_strands_integration.py.
"""

import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import strands_stub
except ImportError:  # run as tests.<module> from the repo root
    from tests import strands_stub

strands_stub.install()  # no-op when strands-agents is installed

from strands.interrupt import Interrupt, InterruptException

from agent import hitl, main as campaign_main, tools
from agent.experiments import get_budget_status
from agent.policies import build_system_prompt

OPTION_A = {"model_id": "resnet18_rgb", "epochs": 2, "batch_size": 16, "learning_rate": 0.001}


class FakeTrainer:
    """Stands in for ml.train_model; counts every launch. lr >= 0.5 fails."""

    def __init__(self, directory, seconds=10.0):
        self.directory = Path(directory)
        self.seconds = seconds
        self.calls = []

    def __call__(self, *, dataset_id, config, **solution_context):
        self.calls.append(config)
        if config["learning_rate"] >= 0.5:
            return {"status": "failed", "message": "CUDA out of memory", "training_seconds": 1.0}
        metrics = self.directory / f"metrics_{len(self.calls)}.json"
        metrics.write_text(json.dumps([
            {"train_loss": 0.5, "val_loss": 0.6, "val_accuracy": 0.8, "learning_rate": 0.001},
        ]), encoding="utf-8")
        return {
            "status": "completed", "experiment_id": f"ml-{len(self.calls)}",
            "best_val_score": 0.8, "primary_metric": "accuracy", "metrics_path": str(metrics),
            "checkpoint_path": "best.pt", "native_resolution": [320, 320],
            "effective_resolution": [320, 320], "training_seconds": self.seconds,
            "device": "cuda", "completed_epochs": config["epochs"],
        }


class FakeToolContext:
    def __init__(self, tool_use_id):
        self.tool_use = {"toolUseId": tool_use_id}
        self.response = None

    def interrupt(self, name, reason=None, response=None):
        if self.response is not None:
            return self.response
        raise InterruptException(Interrupt(f"fake:{self.tool_use['toolUseId']}", name, reason))


def run_args(model="resnet18_rgb", lr=0.0003, epochs=2):
    return {"dataset_id": "m-pv4ger", "model_id": model, "epochs": epochs,
            "batch_size": 16, "learning_rate": lr}


def consultation(trigger="before_experiment", configuration=OPTION_A, recommended="A"):
    return {
        "trigger": trigger,
        "situation": "One run completed at accuracy 0.80; evidence about lr is thin.",
        "options": [
            {"id": "A", "title": "Raise lr", "tradeoffs": "Faster, may be unstable.",
             "configuration": configuration},
            {"id": "B", "title": "End the campaign", "tradeoffs": "Saves budget."},
        ],
        "recommended_option": recommended,
        "rationale": "Cheap test of the main open question.",
    }


class HitlTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.patches = [
            patch("agent.experiments.RESULTS_DIR", root / "campaigns"),
            patch.object(tools, "model_runtime_status", lambda model_id: {"eligible": True, "reason": None}),
            patch.dict(os.environ, {"GOTRAIN_STORE": ""}),
        ]
        for item in self.patches:
            item.start()
        self.trainer = FakeTrainer(root)
        self.tool_uses = 0

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def settings(self, mode, **overrides):
        return {
            "feedback": "E2", "policy": "P0", "seed": 42, "max_experiments": 5,
            "dataset": "m-pv4ger", "model_id": "m", "region": "r", "hitl_mode": mode,
            "failure_threshold": 2, "cost_threshold_usd": None, "unknown_cost_policy": "ask",
            "prices": None, **overrides,
        }

    def controller(self, mode, campaign="c", **overrides):
        if hitl.existing_state(campaign) is None:
            hitl.create_state(campaign, self.settings(mode, **overrides))
        return hitl.HitlController(campaign, hitl.load_state(campaign)["settings"])

    def tool(self, gate, campaign="c", policy="P0", max_experiments=5):
        return tools.make_registered_training_tool(
            version="E2", campaign_id=campaign, seed=42, max_experiments=max_experiments,
            required_dataset="m-pv4ger", training_fn=self.trainer,
            selection_policy=policy, launch_gate=gate,
        )

    def pause(self, gate, **request):
        """Call the consultation tool once; it must raise the Strands interrupt."""
        self.tool_uses += 1
        context = FakeToolContext(f"tooluse-{self.tool_uses}")
        consult = gate.make_tool()
        with self.assertRaises(InterruptException):
            consult(**(request or consultation()), tool_context=context)
        return consult, context

    def decide(self, gate, choice, note="", **request):
        """Pause, record the human's answer, then re-execute as Strands does on resume."""
        consult, context = self.pause(gate, **request)
        pending = hitl.pending_request(hitl.load_state(gate.campaign_id))
        hitl.answer_request(gate.campaign_id, pending["id"], choice, note)
        context.response = {"request_id": pending["id"]}
        return consult(**(request or consultation()), tool_context=context)


class OffModeTests(HitlTestCase):
    def test_off_mode_has_no_gate_no_state_and_unchanged_agent(self):
        args = campaign_main.build_parser().parse_args(["--dataset", "m-pv4ger"])
        self.assertEqual(args.hitl, "off")
        self.assertIsNone(campaign_main.setup_hitl(args, "plain", None))
        self.assertIsNone(hitl.existing_state("plain"))
        captured = {}
        with patch.object(campaign_main, "Agent", lambda **kw: captured.update(kw)), \
                patch.object(campaign_main, "_bedrock_model", lambda *a: None):
            campaign_main.create_campaign_agent(args, "plain")
        self.assertEqual(set(captured), {"model", "system_prompt", "tools"})
        self.assertEqual(captured["system_prompt"], build_system_prompt("P0"))
        names = [getattr(t, "tool_name", None) or t.__name__ for t in captured["tools"]]
        self.assertNotIn("request_human_consultation", names)
        self.assertEqual(self.tool(None, "plain")(**run_args())["status"], "completed")
        self.assertFalse((Path(self.temporary.name) / "campaigns" / "hitl").exists())

    def test_unusual_campaign_ids_still_work_with_hitl_off(self):
        self.assertIsNone(hitl.existing_state("weird/id"))


class AlwaysModeTests(HitlTestCase):
    def test_no_gpu_training_or_budget_use_before_approval(self):
        gate = self.controller("always")
        train = self.tool(gate)
        blocked = train(**run_args())
        self.assertEqual(blocked["error_code"], "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(blocked["triggers"], ["before_experiment"])
        self.pause(gate)
        request = hitl.pending_request(hitl.load_state("c"))
        self.assertEqual(request["interrupt_id"], "fake:tooluse-1")
        self.assertEqual(train(**run_args(lr=0.001))["error_code"], "HUMAN_DECISION_PENDING")
        self.assertEqual(self.trainer.calls, [])
        self.assertEqual(get_budget_status("E2", "c", "m-pv4ger", 42)["launched_experiments"], 0)

    def test_selected_configuration_is_enforced_exactly_once(self):
        gate = self.controller("always")
        answer = self.decide(gate, "A", "go")
        self.assertEqual(answer["human_choice"], "A")
        self.assertEqual(answer["human_note"], "go")
        self.assertEqual(answer["selected_option"]["configuration"], OPTION_A)
        train = self.tool(gate)
        mismatch = train(**run_args(lr=0.002))
        self.assertEqual(mismatch["error_code"], "HUMAN_SELECTION_MISMATCH")
        self.assertEqual(mismatch["approved_configuration"], OPTION_A)
        self.assertEqual(train(**run_args(lr=0.001))["status"], "completed")
        self.assertEqual(train(**run_args(lr=0.003))["error_code"], "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(len(self.trainer.calls), 1)
        request = hitl.load_state("c")["requests"][0]
        self.assertEqual(request["status"], "delivered")
        self.assertEqual(request["approval"]["consumed_by_experiment"], 1)

    def test_delegation_stop_and_unconfigured_options(self):
        gate = self.controller("always")
        self.assertIn("delegated", self.decide(gate, "delegate")["instruction"])
        train = self.tool(gate)
        self.assertEqual(train(**run_args(lr=0.004))["status"], "completed")
        self.decide(gate, "B")  # option without configuration authorizes no run
        self.assertEqual(train(**run_args(lr=0.005))["error_code"], "HUMAN_APPROVAL_REQUIRED")
        self.assertIn("get_campaign_report", self.decide(gate, "stop")["instruction"])
        self.assertEqual(train(**run_args(lr=0.006))["error_code"], "CAMPAIGN_STOPPED_BY_HUMAN")
        self.assertEqual(len(self.trainer.calls), 1)

    def test_reexecution_reuses_the_request(self):
        gate = self.controller("always")
        consult, context = self.pause(gate)
        with self.assertRaises(InterruptException):  # Strands re-runs the same tool use
            consult(**consultation(), tool_context=context)
        self.assertEqual(len(hitl.load_state("c")["requests"]), 1)
        second = FakeToolContext("another-tool-use")
        busy = consult(**consultation(), tool_context=second)
        self.assertEqual(busy["error_code"], "HUMAN_DECISION_PENDING")

    def test_invalid_consultation_is_rejected_and_not_persisted(self):
        gate = self.controller("always")
        bad = consultation(recommended="Z")
        bad["options"][0]["configuration"] = {"optimizer": "sgd"}
        result = gate.make_tool()(**bad, tool_context=FakeToolContext("t"))
        self.assertEqual(result["error_code"], "INVALID_CONSULTATION")
        self.assertIn("recommended_option", result["field_errors"])
        self.assertIn("options[0].configuration", result["field_errors"])
        self.assertEqual(hitl.load_state("c")["requests"], [])

    def test_p1_decision_record_still_required_under_hitl(self):
        gate = self.controller("always")
        self.decide(gate, "A")
        train = self.tool(gate, policy="P1")
        self.assertEqual(train(decision_record={}, **run_args(lr=0.001))["error_code"],
                         "INVALID_DECISION_RECORD")
        record = {key: "x" for key in ("observation", "action", "expected_signal", "cost_justification")}
        self.assertEqual(train(decision_record=record, **run_args(lr=0.001))["status"], "completed")


class ResponseValidationTests(HitlTestCase):
    def test_invalid_duplicate_and_stale_responses_are_rejected(self):
        gate = self.controller("always")
        with self.assertRaisesRegex(hitl.HitlError, "does not exist"):
            hitl.answer_request("c", 1, "A")
        self.pause(gate)
        with self.assertRaisesRegex(hitl.HitlError, "choice must be one of"):
            hitl.answer_request("c", 1, "Z")
        hitl.answer_request("c", 1, "delegate")
        with self.assertRaisesRegex(hitl.HitlError, "duplicate or stale"):
            hitl.answer_request("c", 1, "A")  # duplicate
        # Deliver #1, then #2 is pending: answering #1 again is stale.
        context = FakeToolContext("tooluse-1")
        context.response = {"request_id": 1}
        gate.make_tool()(**consultation(), tool_context=context)
        self.pause(gate)
        with self.assertRaisesRegex(hitl.HitlError, "already delivered"):
            hitl.answer_request("c", 1, "B")
        self.assertEqual(hitl.pending_request(hitl.load_state("c"))["id"], 2)

    def test_response_before_pause_and_mismatched_payload(self):
        gate = self.controller("always")
        gate.open_request("t-1", **consultation())  # created, interrupt not yet raised
        with self.assertRaisesRegex(hitl.HitlError, "not paused"):
            hitl.answer_request("c", 1, "A")
        gate.record_interrupt(1, "i-1")
        hitl.answer_request("c", 1, "A")
        self.assertEqual(gate.deliver(1, {"request_id": 99})["error_code"], "STALE_HUMAN_RESPONSE")
        self.assertEqual(hitl.load_state("c")["requests"][0]["status"], "answered")

    def test_cli_requires_request_id_and_rejects_duplicates(self):
        gate = self.controller("always")
        self.pause(gate)
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            hitl.main(["respond", "--campaign-id", "c", "--choice", "A"])  # missing id
        with redirect_stdout(out):
            hitl.main(["show", "--campaign-id", "c"])
            hitl.main(["respond", "--campaign-id", "c", "--request-id", "1", "--choice", "A"])
        self.assertIn("--request-id 1", out.getvalue())
        with self.assertRaises(SystemExit):
            hitl.main(["respond", "--campaign-id", "c", "--request-id", "1", "--choice", "B"])


class CriticalModeTests(HitlTestCase):
    def test_autonomous_until_repeated_failures(self):
        gate = self.controller("critical")
        train = self.tool(gate)
        self.assertEqual(train(**run_args())["status"], "completed")
        self.assertEqual(train(**run_args(lr=0.5))["status"], "failed")
        self.assertEqual(train(**run_args(lr=0.6))["status"], "failed")
        blocked = train(**run_args(lr=0.7))
        self.assertEqual(blocked["triggers"], ["repeated_failures"])
        self.assertEqual(blocked["trigger_details"]["repeated_failures"],
                         {"consecutive_failed_runs": 2, "threshold": 2})
        self.assertEqual(len(self.trainer.calls), 3)
        self.decide(gate, "delegate", **_rest("repeated_failures"))
        self.assertEqual(train(**run_args(lr=0.002))["status"], "completed")
        self.assertEqual(train(**run_args(lr=0.003))["status"], "completed")  # streak broken

    def test_unknown_cost_asks_by_default_with_structured_reason(self):
        gate = self.controller("critical", cost_threshold_usd=1.0,
                               prices={"gpu_instance_usd_per_hour": 1.0})
        blocked = self.tool(gate)(**run_args(model="convnext_tiny_rgb"))
        self.assertEqual(blocked["error_code"], "HUMAN_APPROVAL_REQUIRED")
        detail = blocked["trigger_details"]["cost_threshold"]
        self.assertEqual(detail["reason"], "estimate_unknown")
        self.assertEqual(detail["threshold_usd"], 1.0)
        self.assertEqual(detail["estimate"]["status"], "unknown")
        self.assertIsNone(detail["estimate"]["estimated_gpu_usd"])
        self.assertEqual(self.trainer.calls, [])

    def test_unknown_cost_allow_policy_and_exceeding_estimate(self):
        self.trainer.seconds = 3600.0  # 2 epochs -> 1800 s/epoch measured
        gate = self.controller("critical", cost_threshold_usd=1.0, unknown_cost_policy="allow",
                               prices={"gpu_instance_usd_per_hour": 1.0})
        train = self.tool(gate)
        self.assertEqual(train(**run_args())["status"], "completed")  # unknown, allowed
        self.assertEqual(train(**run_args(lr=0.002, epochs=1))["status"], "completed")  # ~$0.50
        blocked = train(**run_args(lr=0.003, epochs=4))  # slowest 3600 s/epoch x 4 = $4
        detail = blocked["trigger_details"]["cost_threshold"]
        self.assertEqual(detail["reason"], "estimate_exceeds_threshold")
        self.assertEqual(detail["estimate"]["estimated_gpu_usd"], 4.0)
        self.assertEqual(detail["estimate"]["measured_runs"], 2)

    def test_budget_rules_apply_independently_of_cost_consultation(self):
        gate = self.controller("critical", cost_threshold_usd=1.0, max_experiments=1,
                               prices={"gpu_instance_usd_per_hour": 1.0})
        train = self.tool(gate, max_experiments=1)
        self.decide(gate, "delegate", **_rest("cost_threshold"))
        self.assertEqual(train(**run_args())["status"], "completed")
        exhausted = train(**run_args(model="convnext_tiny_rgb"))  # unknown cost, no budget
        self.assertEqual(exhausted["error_code"], "BUDGET_EXHAUSTED")
        self.assertEqual(len(self.trainer.calls), 1)

    def test_code_change_and_voluntary_consultations_are_recorded(self):
        gate = self.controller("critical")
        self.assertEqual(self.decide(gate, "B", **_rest("code_change", None))["human_choice"], "B")
        self.decide(gate, "delegate", **_rest("insufficient_evidence"))
        triggers = [r["trigger"] for r in hitl.load_state("c")["requests"]]
        self.assertEqual(triggers, ["code_change", "insufficient_evidence"])
        self.assertIn("code_change", gate.instructions())


def _rest(trigger, configuration=OPTION_A):
    return consultation(trigger=trigger, configuration=configuration)


class Result:
    def __init__(self, stop_reason="end_turn", interrupts=()):
        self.stop_reason, self.interrupts = stop_reason, list(interrupts)


class FakeSession:
    def __init__(self, agent):
        self.agent = agent

    def call(self, agent, prompt):
        return agent(prompt)


class RunAndResumeTests(HitlTestCase):
    def test_deferred_run_pauses_and_interactive_run_continues(self):
        gate = self.controller("always")

        def agent(prompt):
            if isinstance(prompt, str):
                self.pause(gate)
                return Result("interrupt", [Interrupt("fake:tooluse-1", hitl.INTERRUPT_NAME)])
            agent.resumed_with = prompt
            return Result()
        self.assertEqual(gate.run(FakeSession(agent), agent, "go"), "paused")
        self.assertEqual(hitl.pending_request(hitl.load_state("c"))["id"], 1)

        gate2 = self.controller("always", campaign="c2")
        answers = iter(["A", "note"])

        def agent2(prompt):
            if isinstance(prompt, str):
                self.pause(gate2)
                return Result("interrupt", [Interrupt(f"fake:tooluse-{self.tool_uses}", "x")])
            agent2.resumed_with = prompt
            return Result()
        outcome = gate2.run(FakeSession(agent2), agent2, "go", interactive=True,
                            input_fn=lambda _: next(answers), output=io.StringIO())
        self.assertEqual(outcome, "finished")
        payload = agent2.resumed_with[0]["interruptResponse"]
        self.assertEqual(payload["response"], {"request_id": 1})
        self.assertEqual(hitl.load_state("c2")["requests"][0]["response"]["via"], "terminal")

    def test_resume_target_guards(self):
        gate = self.controller("always")
        with self.assertRaisesRegex(hitl.HitlError, "Nothing to resume"):
            gate.resume_target()
        self.pause(gate)
        with self.assertRaisesRegex(hitl.HitlError, "still awaits the human"):
            gate.resume_target()
        hitl.answer_request("c", 1, "A")
        with self.assertRaisesRegex(hitl.HitlError, "No saved Strands session"):
            gate.resume_target()
        (hitl.session_storage_dir() / "session_c").mkdir(parents=True)
        self.assertEqual(gate.resume_target()["id"], 1)

    def test_rejected_resume_leaves_decision_undelivered(self):
        gate = self.controller("always")
        self.pause(gate)
        request = hitl.answer_request("c", 1, "A")

        def agent(prompt):
            raise ValueError("Received interrupt responses but agent is not in interrupt state")
        with self.assertRaisesRegex(hitl.HitlError, "Strands rejected the resume"):
            gate.run(FakeSession(agent), agent, gate.interrupt_response(request), resuming=request)
        self.assertEqual(hitl.awaiting_delivery(hitl.load_state("c"))["id"], 1)
        self.assertEqual(self.trainer.calls, [])

    def test_verify_restored_requires_conversation_and_interrupt(self):
        request = {"interrupt_id": "i-1"}
        with self.assertRaisesRegex(hitl.HitlError, "no conversation"):
            hitl.HitlController.verify_restored(types.SimpleNamespace(messages=[]), request)
        lost = types.SimpleNamespace(
            messages=[{"role": "user"}],
            _interrupt_state=types.SimpleNamespace(activated=False, interrupts={}),
        )
        with self.assertRaisesRegex(hitl.HitlError, "interrupt"):
            hitl.HitlController.verify_restored(lost, request)

    def test_terminal_prompt_parsing(self):
        inputs = iter(["?", "d", "note"])
        out = io.StringIO()
        self.assertEqual(
            hitl.prompt_terminal("text", ["A"], input_fn=lambda _: next(inputs), output=out),
            ("delegate", "note"),
        )
        self.assertIsNone(hitl.prompt_terminal("t", ["A"], input_fn=lambda _: "p", output=out))

        def eof(_):
            raise EOFError
        self.assertIsNone(hitl.prompt_terminal("t", ["A"], input_fn=eof, output=out))


if __name__ == "__main__":
    unittest.main()

"""Phase 1C: solution search graph.

Runs the real search tools, the guarded campaign tool, and ml.train_model on
CPU (tiny stand-in models, fake GEO-Bench task). No GPU, Bedrock, network, or
test split. The SDK-level class drives agent.main --search with the real
Strands agent loop and a scripted model; it is not a Bedrock campaign.
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

from agent import main as campaign_main  # noqa: E402
from agent import tools  # noqa: E402
from agent.autonomy import AutonomyPolicy  # noqa: E402
from agent.experiments import get_budget_status  # noqa: E402
from agent.search_graph import SearchGraph, comparability_key  # noqa: E402
from agent.search_tools import make_search_tools  # noqa: E402
from ml.solution import file_sha256  # noqa: E402
from ml.trainer import train_model  # noqa: E402

DATASET = "m-eurosat"


def name_of(tool_object):
    return getattr(tool_object, "tool_name", None) or tool_object.__name__


class SearchCase(unittest.TestCase):
    max_experiments = 7

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
        self.launches = []

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def setup_search(self, campaign="search", version="E2", scope="S1", launch_gate=None,
                     policy="P0"):
        self.graph = SearchGraph(campaign, version, 42, self.max_experiments, DATASET)
        autonomy = AutonomyPolicy(scope=scope)
        self.search = {name_of(t): t for t in make_search_tools(graph=self.graph,
                                                                autonomy=autonomy,
                                                                required_dataset=DATASET)}

        def trainer(dataset_id, config, **context):
            self.launches.append(config)
            if config["learning_rate"] >= 0.5:  # simulated divergence: a failed real run
                return {"status": "failed", "message": "loss became NaN at epoch 1",
                        "training_seconds": 0.4}
            return train_model(dataset_id, {**config, "output_dir": str(self.root / "runs")},
                               **context)
        self.run_tool = tools.make_registered_training_tool(
            version=version, campaign_id=campaign, seed=42, max_experiments=self.max_experiments,
            required_dataset=DATASET, training_fn=trainer, autonomy=autonomy,
            search_graph=self.graph, launch_gate=launch_gate, selection_policy=policy)

    @staticmethod
    def args(lr=0.01, epochs=1, pipeline=None):
        values = {"dataset_id": DATASET, "model_id": "resnet18_rgb", "epochs": epochs,
                  "batch_size": 2, "learning_rate": lr}
        if pipeline is not None:
            values["pipeline"] = pipeline
        return values

    def propose(self, intent, parent=None, claimed=None, **run_args):
        return self.search["propose_branch"](
            intent=intent, observation=f"observed before {intent}",
            hypothesis=f"{intent} should help", expected_signal="higher validation accuracy",
            parent_solution_id=parent, claimed_changes=claimed, **self.args(**run_args))

    def launch(self, **run_args):
        return self.run_tool(**self.args(**run_args))


class EndToEndSearchTests(SearchCase):
    def test_branch_refine_fail_and_return_campaign(self):
        self.setup_search()
        # 1. Baseline root.
        root = self.propose("root")
        self.assertEqual(root["status"], "proposed", root)
        self.assertEqual(self.launch()["status"], "completed")
        # 2-3. Two alternatives from the baseline.
        sgd = self.propose("explore", root["solution_id"], ["pipeline.optimizer"],
                           pipeline={"optimizer": "sgd"})
        fast = self.propose("explore", root["solution_id"], ["training.learning_rate"], lr=0.05)
        self.assertEqual(self.launch(pipeline={"optimizer": "sgd"})["status"], "completed")
        self.assertEqual(self.launch(lr=0.05)["status"], "completed")
        # 4. Refine one branch.
        refined = self.propose("refine", sgd["solution_id"], ["pipeline.scheduler"],
                               pipeline={"optimizer": "sgd", "scheduler": "constant"})
        self.assertEqual(self.launch(pipeline={"optimizer": "sgd", "scheduler": "constant"})
                         ["status"], "completed")
        # 5. A failed branch stays visible but is never a solution.
        diverged = self.propose("exploit", fast["solution_id"], ["training.learning_rate"], lr=0.6)
        self.assertEqual(self.launch(lr=0.6)["status"], "failed")
        # 6. Return to the earlier promising branch (not the latest candidate).
        frontier = self.search["search_frontier"]()
        self.assertIn(diverged["solution_id"], frontier["failed"])
        back = self.propose("revisit", sgd["solution_id"], ["training.epochs"],
                            pipeline={"optimizer": "sgd"}, epochs=2)
        self.assertEqual(back["parents"], [sgd["solution_id"]])
        self.assertEqual(self.launch(pipeline={"optimizer": "sgd"}, epochs=2)["status"],
                         "completed")

        state = self.graph.state()
        nodes = state["nodes"]
        self.assertEqual(len(nodes), 6)
        self.assertEqual(nodes[diverged["solution_id"]]["status"], "failed")
        self.assertIsNone(nodes[diverged["solution_id"]]["validation_score"])
        self.assertEqual(sorted(nodes[root["solution_id"]]["children"]),
                         sorted([sgd["solution_id"], fast["solution_id"]]))
        self.assertEqual(nodes[back["solution_id"]]["parents"], [sgd["solution_id"]])
        self.assertEqual(nodes[refined["solution_id"]]["parents"], [sgd["solution_id"]])
        # 7. Incumbent = best measured validation score; failed runs excluded.
        completed = {sid: n["validation_score"] for sid, n in nodes.items()
                     if n["status"] == "completed"}
        [group] = self.graph.incumbents(nodes)
        best = max(completed.values())
        self.assertEqual(group["validation_score"], best)
        self.assertEqual(sorted([group["incumbent"], *group["tied_with"]]),
                         sorted(s for s, v in completed.items() if v == best))
        self.assertEqual(group["candidates_compared"], 5)
        self.assertEqual(group["direction"], "max")
        # Checkpoint provenance for every completed node.
        for sid in completed:
            detail = self.graph.inspect(sid)
            self.assertEqual(detail["checkpoint_sha256"], file_sha256(detail["checkpoint_path"]))
        # Budget: proposals are free; only launches counted.
        budget = get_budget_status("E2", "search", DATASET, 42, max_experiments=7)
        self.assertEqual(budget["launched_experiments"], 6)
        self.assertEqual(len(self.launches), 6)

    def test_persistence_and_reload(self):
        self.setup_search()
        root = self.propose("root")
        self.launch()
        child = self.propose("explore", root["solution_id"], ["training.learning_rate"], lr=0.05)
        before = self.graph.frontier()
        reloaded = SearchGraph("search", "E2", 42, self.max_experiments, DATASET)
        after = reloaded.frontier()
        self.assertEqual(before["nodes"], after["nodes"])
        self.assertEqual(before["edges"], after["edges"])
        self.assertEqual(after["ready_to_launch"], [child["solution_id"]])
        log = (self.root / "campaigns" / "search" / "search.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["type"] for line in log], ["edge", "edge"])


class GraphIntegrityTests(SearchCase):
    def setUp(self):
        super().setUp()
        self.setup_search()
        self.root_id = self.propose("root")["solution_id"]
        self.sgd = self.propose("explore", self.root_id, ["pipeline.optimizer"],
                                pipeline={"optimizer": "sgd"})["solution_id"]
        self.fast = self.propose("explore", self.root_id, ["training.learning_rate"],
                                 lr=0.05)["solution_id"]

    def test_duplicate_proposals_reuse_the_node_and_edge(self):
        again = self.propose("explore", self.root_id, ["pipeline.optimizer"],
                             pipeline={"optimizer": "sgd"})
        self.assertEqual(again["solution_id"], self.sgd)
        self.assertTrue(again["duplicate_edge"])
        self.assertEqual(len(self.graph.state()["edges"]), 3)

    def test_multiple_parents_for_one_candidate(self):
        via_sgd = self.propose("refine", self.sgd, ["training.learning_rate"],
                               lr=0.05, pipeline={"optimizer": "sgd"})
        via_fast = self.propose("refine", self.fast, ["pipeline.optimizer"],
                                lr=0.05, pipeline={"optimizer": "sgd"})
        self.assertEqual(via_sgd["solution_id"], via_fast["solution_id"])
        self.assertTrue(via_fast["reused_existing_node"])
        node = self.graph.state()["nodes"][via_sgd["solution_id"]]
        self.assertEqual(sorted(node["parents"]), sorted([self.sgd, self.fast]))

    def test_combination_must_include_the_claimed_changes(self):
        combined = self.search["propose_branch"](
            intent="combine", observation="both helped", hypothesis="gains are additive",
            expected_signal="better than either parent", parent_solution_id=self.sgd,
            second_parent_id=self.fast, claimed_changes=["training.learning_rate"],
            changes_from_second_parent=["training.learning_rate"],
            **self.args(lr=0.05, pipeline={"optimizer": "sgd"}))
        self.assertEqual(combined["status"], "proposed", combined)
        self.assertEqual(sorted(combined["parents"]), sorted([self.sgd, self.fast]))
        fake = self.search["propose_branch"](
            intent="combine", observation="x", hypothesis="y", expected_signal="z",
            parent_solution_id=self.sgd, second_parent_id=self.fast,
            claimed_changes=["training.learning_rate"],
            changes_from_second_parent=["training.learning_rate"],
            **self.args(lr=0.02, pipeline={"optimizer": "sgd"}))  # not the donor's value
        self.assertEqual(fake["error_code"], "COMBINE_NOT_PRESENT")

    def test_claimed_changes_must_match_the_actual_delta(self):
        wrong = self.propose("refine", self.sgd, ["training.epochs"],
                             pipeline={"optimizer": "sgd", "scheduler": "step"})
        self.assertEqual(wrong["error_code"], "DELTA_MISMATCH")
        self.assertIn("pipeline.scheduler.name", wrong["actual_delta"])
        self.assertEqual(wrong["claimed_but_unchanged"], ["training.epochs"])
        same = self.propose("refine", self.sgd, [], pipeline={"optimizer": "sgd"})
        self.assertEqual(same["error_code"], "NO_CHANGE")

    def test_cycles_are_rejected(self):
        loop = self.propose("revisit", self.sgd, ["pipeline.optimizer"])  # back to the root
        self.assertEqual(loop["error_code"], "CYCLE")
        unknown = self.propose("explore", "sol_" + "0" * 20, ["training.learning_rate"], lr=0.2)
        self.assertEqual(unknown["error_code"], "UNKNOWN_PARENT")
        orphan = self.propose("explore", None, None, lr=0.3)
        self.assertEqual(orphan["error_code"], "INVALID_PROPOSAL")

    def test_concurrent_proposals_keep_the_graph_consistent(self):
        results, threads = [], []
        rates = [0.011, 0.012, 0.013, 0.014, 0.015, 0.011, 0.012, 0.013]  # 5 distinct, 3 repeats

        def worker(rate):
            results.append(self.propose("explore", self.root_id, ["training.learning_rate"],
                                        lr=rate))
        for rate in rates:
            threads.append(threading.Thread(target=worker, args=(rate,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertTrue(all(r["status"] == "proposed" for r in results), results)
        edges = self.graph.state()["edges"]
        pairs = [(tuple(e["parents"]), e["child"]) for e in edges]
        self.assertEqual(len(pairs), len(set(pairs)))  # no duplicate edges
        self.assertEqual(len(edges), 3 + 5)


class DecisionInfluenceTests(SearchCase):
    """The graph changes what the agent can do, not only what is recorded."""

    def test_only_proposed_candidates_can_launch(self):
        self.setup_search()
        rejected = self.launch()
        self.assertEqual(rejected["error_code"], "NOT_IN_SEARCH_GRAPH")
        self.assertEqual(self.launches, [])
        self.propose("root")
        self.assertEqual(self.launch()["status"], "completed")
        again = self.launch()
        self.assertEqual(again["error_code"], "DUPLICATE_CONFIGURATION")  # node runs once

    def test_branching_from_an_earlier_node_and_concluding(self):
        self.setup_search()
        root = self.propose("root")
        self.launch()
        first = self.propose("explore", root["solution_id"], ["training.learning_rate"], lr=0.6)
        self.launch(lr=0.6)
        # The latest candidate failed; the frontier offers the earlier root to branch from.
        frontier = self.search["search_frontier"]()
        self.assertIn(first["solution_id"], frontier["failed"])
        self.assertEqual(frontier["incumbents"][0]["incumbent"], root["solution_id"])
        branch = self.propose("explore", root["solution_id"], ["pipeline.optimizer"],
                              pipeline={"optimizer": "sgd"})
        self.assertEqual(self.search["search_frontier"]()["ready_to_launch"],
                         [branch["solution_id"]])
        done = self.search["conclude_search"]("budget better kept", root["solution_id"])
        self.assertEqual(done["status"], "concluded")
        self.assertEqual(self.launch(pipeline={"optimizer": "sgd"})["error_code"],
                         "SEARCH_CONCLUDED")
        self.assertEqual(self.propose("explore", root["solution_id"], ["training.epochs"],
                                      epochs=3)["error_code"], "SEARCH_CONCLUDED")

    def test_budget_is_still_enforced(self):
        self.max_experiments = 1
        self.setup_search()
        root = self.propose("root")
        self.launch()
        self.propose("explore", root["solution_id"], ["training.learning_rate"], lr=0.05)
        self.assertEqual(self.launch(lr=0.05)["error_code"], "BUDGET_EXHAUSTED")

    def test_tools_and_gate_exist_only_in_search_mode(self):
        args = campaign_main.build_parser().parse_args(["--dataset", DATASET, "--feedback", "E2"])
        self.assertIsNone(campaign_main.search_graph_for(args, "plain"))
        plain = tools.make_registered_training_tool(
            version="E2", campaign_id="plain", seed=42, max_experiments=2,
            required_dataset=DATASET, training_fn=lambda **kw: {"status": "failed"})
        self.assertNotEqual(plain(**self.args()).get("error_code"), "NOT_IN_SEARCH_GRAPH")


class BoundaryTests(SearchCase):
    def test_e0_and_e1_cannot_use_the_graph(self):
        for version in ("E0", "E1"):
            with self.subTest(version):
                graph = SearchGraph("x", version, 42, 3, DATASET)
                with self.assertRaisesRegex(ValueError, "history"):
                    make_search_tools(graph=graph, autonomy=AutonomyPolicy(scope="S1"))
                args = campaign_main.build_parser().parse_args(
                    ["--search", "--feedback", version, "--dataset", DATASET])
                with self.assertRaises(SystemExit):
                    campaign_main.search_graph_for(args, "x")

    def test_scope_limits_still_apply(self):
        self.setup_search(scope="S0")
        refused = self.search["propose_branch"](
            intent="root", observation="o", hypothesis="h", expected_signal="e",
            **self.args(pipeline={"optimizer": "sgd"}))
        self.assertEqual(refused["error_code"], "SCOPE_NOT_PERMITTED")
        self.setup_search(campaign="s1", scope="S1")
        generated = self.search["propose_branch"](
            intent="root", observation="o", hypothesis="h", expected_signal="e",
            **self.args(pipeline={"loss": {"name": "custom", "component_id": "cl_" + "a" * 20}}))
        self.assertEqual(generated["error_code"], "SCOPE_NOT_PERMITTED")

    def test_p1_decision_records_still_required(self):
        self.setup_search(policy="P1")
        self.propose("root")
        response = self.run_tool(decision_record={}, **self.args())
        self.assertEqual(response["error_code"], "INVALID_DECISION_RECORD")

    def test_hitl_approval_still_gates_graph_nodes(self):
        from agent import hitl

        hitl.create_state("hitl-search", {
            "feedback": "E2", "policy": "P0", "seed": 42, "max_experiments": 3,
            "dataset": DATASET, "model_id": "m", "region": "r", "hitl_mode": "always",
            "failure_threshold": 2, "cost_threshold_usd": None, "unknown_cost_policy": "ask",
            "prices": None})
        gate = hitl.HitlController("hitl-search", hitl.load_state("hitl-search")["settings"])
        self.setup_search(campaign="hitl-search", launch_gate=gate)
        self.assertEqual(self.launch()["error_code"], "NOT_IN_SEARCH_GRAPH")
        self.propose("root")
        self.assertEqual(self.launch()["error_code"], "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(self.launches, [])


class IncumbentTests(unittest.TestCase):
    @staticmethod
    def node(sid, score, metric="accuracy", status="completed", task="classification",
             image_size=None, dataset="m-eurosat"):
        return {"solution_id": sid, "status": status, "validation_score": score,
                "primary_metric": metric,
                "spec": {"dataset": {"dataset_id": dataset, "task_type": task},
                         "inputs": {"image_size": image_size, "native_resolution": [64, 64]}}}

    def test_groups_direction_failures_and_unknown_metrics(self):
        graph = SearchGraph("i", "E2", 42, 3)
        nodes = {n["solution_id"]: n for n in (
            self.node("a", 0.7), self.node("b", 0.9), self.node("c", None, status="failed"),
            self.node("d", 0.95, metric="mystery"),
            self.node("s1", 0.4, "mean_iou", task="segmentation"),
            self.node("s2", 0.6, "mean_iou", task="segmentation", image_size=128),
        )}
        groups = {(g["group"]["primary_metric"], g["group"]["evaluation_resolution"]): g
                  for g in graph.incumbents(nodes)}
        self.assertEqual(groups[("accuracy", None)]["incumbent"], "b")
        self.assertEqual(groups[("accuracy", None)]["candidates_compared"], 2)
        # Different evaluation resolutions are different protocols: never ranked together.
        self.assertEqual(groups[("mean_iou", "(64, 64)")]["incumbent"], "s1")
        self.assertEqual(groups[("mean_iou", "128")]["incumbent"], "s2")
        self.assertNotIn(("mystery", None), groups)
        self.assertNotEqual(comparability_key(nodes["a"]["spec"], "accuracy"),
                            comparability_key(self.node("x", 1, dataset="m-pv4ger")["spec"],
                                              "accuracy"))


REAL_STRANDS = strands_stub.real_strands_available()


@unittest.skipUnless(REAL_STRANDS, "strands-agents is not installed")
class StrandsSearchCampaignTests(SearchCase):
    """agent.main --search through the real Strands agent loop with a scripted model."""

    def test_scripted_agent_branches_through_the_graph(self):
        from strands import Agent

        try:
            from test_strands_integration import ScriptedModel
        except ImportError:
            from tests.test_strands_integration import ScriptedModel
        script, seen = [], []
        real_factory = tools.make_registered_training_tool

        def factory(**kwargs):
            def trainer(dataset_id, config, **context):
                return train_model(dataset_id, {**config, "output_dir": str(self.root / "runs")},
                                   **context)
            return real_factory(training_fn=trainer, **kwargs)
        run = {"dataset_id": DATASET, "model_id": "resnet18_rgb", "epochs": 1,
               "batch_size": 2, "learning_rate": 0.01}
        script += [
            ("tool", ("run_campaign_experiment", run)),  # refused: not proposed yet
            ("tool", ("propose_branch", {"intent": "root", "observation": "start",
                                         "hypothesis": "baseline", "expected_signal": "any",
                                         **run})),
            ("tool", ("run_campaign_experiment", run)),
            ("tool", ("search_frontier", {})),
            ("tool", ("propose_branch", {"intent": "explore", "observation": "baseline done",
                                         "hypothesis": "higher lr", "expected_signal": "higher acc",
                                         "parent_solution_id": "__ROOT__",
                                         "claimed_changes": ["training.learning_rate"],
                                         **{**run, "learning_rate": 0.05}})),
            ("tool", ("conclude_search", {"reason": "budget"})),
            ("text", "done"),
        ]

        class RootFillingModel(ScriptedModel):
            async def stream(inner, messages, tool_specs=None, system_prompt=None, **kwargs):
                text = json.dumps(messages, default=str)
                if script and script[0][0] == "tool" and script[0][1][1].get(
                        "parent_solution_id") == "__ROOT__":
                    import re
                    root_id = re.findall(r"sol_[0-9a-f]{20}", text)[0]
                    script[0][1][1]["parent_solution_id"] = root_id
                async for event in ScriptedModel.stream(inner, messages, tool_specs,
                                                        system_prompt, **kwargs):
                    yield event
        with patch.object(campaign_main, "_bedrock_model",
                          lambda *a: RootFillingModel(script, seen)), \
                patch.object(campaign_main, "make_registered_training_tool", factory), \
                patch.object(campaign_main, "Agent",
                             lambda **kw: Agent(callback_handler=None, **kw)):
            out = io.StringIO()
            with redirect_stdout(out):
                campaign_main.run_campaign(campaign_main.build_parser().parse_args(
                    ["--campaign-id", "sdk-search", "--dataset", DATASET, "--feedback", "E2",
                     "--search", "--max-experiments", "3"]))
        output = out.getvalue()
        report = json.loads(output[output.index("{", output.index("FINAL PERSISTED")):])
        graph = report["search_graph"]
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(report["experiments"]), 1)
        self.assertIsNotNone(graph["concluded"])
        self.assertIn("NOT_IN_SEARCH_GRAPH", seen[1])  # the unproposed launch was refused


if __name__ == "__main__":
    unittest.main()

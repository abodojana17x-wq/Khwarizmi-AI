import unittest

from rafig.language.semantic_representation import SemanticAnalyzer
from rafig.reasoning import (
    CandidateAction,
    CausalRelation,
    Constraint,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    InferenceRule,
    PlanStatus,
    ReasoningEngine,
    Result,
    WorkStatus,
)


EXAMPLE_REQUEST = (
    "Create a Python program that reads files, groups them by extension, "
    "and moves them into folders."
)


class ReasoningEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ReasoningEngine()

    def test_example_produces_a_structured_plan_not_code(self) -> None:
        report = self.engine.reason(EXAMPLE_REQUEST)
        plan = report.plan

        self.assertGreaterEqual(len(plan.goals), 4)
        self.assertEqual(plan.primary_goal.parent_goal_id, None)
        actions = [task.action for task in plan.tasks]
        self.assertEqual(actions, ["design", "read", "group", "move", "verify"])
        self.assertTrue(all(task.subtasks for task in plan.tasks))
        self.assertFalse(plan.metadata["executes_operations"])
        self.assertNotIn("def ", str(plan.to_dict()))

    def test_tracks_assumptions_constraints_and_hypotheses(self) -> None:
        plan = self.engine.create_plan(EXAMPLE_REQUEST)
        constraint_text = {item.description for item in plan.constraints}

        self.assertIn("Implementation language is Python", constraint_text)
        self.assertIn("Files are grouped using their extension", constraint_text)
        self.assertGreaterEqual(len(plan.assumptions), 2)
        self.assertEqual(len(plan.hypotheses), len(plan.assumptions))

    def test_tasks_have_ordering_and_causal_relationships(self) -> None:
        plan = self.engine.create_plan(EXAMPLE_REQUEST)

        self.assertEqual(plan.tasks[0].dependencies, [])
        for earlier, later in zip(plan.tasks, plan.tasks[1:]):
            self.assertEqual(later.dependencies, [earlier.id])
            self.assertTrue(
                any(
                    relation.cause == earlier.id and relation.effect == later.id
                    for relation in plan.causal_relations
                )
            )

    def test_possible_actions_are_compared_transparently(self) -> None:
        comparison = self.engine.compare_actions(
            [
                CandidateAction("fast", utility=0.8, feasibility=1.0, estimated_cost=0.1, risk=0.5),
                CandidateAction("safe", utility=0.9, feasibility=1.0, estimated_cost=0.3, risk=0.05),
            ]
        )

        self.assertEqual(comparison.selected_action, "safe")
        self.assertTrue(all(option.score is not None for option in comparison.ranked_options))
        self.assertIn("utility", comparison.rationale)

    def test_forward_chaining_performs_basic_logical_inference(self) -> None:
        self.engine.add_rule(InferenceRule(("files read",), "files available", "read_output"))
        self.engine.add_rule(
            InferenceRule(("files available", "grouping key known"), "files can be grouped", "group_preconditions")
        )

        facts = self.engine.infer({"files read", "grouping key known"})

        self.assertIn("files available", facts)
        self.assertIn("files can be grouped", facts)

    def test_hypothesis_changes_only_from_linked_evidence(self) -> None:
        hypothesis = Hypothesis("source exists")
        evidence = Evidence("directory was found", supports=["source exists"], confidence=0.9)

        updated = self.engine.evaluate_hypothesis(hypothesis, [evidence])

        self.assertEqual(updated.status, HypothesisStatus.SUPPORTED)
        self.assertEqual(updated.supporting_evidence_ids, [evidence.id])

    def test_causal_reasoner_predicts_transitive_effects(self) -> None:
        self.engine.add_causal_relation(CausalRelation("read files", "files available"))
        self.engine.add_causal_relation(CausalRelation("files available", "grouping enabled"))

        effects = self.engine.predict_effects({"read files"})

        self.assertEqual(effects, {"files available", "grouping enabled"})

    def test_goal_completion_is_derived_from_successful_results(self) -> None:
        plan = self.engine.create_plan(EXAMPLE_REQUEST)
        while plan.status != PlanStatus.COMPLETED:
            ready = plan.ready_tasks()
            self.assertTrue(ready)
            for task in ready:
                self.engine.record_result(plan, Result(task.id, True, output={"ok": True}))

        self.assertTrue(self.engine.check_goal_completion(plan))
        self.assertTrue(all(goal.status == WorkStatus.COMPLETED for goal in plan.goals))
        self.assertEqual(plan.progress, 1.0)

    def test_hard_constraint_failure_prevents_goal_completion(self) -> None:
        plan = self.engine.create_plan("Create a report", [Constraint("Output must be text")])
        plan.constraints[-1].satisfied = False
        for task in plan.tasks:
            self.engine.record_result(plan, task.id, True)

        self.assertFalse(self.engine.check_goal_completion(plan))
        self.assertEqual(plan.primary_goal.status, WorkStatus.FAILED)

    def test_failed_operation_revises_plan_and_redirects_dependents(self) -> None:
        plan = self.engine.create_plan(EXAMPLE_REQUEST)
        design, read, group = plan.tasks[:3]
        self.engine.record_result(plan, design.id, True)

        revision = self.engine.record_result(
            plan,
            read.id,
            False,
            error="source temporarily unavailable",
        )

        self.assertIsNotNone(revision)
        self.assertEqual(plan.status, PlanStatus.REVISED)
        self.assertIsNotNone(read.replaced_by_task_id)
        retry = plan.get_task(read.replaced_by_task_id)
        self.assertEqual(retry.replaces_task_id, read.id)
        self.assertNotEqual(retry.selected_strategy, read.selected_strategy)
        self.assertEqual(group.dependencies, [retry.id])
        self.assertEqual(len(revision.added_task_ids), 2)

    def test_semantic_representation_can_be_used_as_input(self) -> None:
        semantic = SemanticAnalyzer().analyze("Fix this Python code")

        report = self.engine.reason(semantic)

        self.assertEqual(report.semantic_input, semantic)
        self.assertIn("repair", [task.action for task in report.tasks])


if __name__ == "__main__":
    unittest.main()

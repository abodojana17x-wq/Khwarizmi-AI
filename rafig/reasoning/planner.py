"""Dependency-aware symbolic plan construction."""

from __future__ import annotations

from typing import Any, Iterable

from .decomposition import ActionMention, RequestDecomposer
from .evaluation import ActionEvaluator
from .models import (
    CandidateAction,
    CausalRelation,
    Constraint,
    Evidence,
    Goal,
    Hypothesis,
    Plan,
    PlanStatus,
    Subtask,
    Task,
)


class Planner:
    """Turn a request into explicit goals, tasks, subtasks, and dependencies."""

    def __init__(
        self,
        decomposer: RequestDecomposer | None = None,
        evaluator: ActionEvaluator | None = None,
    ) -> None:
        self.decomposer = decomposer or RequestDecomposer()
        self.evaluator = evaluator or ActionEvaluator()

    def create_plan(
        self,
        request: str,
        semantic_input: Any = None,
        additional_constraints: Iterable[Constraint | str] | None = None,
    ) -> Plan:
        text = " ".join(request.strip().split())
        if not text:
            raise ValueError("Cannot create a plan for an empty request")

        mentions = self.decomposer.extract_actions(text)
        if not mentions:
            mentions = [self._mention_from_semantics(text, semantic_input)]
        goals = self.decomposer.decompose_goals(text, mentions)
        primary = goals[0]
        constraints = self.decomposer.extract_constraints(text, semantic_input)
        constraints.extend(self._coerce_constraints(additional_constraints or []))
        constraints = self._deduplicate_constraints(constraints)
        assumptions = self.decomposer.identify_assumptions(text, mentions)
        evidence = [
            Evidence(
                statement=f"The user requested: {text}",
                source="user_request",
                confidence=1.0,
                supports=[goal.id for goal in goals],
            )
        ]
        hypotheses = [
            Hypothesis(statement=assumption.statement, confidence=assumption.confidence)
            for assumption in assumptions
        ]

        plan = Plan(
            request=text,
            goals=goals,
            assumptions=assumptions,
            constraints=constraints,
            evidence=evidence,
            hypotheses=hypotheses,
            status=PlanStatus.ACTIVE,
            metadata={
                "planner": "symbolic",
                "executes_operations": False,
                "semantic_input_available": semantic_input is not None,
            },
        )
        self._build_tasks(plan, mentions)
        return plan

    def identify_subtasks(self, task: Task) -> list[Subtask]:
        """Build procedural subtasks from the action schema of a task."""
        target = task.target or "requested item"
        descriptions = self._subtask_descriptions(task.action, target)
        subtasks: list[Subtask] = []
        previous_id: str | None = None
        for action, description in descriptions:
            subtask = Subtask(
                description=description,
                action=action,
                target=target,
                parent_task_id=task.id,
                dependencies=[previous_id] if previous_id else [],
            )
            subtasks.append(subtask)
            previous_id = subtask.id
        return subtasks

    def _build_tasks(self, plan: Plan, mentions: list[ActionMention]) -> None:
        primary = plan.primary_goal
        assert primary is not None
        operational = [
            mention
            for mention in mentions
            if not self.decomposer.is_wrapper_action(mention, mentions)
        ]
        wrapper = next(
            (mention for mention in mentions if self.decomposer.is_wrapper_action(mention, mentions)),
            None,
        )
        child_goals = [goal for goal in plan.goals if goal.parent_goal_id == primary.id]
        child_by_description = {goal.description.lower(): goal for goal in child_goals}

        task_specs: list[tuple[str, str, Goal]] = []
        if wrapper is not None:
            task_specs.append(("design", wrapper.target, primary))
        for mention in operational:
            description = self.decomposer.describe_action(mention.action, mention.target)
            goal = child_by_description.get(description.lower(), primary)
            task_specs.append((mention.action, mention.target, goal))
        if not task_specs:
            task_specs.append(("understand", plan.request, primary))
        task_specs.append(("verify", primary.description, primary))

        previous: Task | None = None
        for action, target, goal in task_specs:
            description = self.decomposer.describe_action(action, target)
            candidates = self._candidate_actions(action, target)
            comparison = self.evaluator.compare(candidates, plan.constraints)
            task = Task(
                description=description,
                action=action,
                target=target,
                goal_id=goal.id,
                dependencies=[previous.id] if previous else [],
                constraints=[constraint.id for constraint in plan.constraints],
                candidate_actions=comparison.ranked_options,
                selected_strategy=comparison.selected_action,
                requires=self._required_resources(action),
                produces=self._produced_resources(action),
            )
            task.subtasks = self.identify_subtasks(task)
            plan.tasks.append(task)
            goal.task_ids.append(task.id)
            plan.action_comparisons.append(comparison)
            if previous is not None:
                plan.causal_relations.append(
                    CausalRelation(
                        cause=previous.id,
                        effect=task.id,
                        relation="enables",
                        conditions=[f"{previous.id}:completed"],
                        confidence=1.0,
                        evidence_ids=[plan.evidence[0].id],
                    )
                )
            previous = task

    @staticmethod
    def _mention_from_semantics(text: str, semantic_input: Any) -> ActionMention:
        action = getattr(semantic_input, "action", None)
        target = getattr(semantic_input, "object", None)
        if isinstance(semantic_input, dict):
            action = semantic_input.get("action", action)
            target = semantic_input.get("object", target)
        action = str(action or "understand")
        target = str(target or text)
        return ActionMention(action=action, surface=action, target=target, start=0, end=0)

    @staticmethod
    def _coerce_constraints(items: Iterable[Constraint | str]) -> list[Constraint]:
        return [item if isinstance(item, Constraint) else Constraint(str(item), source="caller") for item in items]

    @staticmethod
    def _deduplicate_constraints(constraints: Iterable[Constraint]) -> list[Constraint]:
        unique: list[Constraint] = []
        seen: set[str] = set()
        for constraint in constraints:
            key = " ".join(constraint.description.lower().split())
            if key not in seen:
                seen.add(key)
                unique.append(constraint)
        return unique

    @staticmethod
    def _required_resources(action: str) -> list[str]:
        return {
            "design": ["request", "constraints"],
            "read": ["source_location"],
            "group": ["items", "grouping_key"],
            "move": ["items", "destination"],
            "copy": ["items", "destination"],
            "delete": ["items", "permission"],
            "repair": ["failing_artifact", "failure_evidence"],
            "validate": ["candidate_outcome", "success_criteria"],
            "verify": ["operation_results", "success_criteria"],
        }.get(action, ["request"])

    @staticmethod
    def _produced_resources(action: str) -> list[str]:
        return {
            "design": ["requirements", "success_criteria"],
            "read": ["items", "item_metadata"],
            "group": ["grouped_items", "destinations"],
            "move": ["relocated_items"],
            "copy": ["copied_items"],
            "delete": ["removed_items"],
            "repair": ["repaired_artifact"],
            "validate": ["validation_result"],
            "verify": ["completion_evidence"],
            "understand": ["clarified_request"],
        }.get(action, [f"{action}_result"])

    @staticmethod
    def _subtask_descriptions(action: str, target: str) -> list[tuple[str, str]]:
        schemas: dict[str, list[tuple[str, str]]] = {
            "design": [
                ("extract", f"Extract required behavior and constraints for {target}"),
                ("define", f"Define inputs, outputs, and completion criteria for {target}"),
            ],
            "read": [
                ("validate", f"Validate the source for {target}"),
                ("enumerate", f"Enumerate {target}"),
                ("observe", f"Collect the metadata needed from {target}"),
            ],
            "group": [
                ("identify_key", f"Identify the grouping key for {target}"),
                ("assign", f"Assign each item in {target} to one group"),
                ("validate", f"Check that every item in {target} has one group"),
            ],
            "move": [
                ("prepare", f"Prepare and validate destinations for {target}"),
                ("transfer", f"Move each item in {target} to its destination"),
                ("validate", f"Verify each move for {target}"),
            ],
            "copy": [
                ("prepare", f"Prepare destinations for {target}"),
                ("transfer", f"Copy {target}"),
                ("validate", f"Verify copied {target}"),
            ],
            "repair": [
                ("diagnose", f"Collect evidence about the failure in {target}"),
                ("isolate", f"Identify the cause of the failure in {target}"),
                ("repair", f"Apply the least disruptive repair to {target}"),
                ("validate", f"Verify the repaired {target}"),
            ],
            "verify": [
                ("collect", f"Collect results for {target}"),
                ("compare", f"Compare results with success criteria for {target}"),
            ],
            "understand": [
                ("extract", f"Extract known information from {target}"),
                ("clarify", f"Identify missing information in {target}"),
            ],
        }
        if action in schemas:
            return schemas[action]
        return [
            ("prepare", f"Prepare inputs for {action} {target}"),
            (action, f"{action.capitalize()} {target}"),
            ("validate", f"Validate the result of {action} {target}"),
        ]

    @staticmethod
    def _candidate_actions(action: str, target: str) -> list[CandidateAction]:
        specific: dict[str, list[CandidateAction]] = {
            "read": [
                CandidateAction("single_pass", f"Read {target} once", estimated_cost=0.25, risk=0.25, utility=0.85),
                CandidateAction("incremental_scan", f"Read {target} in bounded batches", estimated_cost=0.45, risk=0.15, utility=0.8),
            ],
            "group": [
                CandidateAction("direct_classification", f"Classify {target} in one pass", estimated_cost=0.25, risk=0.2, utility=0.9),
                CandidateAction("staged_classification", f"Collect keys before classifying {target}", estimated_cost=0.45, risk=0.1, utility=0.82),
            ],
            "move": [
                CandidateAction("direct_move", f"Move {target} directly", estimated_cost=0.25, risk=0.4, utility=0.9),
                CandidateAction("copy_verify_then_remove", f"Copy, verify, then remove {target}", estimated_cost=0.65, risk=0.12, utility=0.92),
            ],
            "repair": [
                CandidateAction("minimal_repair", f"Apply the smallest repair to {target}", estimated_cost=0.3, risk=0.25, utility=0.9),
                CandidateAction("replace_component", f"Replace the failing part of {target}", estimated_cost=0.6, risk=0.45, utility=0.75),
            ],
            "verify": [
                CandidateAction("criteria_check", f"Check all explicit criteria for {target}", estimated_cost=0.25, risk=0.05, utility=0.95),
                CandidateAction("sample_check", f"Check a sample of outcomes for {target}", estimated_cost=0.1, risk=0.35, utility=0.65),
            ],
        }
        if action in specific:
            return specific[action]
        return [
            CandidateAction("direct", f"Perform {action} for {target} directly", estimated_cost=0.3, risk=0.3, utility=0.85),
            CandidateAction("guarded", f"Validate before and after {action} for {target}", estimated_cost=0.5, risk=0.15, utility=0.88),
        ]

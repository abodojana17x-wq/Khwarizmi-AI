"""High-level orchestration for RAFIQ's phase 06 reasoning engine."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol

from .evaluation import ActionEvaluator
from .inference import CausalReasoner, InferenceEngine, InferenceReport, InferenceRule
from .models import (
    ActionComparison,
    AssumptionStatus,
    CandidateAction,
    CausalRelation,
    Constraint,
    ConstraintKind,
    Evidence,
    Goal,
    Hypothesis,
    Plan,
    PlanRevision,
    PlanStatus,
    ReasoningReport,
    Result,
    Subtask,
    Task,
    WorkStatus,
)
from .planner import Planner


class AnalyzerPort(Protocol):
    """Interface implemented by language and semantic analyzers."""

    def analyze(self, text: str) -> Any: ...


class MemoryPort(Protocol):
    """Minimal future-facing interface for a memory component."""

    def store(self, item: Any) -> Any: ...

    def retrieve(self, query: Any) -> Iterable[Any]: ...


class PythonBrainPort(Protocol):
    """Future execution interface; phase 06 never invokes it automatically."""

    def execute(self, task: Task) -> Result: ...


class ReasoningEngine:
    """Create and revise plans using deterministic symbolic operations.

    The engine plans only.  It does not generate Python code and does not run
    plan operations.  A later Python Brain can execute a ready task and feed
    its :class:`Result` back through :meth:`record_result`.
    """

    def __init__(
        self,
        planner: Planner | None = None,
        language_analyzer: AnalyzerPort | None = None,
        semantic_analyzer: AnalyzerPort | None = None,
        memory: MemoryPort | None = None,
        python_brain: PythonBrainPort | None = None,
        max_revisions: int = 3,
    ) -> None:
        if max_revisions < 0:
            raise ValueError("max_revisions cannot be negative")
        self.planner = planner or Planner()
        self.action_evaluator: ActionEvaluator = self.planner.evaluator
        self.language_analyzer = language_analyzer or self._default_language_analyzer()
        self.semantic_analyzer = semantic_analyzer or self._default_semantic_analyzer()
        self.memory = memory
        self.python_brain = python_brain
        self.max_revisions = max_revisions
        self.inference_engine = InferenceEngine(self._base_rules())
        self.causal_reasoner = CausalReasoner()
        self.last_report: ReasoningReport | None = None

    def reason(
        self,
        request: str | Any,
        constraints: Iterable[Constraint | str] | None = None,
    ) -> ReasoningReport:
        """Analyze a request and return inspectable reasoning plus a plan."""
        text, supplied_semantics = self._request_text(request)
        language_input = None
        semantic_input = supplied_semantics
        if isinstance(request, str):
            if self.language_analyzer is not None:
                language_input = self.language_analyzer.analyze(text)
            if self.semantic_analyzer is not None:
                semantic_input = self.semantic_analyzer.analyze(text)

        plan = self.planner.create_plan(text, semantic_input, constraints)
        facts = self._facts_from_plan(plan)
        inference = self.inference_engine.infer_with_trace(facts)
        inferred_evidence = self.inference_engine.evidence_from_report(inference)
        plan.evidence.extend(inferred_evidence)
        for relation in plan.causal_relations:
            self.causal_reasoner.add_relation(relation)

        report = ReasoningReport(
            request=text,
            plan=plan,
            inferred_facts=inference.facts,
            semantic_input=semantic_input,
            language_input=language_input,
        )
        self.last_report = report
        self._store(report)
        return report

    analyze = reason

    def create_plan(
        self,
        request: str | Any,
        constraints: Iterable[Constraint | str] | None = None,
    ) -> Plan:
        """Convenience method returning only the structured plan."""
        return self.reason(request, constraints).plan

    plan_request = create_plan

    def break_into_goals(self, request: str | Any) -> list[Goal]:
        return self.create_plan(request).goals

    decompose_request = break_into_goals

    def identify_subtasks(self, item: Task | Goal, plan: Plan | None = None) -> list[Subtask]:
        if isinstance(item, Task):
            return item.subtasks or self.planner.identify_subtasks(item)
        if plan is None:
            raise ValueError("A plan is required when identifying subtasks for a goal")
        return [subtask for task in plan.tasks_for_goal(item.id) for subtask in task.subtasks]

    def compare_actions(
        self,
        actions: Iterable[CandidateAction],
        constraints: Iterable[Constraint] | None = None,
    ) -> ActionComparison:
        return self.action_evaluator.compare(actions, constraints)

    def add_rule(self, rule: InferenceRule) -> None:
        self.inference_engine.add_rule(rule)

    def infer(self, facts: Iterable[str | Evidence]) -> set[str]:
        return self.inference_engine.infer(facts)

    def infer_with_trace(self, facts: Iterable[str | Evidence]) -> InferenceReport:
        return self.inference_engine.infer_with_trace(facts)

    def add_causal_relation(self, relation: CausalRelation) -> None:
        self.causal_reasoner.add_relation(relation)

    def predict_effects(
        self,
        causes: Iterable[str],
        available_conditions: Iterable[str] | None = None,
    ) -> set[str]:
        return self.causal_reasoner.predict_effects(causes, available_conditions)

    def evaluate_hypothesis(self, hypothesis: Hypothesis, evidence: Iterable[Evidence]) -> Hypothesis:
        return self.inference_engine.evaluate_hypothesis(hypothesis, evidence)

    def start_task(self, plan: Plan, task_id: str) -> Task:
        task = plan.get_task(task_id)
        ready_ids = {item.id for item in plan.ready_tasks()}
        if task.id not in ready_ids and task.status != WorkStatus.READY:
            raise RuntimeError(f"Task {task_id} has incomplete dependencies")
        task.status = WorkStatus.IN_PROGRESS
        plan.status = PlanStatus.ACTIVE
        for subtask in task.subtasks:
            if not subtask.dependencies:
                subtask.status = WorkStatus.READY
        return task

    def record_result(
        self,
        plan: Plan,
        result: Result | str,
        success: bool | None = None,
        output: Any = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        revise_on_failure: bool = True,
    ) -> PlanRevision | None:
        """Track an outcome, update goal state, and optionally revise on failure.

        ``result`` can be a fully populated :class:`Result` or a task id plus
        the remaining keyword arguments.  Operations themselves happen
        outside this phase.
        """
        if isinstance(result, str):
            if success is None:
                raise ValueError("success must be supplied with a task id")
            result = Result(
                task_id=result,
                success=success,
                output=output,
                error=error,
                metadata=dict(metadata or {}),
            )
        task = plan.get_task(result.task_id)
        task.attempt_count += 1
        task.result_ids.append(result.id)
        if task.selected_strategy and task.selected_strategy not in task.attempted_strategies:
            task.attempted_strategies.append(task.selected_strategy)
        plan.results.append(result)

        if result.success:
            task.status = WorkStatus.COMPLETED
            for subtask in task.subtasks:
                subtask.status = WorkStatus.COMPLETED
                subtask.result_ids.append(result.id)
            if not result.evidence:
                result.evidence.append(
                    Evidence(
                        statement=f"Task completed: {task.description}",
                        source=f"result:{result.id}",
                        supports=[task.id, task.goal_id or ""],
                    )
                )
            plan.evidence.extend(result.evidence)
            self._evaluate_constraints(plan, result)
            self.check_goal_completion(plan)
            self._store(result)
            return None

        task.status = WorkStatus.FAILED
        for subtask in task.subtasks:
            if subtask.status in {WorkStatus.READY, WorkStatus.IN_PROGRESS}:
                subtask.status = WorkStatus.FAILED
        if not result.evidence:
            result.evidence.append(
                Evidence(
                    statement=f"Task failed: {task.description}",
                    source=f"result:{result.id}",
                    confidence=1.0,
                    contradicts=[task.id, task.goal_id or ""],
                    metadata={"error": result.error},
                )
            )
        plan.evidence.extend(result.evidence)
        self._store(result)
        if revise_on_failure:
            return self.revise_plan(plan, task.id, result.error or "operation returned failure")
        self._mark_goal_failed_if_unreplaced(plan, task)
        return None

    def revise_plan(self, plan: Plan, failed_task_id: str, reason: str) -> PlanRevision | None:
        """Replace a failed operation with diagnosis plus an untried strategy."""
        failed = plan.get_task(failed_task_id)
        if failed.status != WorkStatus.FAILED:
            raise ValueError("Only a failed task can trigger plan revision")
        if len(plan.revisions) >= self.max_revisions:
            plan.status = PlanStatus.FAILED
            self._mark_goal_failed_if_unreplaced(plan, failed)
            return None

        attempted = set(failed.attempted_strategies)
        alternative = next(
            (candidate for candidate in failed.candidate_actions if candidate.name not in attempted),
            None,
        )
        if alternative is None:
            alternative = CandidateAction(
                name="diagnose_then_retry",
                description=f"Diagnose '{reason}' before retrying {failed.description}",
                estimated_cost=0.65,
                risk=0.25,
                utility=0.7,
            )

        diagnosis = Task(
            description=f"Diagnose failure: {reason}",
            action="diagnose",
            target=failed.description,
            goal_id=failed.goal_id,
            dependencies=list(failed.dependencies),
            constraints=list(failed.constraints),
            candidate_actions=[
                CandidateAction(
                    "inspect_failure_evidence",
                    "Inspect the failed result and validate prerequisites",
                    estimated_cost=0.3,
                    risk=0.05,
                    utility=0.9,
                )
            ],
            selected_strategy="inspect_failure_evidence",
            requires=["failed_result", "task_context"],
            produces=["failure_cause", "recovery_preconditions"],
            metadata={"revision_for": failed.id},
        )
        diagnosis.subtasks = self.planner.identify_subtasks(diagnosis)
        retry = Task(
            description=f"Retry {failed.description} using {alternative.name}",
            action=failed.action,
            target=failed.target,
            goal_id=failed.goal_id,
            dependencies=[diagnosis.id],
            constraints=list(failed.constraints),
            candidate_actions=list(failed.candidate_actions),
            selected_strategy=alternative.name,
            attempted_strategies=list(failed.attempted_strategies),
            requires=list(failed.requires),
            produces=list(failed.produces),
            attempt_count=failed.attempt_count,
            replaces_task_id=failed.id,
            metadata={**failed.metadata, "revision_reason": reason},
        )
        retry.subtasks = self.planner.identify_subtasks(retry)
        failed.replaced_by_task_id = retry.id

        failed_index = plan.tasks.index(failed)
        plan.tasks[failed_index + 1:failed_index + 1] = [diagnosis, retry]
        redirected: list[str] = []
        for task in plan.tasks:
            if task.id in {failed.id, diagnosis.id, retry.id}:
                continue
            if failed.id in task.dependencies:
                task.dependencies = [retry.id if dependency == failed.id else dependency for dependency in task.dependencies]
                redirected.append(task.id)

        if failed.goal_id:
            goal = plan.get_goal(failed.goal_id)
            goal.task_ids.extend([diagnosis.id, retry.id])
            goal.status = WorkStatus.IN_PROGRESS
        self._revise_causal_relations(plan, failed, diagnosis, retry)
        for relation in plan.causal_relations:
            self.causal_reasoner.add_relation(relation)
        revision = PlanRevision(
            failed_task_id=failed.id,
            reason=reason,
            added_task_ids=[diagnosis.id, retry.id],
            removed_dependency_ids=redirected,
            selected_alternative=alternative.name,
            revision_number=len(plan.revisions) + 1,
        )
        plan.revisions.append(revision)
        plan.status = PlanStatus.REVISED
        self._store(revision)
        return revision

    revise_on_failure = revise_plan

    def check_goal_completion(self, plan: Plan, goal_id: str | None = None) -> bool:
        """Evaluate completion from active tasks, child goals, and constraints."""
        for goal in reversed(plan.goals):
            self._update_goal_status(plan, goal)
        if plan.goals and all(goal.status == WorkStatus.COMPLETED for goal in plan.goals):
            plan.status = PlanStatus.COMPLETED
        elif any(goal.status == WorkStatus.FAILED for goal in plan.goals if goal.parent_goal_id is None):
            plan.status = PlanStatus.FAILED
        selected = plan.get_goal(goal_id) if goal_id else plan.primary_goal
        return bool(selected and selected.status == WorkStatus.COMPLETED)

    is_goal_complete = check_goal_completion
    check_goal_completed = check_goal_completion

    def confirm_assumption(self, plan: Plan, assumption_id: str, evidence: Evidence | None = None) -> None:
        assumption = next(item for item in plan.assumptions if item.id == assumption_id)
        assumption.status = AssumptionStatus.CONFIRMED
        assumption.confidence = 1.0
        if evidence:
            assumption.evidence_ids.append(evidence.id)
            plan.evidence.append(evidence)

    def reject_assumption(self, plan: Plan, assumption_id: str, evidence: Evidence | None = None) -> None:
        assumption = next(item for item in plan.assumptions if item.id == assumption_id)
        assumption.status = AssumptionStatus.REJECTED
        assumption.confidence = 0.0
        if evidence:
            assumption.evidence_ids.append(evidence.id)
            plan.evidence.append(evidence)

    @staticmethod
    def _base_rules() -> list[InferenceRule]:
        return [
            InferenceRule(("request received",), "a goal must be identified", "request_implies_goal"),
            InferenceRule(("a goal must be identified", "actions identified"), "a plan can be created", "goal_and_actions_enable_plan"),
            InferenceRule(("task dependencies recorded",), "task execution order is constrained", "dependencies_imply_order"),
            InferenceRule(("success criteria recorded", "operation results observable"), "goal completion can be checked", "observable_completion"),
        ]

    @staticmethod
    def _facts_from_plan(plan: Plan) -> set[str]:
        facts = {"request received", "a goal must be identified"}
        if plan.goals:
            facts.add("goals identified")
        if plan.tasks:
            facts.add("actions identified")
        if any(task.dependencies for task in plan.tasks):
            facts.add("task dependencies recorded")
        if any(goal.success_criteria for goal in plan.goals):
            facts.add("success criteria recorded")
        if any("results can be observed" in item.statement.lower() for item in plan.assumptions):
            facts.add("operation results observable")
        if plan.constraints:
            facts.add("constraints recorded")
        if plan.assumptions:
            facts.add("assumptions recorded")
        return facts

    @staticmethod
    def _request_text(request: str | Any) -> tuple[str, Any]:
        if isinstance(request, str):
            text = request
            semantic = None
        elif isinstance(request, Mapping):
            text = str(request.get("raw_text") or request.get("text") or request.get("request") or "")
            semantic = request
        else:
            text = str(getattr(request, "raw_text", "") or getattr(request, "text", ""))
            semantic = request
        if not text.strip():
            raise ValueError("A reasoning request must contain text")
        return text, semantic

    @staticmethod
    def _default_language_analyzer() -> AnalyzerPort | None:
        try:
            from rafig.language.language_understanding import LanguageAnalyzer

            return LanguageAnalyzer()
        except ImportError:  # pragma: no cover - allows the module to stay independently reusable
            return None

    @staticmethod
    def _default_semantic_analyzer() -> AnalyzerPort | None:
        try:
            from rafig.language.semantic_representation import SemanticAnalyzer

            return SemanticAnalyzer()
        except ImportError:  # pragma: no cover
            return None

    @staticmethod
    def _evaluate_constraints(plan: Plan, result: Result) -> None:
        context: dict[str, Any] = dict(result.metadata)
        if isinstance(result.output, Mapping):
            context.update(result.output)
        for constraint in plan.constraints:
            constraint.evaluate(context)

    def _update_goal_status(self, plan: Plan, goal: Goal) -> None:
        active_tasks = plan.tasks_for_goal(goal.id)
        child_goals = [plan.get_goal(item) for item in goal.child_goal_ids]
        hard_constraint_failed = any(
            constraint.kind == ConstraintKind.HARD
            and constraint.satisfied is False
            and (not constraint.applies_to or goal.id in constraint.applies_to)
            for constraint in plan.constraints
        )
        if hard_constraint_failed:
            goal.status = WorkStatus.FAILED
            return
        if any(task.status == WorkStatus.FAILED for task in active_tasks):
            goal.status = WorkStatus.FAILED
            return
        tasks_complete = bool(active_tasks) and all(task.status == WorkStatus.COMPLETED for task in active_tasks)
        children_complete = all(child.status == WorkStatus.COMPLETED for child in child_goals)
        if tasks_complete and children_complete:
            goal.status = WorkStatus.COMPLETED
            goal.evidence_ids = [
                evidence.id
                for result in plan.results
                if result.success
                and plan.get_task(result.task_id).goal_id == goal.id
                for evidence in result.evidence
            ]
        elif any(task.status == WorkStatus.IN_PROGRESS for task in active_tasks) or any(
            child.status in {WorkStatus.IN_PROGRESS, WorkStatus.COMPLETED} for child in child_goals
        ):
            goal.status = WorkStatus.IN_PROGRESS
        else:
            goal.status = WorkStatus.PENDING

    @staticmethod
    def _mark_goal_failed_if_unreplaced(plan: Plan, task: Task) -> None:
        if task.replaced_by_task_id is None and task.goal_id:
            plan.get_goal(task.goal_id).status = WorkStatus.FAILED
            plan.status = PlanStatus.FAILED

    @staticmethod
    def _revise_causal_relations(plan: Plan, failed: Task, diagnosis: Task, retry: Task) -> None:
        for relation in plan.causal_relations:
            if relation.effect == failed.id:
                relation.effect = diagnosis.id
                relation.conditions = [
                    f"{dependency}:completed" for dependency in diagnosis.dependencies
                ]
            if relation.cause == failed.id:
                relation.cause = retry.id
                relation.conditions = [f"{retry.id}:completed"]
        plan.causal_relations.extend(
            [
                CausalRelation(
                    cause=f"{failed.id}:failed",
                    effect=diagnosis.id,
                    relation="triggers",
                ),
                CausalRelation(
                    cause=diagnosis.id,
                    effect=retry.id,
                    relation="enables",
                    conditions=[f"{diagnosis.id}:completed"],
                ),
            ]
        )

    def _store(self, item: Any) -> None:
        if self.memory is not None:
            self.memory.store(item)

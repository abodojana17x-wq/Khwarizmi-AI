"""Explicit data structures used by RAFIQ's symbolic reasoning engine.

The models in this module deliberately contain data, state transitions, and
small validation helpers only.  They do not call an AI model or execute a
plan.  Keeping them independent makes it possible for later memory and Python
components to store or consume plans without depending on the planner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping
from uuid import uuid4


def _identifier(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class WorkStatus(str, Enum):
    """Lifecycle shared by goals, tasks, and subtasks."""

    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    REVISED = "revised"
    COMPLETED = "completed"
    FAILED = "failed"


class ConstraintKind(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class HypothesisStatus(str, Enum):
    UNTESTED = "untested"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class AssumptionStatus(str, Enum):
    ASSUMED = "assumed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(slots=True)
class Evidence:
    """An observation used to support or contradict propositions."""

    statement: str
    source: str = "reasoning"
    confidence: float = 1.0
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _identifier("evidence"))

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass(slots=True)
class Hypothesis:
    """A proposition whose truth has not necessarily been established."""

    statement: str
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _identifier("hypothesis"))

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass(slots=True)
class Assumption:
    """An explicit, revisable fact adopted while information is missing."""

    statement: str
    reason: str = "required to make a plan"
    status: AssumptionStatus = AssumptionStatus.ASSUMED
    confidence: float = 0.5
    evidence_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _identifier("assumption"))

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass(slots=True)
class Constraint:
    """A hard or soft condition that a plan or result should satisfy.

    ``key``, ``operator``, and ``value`` form an optional machine-checkable
    condition.  A prose-only constraint remains tracked with ``satisfied`` set
    to ``None`` until another component supplies evidence.
    """

    description: str
    kind: ConstraintKind = ConstraintKind.HARD
    source: str = "request"
    key: str | None = None
    operator: str = "equals"
    value: Any = None
    applies_to: list[str] = field(default_factory=list)
    satisfied: bool | None = None
    id: str = field(default_factory=lambda: _identifier("constraint"))

    def evaluate(self, context: Mapping[str, Any]) -> bool | None:
        if self.key is None or self.key not in context:
            return self.satisfied
        actual = context[self.key]
        if self.operator == "equals":
            result = actual == self.value
        elif self.operator == "not_equals":
            result = actual != self.value
        elif self.operator == "contains":
            try:
                result = self.value in actual
            except TypeError:
                result = False
        elif self.operator == "excludes":
            try:
                result = self.value not in actual
            except TypeError:
                result = True
        elif self.operator == "exists":
            result = actual is not None
        else:
            raise ValueError(f"Unsupported constraint operator: {self.operator}")
        self.satisfied = bool(result)
        return self.satisfied


@dataclass(slots=True)
class CandidateAction:
    """One possible strategy for performing an action."""

    name: str
    description: str = ""
    expected_effects: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    estimated_cost: float = 0.5
    risk: float = 0.5
    utility: float = 0.5
    feasibility: float = 1.0
    constraint_satisfaction: float = 1.0
    evidence_support: float = 0.0
    score: float | None = None

    def __post_init__(self) -> None:
        for attribute in (
            "estimated_cost",
            "risk",
            "utility",
            "feasibility",
            "constraint_satisfaction",
            "evidence_support",
        ):
            value = max(0.0, min(1.0, float(getattr(self, attribute))))
            setattr(self, attribute, value)


@dataclass(slots=True)
class ActionComparison:
    """The scored outcome of comparing several candidate actions."""

    ranked_options: list[CandidateAction]
    selected_action: str | None
    rationale: str


@dataclass(slots=True)
class Subtask:
    """A small operation that contributes to a parent task."""

    description: str
    action: str = "perform"
    target: str = ""
    parent_task_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    status: WorkStatus = WorkStatus.PENDING
    result_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _identifier("subtask"))


@dataclass(slots=True)
class Task:
    """An executable unit in a plan; execution is delegated to later phases."""

    description: str
    action: str = "perform"
    target: str = ""
    goal_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    subtasks: list[Subtask] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    candidate_actions: list[CandidateAction] = field(default_factory=list)
    selected_strategy: str | None = None
    attempted_strategies: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    status: WorkStatus = WorkStatus.PENDING
    result_ids: list[str] = field(default_factory=list)
    attempt_count: int = 0
    replaces_task_id: str | None = None
    replaced_by_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _identifier("task"))

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            WorkStatus.COMPLETED,
            WorkStatus.FAILED,
            WorkStatus.SKIPPED,
        }


@dataclass(slots=True)
class Goal:
    """A desired state and the criteria used to recognize completion."""

    description: str
    success_criteria: list[str] = field(default_factory=list)
    priority: int = 1
    parent_goal_id: str | None = None
    child_goal_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    status: WorkStatus = WorkStatus.PENDING
    evidence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _identifier("goal"))

    @property
    def is_complete(self) -> bool:
        return self.status == WorkStatus.COMPLETED


@dataclass(slots=True)
class Result:
    """Observed outcome of trying a task or subtask."""

    task_id: str
    success: bool
    output: Any = None
    error: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _identifier("result"))


@dataclass(slots=True)
class CausalRelation:
    """A directed causal assertion used for prediction and explanation."""

    cause: str
    effect: str
    relation: str = "enables"
    conditions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _identifier("cause"))

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass(slots=True)
class PlanRevision:
    """Audit record describing why and how a plan changed."""

    failed_task_id: str
    reason: str
    added_task_ids: list[str] = field(default_factory=list)
    removed_dependency_ids: list[str] = field(default_factory=list)
    selected_alternative: str | None = None
    revision_number: int = 1
    id: str = field(default_factory=lambda: _identifier("revision"))


@dataclass(slots=True)
class Plan:
    """A dependency-aware, inspectable plan."""

    request: str
    goals: list[Goal] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    causal_relations: list[CausalRelation] = field(default_factory=list)
    action_comparisons: list[ActionComparison] = field(default_factory=list)
    revisions: list[PlanRevision] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _identifier("plan"))

    @property
    def primary_goal(self) -> Goal | None:
        return next((goal for goal in self.goals if goal.parent_goal_id is None), None)

    @property
    def progress(self) -> float:
        active = [task for task in self.tasks if task.replaced_by_task_id is None]
        if not active:
            return 0.0
        completed = sum(task.status == WorkStatus.COMPLETED for task in active)
        return completed / len(active)

    def get_goal(self, goal_id: str) -> Goal:
        for goal in self.goals:
            if goal.id == goal_id:
                return goal
        raise KeyError(f"Unknown goal: {goal_id}")

    def get_task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"Unknown task: {task_id}")

    def ready_tasks(self) -> list[Task]:
        """Return pending tasks whose active dependencies have completed."""
        status_by_id = {task.id: task.status for task in self.tasks}
        ready: list[Task] = []
        for task in self.tasks:
            if task.status not in {WorkStatus.PENDING, WorkStatus.READY}:
                continue
            if all(status_by_id.get(dep) == WorkStatus.COMPLETED for dep in task.dependencies):
                task.status = WorkStatus.READY
                ready.append(task)
        return ready

    def next_tasks(self) -> list[Task]:
        return self.ready_tasks()

    def tasks_for_goal(self, goal_id: str, include_replaced: bool = False) -> list[Task]:
        return [
            task
            for task in self.tasks
            if task.goal_id == goal_id and (include_replaced or task.replaced_by_task_id is None)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly structured representation of this plan."""
        return _to_plain_data(self)


@dataclass(slots=True)
class ReasoningReport:
    """Complete reasoning output while keeping the plan as the main product."""

    request: str
    plan: Plan
    inferred_facts: set[str] = field(default_factory=set)
    semantic_input: Any = None
    language_input: Any = None

    @property
    def goals(self) -> list[Goal]:
        return self.plan.goals

    @property
    def tasks(self) -> list[Task]:
        return self.plan.tasks

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_data(self)


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    return value


def unique_descriptions(items: Iterable[Constraint | Assumption]) -> list[Any]:
    """Deduplicate description-bearing records while preserving order."""
    seen: set[str] = set()
    unique: list[Any] = []
    for item in items:
        key = " ".join(item.description.lower().split()) if isinstance(item, Constraint) else " ".join(item.statement.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

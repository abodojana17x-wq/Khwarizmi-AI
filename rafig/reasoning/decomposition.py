"""Rule-based request decomposition for the reasoning engine.

The decomposer uses grammar-like action recognition and clause boundaries.  It
is intentionally small and extensible: adding an action changes procedural
knowledge rather than adding a fixed response to a whole request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import Assumption, Constraint, ConstraintKind, Goal, unique_descriptions


@dataclass(slots=True)
class ActionMention:
    action: str
    surface: str
    target: str
    start: int
    end: int
    modifier: str = ""


class RequestDecomposer:
    """Extract goals, actions, constraints, and assumptions from a request."""

    ACTION_TERMS: dict[str, tuple[str, ...]] = {
        "create": ("create", "build", "make", "develop", "implement", "write", "generate", "أنشئ", "انشئ", "اعمل", "أعمل", "اكتب", "نفذ"),
        "repair": ("repair", "fix", "debug", "correct", "اصلح", "أصلح", "تصلح", "tesala7", "salah"),
        "read": ("read", "reads", "load", "loads", "scan", "scans", "اقرأ", "يقرا", "يقرأ"),
        "group": ("group", "groups", "classify", "classifies", "organize", "organizes", "sort", "sorts", "جمع", "صنف", "رتب"),
        "move": ("move", "moves", "relocate", "relocates", "نقل", "انقل", "ينقل"),
        "copy": ("copy", "copies", "duplicate", "duplicates", "انسخ", "ينسخ"),
        "delete": ("delete", "deletes", "remove", "removes", "احذف", "يمسح"),
        "analyze": ("analyze", "analyse", "inspect", "examine", "حلل", "افحص"),
        "validate": ("validate", "verify", "check", "test", "تحقق", "اختبر"),
        "save": ("save", "store", "persist", "احفظ", "خزن"),
        "convert": ("convert", "transform", "translate", "حول", "ترجم"),
        "calculate": ("calculate", "compute", "count", "sum", "احسب", "عد"),
        "display": ("display", "show", "print", "report", "اعرض", "اطبع"),
    }

    _WRAPPER_TARGETS = re.compile(r"\b(?:program|script|application|app|tool|code|برنامج|تطبيق|كود)\b", re.IGNORECASE)
    _PRONOUN_PREFIX = re.compile(r"^(?:them|it|these|those|each one|ها|هم)\b\s*", re.IGNORECASE)

    def __init__(self) -> None:
        term_to_action = {
            term.lower(): action
            for action, terms in self.ACTION_TERMS.items()
            for term in terms
        }
        expression = "|".join(re.escape(term) for term in sorted(term_to_action, key=len, reverse=True))
        self._action_pattern = re.compile(rf"(?<!\w)({expression})(?!\w)", re.IGNORECASE)
        self._term_to_action = term_to_action

    def extract_actions(self, text: str) -> list[ActionMention]:
        normalized = " ".join(text.strip().split())
        matches = list(self._action_pattern.finditer(normalized))
        mentions: list[ActionMention] = []
        antecedent = ""
        for index, match in enumerate(matches):
            segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            target = normalized[match.end():segment_end]
            target = re.sub(r"^[\s:,-]*(?:to\s+)?", "", target, flags=re.IGNORECASE)
            target = re.sub(r"\s*(?:\band\b|\bthen\b|و)\s*$", "", target, flags=re.IGNORECASE)
            target = re.sub(r"\s*[,;]\s*$", "", target)
            target = re.sub(r"^(?:a|an|the)\s+", "", target, flags=re.IGNORECASE)
            target = re.sub(r"\s+\b(?:that|which|who)\s*$", "", target, flags=re.IGNORECASE)
            target = target.strip(" .!?،؛")

            pronoun = self._PRONOUN_PREFIX.match(target)
            if pronoun and antecedent:
                target = f"{antecedent} {target[pronoun.end():]}".strip()
            action = self._term_to_action[match.group(1).lower()]
            modifier = self._extract_modifier(target)
            mentions.append(
                ActionMention(
                    action=action,
                    surface=match.group(1),
                    target=target or "requested item",
                    start=match.start(),
                    end=match.end(),
                    modifier=modifier,
                )
            )
            if not pronoun:
                candidate = self._antecedent_from_target(target)
                if candidate:
                    antecedent = candidate
        return mentions

    def decompose_goals(self, text: str, actions: Iterable[ActionMention] | None = None) -> list[Goal]:
        normalized = " ".join(text.strip().split()).strip()
        if not normalized:
            raise ValueError("Cannot reason about an empty request")
        actions = list(actions if actions is not None else self.extract_actions(normalized))
        primary = Goal(
            description=normalized.rstrip(".!?"),
            success_criteria=[
                f"The requested outcome is observable: {normalized.rstrip('.!?')}"
            ],
            priority=1,
        )
        goals = [primary]
        operational = self._operational_actions(actions)
        if len(operational) > 1:
            for mention in operational:
                child = Goal(
                    description=self.describe_action(mention.action, mention.target),
                    success_criteria=[f"{self.describe_action(mention.action, mention.target)} succeeds"],
                    priority=primary.priority,
                    parent_goal_id=primary.id,
                )
                primary.child_goal_ids.append(child.id)
                goals.append(child)
                primary.success_criteria.extend(child.success_criteria)
        return goals

    def extract_constraints(self, text: str, semantic_input: Any = None) -> list[Constraint]:
        lowered = text.lower()
        constraints: list[Constraint] = []

        semantic_constraints = getattr(semantic_input, "constraints", None)
        if semantic_constraints is None and isinstance(semantic_input, dict):
            semantic_constraints = semantic_input.get("constraints", [])
        for item in semantic_constraints or []:
            value = str(item).replace("_", " ")
            if str(item).lower() == "produce_python":
                constraints.append(
                    Constraint(
                        "Implementation language is Python",
                        key="implementation_language",
                        value="python",
                        source="semantic representation",
                    )
                )
            else:
                constraints.append(Constraint(value, source="semantic representation"))

        if re.search(r"\bpython\b|بايثون", lowered):
            constraints.append(
                Constraint(
                    "Implementation language is Python",
                    key="implementation_language",
                    value="python",
                )
            )
        if re.search(r"\bby\s+(?:file\s+)?extension\b|حسب\s+الامتداد", lowered):
            constraints.append(
                Constraint(
                    "Files are grouped using their extension",
                    key="grouping_key",
                    value="extension",
                )
            )
        if re.search(r"\binto\s+(?:separate\s+)?folders?\b|إلى\s+مجلد", lowered):
            constraints.append(
                Constraint(
                    "Grouped items are placed into folders",
                    key="destination_type",
                    value="folder",
                )
            )

        patterns = (
            (r"\bmust\s+([^,.!?;]+)", ConstraintKind.HARD, "Must {value}"),
            (r"\bshould\s+([^,.!?;]+)", ConstraintKind.SOFT, "Should {value}"),
            (r"\bwithout\s+([^,.!?;]+)", ConstraintKind.HARD, "Must not use {value}"),
            (r"\b(?:do not|don't|never)\s+([^,.!?;]+)", ConstraintKind.HARD, "Must not {value}"),
            (r"\bonly\s+([^,.!?;]+)", ConstraintKind.HARD, "Only {value}"),
        )
        for pattern, kind, template in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1).strip()
                constraints.append(Constraint(template.format(value=value), kind=kind))
        return unique_descriptions(constraints)

    def identify_assumptions(self, text: str, actions: Iterable[ActionMention]) -> list[Assumption]:
        lowered = text.lower()
        action_names = {item.action for item in actions}
        assumptions: list[Assumption] = []
        if action_names & {"read", "move", "copy", "delete"}:
            if not re.search(r"\b(?:from|source|input|path|directory|current directory)\b|مسار|مجلد المصدر", lowered):
                assumptions.append(
                    Assumption(
                        "A source location will be supplied before execution",
                        "the request mentions input items but no unambiguous source location",
                        confidence=0.45,
                    )
                )
        if "move" in action_names and not re.search(r"\b(?:overwrite|collision|duplicate|existing file|replace)\b|تعارض|استبدال", lowered):
            assumptions.append(
                Assumption(
                    "Existing destination-name collisions will not be overwritten without a policy",
                    "the collision policy is unspecified",
                    confidence=0.4,
                )
            )
        if "read" in action_names and "file" in lowered and not re.search(r"\brecursive(?:ly)?\b|subfolders?|subdirectories", lowered):
            assumptions.append(
                Assumption(
                    "Only the selected directory level is processed unless recursion is requested",
                    "recursive traversal is unspecified",
                    confidence=0.5,
                )
            )
        if actions:
            assumptions.append(
                Assumption(
                    "Operation results can be observed for completion checks",
                    "the engine needs evidence to determine whether goals are complete",
                    confidence=0.7,
                )
            )
        return unique_descriptions(assumptions)

    def is_wrapper_action(self, mention: ActionMention, actions: Iterable[ActionMention]) -> bool:
        return mention.action == "create" and len(list(actions)) > 1 and bool(self._WRAPPER_TARGETS.search(mention.target))

    def _operational_actions(self, actions: list[ActionMention]) -> list[ActionMention]:
        return [mention for mention in actions if not self.is_wrapper_action(mention, actions)]

    @staticmethod
    def describe_action(action: str, target: str) -> str:
        labels = {
            "create": "Create",
            "repair": "Repair",
            "read": "Read",
            "group": "Group",
            "move": "Move",
            "copy": "Copy",
            "delete": "Delete",
            "analyze": "Analyze",
            "validate": "Validate",
            "save": "Save",
            "convert": "Convert",
            "calculate": "Calculate",
            "display": "Display",
            "design": "Define requirements and structure for",
            "verify": "Verify completion of",
            "understand": "Clarify and understand",
        }
        return f"{labels.get(action, action.capitalize())} {target}".strip()

    @staticmethod
    def _extract_modifier(target: str) -> str:
        match = re.search(r"\b(by|into|from|using|with|without)\s+(.+)$", target, re.IGNORECASE)
        return match.group(0) if match else ""

    @staticmethod
    def _antecedent_from_target(target: str) -> str:
        cleaned = re.split(r"\b(?:by|into|from|using|with|without)\b", target, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = re.sub(r"^(?:all|each|the|a|an)\s+", "", cleaned.strip(), flags=re.IGNORECASE)
        if not cleaned or len(cleaned.split()) > 4:
            return ""
        return cleaned

from __future__ import annotations

from typing import Any, Protocol
from ..models import Dataset, GoalSpec
from .prompts import GOAL_INTERPRETER_SYSTEM_PROMPT, build_goal_interpreter_user_prompt


class JSONCompleter(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


def _ensure_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} debe ser una lista.")
    return value


def _ensure_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} debe ser un objeto JSON.")
    return value


def validate_goal_spec(goal_spec: dict[str, Any], dataset: Dataset) -> GoalSpec:
    role_id = goal_spec.get("role_id")
    if role_id not in dataset.roles:
        raise ValueError("role_id no existe en el dataset")

    unknown = [str(item) for item in _ensure_list(goal_spec.get("unknown_skill_mentions"), "unknown_skill_mentions")]

    valid_target_skills: set[str] = set()
    for skill_id in _ensure_list(goal_spec.get("target_skill_ids"), "target_skill_ids"):
        skill_id = str(skill_id)
        if skill_id in dataset.skills:
            valid_target_skills.add(skill_id)
        else:
            unknown.append(skill_id)

    valid_initial_skills: set[str] = set()
    for skill_id in _ensure_list(goal_spec.get("initial_skill_ids"), "initial_skill_ids"):
        skill_id = str(skill_id)
        if skill_id in dataset.skills:
            valid_initial_skills.add(skill_id)
        else:
            unknown.append(skill_id)

    valid_mentioned_skills: set[str] = set()
    for skill_id in _ensure_list(goal_spec.get("mentioned_skill_ids"), "mentioned_skill_ids"):
        skill_id = str(skill_id)
        if skill_id in dataset.skills:
            valid_mentioned_skills.add(skill_id)
        else:
            unknown.append(skill_id)

    try:
        confidence = float(goal_spec.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return GoalSpec(
        role_id=str(role_id),
        target_skill_ids=valid_target_skills,
        initial_skill_ids=valid_initial_skills,
        mentioned_skill_ids=valid_mentioned_skills,
        constraints=_ensure_dict(goal_spec.get("constraints"), "constraints"),
        ignored_constraints=[
            str(item) for item in _ensure_list(goal_spec.get("ignored_constraints"), "ignored_constraints")
        ],
        unknown_skill_mentions=unknown,
        confidence=confidence,
    )


def interpret_goal(
    user_text: str,
    dataset: Dataset,
    llm_client: JSONCompleter,
) -> GoalSpec:
    user_prompt = build_goal_interpreter_user_prompt(user_text, dataset)
    raw_goal_spec = llm_client.complete_json(
        GOAL_INTERPRETER_SYSTEM_PROMPT,
        user_prompt,
    )
    return validate_goal_spec(raw_goal_spec, dataset)

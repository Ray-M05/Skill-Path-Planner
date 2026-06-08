"""Tests del rol a medida (custom_) como fallback del interpreter LLM."""
from __future__ import annotations

import pytest

from src.dataset.loader import load_dataset
from src.llm.evaluator import build_mock_evaluation_response
from src.llm.interpreter import resolve_role, validate_goal_spec
from src.llm.prompts import build_plan_evaluator_user_prompt
from src.models import GoalSpec, PlanResult, StudentProfile
from src.scoring import compute_score_with_breakdown


def _raw_custom_goal(**overrides) -> dict:
    raw = {
        "role_id": "custom_disenador_de_videojuegos",
        "role_name": "Disenador de videojuegos",
        "target_skill_ids": ["skill_python_intermediate", "skill_sql_basic"],
        "initial_skill_ids": [],
        "mentioned_skill_ids": [],
        "constraints": {},
        "ignored_constraints": [],
        "unknown_skill_mentions": [],
        "confidence": 0.4,
    }
    raw.update(overrides)
    return raw


def _make_plan(reached: set[str]) -> PlanResult:
    return PlanResult(
        planner_name="astar",
        course_ids=[],
        reached_skills=reached,
        total_weeks=10,
        total_difficulty=0.5,
        expanded_nodes=5,
        max_frontier_size=3,
        runtime_seconds=0.01,
        valid=True,
    )


def test_validate_goal_spec_accepts_custom_role() -> None:
    dataset = load_dataset("data")

    goal = validate_goal_spec(_raw_custom_goal(), dataset)

    assert goal.role_id == "custom_disenador_de_videojuegos"
    assert goal.role_name == "Disenador de videojuegos"
    assert goal.target_skill_ids == {"skill_python_intermediate", "skill_sql_basic"}


def test_validate_goal_spec_custom_requires_target_skills() -> None:
    dataset = load_dataset("data")

    with pytest.raises(ValueError, match="custom_"):
        validate_goal_spec(_raw_custom_goal(target_skill_ids=[]), dataset)


def test_validate_goal_spec_rejects_unknown_role_without_custom_prefix() -> None:
    dataset = load_dataset("data")

    with pytest.raises(ValueError, match="role_id"):
        validate_goal_spec(
            _raw_custom_goal(role_id="role_inexistente", role_name=None),
            dataset,
        )


def test_resolve_role_returns_dataset_role_for_existing() -> None:
    dataset = load_dataset("data")
    goal = GoalSpec(
        role_id="role_data_analyst",
        target_skill_ids={"skill_sql_basic"},
        initial_skill_ids=set(),
        mentioned_skill_ids=set(),
        constraints={},
        ignored_constraints=[],
        unknown_skill_mentions=[],
        confidence=0.9,
    )

    role = resolve_role(goal, dataset)

    assert role is dataset.roles["role_data_analyst"]


def test_resolve_role_builds_synthetic_role_for_custom() -> None:
    dataset = load_dataset("data")
    goal = validate_goal_spec(_raw_custom_goal(), dataset)

    role = resolve_role(goal, dataset)

    assert role.id == "custom_disenador_de_videojuegos"
    assert role.name == "Disenador de videojuegos"
    assert role.required_skills == frozenset({"skill_python_intermediate", "skill_sql_basic"})
    assert role.recommended_skills == frozenset()


def test_resolve_role_deslugs_name_when_role_name_missing() -> None:
    dataset = load_dataset("data")
    goal = validate_goal_spec(_raw_custom_goal(role_name=None), dataset)

    role = resolve_role(goal, dataset)

    # custom_disenador_de_videojuegos -> "Disenador de videojuegos"
    assert role.name == "Disenador de videojuegos"


def test_scoring_with_custom_role_gives_full_required_coverage() -> None:
    dataset = load_dataset("data")
    goal = validate_goal_spec(_raw_custom_goal(), dataset)
    role = resolve_role(goal, dataset)
    profile = StudentProfile(
        id="p",
        initial_skills=frozenset(),
        max_weeks=72,
        max_weekly_hours=10,
        risk_tolerance=0.5,
    )
    plan = _make_plan(reached={"skill_python_intermediate", "skill_sql_basic"})

    _, breakdown = compute_score_with_breakdown(plan, role, profile)

    # required = w_required * 1.0 (sin MC ni LLM => w_required = 0.65); recomendadas vacias => 0
    assert breakdown["required"] == 0.65
    assert breakdown["recommended"] == 0.0


def test_evaluator_prompt_strips_custom_prefix() -> None:
    dataset = load_dataset("data")
    goal = validate_goal_spec(_raw_custom_goal(), dataset)
    role = resolve_role(goal, dataset)
    plan = _make_plan(reached={"skill_python_intermediate", "skill_sql_basic"})

    prompt = build_plan_evaluator_user_prompt(plan, role, goal, dataset)

    assert "custom_" not in prompt
    assert "Disenador de videojuegos" in prompt


def test_mock_evaluation_runs_with_custom_role() -> None:
    dataset = load_dataset("data")
    goal = validate_goal_spec(_raw_custom_goal(), dataset)
    role = resolve_role(goal, dataset)
    profile = StudentProfile(
        id="p",
        initial_skills=frozenset(),
        max_weeks=72,
        max_weekly_hours=10,
        risk_tolerance=0.5,
    )
    plan = _make_plan(reached={"skill_python_intermediate", "skill_sql_basic"})

    evaluation = build_mock_evaluation_response(plan, role, profile, dataset)

    assert evaluation["goal_alignment"] == 1.0
    assert 0.0 <= evaluation["global_quality"] <= 1.0

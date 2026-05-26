from __future__ import annotations

import pytest

from src.dataset.loader import load_dataset
from src.llm.evaluator import (
    build_mock_evaluation_response,
    evaluate_plan_with_llm,
    validate_llm_evaluation,
)
from src.models import Course, GoalSpec, PlanResult, Role, StudentProfile

_EVAL_KEYS = {
    "goal_alignment",
    "pedagogical_coherence",
    "profile_realism",
    "non_redundancy",
    "professional_value",
    "global_quality",
    "main_strengths",
    "main_weaknesses",
    "justification",
}

_NUMERIC_FIELDS = [
    "goal_alignment",
    "pedagogical_coherence",
    "profile_realism",
    "non_redundancy",
    "professional_value",
    "global_quality",
]


def _make_profile(initial_skills: frozenset[str] | None = None) -> StudentProfile:
    return StudentProfile(
        id="test_profile",
        initial_skills=initial_skills or frozenset(),
        max_weeks=52,
        max_weekly_hours=10,
        risk_tolerance=0.5,
    )


def _make_role(
    role_id: str = "role_test",
    required: frozenset[str] | None = None,
    recommended: frozenset[str] | None = None,
) -> Role:
    return Role(
        id=role_id,
        name="Test Role",
        required_skills=required or frozenset({"skill_a", "skill_b"}),
        recommended_skills=recommended or frozenset({"skill_c"}),
    )


def _make_plan(
    course_ids: list[str],
    reached_skills: set[str],
    valid: bool = True,
) -> PlanResult:
    return PlanResult(
        planner_name="test",
        course_ids=course_ids,
        reached_skills=reached_skills,
        total_weeks=10,
        total_difficulty=0.5,
        expanded_nodes=1,
        max_frontier_size=1,
        runtime_seconds=0.01,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# validate_llm_evaluation
# ---------------------------------------------------------------------------


def test_validate_llm_evaluation_accepts_valid_input() -> None:
    raw = {
        "goal_alignment": 0.9,
        "pedagogical_coherence": 0.85,
        "profile_realism": 0.95,
        "non_redundancy": 0.8,
        "professional_value": 0.7,
        "global_quality": 0.87,
        "main_strengths": ["Buena cobertura"],
        "main_weaknesses": [],
        "justification": "Plan aceptable.",
    }
    result = validate_llm_evaluation(raw)
    assert result is raw


def test_validate_llm_evaluation_rejects_out_of_range() -> None:
    raw = {
        "goal_alignment": 1.5,
        "pedagogical_coherence": 0.85,
        "profile_realism": 0.95,
        "non_redundancy": 0.8,
        "professional_value": 0.7,
        "global_quality": 0.87,
        "main_strengths": [],
        "main_weaknesses": [],
        "justification": "x",
    }
    with pytest.raises(ValueError, match="goal_alignment"):
        validate_llm_evaluation(raw)


def test_validate_llm_evaluation_rejects_non_numeric_field() -> None:
    raw = {
        "goal_alignment": "alto",
        "pedagogical_coherence": 0.8,
        "profile_realism": 0.9,
        "non_redundancy": 0.8,
        "professional_value": 0.7,
        "global_quality": 0.8,
        "main_strengths": [],
        "main_weaknesses": [],
        "justification": "x",
    }
    with pytest.raises(ValueError, match="goal_alignment"):
        validate_llm_evaluation(raw)


def test_validate_llm_evaluation_rejects_missing_justification() -> None:
    raw = {
        "goal_alignment": 0.9,
        "pedagogical_coherence": 0.85,
        "profile_realism": 0.95,
        "non_redundancy": 0.8,
        "professional_value": 0.7,
        "global_quality": 0.87,
        "main_strengths": [],
        "main_weaknesses": [],
        "justification": 42,
    }
    with pytest.raises(ValueError, match="justification"):
        validate_llm_evaluation(raw)


def test_validate_llm_evaluation_rejects_non_list_strengths() -> None:
    raw = {
        "goal_alignment": 0.9,
        "pedagogical_coherence": 0.85,
        "profile_realism": 0.95,
        "non_redundancy": 0.8,
        "professional_value": 0.7,
        "global_quality": 0.87,
        "main_strengths": "bueno",
        "main_weaknesses": [],
        "justification": "x",
    }
    with pytest.raises(ValueError, match="main_strengths"):
        validate_llm_evaluation(raw)


# ---------------------------------------------------------------------------
# build_mock_evaluation_response
# ---------------------------------------------------------------------------


def test_build_mock_evaluation_returns_expected_keys() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_data_analyst"]
    plan = _make_plan(
        course_ids=["course_python_intermediate"],
        reached_skills={"skill_python_basic", "skill_python_intermediate"},
    )

    result = build_mock_evaluation_response(plan, role, profile, dataset)

    assert set(result) == _EVAL_KEYS


def test_build_mock_evaluation_values_in_range() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_data_analyst"]
    plan = _make_plan(
        course_ids=["course_python_intermediate", "course_statistics_basic"],
        reached_skills={"skill_python_basic", "skill_python_intermediate", "skill_statistics_basic"},
    )

    result = build_mock_evaluation_response(plan, role, profile, dataset)

    for field in _NUMERIC_FIELDS:
        assert 0.0 <= result[field] <= 1.0, f"{field} = {result[field]} out of [0, 1]"


def test_build_mock_evaluation_justification_is_string() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_data_analyst"]
    plan = _make_plan(
        course_ids=["course_python_intermediate"],
        reached_skills={"skill_python_basic"},
    )

    result = build_mock_evaluation_response(plan, role, profile, dataset)

    assert isinstance(result["justification"], str)
    assert len(result["justification"]) > 0


def test_build_mock_evaluation_goal_alignment_perfect_plan() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_data_analyst"]
    plan = _make_plan(
        course_ids=list(dataset.courses.keys())[:3],
        reached_skills=set(role.required_skills),
    )

    result = build_mock_evaluation_response(plan, role, profile, dataset)

    assert result["goal_alignment"] == 1.0


def test_build_mock_evaluation_empty_plan_has_low_goal_alignment() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_data_analyst"]
    plan = _make_plan(course_ids=[], reached_skills=set())

    result = build_mock_evaluation_response(plan, role, profile, dataset)

    assert result["goal_alignment"] == 0.0


# ---------------------------------------------------------------------------
# evaluate_plan_with_llm
# ---------------------------------------------------------------------------


def test_evaluate_plan_with_llm_mock_returns_expected_keys() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    goal = GoalSpec(
        role_id=role.id,
        target_skill_ids=set(role.required_skills),
        initial_skill_ids=set(),
        mentioned_skill_ids=set(),
        constraints={"max_weeks": 52, "max_weekly_hours": 10},
        ignored_constraints=[],
        unknown_skill_mentions=[],
        confidence=0.9,
    )
    plan = _make_plan(
        course_ids=["course_python_intermediate"],
        reached_skills={"skill_python_basic", "skill_python_intermediate"},
    )

    result = evaluate_plan_with_llm(plan, role, goal, dataset, llm_client=None, use_mock=True)

    assert set(result) == _EVAL_KEYS
    for field in _NUMERIC_FIELDS:
        assert 0.0 <= result[field] <= 1.0


def test_evaluate_plan_with_llm_skips_invalid_plan() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    goal = GoalSpec(
        role_id=role.id,
        target_skill_ids=set(role.required_skills),
        initial_skill_ids=set(),
        mentioned_skill_ids=set(),
        constraints={},
        ignored_constraints=[],
        unknown_skill_mentions=[],
        confidence=0.5,
    )
    plan = _make_plan(course_ids=[], reached_skills=set(), valid=False)

    result = evaluate_plan_with_llm(plan, role, goal, dataset, llm_client=None, use_mock=True)

    assert result.get("skipped") is True


def test_evaluate_plan_with_llm_skips_plan_with_no_courses() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    goal = GoalSpec(
        role_id=role.id,
        target_skill_ids=set(role.required_skills),
        initial_skill_ids=set(),
        mentioned_skill_ids=set(),
        constraints={},
        ignored_constraints=[],
        unknown_skill_mentions=[],
        confidence=0.5,
    )
    plan = _make_plan(course_ids=[], reached_skills={"skill_python_basic"}, valid=True)

    result = evaluate_plan_with_llm(plan, role, goal, dataset, llm_client=None, use_mock=True)

    assert result.get("skipped") is True


def test_evaluate_plan_with_llm_mock_respects_null_constraints() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    goal = GoalSpec(
        role_id=role.id,
        target_skill_ids=set(role.required_skills),
        initial_skill_ids=set(),
        mentioned_skill_ids=set(),
        constraints={"max_weeks": None, "max_weekly_hours": None},
        ignored_constraints=[],
        unknown_skill_mentions=[],
        confidence=0.8,
    )
    plan = _make_plan(
        course_ids=["course_python_intermediate"],
        reached_skills={"skill_python_basic"},
    )

    result = evaluate_plan_with_llm(plan, role, goal, dataset, llm_client=None, use_mock=True)

    assert set(result) == _EVAL_KEYS

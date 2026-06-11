from __future__ import annotations

import pytest
from src.dataset.loader import load_dataset
from src.models import PlanResult, Role, StudentProfile
from src.scoring import (
    compute_final_score,
    normalize_value,
    rank_plans,
    recommended_coverage_score,
    required_coverage_score,
)


def _make_profile(max_weeks: int = 72, max_weekly_hours: int = 10) -> StudentProfile:
    return StudentProfile(
        id="test",
        initial_skills=frozenset({"skill_python_basic"}),
        max_weeks=max_weeks,
        max_weekly_hours=max_weekly_hours,
        risk_tolerance=0.5,
    )


def _make_role(required: frozenset[str], recommended: frozenset[str]) -> Role:
    return Role(
        id="role_test",
        name="Test",
        required_skills=required,
        recommended_skills=recommended,
    )


def _make_plan(
    course_ids: list[str],
    reached_skills: set[str],
    total_weeks: int = 10,
    total_difficulty: float = 0.5,
    valid: bool = True,
    monte_carlo: dict | None = None,
    llm_evaluation: dict | None = None,
    final_score: float | None = None,
) -> PlanResult:
    return PlanResult(
        planner_name="astar",
        course_ids=course_ids,
        reached_skills=reached_skills,
        total_weeks=total_weeks,
        total_difficulty=total_difficulty,
        expanded_nodes=5,
        max_frontier_size=3,
        runtime_seconds=0.01,
        valid=valid,
        monte_carlo=monte_carlo,
        llm_evaluation=llm_evaluation,
        final_score=final_score,
    )


def test_recommended_coverage_full_coverage() -> None:
    assert recommended_coverage_score({"a", "b", "c"}, frozenset({"a", "b"})) == 1.0


def test_recommended_coverage_partial() -> None:
    score = recommended_coverage_score({"a"}, frozenset({"a", "b"}))
    assert score == 0.5


def test_recommended_coverage_no_recommended() -> None:
    assert recommended_coverage_score({"a", "b"}, frozenset()) == 0.0


def test_recommended_coverage_no_overlap() -> None:
    assert recommended_coverage_score({"x"}, frozenset({"a", "b"})) == 0.0


def test_required_coverage_full() -> None:
    assert required_coverage_score({"a", "b", "c"}, frozenset({"a", "b"})) == 1.0


def test_required_coverage_partial() -> None:
    assert required_coverage_score({"a"}, frozenset({"a", "b"})) == 0.5


def test_required_coverage_empty_required_returns_one() -> None:
    assert required_coverage_score({"a"}, frozenset()) == 1.0


def test_valid_plan_score_is_non_negative_without_mc_or_llm() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = _make_profile()
    # Plan válido que solo cubre required_skills (rec_coverage puede ser 0)
    plan = _make_plan(["c1"], reached_skills=set(role.required_skills))

    score = compute_final_score(plan, role, profile)

    assert score >= 0.0


def test_normalize_value_within_range() -> None:
    assert normalize_value(5.0, 10.0) == 0.5


def test_normalize_value_clamps_to_one() -> None:
    assert normalize_value(20.0, 10.0) == 1.0


def test_normalize_value_zero_max() -> None:
    assert normalize_value(5.0, 0.0) == 0.0


def test_normalize_value_exact_max() -> None:
    assert normalize_value(10.0, 10.0) == 1.0


def test_compute_final_score_returns_float() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = _make_profile()
    plan = _make_plan(["course_python_intermediate"], reached_skills=set(role.required_skills))

    score = compute_final_score(plan, role, profile)

    assert isinstance(score, float)


def test_compute_final_score_higher_mc_raises_score() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = _make_profile()
    reached = set(role.required_skills) | set(role.recommended_skills)

    plan_low_mc = _make_plan(
        ["c1"], reached_skills=reached,
        monte_carlo={"success_probability": 0.2},
    )
    plan_high_mc = _make_plan(
        ["c1"], reached_skills=reached,
        monte_carlo={"success_probability": 0.9},
    )

    assert compute_final_score(plan_high_mc, role, profile) > compute_final_score(plan_low_mc, role, profile)


def test_compute_final_score_higher_llm_raises_score() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = _make_profile()
    reached = set(role.required_skills)

    plan_low = _make_plan(
        ["c1"], reached_skills=reached,
        llm_evaluation={"global_quality": 0.2},
    )
    plan_high = _make_plan(
        ["c1"], reached_skills=reached,
        llm_evaluation={"global_quality": 0.9},
    )

    assert compute_final_score(plan_high, role, profile) > compute_final_score(plan_low, role, profile)


def test_compute_final_score_more_weeks_lowers_score() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = _make_profile(max_weeks=72)
    reached = set(role.required_skills)

    plan_short = _make_plan(["c1"], reached_skills=reached, total_weeks=20)
    plan_long = _make_plan(["c1"], reached_skills=reached, total_weeks=60)

    assert compute_final_score(plan_short, role, profile) > compute_final_score(plan_long, role, profile)


def test_compute_final_score_skipped_mc_treated_as_zero() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = _make_profile()
    reached = set(role.required_skills)

    plan_no_mc = _make_plan(["c1"], reached_skills=reached)
    plan_skipped = _make_plan(
        ["c1"], reached_skills=reached,
        monte_carlo={"skipped": True, "reason": "invalid"},
    )

    assert compute_final_score(plan_no_mc, role, profile) == compute_final_score(plan_skipped, role, profile)


def test_compute_final_score_skipped_llm_treated_as_zero() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = _make_profile()
    reached = set(role.required_skills)

    plan_no_llm = _make_plan(["c1"], reached_skills=reached)
    plan_skipped = _make_plan(
        ["c1"], reached_skills=reached,
        llm_evaluation={"skipped": True, "reason": "invalid"},
    )

    assert compute_final_score(plan_no_llm, role, profile) == compute_final_score(plan_skipped, role, profile)

def test_rank_plans_orders_descending() -> None:
    plans = [
        _make_plan(["c1"], reached_skills=set(), final_score=0.3),
        _make_plan(["c2"], reached_skills=set(), final_score=0.8),
        _make_plan(["c3"], reached_skills=set(), final_score=0.5),
    ]

    ranked = rank_plans(plans)

    scores = [p.final_score for p in ranked]
    assert scores == [0.8, 0.5, 0.3]


def test_rank_plans_none_score_goes_last() -> None:
    plans = [
        _make_plan(["c1"], reached_skills=set(), final_score=None),
        _make_plan(["c2"], reached_skills=set(), final_score=0.4),
    ]

    ranked = rank_plans(plans)

    assert ranked[0].final_score == 0.4
    assert ranked[1].final_score is None


def test_rank_plans_single_plan() -> None:
    plans = [_make_plan(["c1"], reached_skills=set(), final_score=0.7)]

    ranked = rank_plans(plans)

    assert len(ranked) == 1
    assert ranked[0].final_score == 0.7


def test_rank_plans_integration_with_dataset() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_ml_engineer"]

    from src.planners.astar import astar_k_plans

    plans = astar_k_plans(profile.initial_skills, role.required_skills, dataset.courses, profile, k=3)
    for p in plans:
        p.final_score = compute_final_score(p, role, profile)

    ranked = rank_plans(plans)

    assert len(ranked) >= 1
    for i in range(len(ranked) - 1):
        s1 = ranked[i].final_score or -999
        s2 = ranked[i + 1].final_score or -999
        assert s1 >= s2

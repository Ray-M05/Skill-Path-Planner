import numpy as np

from src.dataset.loader import load_dataset
from src.models import Course, PlanResult, StudentProfile
from src.simulation.monte_carlo import (
    MonteCarloParams,
    attach_monte_carlo_to_plan,
    evaluate_plan_monte_carlo,
    simulate_plan_once,
)

_EXPECTED_KEYS = {
    "runs",
    "seed",
    "params",
    "success_probability",
    "success_ci_95",
    "expected_weeks",
    "weeks_p10",
    "weeks_p50",
    "weeks_p90",
    "expected_failed_courses",
    "failed_courses_std",
    "risk_score",
}


def test_evaluate_plan_monte_carlo_returns_expected_keys() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    course_ids = ["course_python_intermediate", "course_statistics_basic"]

    result = evaluate_plan_monte_carlo(course_ids, dataset.courses, profile, runs=20, seed=7)

    assert set(result) == _EXPECTED_KEYS
    assert result["runs"] == 20
    assert 0.0 <= result["success_probability"] <= 1.0


def test_monte_carlo_is_reproducible_with_fixed_seed() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    course_ids = ["course_python_intermediate", "course_statistics_basic"]

    left = evaluate_plan_monte_carlo(course_ids, dataset.courses, profile, runs=50, seed=123)
    right = evaluate_plan_monte_carlo(course_ids, dataset.courses, profile, runs=50, seed=123)

    assert left == right


def test_easy_courses_have_higher_success_probability_than_hard_courses() -> None:
    # max_failed_courses=0: cualquier fallo abandona el plan inmediatamente,
    # lo que asegura que la diferencia de p_pass entre cursos se refleje en success_probability
    params = MonteCarloParams(max_failed_courses=0)
    profile = StudentProfile(
        id="profile_test",
        initial_skills=frozenset(),
        max_weeks=100,
        max_weekly_hours=10,
        risk_tolerance=0.5,
    )
    easy_course = Course(
        id="course_easy",
        name="Easy",
        description="",
        prerequisites=frozenset(),
        outcomes=frozenset({"skill_easy"}),
        duration_weeks=1,
        weekly_hours=1,
        difficulty=0.1,
        pass_base_probability=0.98,
    )
    hard_course = Course(
        id="course_hard",
        name="Hard",
        description="",
        prerequisites=frozenset(),
        outcomes=frozenset({"skill_hard"}),
        duration_weeks=1,
        weekly_hours=10,
        difficulty=1.0,
        pass_base_probability=0.35,
    )

    easy = evaluate_plan_monte_carlo(
        ["course_easy"], {"course_easy": easy_course}, profile, runs=200, seed=3, params=params
    )
    hard = evaluate_plan_monte_carlo(
        ["course_hard"], {"course_hard": hard_course}, profile, runs=200, seed=3, params=params
    )

    assert easy["success_probability"] > hard["success_probability"]


def test_success_ci_95_bounds() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    course_ids = ["course_python_intermediate"]

    result = evaluate_plan_monte_carlo(course_ids, dataset.courses, profile, runs=100, seed=42)
    ci = result["success_ci_95"]

    assert len(ci) == 2
    assert ci[0] <= result["success_probability"] <= ci[1]
    assert 0.0 <= ci[0] <= 1.0
    assert 0.0 <= ci[1] <= 1.0


def test_percentiles_ordered() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    course_ids = ["course_python_intermediate", "course_statistics_basic"]

    result = evaluate_plan_monte_carlo(course_ids, dataset.courses, profile, runs=100, seed=5)

    assert result["weeks_p10"] <= result["weeks_p50"] <= result["weeks_p90"]


def test_simulate_plan_once_tracks_failures() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    course_ids = ["course_deep_learning_basic"]
    params = MonteCarloParams()

    result = simulate_plan_once(
        course_ids,
        dataset.courses,
        profile,
        np.random.default_rng(1),
        params,
    )

    assert {"success", "weeks_used", "failed_courses"}.issubset(result)


def test_custom_params_are_reflected_in_output() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    course_ids = ["course_python_intermediate"]
    params = MonteCarloParams(difficulty_penalty=0.30)

    result = evaluate_plan_monte_carlo(
        course_ids, dataset.courses, profile, runs=50, seed=9, params=params
    )

    assert result["params"]["difficulty_penalty"] == 0.30


def test_attach_monte_carlo_to_valid_plan() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    plan = PlanResult(
        planner_name="test",
        course_ids=["course_python_intermediate"],
        reached_skills={"skill_python_basic", "skill_python_intermediate"},
        total_weeks=6,
        total_difficulty=0.35,
        expanded_nodes=1,
        max_frontier_size=1,
        runtime_seconds=0.01,
        valid=True,
    )

    result = attach_monte_carlo_to_plan(plan, dataset.courses, profile, runs=10, seed=2)

    assert result.monte_carlo is not None
    assert result.monte_carlo["runs"] == 10
    assert set(result.monte_carlo) == _EXPECTED_KEYS


def test_attach_monte_carlo_skips_invalid_plan() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    plan = PlanResult(
        planner_name="test",
        course_ids=[],
        reached_skills=set(),
        total_weeks=0,
        total_difficulty=0.0,
        expanded_nodes=0,
        max_frontier_size=0,
        runtime_seconds=0.01,
        valid=False,
    )

    result = attach_monte_carlo_to_plan(plan, dataset.courses, profile)

    assert result.monte_carlo == {
        "skipped": True,
        "reason": "Monte Carlo solo se ejecuta sobre planes validos con cursos.",
    }

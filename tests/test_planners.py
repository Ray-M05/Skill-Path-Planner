from src.dataset.loader import load_dataset
from src.models import StudentProfile
from src.planners.astar import astar_k_plans, astar_plan, g_cost, h_cost
from src.planners.greedy import greedy_plan, greedy_score
from src.planners.ucs import path_cost, ucs_plan
from src.models import SearchState


def test_greedy_score_rewards_new_target_skills() -> None:
    dataset = load_dataset("data")
    course = dataset.courses["course_python_intermediate"]

    assert greedy_score(course, {"skill_python_basic"}, {"skill_python_intermediate"}) > 0
    assert greedy_score(course, {"skill_python_intermediate"}, {"skill_python_intermediate"}) == 0


def test_greedy_finds_valid_ml_engineer_plan() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    target_skills = dataset.roles["role_ml_engineer"].required_skills

    result = greedy_plan(profile.initial_skills, target_skills, dataset.courses, profile)

    assert result.valid
    assert set(target_skills).issubset(result.reached_skills)
    assert len(result.course_ids) > 0


def test_ucs_finds_valid_ml_engineer_plan() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    target_skills = dataset.roles["role_ml_engineer"].required_skills

    result = ucs_plan(profile.initial_skills, target_skills, dataset.courses, profile, max_nodes=50_000)

    assert result.valid
    assert set(target_skills).issubset(result.reached_skills)
    assert result.expanded_nodes > 0


def test_astar_finds_valid_ml_engineer_plan() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    target_skills = dataset.roles["role_ml_engineer"].required_skills

    result = astar_plan(profile.initial_skills, target_skills, dataset.courses, profile)

    assert result.valid
    assert set(target_skills).issubset(result.reached_skills)
    assert result.max_frontier_size > 0


def test_astar_h_cost_returns_zero_when_goal_is_reached() -> None:
    dataset = load_dataset("data")
    state = SearchState(
        skills=frozenset({"skill_sql_basic"}),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )

    assert h_cost(state, {"skill_sql_basic"}, dataset.courses) == 0


def test_astar_h_cost_is_positive_and_admissible_units() -> None:
    dataset = load_dataset("data")
    state = SearchState(
        skills=frozenset(),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )

    value = h_cost(
        state,
        {"skill_python_intermediate", "skill_statistics_basic"},
        dataset.courses,
    )
    assert value > 0
    min_duration = min(c.duration_weeks for c in dataset.courses.values())
    assert value >= min_duration


def test_astar_k_plans_returns_unique_valid_plans() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    target_skills = dataset.roles["role_data_analyst"].required_skills

    plans = astar_k_plans(profile.initial_skills, target_skills, dataset.courses, profile, k=2)

    assert plans
    assert all(plan.valid for plan in plans)
    assert len({tuple(plan.course_ids) for plan in plans}) == len(plans)


def test_planners_fail_when_profile_limits_make_goal_impossible() -> None:
    dataset = load_dataset("data")
    profile = StudentProfile(
        id="profile_impossible",
        initial_skills=frozenset({"skill_python_basic"}),
        max_weeks=4,
        max_weekly_hours=10,
        risk_tolerance=0.4,
    )
    target_skills = dataset.roles["role_ml_engineer"].required_skills

    assert not greedy_plan(profile.initial_skills, target_skills, dataset.courses, profile).valid
    assert not ucs_plan(profile.initial_skills, target_skills, dataset.courses, profile).valid
    assert not astar_plan(profile.initial_skills, target_skills, dataset.courses, profile).valid


def test_cost_functions_follow_plan_definition() -> None:
    state = SearchState(
        skills=frozenset({"skill_python_basic"}),
        taken_courses=("course_python_intermediate", "course_statistics_basic"),
        weeks_used=12,
        difficulty_sum=0.75,
    )

    assert path_cost(state) == 14.5
    assert g_cost(state) == 14.5
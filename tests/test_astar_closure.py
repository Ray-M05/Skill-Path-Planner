from __future__ import annotations

from src.dataset.loader import load_dataset
from src.models import SearchState, StudentProfile
from src.planners.astar import astar_plan, h_cost
from src.planners.common import build_skill_producers, prerequisite_closure
from src.planners.ucs import ucs_plan

_DEEP_TARGET = {"skill_computer_vision", "skill_linear_algebra", "skill_python_advanced"}
_INITIAL = {"skill_python_basic"}


def _deep_profile() -> StudentProfile:
    return StudentProfile(
        id="t",
        initial_skills=frozenset(_INITIAL),
        max_weeks=120,
        max_weekly_hours=12,
        risk_tolerance=0.5,
    )


def test_build_skill_producers_maps_skill_to_course() -> None:
    ds = load_dataset("data")
    producers = build_skill_producers(ds.courses)

    assert producers["skill_python_intermediate"].id == "course_python_intermediate"
    assert producers["skill_python_basic"].id == "course_python_basic"


def test_prerequisite_closure_includes_transitive_prereqs() -> None:
    ds = load_dataset("data")
    closure = prerequisite_closure({"skill_computer_vision"}, ds.courses)

    for skill_id in (
        "skill_computer_vision",
        "skill_deep_learning_basic",
        "skill_machine_learning_basic",
        "skill_python_intermediate",
        "skill_statistics_basic",
        "skill_linear_algebra",
        "skill_python_basic",
    ):
        assert skill_id in closure
    assert "skill_html_css" not in closure


def test_h_cost_is_prereq_aware_and_admissible() -> None:
    ds = load_dataset("data")
    state = SearchState(skills=frozenset(_INITIAL), taken_courses=(), weeks_used=0, difficulty_sum=0.0)

    h0 = h_cost(state, _DEEP_TARGET, ds.courses)
    assert h0 > 30

    plan = astar_plan(set(_INITIAL), set(_DEEP_TARGET), ds.courses, _deep_profile())
    assert plan.valid
    g_plan = plan.total_weeks + 2.0 * plan.total_difficulty + 0.5 * len(plan.course_ids)
    assert h0 <= g_plan + 1e-9


def test_astar_solves_deep_target_efficiently() -> None:
    ds = load_dataset("data")
    plan = astar_plan(set(_INITIAL), set(_DEEP_TARGET), ds.courses, _deep_profile())

    assert plan.valid
    assert set(_DEEP_TARGET).issubset(plan.reached_skills)
    assert plan.expanded_nodes < 50000


def test_ucs_unpruned_solves_shallow_target() -> None:
    ds = load_dataset("data")
    target = {"skill_python_intermediate", "skill_statistics_basic"}
    plan = ucs_plan(set(_INITIAL), target, ds.courses, _deep_profile())

    assert plan.valid
    assert target.issubset(plan.reached_skills)


def test_ucs_node_cap_prevents_runaway() -> None:
    ds = load_dataset("data")
    target = {"skill_computer_vision", "skill_reinforcement_learning"}
    plan = ucs_plan(set(_INITIAL), target, ds.courses, _deep_profile(), max_nodes=500)

    assert not plan.valid
    assert plan.expanded_nodes <= 500

from src.dataset.loader import load_dataset
from src.models import SearchState
from src.planners.common import apply_course, is_applicable, is_goal


def test_is_goal_returns_true_when_target_skills_are_reached() -> None:
    state = SearchState(
        skills=frozenset({"skill_python_basic", "skill_sql_basic"}),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )

    assert is_goal(state, {"skill_sql_basic"})
    assert not is_goal(state, {"skill_machine_learning_basic"})


def test_course_is_applicable_only_when_prerequisites_are_met_and_not_taken() -> None:
    dataset = load_dataset("data")
    course = dataset.courses["course_python_intermediate"]
    state = SearchState(
        skills=frozenset({"skill_python_basic"}),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )
    already_taken = SearchState(
        skills=frozenset({"skill_python_basic", "skill_python_intermediate"}),
        taken_courses=("course_python_intermediate",),
        weeks_used=6,
        difficulty_sum=0.35,
    )
    missing_prereq = SearchState(
        skills=frozenset(),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )

    assert is_applicable(course, state)
    assert not is_applicable(course, already_taken)
    assert not is_applicable(course, missing_prereq)


def test_apply_course_adds_outcomes_and_accumulates_costs() -> None:
    dataset = load_dataset("data")
    course = dataset.courses["course_python_intermediate"]
    state = SearchState(
        skills=frozenset({"skill_python_basic"}),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )

    next_state = apply_course(course, state)

    assert next_state.skills == frozenset({"skill_python_basic", "skill_python_intermediate"})
    assert next_state.taken_courses == ("course_python_intermediate",)
    assert next_state.weeks_used == 6
    assert next_state.difficulty_sum == 0.35
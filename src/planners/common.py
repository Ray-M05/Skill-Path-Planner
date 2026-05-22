from __future__ import annotations

import time

from ..models import Course, PlanResult, SearchState, StudentProfile
from ..validators import validate_plan


def is_goal(state: SearchState, target_skills: set[str]) -> bool:
    return target_skills.issubset(state.skills)


def is_applicable(course: Course, state: SearchState) -> bool:
    return course.prerequisites.issubset(state.skills) and course.id not in state.taken_courses


def apply_course(course: Course, state: SearchState) -> SearchState:
    return SearchState(
        skills=frozenset(set(state.skills) | course.outcomes),
        taken_courses=state.taken_courses + (course.id,),
        weeks_used=state.weeks_used + course.duration_weeks,
        difficulty_sum=state.difficulty_sum + course.difficulty,
    )


def violates_profile_limits(course: Course, state: SearchState, profile: StudentProfile) -> bool:
    if course.weekly_hours > profile.max_weekly_hours:
        return True
    return state.weeks_used + course.duration_weeks > profile.max_weeks


def make_plan_result(
    planner_name: str,
    state: SearchState,
    initial_skills: set[str],
    target_skills: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    expanded_nodes: int,
    max_frontier_size: int,
    start_time: float,
) -> PlanResult:
    course_ids = list(state.taken_courses)
    valid, validation_errors = validate_plan(
        course_ids,
        initial_skills,
        target_skills,
        courses,
        profile,
    )
    return PlanResult(
        planner_name=planner_name,
        course_ids=course_ids,
        reached_skills=set(state.skills),
        total_weeks=state.weeks_used,
        total_difficulty=state.difficulty_sum,
        expanded_nodes=expanded_nodes,
        max_frontier_size=max_frontier_size,
        runtime_seconds=time.perf_counter() - start_time,
        valid=valid,
        validation_errors=validation_errors,
    )


def make_failure_result(
    planner_name: str,
    initial_skills: set[str],
    expanded_nodes: int,
    max_frontier_size: int,
    start_time: float,
    message: str,
) -> PlanResult:
    return PlanResult(
        planner_name=planner_name,
        course_ids=[],
        reached_skills=set(initial_skills),
        total_weeks=0,
        total_difficulty=0.0,
        expanded_nodes=expanded_nodes,
        max_frontier_size=max_frontier_size,
        runtime_seconds=time.perf_counter() - start_time,
        valid=False,
        validation_errors=[message],
    )

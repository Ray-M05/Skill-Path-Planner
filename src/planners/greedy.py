from __future__ import annotations

import time

from ..models import Course, PlanResult, SearchState, StudentProfile
from .common import (
    apply_course,
    is_applicable,
    is_goal,
    make_failure_result,
    make_plan_result,
    violates_profile_limits,
)


def greedy_score(
    course: Course,
    current_skills: set[str],
    target_skills: set[str],
    difficulty_weight: float = 4.0,
) -> float:
    new_target_skills = (course.outcomes & target_skills) - current_skills
    if not new_target_skills:
        return 0.0
    return len(new_target_skills) / (
        1 + course.duration_weeks + difficulty_weight * course.difficulty
    )


def greedy_plan(
    initial_skills: set[str],
    target_skills: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
) -> PlanResult:
    start_time = time.perf_counter()
    state = SearchState(
        skills=frozenset(initial_skills),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )
    expanded_nodes = 0
    max_frontier_size = 0

    while not is_goal(state, target_skills):
        expanded_nodes += 1
        candidates: list[tuple[float, str, Course]] = []
        current_skills = set(state.skills)

        for course in courses.values():
            if not is_applicable(course, state):
                continue
            if violates_profile_limits(course, state, profile):
                continue
            score = greedy_score(course, current_skills, target_skills)
            if score > 0:
                candidates.append((score, course.id, course))

        max_frontier_size = max(max_frontier_size, len(candidates))
        if not candidates:
            return make_failure_result(
                "greedy",
                initial_skills,
                expanded_nodes,
                max_frontier_size,
                start_time,
                "Greedy no encontro un curso aplicable que acerque al objetivo.",
            )

        _, _, selected_course = max(candidates, key=lambda item: (item[0], -item[2].duration_weeks, item[1]))
        state = apply_course(selected_course, state)

    return make_plan_result(
        "greedy",
        state,
        initial_skills,
        target_skills,
        courses,
        profile,
        expanded_nodes,
        max_frontier_size,
        start_time,
    )

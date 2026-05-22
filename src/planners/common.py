from __future__ import annotations

from ..models import Course, SearchState


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

from __future__ import annotations

import time

from ..models import Course, PlanResult, SearchState, StudentProfile
from ..validators import validate_plan


def course_h_cost(course: Course) -> float:
    # Componente del costo de un curso para la heuristica (semanas + dificultad).
    return course.duration_weeks + 2.0 * course.difficulty


def build_skill_producers(courses: dict[str, Course]) -> dict[str, Course]:
    # Mapa skill_id -> curso que la produce (el de menor course_h_cost si hubiera varios).
    producers: dict[str, Course] = {}
    for course in courses.values():
        for skill_id in course.outcomes:
            current = producers.get(skill_id)
            if current is None or course_h_cost(course) < course_h_cost(current):
                producers[skill_id] = course
    return producers


def prerequisite_closure(
    target_skills: set[str],
    courses: dict[str, Course],
    producers: dict[str, Course] | None = None,
) -> set[str]:
    """ Devuelve todas las skills producibles que hay que adquirir para alcanzar los objetivos:
    los propios objetivos mas sus prerequisitos transitivos que tengan curso productor.
    Las skills base sin productor se excluyen (se asumen como conocimiento inicial; si faltan,
    la busqueda fallara por si sola al no poder satisfacer los prerequisitos).
    """
    if producers is None:
        producers = build_skill_producers(courses)
    closure: set[str] = set()
    stack = list(target_skills)
    while stack:
        skill_id = stack.pop()
        if skill_id in closure:
            continue
        course = producers.get(skill_id)
        if course is None:
            continue
        closure.add(skill_id)
        stack.extend(course.prerequisites)
    return closure


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

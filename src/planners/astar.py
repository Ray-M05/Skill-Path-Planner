from __future__ import annotations

import heapq
import itertools
import math
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


def g_cost(state: SearchState) -> float:
    return (
        state.weeks_used
        + 2.0 * state.difficulty_sum
        + 0.5 * len(state.taken_courses)
    )


def h_cost(state: SearchState, target_skills: set[str], courses: dict[str, Course]) -> float:
    missing = target_skills - set(state.skills)
    if not missing:
        return 0.0

    min_duration = min(c.duration_weeks for c in courses.values())
    min_difficulty = min(c.difficulty for c in courses.values())
    max_gain = 0
    for course in courses.values():
        objective_gain = len(course.outcomes & missing)
        if objective_gain > max_gain:
            max_gain = objective_gain

    if max_gain == 0:
        return math.inf

    n_courses_needed = math.ceil(len(missing) / max_gain)
    return n_courses_needed * (min_duration + 2.0 * min_difficulty + 0.5)


def astar_plan(
    initial_skills: set[str],
    target_skills: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    max_nodes: int = 0,
) -> PlanResult:
    plans = astar_k_plans(initial_skills, target_skills, courses, profile, k=1, max_nodes=max_nodes)
    if plans:
        plans[0].planner_name = "astar"
        return plans[0]

    start_time = time.perf_counter()
    msg = (
        f"A* no encontro una trayectoria valida (limite de {max_nodes} nodos alcanzado)."
        if max_nodes > 0
        else "A* no encontro una trayectoria valida."
    )
    return make_failure_result(
        "astar",
        initial_skills,
        expanded_nodes=0,
        max_frontier_size=0,
        start_time=start_time,
        message=msg,
    )


def astar_k_plans(
    initial_skills: set[str],
    target_skills: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    k: int = 3,
    max_nodes: int = 0,
) -> list[PlanResult]:
    """Genera hasta k planes con A*. max_nodes=0 significa sin limite de expansion."""
    start_time = time.perf_counter()
    initial_state = SearchState(
        skills=frozenset(initial_skills),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )
    initial_h = h_cost(initial_state, target_skills, courses)
    if math.isinf(initial_h):
        return []

    counter = itertools.count()
    frontier: list[tuple[float, int, SearchState]] = [
        (g_cost(initial_state) + initial_h, next(counter), initial_state)
    ]
    best_cost_by_key: dict[tuple[frozenset[str], tuple[str, ...]], float] = {}
    plans: list[PlanResult] = []
    seen_plan_course_ids: set[tuple[str, ...]] = set()
    expanded_nodes = 0
    max_frontier_size = 1

    while frontier and len(plans) < k:
        if max_nodes > 0 and expanded_nodes >= max_nodes:
            break

        max_frontier_size = max(max_frontier_size, len(frontier))
        _, _, state = heapq.heappop(frontier)
        state_cost = g_cost(state)
        state_key = (state.skills, state.taken_courses)
        if state_key in best_cost_by_key and best_cost_by_key[state_key] <= state_cost:
            continue
        best_cost_by_key[state_key] = state_cost
        expanded_nodes += 1

        if is_goal(state, target_skills):
            plan_key = state.taken_courses
            if plan_key not in seen_plan_course_ids:
                seen_plan_course_ids.add(plan_key)
                result = make_plan_result(
                    "astar",
                    state,
                    initial_skills,
                    target_skills,
                    courses,
                    profile,
                    expanded_nodes,
                    max_frontier_size,
                    start_time,
                )
                plans.append(result)
            continue

        for course in courses.values():
            if not is_applicable(course, state):
                continue
            if violates_profile_limits(course, state, profile):
                continue
            next_state = apply_course(course, state)
            heuristic = h_cost(next_state, target_skills, courses)
            if math.isinf(heuristic):
                continue
            priority = g_cost(next_state) + heuristic
            heapq.heappush(frontier, (priority, next(counter), next_state))

    return plans

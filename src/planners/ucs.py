from __future__ import annotations

import heapq
import itertools
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

_DEFAULT_MAX_NODES = 5000


def path_cost(state: SearchState) -> float:
    return (
        state.weeks_used
        + 2.0 * state.difficulty_sum
        + 0.5 * len(state.taken_courses)
    )


def ucs_plan(
    initial_skills: set[str],
    target_skills: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    max_nodes: int = 0,
) -> PlanResult:
    start_time = time.perf_counter()
    # Dijkstra 
    cap = max_nodes if max_nodes > 0 else _DEFAULT_MAX_NODES
    initial_state = SearchState(
        skills=frozenset(initial_skills),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )
    counter = itertools.count()
    frontier: list[tuple[float, int, SearchState]] = [(0.0, next(counter), initial_state)]
    closed: dict[frozenset[str], float] = {}
    expanded_nodes = 0
    max_frontier_size = 1

    while frontier:
        if expanded_nodes >= cap:
            return make_failure_result(
                "ucs",
                initial_skills,
                expanded_nodes,
                max_frontier_size,
                start_time,
                f"UCS no encontro plan dentro del limite de {cap} nodos.",
            )
        max_frontier_size = max(max_frontier_size, len(frontier))
        _, _, state = heapq.heappop(frontier)
        state_cost = path_cost(state)
        state_key = state.skills
        if state_key in closed and closed[state_key] <= state_cost:
            continue
        closed[state_key] = state_cost
        expanded_nodes += 1

        if is_goal(state, target_skills):
            return make_plan_result(
                "ucs",
                state,
                initial_skills,
                target_skills,
                courses,
                profile,
                expanded_nodes,
                max_frontier_size,
                start_time,
            )

        for course in courses.values():
            if not is_applicable(course, state):
                continue
            if violates_profile_limits(course, state, profile):
                continue
            next_state = apply_course(course, state)
            heapq.heappush(frontier, (path_cost(next_state), next(counter), next_state))

    return make_failure_result(
        "ucs",
        initial_skills,
        expanded_nodes,
        max_frontier_size,
        start_time,
        "UCS no encontro una trayectoria valida.",
    )

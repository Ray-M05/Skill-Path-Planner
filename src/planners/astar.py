from __future__ import annotations

import heapq
import itertools
import math
import time
from ..models import Course, PlanResult, SearchState, StudentProfile
from .common import (
    apply_course,
    build_skill_producers,
    course_h_cost,
    is_applicable,
    is_goal,
    make_failure_result,
    make_plan_result,
    prerequisite_closure,
    violates_profile_limits,
)

_REC_BONUS = 0.5  # costo por cada skill recomendada cubierta


def g_cost(state: SearchState, recommended_skills: frozenset[str] | None = None) -> float:
    base = (
        state.weeks_used
        + 2.0 * state.difficulty_sum
        + 0.5 * len(state.taken_courses)
    )
    if recommended_skills:
        covered = len(recommended_skills & set(state.skills))
        base -= _REC_BONUS * covered
    return base


def h_cost(
    state: SearchState,
    target_skills: set[str],
    courses: dict[str, Course],
    producers: dict[str, Course] | None = None,
    needed_closure: set[str] | None = None,
) -> float:
    missing = target_skills - set(state.skills)
    if not missing:
        return 0.0

    if producers is None:
        producers = build_skill_producers(courses)

    # Objetivo inalcanzable: alguna skill objetivo no la produce ningun curso.
    if any(producers.get(skill_id) is None for skill_id in missing):
        return math.inf

    if needed_closure is None:
        needed_closure = prerequisite_closure(set(target_skills), courses, producers)

    remaining = needed_closure - set(state.skills)
    return sum(course_h_cost(producers[skill_id]) for skill_id in remaining)


def astar_plan(
    initial_skills: set[str],
    target_skills: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    max_nodes: int = 0,
    recommended_skills: frozenset[str] | None = None,
) -> PlanResult:
    plans = astar_k_plans(
        initial_skills, target_skills, courses, profile,
        k=1, max_nodes=max_nodes, recommended_skills=recommended_skills,
    )
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
    recommended_skills: frozenset[str] | None = None,
) -> list[PlanResult]:
    """Genera hasta k planes con A*."""
    rec = recommended_skills or frozenset()
    start_time = time.perf_counter()
    producers = build_skill_producers(courses)
    needed_closure = prerequisite_closure(set(target_skills), courses, producers)
    initial_state = SearchState(
        skills=frozenset(initial_skills),
        taken_courses=(),
        weeks_used=0,
        difficulty_sum=0.0,
    )
    initial_h = h_cost(initial_state, target_skills, courses, producers, needed_closure)
    if math.isinf(initial_h):
        return []

    counter = itertools.count()
    frontier: list[tuple[float, int, SearchState]] = [
        (g_cost(initial_state, rec) + initial_h, next(counter), initial_state)
    ]
    # La diversidad entre k-planes surge de `seen_plan_course_ids`
    # que descarta planes con exactamente los mismos cursos.
    best_cost_by_key: dict[frozenset[str], float] = {}
    plans: list[PlanResult] = []
    seen_plan_course_ids: set[tuple[str, ...]] = set()
    expanded_nodes = 0
    max_frontier_size = 1

    while frontier and len(plans) < k:
        if max_nodes > 0 and expanded_nodes >= max_nodes:
            break

        max_frontier_size = max(max_frontier_size, len(frontier))
        _, _, state = heapq.heappop(frontier)
        state_cost = g_cost(state, rec)
        state_key = state.skills
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

        # Poda: solo cursos que producen una skill aun pendiente del cierre de prerequisitos.
        # Acota los estados alcanzables a subconjuntos del cierre y descarta cursos
        #  irrelevantes o ya cubiertos sin perder optimalidad.
        needed_now = needed_closure - set(state.skills)
        for course in courses.values():
            if course.outcomes.isdisjoint(needed_now):
                continue
            if not is_applicable(course, state):
                continue
            if violates_profile_limits(course, state, profile):
                continue
            next_state = apply_course(course, state)
            heuristic = h_cost(next_state, target_skills, courses, producers, needed_closure)
            if math.isinf(heuristic):
                continue
            priority = g_cost(next_state, rec) + heuristic
            heapq.heappush(frontier, (priority, next(counter), next_state))

    return plans

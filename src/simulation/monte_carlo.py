from __future__ import annotations

import random
from statistics import mean
from typing import Any

from ..models import Course, PlanResult, StudentProfile


def adjusted_pass_probability(course: Course, profile: StudentProfile) -> float:
    probability = course.pass_base_probability
    probability -= 0.10 * course.difficulty
    probability += 0.05 * profile.risk_tolerance
    probability += 0.03 * max(profile.max_weekly_hours - course.weekly_hours, 0)
    return max(0.0, min(1.0, probability))


def simulate_plan_once(
    course_ids: list[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    rng: random.Random,
    max_attempts_per_course: int = 2,
) -> dict[str, Any]:
    weeks_used = 0
    failed_courses = 0
    completed_courses: list[str] = []

    for course_id in course_ids:
        course = courses[course_id]
        passed = False

        for attempt in range(max_attempts_per_course):
            weeks_used += course.duration_weeks
            if rng.random() <= adjusted_pass_probability(course, profile):
                passed = True
                break
            failed_courses += 1

        if not passed:
            return {
                "success": False,
                "weeks_used": weeks_used,
                "failed_courses": failed_courses,
                "completed_courses": completed_courses,
                "failed_at_course_id": course_id,
            }

        completed_courses.append(course_id)

    return {
        "success": True,
        "weeks_used": weeks_used,
        "failed_courses": failed_courses,
        "completed_courses": completed_courses,
        "failed_at_course_id": None,
    }


def evaluate_plan_monte_carlo(
    course_ids: list[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    runs: int = 500,
    seed: int | None = None,
    max_attempts_per_course: int = 2,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs debe ser mayor que 0.")
    if max_attempts_per_course <= 0:
        raise ValueError("max_attempts_per_course debe ser mayor que 0.")

    rng = random.Random(seed)
    outcomes = [
        simulate_plan_once(
            course_ids,
            courses,
            profile,
            rng,
            max_attempts_per_course=max_attempts_per_course,
        )
        for _ in range(runs)
    ]
    successes = [outcome for outcome in outcomes if outcome["success"]]
    failed_at_counts: dict[str, int] = {}
    for outcome in outcomes:
        failed_at = outcome["failed_at_course_id"]
        if failed_at is not None:
            failed_at_counts[failed_at] = failed_at_counts.get(failed_at, 0) + 1

    return {
        "runs": runs,
        "seed": seed,
        "max_attempts_per_course": max_attempts_per_course,
        "success_probability": len(successes) / runs,
        "expected_weeks": mean(outcome["weeks_used"] for outcome in outcomes),
        "expected_failed_courses": mean(outcome["failed_courses"] for outcome in outcomes),
        "success_expected_weeks": mean(outcome["weeks_used"] for outcome in successes)
        if successes
        else None,
        "failed_at_counts": failed_at_counts,
        "course_pass_probabilities": {
            course_id: adjusted_pass_probability(courses[course_id], profile)
            for course_id in course_ids
        },
    }


def attach_monte_carlo_to_plan(
    plan: PlanResult,
    courses: dict[str, Course],
    profile: StudentProfile,
    runs: int = 500,
    seed: int | None = None,
    max_attempts_per_course: int = 2,
) -> PlanResult:
    if not plan.valid or not plan.course_ids:
        plan.monte_carlo = {
            "skipped": True,
            "reason": "Monte Carlo solo se ejecuta sobre planes validos con cursos.",
        }
        return plan

    plan.monte_carlo = evaluate_plan_monte_carlo(
        plan.course_ids,
        courses,
        profile,
        runs=runs,
        seed=seed,
        max_attempts_per_course=max_attempts_per_course,
    )
    return plan

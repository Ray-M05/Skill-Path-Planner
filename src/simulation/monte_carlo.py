from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..models import Course, PlanResult, StudentProfile


@dataclass
class MonteCarloParams:
    difficulty_penalty: float = 0.15
    overload_penalty_factor: float = 0.05
    fatigue_penalty: float = 0.03
    fatigue_decay: float = 0.9
    retry_weeks_multiplier: float = 2.0
    max_failed_courses: int = 2
    duration_noise_base: float = 0.10
    duration_noise_difficulty_factor: float = 0.05


def _wilson_ci_95(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def simulate_plan_once(
    course_ids: list[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    rng: np.random.Generator,
    params: MonteCarloParams,
) -> dict[str, Any]:
    total_weeks = 0.0
    failed_courses = 0
    accumulated_fatigue = 0.0

    for course_id in course_ids:
        course = courses[course_id]
        overload = max(0, course.weekly_hours - profile.max_weekly_hours)

        p_pass = (
            course.pass_base_probability
            - params.difficulty_penalty * course.difficulty
            - params.overload_penalty_factor * overload
            - params.fatigue_penalty * accumulated_fatigue
        )
        p_pass = min(0.98, max(0.05, p_pass))

        duration_std = params.duration_noise_base + params.duration_noise_difficulty_factor * course.difficulty
        actual_weeks = course.duration_weeks * float(rng.normal(1.0, duration_std))
        actual_weeks = max(course.duration_weeks * 0.5, actual_weeks)

        if rng.random() <= p_pass:
            total_weeks += actual_weeks
        else:
            failed_courses += 1
            total_weeks += params.retry_weeks_multiplier * actual_weeks
            if failed_courses > params.max_failed_courses:
                return {
                    "success": False,
                    "weeks_used": total_weeks,
                    "failed_courses": failed_courses,
                }

        accumulated_fatigue = accumulated_fatigue * params.fatigue_decay + course.difficulty

    success = total_weeks <= profile.max_weeks
    return {
        "success": success,
        "weeks_used": total_weeks,
        "failed_courses": failed_courses,
    }


def evaluate_plan_monte_carlo(
    course_ids: list[str],
    courses: dict[str, Course],
    profile: StudentProfile,
    runs: int = 500,
    seed: int | None = None,
    params: MonteCarloParams | None = None,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs debe ser mayor que 0.")
    if params is None:
        params = MonteCarloParams()

    rng = np.random.default_rng(seed)
    outcomes = [
        simulate_plan_once(course_ids, courses, profile, rng, params)
        for _ in range(runs)
    ]

    successes_count = sum(1 for o in outcomes if o["success"])
    weeks_arr = np.array([o["weeks_used"] for o in outcomes])
    failed_arr = np.array([o["failed_courses"] for o in outcomes])

    success_prob = successes_count / runs
    ci_low, ci_high = _wilson_ci_95(successes_count, runs)

    p50 = float(np.percentile(weeks_arr, 50))
    p90 = float(np.percentile(weeks_arr, 90))
    risk_score = (p90 - p50) / max(p50, 1.0)

    return {
        "runs": runs,
        "seed": seed,
        "params": {
            "difficulty_penalty": params.difficulty_penalty,
            "overload_penalty_factor": params.overload_penalty_factor,
            "fatigue_penalty": params.fatigue_penalty,
            "fatigue_decay": params.fatigue_decay,
            "retry_weeks_multiplier": params.retry_weeks_multiplier,
            "max_failed_courses": params.max_failed_courses,
        },
        "success_probability": success_prob,
        "success_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "expected_weeks": float(np.mean(weeks_arr)),
        "weeks_p10": float(np.percentile(weeks_arr, 10)),
        "weeks_p50": p50,
        "weeks_p90": p90,
        "expected_failed_courses": float(np.mean(failed_arr)),
        "failed_courses_std": float(np.std(failed_arr)),
        "risk_score": round(risk_score, 4),
    }


def attach_monte_carlo_to_plan(
    plan: PlanResult,
    courses: dict[str, Course],
    profile: StudentProfile,
    runs: int = 500,
    seed: int | None = None,
    params: MonteCarloParams | None = None,
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
        params=params,
    )
    return plan

from __future__ import annotations

from typing import Any

from .models import PlanResult, Role, StudentProfile


def recommended_coverage_score(reached_skills: set[str], recommended_skills: frozenset[str]) -> float:
    if not recommended_skills:
        return 0.0
    return len(reached_skills & recommended_skills) / len(recommended_skills)


def required_coverage_score(reached_skills: set[str], required_skills: frozenset[str]) -> float:
    if not required_skills:
        return 1.0
    return len(reached_skills & required_skills) / len(required_skills)


def normalize_value(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return min(1.0, value / max_value)


def compute_score_with_breakdown(
    plan: PlanResult,
    role: Role,
    profile: StudentProfile,
) -> tuple[float, dict[str, Any]]:
    """Devuelve (score, breakdown) donde breakdown muestra la contribución de cada señal.

    Pesos base con todas las señales activas:
      +0.20 required_coverage — siempre 1.0 para planes válidos; base garantizada
      +0.15 rec_coverage      — discrimina entre planes alternativos
      +0.25 MC success_probability
      +0.20 LLM global_quality
      -0.20 tiempo relativo al presupuesto
      -0.10 dificultad media por curso

    Cuando MC o LLM no están activos, su peso se transfiere a w_required para
    mantener el score base positivo en cualquier plan válido.
    """
    req_coverage = required_coverage_score(plan.reached_skills, role.required_skills)
    rec_coverage = recommended_coverage_score(plan.reached_skills, role.recommended_skills)
    time_penalty = normalize_value(plan.total_weeks, profile.max_weeks)
    difficulty_penalty = min(1.0, plan.total_difficulty / max(1, len(plan.course_ids)))

    mc = plan.monte_carlo
    has_mc = mc is not None and not mc.get("skipped")
    mc_success = mc["success_probability"] if has_mc else 0.0

    llm = plan.llm_evaluation
    has_llm = llm is not None and not llm.get("skipped")
    llm_quality = llm["global_quality"] if has_llm else 0.0

    w_required = 0.20 + (0.0 if has_mc else 0.25) + (0.0 if has_llm else 0.20)

    score = round(
        w_required * req_coverage
        + 0.15 * rec_coverage
        - 0.20 * time_penalty
        - 0.10 * difficulty_penalty
        + 0.25 * mc_success
        + 0.20 * llm_quality,
        4,
    )

    breakdown: dict[str, Any] = {
        "required": round(w_required * req_coverage, 4),
        "recommended": round(0.15 * rec_coverage, 4),
        "time_penalty": round(-0.20 * time_penalty, 4),
        "difficulty_penalty": round(-0.10 * difficulty_penalty, 4),
        "mc_success": round(0.25 * mc_success, 4),
        "llm_quality": round(0.20 * llm_quality, 4),
        "signals": {
            "has_mc": has_mc,
            "has_llm": has_llm,
            "w_required": round(w_required, 4),
        },
    }

    return score, breakdown


def compute_final_score(plan: PlanResult, role: Role, profile: StudentProfile) -> float:
    score, _ = compute_score_with_breakdown(plan, role, profile)
    return score


def rank_plans(plans: list[PlanResult]) -> list[PlanResult]:
    return sorted(
        plans,
        key=lambda p: p.final_score if p.final_score is not None else -999.0,
        reverse=True,
    )

from __future__ import annotations

from .models import PlanResult, Role, StudentProfile


def recommended_coverage_score(reached_skills: set[str], recommended_skills: frozenset[str]) -> float:
    if not recommended_skills:
        return 0.0
    return len(reached_skills & recommended_skills) / len(recommended_skills)


def normalize_value(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return min(1.0, value / max_value)


def required_coverage_score(reached_skills: set[str], required_skills: frozenset[str]) -> float:
    if not required_skills:
        return 1.0
    return len(reached_skills & required_skills) / len(required_skills)


def compute_final_score(plan: PlanResult, role: Role, profile: StudentProfile) -> float:
    """Score combinado con redistribución de pesos para señales ausentes.

    Términos positivos (pesos base con todas las señales activas):
      +0.20 required_coverage  — siempre 1.0 para planes válidos; base garantizada
      +0.15 rec_coverage       — discrimina entre planes alternativos
      +0.25 MC success_probability
      +0.20 LLM global_quality

    Términos negativos:
      -0.20 tiempo relativo al presupuesto
      -0.10 dificultad media por curso
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

    return round(
        w_required * req_coverage
        + 0.15 * rec_coverage
        - 0.20 * time_penalty
        - 0.10 * difficulty_penalty
        + 0.25 * mc_success
        + 0.20 * llm_quality,
        4,
    )


def rank_plans(plans: list[PlanResult]) -> list[PlanResult]:
    return sorted(
        plans,
        key=lambda p: p.final_score if p.final_score is not None else -999.0,
        reverse=True,
    )

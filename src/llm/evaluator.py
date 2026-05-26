from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import Dataset, GoalSpec, PlanResult, Role, StudentProfile

if TYPE_CHECKING:
    from .client import LLMClient


_NUMERIC_FIELDS = [
    "goal_alignment",
    "pedagogical_coherence",
    "profile_realism",
    "non_redundancy",
    "professional_value",
    "global_quality",
]


def validate_llm_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    for field_name in _NUMERIC_FIELDS:
        value = evaluation.get(field_name)
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{field_name} debe ser numérico, recibido: {type(value).__name__}"
            )
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{field_name} debe estar entre 0 y 1, recibido: {value}"
            )
    if not isinstance(evaluation.get("main_strengths"), list):
        raise ValueError("main_strengths debe ser una lista")
    if not isinstance(evaluation.get("main_weaknesses"), list):
        raise ValueError("main_weaknesses debe ser una lista")
    if not isinstance(evaluation.get("justification"), str):
        raise ValueError("justification debe ser un string")
    return evaluation


def build_mock_evaluation_response(
    plan: PlanResult,
    role: Role,
    profile: StudentProfile,
    dataset: Dataset,
) -> dict[str, Any]:
    """Evaluación determinista para modo mock, sin llamada LLM."""
    courses = [dataset.courses[cid] for cid in plan.course_ids if cid in dataset.courses]

    # Alineación con el objetivo: fracción de required_skills alcanzadas
    goal_alignment = (
        len(plan.reached_skills & role.required_skills) / max(len(role.required_skills), 1)
    )

    # Coherencia pedagógica: los cursos respetan el orden de prerrequisitos acumulado
    skills_so_far: set[str] = set(profile.initial_skills)
    violations = 0
    for course in courses:
        if not course.prerequisites.issubset(skills_so_far):
            violations += 1
        skills_so_far |= course.outcomes
    pedagogical_coherence = max(0.0, 1.0 - violations / max(len(courses), 1))

    # Realismo del perfil: cursos dentro de los límites de horas semanales
    overloaded = sum(1 for c in courses if c.weekly_hours > profile.max_weekly_hours)
    profile_realism = max(0.0, 1.0 - overloaded / max(len(courses), 1))

    # No redundancia: outcomes únicos sobre total de outcomes del plan
    all_outcomes: list[str] = [o for c in courses for o in c.outcomes]
    non_redundancy = len(set(all_outcomes)) / max(len(all_outcomes), 1)

    # Valor profesional: cobertura de recommended_skills del rol
    if role.recommended_skills:
        professional_value = (
            len(plan.reached_skills & role.recommended_skills) / len(role.recommended_skills)
        )
    else:
        professional_value = 0.5

    global_quality = round(
        0.30 * goal_alignment
        + 0.25 * pedagogical_coherence
        + 0.20 * profile_realism
        + 0.10 * non_redundancy
        + 0.15 * professional_value,
        4,
    )

    strengths: list[str] = []
    weaknesses: list[str] = []
    if goal_alignment >= 0.9:
        strengths.append("Trayectoria cubre las habilidades requeridas del rol")
    if pedagogical_coherence >= 0.9:
        strengths.append("Orden pedagógico correcto: prerrequisitos respetados")
    if professional_value >= 0.5:
        strengths.append("Incorpora habilidades recomendadas del rol")
    if overloaded > 0:
        weaknesses.append(
            f"{overloaded} curso(s) exceden la carga semanal del perfil"
        )
    if non_redundancy < 0.7:
        weaknesses.append("Posible redundancia entre outcomes de distintos cursos")
    if goal_alignment < 0.8:
        weaknesses.append("Cobertura de habilidades objetivo incompleta")

    return {
        "goal_alignment": round(goal_alignment, 4),
        "pedagogical_coherence": round(pedagogical_coherence, 4),
        "profile_realism": round(profile_realism, 4),
        "non_redundancy": round(non_redundancy, 4),
        "professional_value": round(professional_value, 4),
        "global_quality": global_quality,
        "main_strengths": strengths,
        "main_weaknesses": weaknesses,
        "justification": (
            f"Evaluación automática (mock): {len(courses)} curso(s), "
            f"cobertura de objetivo {goal_alignment:.0%}, "
            f"coherencia pedagógica {pedagogical_coherence:.0%}, "
            f"calidad global {global_quality:.2f}."
        ),
    }


def evaluate_plan_with_llm(
    plan: PlanResult,
    role: Role,
    goal_spec: GoalSpec,
    dataset: Dataset,
    llm_client: LLMClient,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Evalúa cualitativamente un plan usando el LLM Evaluator.
    """
    if not plan.valid or not plan.course_ids:
        return {
            "skipped": True,
            "reason": "La evaluación LLM solo se ejecuta sobre planes válidos con cursos.",
        }

    max_weeks = goal_spec.constraints.get("max_weeks") or 999
    max_weekly_hours = goal_spec.constraints.get("max_weekly_hours") or 999
    profile = StudentProfile(
        id="eval_profile",
        initial_skills=frozenset(goal_spec.initial_skill_ids),
        max_weeks=int(max_weeks),
        max_weekly_hours=int(max_weekly_hours),
        risk_tolerance=0.5,
    )

    if use_mock:
        raw = build_mock_evaluation_response(plan, role, profile, dataset)
        return validate_llm_evaluation(raw)

    from .prompts import PLAN_EVALUATOR_SYSTEM_PROMPT, build_plan_evaluator_user_prompt

    user_prompt = build_plan_evaluator_user_prompt(plan, role, goal_spec, dataset)
    raw = llm_client.complete_json(PLAN_EVALUATOR_SYSTEM_PROMPT, user_prompt)
    return validate_llm_evaluation(raw)

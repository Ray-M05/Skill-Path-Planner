from __future__ import annotations

import json
from typing import Any
from ..dataset.resolver import RoleResolver, SkillResolver
from ..models import Dataset, GoalSpec, PlanResult, Role


GOAL_INTERPRETER_SYSTEM_PROMPT = """Eres un componente de extraccion estructurada para un sistema de planificacion de trayectorias profesionales.

Reglas obligatorias:
1. Devuelve exclusivamente JSON valido.
2. No inventes IDs de habilidades ni de cursos.
3. El campo role_id debe ser uno de los roles permitidos. Excepcion: si NINGUN rol permitido encaja razonablemente con el objetivo, crea un rol a medida con un id nuevo que empiece con el prefijo "custom_" seguido de un slug en minusculas con guiones bajos derivado del nombre (ejemplo: "custom_disenador_de_videojuegos").
4. Los campos target_skill_ids, initial_skill_ids y mentioned_skill_ids solo pueden contener skill_id incluidos en el catalogo permitido.
5. Si el usuario menciona una habilidad que no existe en el catalogo, no la conviertas en skill_id. Colocala en unknown_skill_mentions.
6. El sistema no modela costo monetario ni presupuesto.
7. Si el usuario menciona "barato", "poco presupuesto", "gratis" o expresiones parecidas, colocalo en ignored_constraints.
8. Prefiere SIEMPRE un rol existente. Solo crea un rol "custom_" cuando el objetivo claramente no corresponda a ningun rol del catalogo. Si el objetivo se acerca mas a un rol existente que a un perfil nuevo, usa el rol existente y baja confidence si el encaje es imperfecto.
9. Cuando crees un rol "custom_": coloca en role_name un nombre de profesion conciso y legible (sin el prefijo "custom_" y sin IDs), y coloca en target_skill_ids las habilidades del catalogo necesarias para ese perfil (seran las habilidades requeridas del rol a medida). Si usas un rol existente, deja role_name en null.
10. Extrae initial_skill_ids solo desde habilidades que el usuario dice tener actualmente.
11. Extrae restricciones como max_weeks, max_weekly_hours, preferred_pace, preferred_difficulty y preferences desde el texto del usuario.
12. No expliques nada fuera del JSON.
"""

def _to_pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_goal_interpreter_user_prompt(
    user_text: str,
    dataset: Dataset,
) -> str:
    allowed_roles = RoleResolver(dataset.roles).get_allowed_role_catalog_for_prompt()
    allowed_skills = SkillResolver(dataset.skills).get_allowed_skill_catalog_for_prompt()
    output_schema = {
        "role_id": "uno de los roles permitidos o un id nuevo con prefijo custom_",
        "role_name": None,
        "target_skill_ids": ["skill_id_1", "skill_id_2"],
        "initial_skill_ids": ["skill_id_1"],
        "mentioned_skill_ids": ["skill_id_1"],
        "constraints": {
            "max_weeks": None,
            "max_weekly_hours": None,
            "preferred_pace": None,
            "preferred_difficulty": None,
            "preferences": [],
        },
        "ignored_constraints": [],
        "unknown_skill_mentions": [],
        "confidence": 0.0,
    }

    return "\n\n".join(
        [
            f'Texto del usuario:\n"{user_text}"',
            f"Roles permitidos:\n{_to_pretty_json(allowed_roles)}",
            f"Habilidades permitidas:\n{_to_pretty_json(allowed_skills)}",
            f"Devuelve este JSON:\n{_to_pretty_json(output_schema)}",
        ]
    )


# Evaluador de trayectorias
PLAN_EVALUATOR_SYSTEM_PROMPT = """Eres un evaluador de calidad de trayectorias profesionales de aprendizaje.

Reglas obligatorias:
1. Devuelve exclusivamente JSON valido.
2. Evalua la calidad CUALITATIVA de la trayectoria: orden pedagogico, coherencia, adecuacion al perfil y valor profesional.
3. IMPORTANTE: La trayectoria ya paso validacion formal automatica (prerrequisitos, horas semanales por curso, duracion total). No re-valides esas restricciones ni las menciones como debilidades.
4. Los cursos se toman de forma secuencial, uno a la vez. max_weekly_hours aplica a cada curso individualmente, nunca a la suma de todos. No sumes horas entre cursos.
5. No evalues costo economico. El sistema no modela costo de cursos.
6. Penaliza trayectorias con orden pedagogico debil (cursos avanzados antes que fundamentos).
7. Penaliza trayectorias redundantes (cursos que aportan habilidades ya cubiertas).
8. Penaliza trayectorias poco alineadas con el rol objetivo.
9. Concentra profile_realism en: si el nivel de dificultad acumulada es apropiado para el perfil inicial, y si el orden es pedagogicamente progresivo.
10. Todos los valores numericos deben estar entre 0.0 y 1.0.
11. No inventes cursos ni habilidades.
12. No escribas texto fuera del JSON.
"""

_EVALUATOR_OUTPUT_SCHEMA = {
    "goal_alignment": 0.0,
    "pedagogical_coherence": 0.0,
    "profile_realism": 0.0,
    "non_redundancy": 0.0,
    "professional_value": 0.0,
    "global_quality": 0.0,
    "main_strengths": [],
    "main_weaknesses": [],
    "justification": "",
}


def build_plan_evaluator_user_prompt(
    plan: PlanResult,
    role: Role,
    goal_spec: GoalSpec,
    dataset: Dataset,
) -> str:
    role_info = {
        "id": role.id.removeprefix("custom_"),
        "name": role.name,
        "required_skills": sorted(role.required_skills),
        "recommended_skills": sorted(role.recommended_skills),
    }
    profile_info = {
        "initial_skill_ids": sorted(goal_spec.initial_skill_ids),
        "max_weeks": goal_spec.constraints.get("max_weeks"),
        "max_weekly_hours": goal_spec.constraints.get("max_weekly_hours"),
    }
    target_skills_info = [
        {"id": sid, "name": dataset.skills[sid].name}
        for sid in sorted(goal_spec.target_skill_ids)
        if sid in dataset.skills
    ]
    plan_courses_info = [
        {
            "id": cid,
            "name": dataset.courses[cid].name,
            "description": dataset.courses[cid].description,
            "duration_weeks": dataset.courses[cid].duration_weeks,
            "weekly_hours": dataset.courses[cid].weekly_hours,
            "difficulty": dataset.courses[cid].difficulty,
            "outcomes": sorted(dataset.courses[cid].outcomes),
        }
        for cid in plan.course_ids
        if cid in dataset.courses
    ]

    plan_summary = {
        "total_weeks": plan.total_weeks,
        "validated": True,
        "note": "Esta trayectoria paso validacion formal: prerrequisitos, horas semanales por curso y duracion total dentro del presupuesto.",
    }

    return "\n\n".join(
        [
            f"Rol objetivo:\n{_to_pretty_json(role_info)}",
            f"Perfil inicial:\n{_to_pretty_json(profile_info)}",
            f"Habilidades objetivo:\n{_to_pretty_json(target_skills_info)}",
            f"Resumen de validacion:\n{_to_pretty_json(plan_summary)}",
            f"Trayectoria propuesta ({len(plan_courses_info)} cursos en orden secuencial):\n{_to_pretty_json(plan_courses_info)}",
            f"Devuelve exactamente este JSON:\n{_to_pretty_json(_EVALUATOR_OUTPUT_SCHEMA)}",
        ]
    )
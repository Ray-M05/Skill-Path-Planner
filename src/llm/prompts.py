from __future__ import annotations

import json
from typing import Any
from ..dataset.resolver import RoleResolver, SkillResolver
from ..models import Dataset


GOAL_INTERPRETER_SYSTEM_PROMPT = """Eres un componente de extraccion estructurada para un sistema de planificacion de trayectorias profesionales.

Reglas obligatorias:
1. Devuelve exclusivamente JSON valido.
2. No inventes IDs.
3. El campo role_id debe ser uno de los roles permitidos.
4. Los campos target_skill_ids, initial_skill_ids y mentioned_skill_ids solo pueden contener skill_id incluidos en el catalogo permitido.
5. Si el usuario menciona una habilidad que no existe en el catalogo, no la conviertas en skill_id. Colocala en unknown_skill_mentions.
6. El sistema no modela costo monetario ni presupuesto.
7. Si el usuario menciona "barato", "poco presupuesto", "gratis" o expresiones parecidas, colocalo en ignored_constraints.
8. Si el objetivo es ambiguo, escoge el role_id mas cercano y baja confidence.
9. Extrae initial_skill_ids solo desde habilidades que el usuario dice tener actualmente.
10. Extrae restricciones como max_weeks, max_weekly_hours, preferred_pace, preferred_difficulty y preferences desde el texto del usuario.
11. No expliques nada fuera del JSON.
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
        "role_id": "uno de los roles permitidos",
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

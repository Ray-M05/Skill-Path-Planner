from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from typing import Any

from .config import Settings, load_settings
from .dataset.loader import load_dataset
from .dataset.resolver import normalize_text
from .llm.client import LLMClient
from .llm.interpreter import validate_goal_spec
from .llm.prompts import GOAL_INTERPRETER_SYSTEM_PROMPT, build_goal_interpreter_user_prompt
from .models import Dataset, GoalSpec, StudentProfile
from .validators import validate_plan


def _pretty_json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=default)


def _section(title: str, body: str) -> str:
    line = "=" * 78
    return f"\n{line}\n{title}\n{line}\n{body}"


def _parse_course_ids(raw_courses: str | None) -> list[str]:
    if not raw_courses:
        return []
    return [course_id.strip() for course_id in raw_courses.split(",") if course_id.strip()]


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _extract_constraints(user_text: str) -> dict[str, Any]:
    normalized = normalize_text(user_text)
    months = _first_int(r"(\d+)\s+mes(?:es)?", normalized)
    weeks = _first_int(r"(\d+)\s+semana(?:s)?", normalized)
    weekly_hours = _first_int(r"(\d+)\s+hora(?:s)?\s+semanal(?:es)?", normalized)

    max_weeks = weeks
    preferred_pace = None
    if months is not None:
        max_weeks = months * 4
        preferred_pace = f"{months} meses"
    elif weeks is not None:
        preferred_pace = f"{weeks} semanas"

    return {
        "max_weeks": max_weeks,
        "max_weekly_hours": weekly_hours,
        "preferred_pace": preferred_pace,
        "preferred_difficulty": None,
        "preferences": [],
    }


def _extract_ignored_constraints(user_text: str) -> list[str]:
    normalized = normalize_text(user_text)
    ignored: list[str] = []
    for keyword in ("gratis", "barato", "poco presupuesto"):
        if keyword in normalized:
            ignored.append(keyword)
    return ignored


def _skill_is_mentioned(user_text: str, skill_name_or_alias: str) -> bool:
    normalized_text = normalize_text(user_text)
    normalized_skill = normalize_text(skill_name_or_alias)
    return bool(normalized_skill and normalized_skill in normalized_text)


def build_mock_goal_response(user_text: str, dataset: Dataset) -> dict[str, Any]:
    normalized = normalize_text(user_text)
    if "machine learning" in normalized or "ml" in normalized:
        role_id = "role_ml_engineer"
        confidence = 0.85
    elif "analista" in normalized or "analisis de datos" in normalized or "datos" in normalized:
        role_id = "role_data_analyst"
        confidence = 0.85
    else:
        role_id = "role_data_analyst"
        confidence = 0.45

    mentioned_skill_ids: set[str] = set()
    for skill in dataset.skills.values():
        candidates = [skill.name, *skill.aliases]
        if any(_skill_is_mentioned(user_text, candidate) for candidate in candidates):
            mentioned_skill_ids.add(skill.id)

    initial_skill_ids = set(mentioned_skill_ids)
    role = dataset.roles[role_id]
    unknown_skill_mentions = []
    if "blockchain" in normalized:
        unknown_skill_mentions.append("blockchain")

    return {
        "role_id": role_id,
        "target_skill_ids": sorted(role.required_skills),
        "initial_skill_ids": sorted(initial_skill_ids),
        "mentioned_skill_ids": sorted(mentioned_skill_ids),
        "constraints": _extract_constraints(user_text),
        "ignored_constraints": _extract_ignored_constraints(user_text),
        "unknown_skill_mentions": unknown_skill_mentions,
        "confidence": confidence,
    }


def build_profile_from_goal(goal: GoalSpec) -> StudentProfile:
    max_weeks = goal.constraints.get("max_weeks") or 999
    max_weekly_hours = goal.constraints.get("max_weekly_hours") or 999
    return StudentProfile(
        id="profile_from_query",
        initial_skills=set(goal.initial_skill_ids),
        max_weeks=int(max_weeks),
        max_weekly_hours=int(max_weekly_hours),
        risk_tolerance=0.5,
    )


def run_cli(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.data_dir)
    settings = load_settings()
    provider = args.provider or settings.llm_provider
    user_prompt = build_goal_interpreter_user_prompt(args.goal, dataset)

    print(_section("QUERY EN LENGUAJE NATURAL", args.goal))
    print(_section("SYSTEM PROMPT ENVIADO AL LLM", GOAL_INTERPRETER_SYSTEM_PROMPT))
    print(_section("USER PROMPT ENVIADO AL LLM", user_prompt))

    if provider == "mock":
        raw_goal_spec = build_mock_goal_response(args.goal, dataset)
        client_description = "mock local sin llamada externa"
    else:
        real_settings = Settings(
            llm_provider=provider,
            llm_model=args.model or settings.llm_model,
            llm_api_key=settings.llm_api_key,
            llm_temperature=settings.llm_temperature,
            llm_max_output_tokens=settings.llm_max_output_tokens,
            llm_timeout_seconds=settings.llm_timeout_seconds,
            llm_cache=settings.llm_cache,
            llm_live_tests=settings.llm_live_tests,
        )
        client = LLMClient(real_settings)
        raw_goal_spec = client.complete_json(GOAL_INTERPRETER_SYSTEM_PROMPT, user_prompt)
        client_description = f"{provider} real con modelo {real_settings.llm_model}"

    print(_section(f"RESPUESTA CRUDA DEL LLM ({client_description})", _pretty_json(raw_goal_spec)))

    goal = validate_goal_spec(raw_goal_spec, dataset)
    print(_section("GOALSPEC VALIDADO", _pretty_json(asdict(goal))))

    course_ids = _parse_course_ids(args.courses)
    validation_profile = build_profile_from_goal(goal)
    valid, errors = validate_plan(
        course_ids,
        validation_profile.initial_skills,
        goal.target_skill_ids,
        dataset.courses,
        validation_profile,
    )
    validation_result = {
        "course_ids": course_ids,
        "initial_skills": sorted(validation_profile.initial_skills),
        "target_skills": sorted(goal.target_skill_ids),
        "max_weeks": validation_profile.max_weeks,
        "max_weekly_hours": validation_profile.max_weekly_hours,
        "valid": valid,
        "errors": errors,
    }
    print(_section("VALIDACION FORMAL DE LA TRAYECTORIA", _pretty_json(validation_result)))

    if not course_ids:
        print(
            _section(
                "NOTA",
                "No se pasaron cursos con --courses. Todavia no hay planificador automatico; "
                "por ahora esta CLI valida una trayectoria manual.",
            )
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI preliminar del Skill Path Planner.")
    parser.add_argument(
        "--goal",
        required=True,
        help="Objetivo profesional en lenguaje natural.",
    )
    parser.add_argument(
        "--courses",
        default="",
        help="IDs de cursos separados por coma para validar una trayectoria manual.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini"],
        default=None,
        help="Proveedor LLM. Si se omite, usa LLM_PROVIDER del .env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo LLM real. Si se omite, usa LLM_MODEL del .env.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directorio del dataset.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    raise SystemExit(run_cli(parser.parse_args()))


if __name__ == "__main__":
    main()

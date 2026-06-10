from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import asdict
from typing import Any

from .config import Settings, load_settings
from .dataset.loader import load_dataset
from .dataset.resolver import normalize_text
from .llm.client import LLMClient
from .llm.evaluator import evaluate_plan_with_llm
from .llm.interpreter import CUSTOM_ROLE_PREFIX, resolve_role, validate_goal_spec
from .llm.prompts import GOAL_INTERPRETER_SYSTEM_PROMPT, build_goal_interpreter_user_prompt
from .metrics import extract_metrics_row, rows_to_csv_string
from .models import Dataset, GoalSpec, PlanResult, StudentProfile
from .planners.astar import astar_k_plans, astar_plan
from .planners.greedy import greedy_plan
from .planners.ucs import ucs_plan
from .scoring import compute_score_with_breakdown, rank_plans
from .simulation.monte_carlo import attach_monte_carlo_to_plan, evaluate_plan_monte_carlo
from .validators import validate_plan

DEFAULT_MAX_NODES = 200000


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


# Palabras vacias para emparejar el objetivo con el nombre de un rol del catalogo.
_GOAL_STOPWORDS = {
    "quiero", "ser", "un", "una", "de", "del", "la", "el", "los", "las", "y", "en",
    "para", "como", "me", "gustaria", "mi", "meta", "es", "convertirme", "trabajar",
    "dedicarme", "prepararme", "llegar", "busco", "trayectoria", "a", "con", "que",
    "puedo", "estudiar", "semanales", "horas", "se", "lo", "mas", "rapido", "posible",
}


def _match_role_id(user_text: str, dataset: Dataset) -> tuple[str | None, float]:
    """Empareja el texto del usuario con el rol mas parecido del dataset cargado.
    """
    norm = normalize_text(user_text)
    text_tokens = set(norm.split()) - _GOAL_STOPWORDS
    best_id: str | None = None
    best_score = 0.0
    for role in dataset.roles.values():
        role_norm = normalize_text(role.name)
        role_tokens = set(role_norm.split()) - _GOAL_STOPWORDS
        if not role_tokens:
            continue
        score = len(text_tokens & role_tokens) / len(role_tokens)
        if role_norm and role_norm in norm:
            score += 1.0  # el nombre del rol aparece literal en la consulta
        if score > best_score:
            best_score = score
            best_id = role.id
    return best_id, best_score


def _slug_from_query(user_text: str) -> str:
    """Construye un slug a partir de las palabras significativas de la consulta,
    para nombrar el rol a medida cuando no se reconoce ninguno (p. ej. 'astronauta')."""
    tokens = [tok for tok in normalize_text(user_text).split() if tok not in _GOAL_STOPWORDS]
    return "_".join(tokens)


def build_mock_goal_response(user_text: str, dataset: Dataset) -> dict[str, Any]:
    """Interprete mock agnostico al dataset: elige el rol existente mas parecido por
    nombre y detecta skills mencionadas por nombre/alias."""
    mentioned_skill_ids: set[str] = set()
    for skill in dataset.skills.values():
        candidates = [skill.name, *skill.aliases]
        if any(_skill_is_mentioned(user_text, candidate) for candidate in candidates):
            mentioned_skill_ids.add(skill.id)

    base = {
        "role_name": None,
        "initial_skill_ids": sorted(mentioned_skill_ids),
        "mentioned_skill_ids": sorted(mentioned_skill_ids),
        "constraints": _extract_constraints(user_text),
        "ignored_constraints": _extract_ignored_constraints(user_text),
        "unknown_skill_mentions": [],
    }

    role_id, score = _match_role_id(user_text, dataset)
    if role_id is None:
        # Ningun rol reconocible: rol a medida sin objetivo -> no planificable.
        slug = _slug_from_query(user_text) or "desconocido"
        return {
            **base,
            "role_id": f"{CUSTOM_ROLE_PREFIX}{slug}",
            "target_skill_ids": [],
            "confidence": 0.1,
        }

    role = dataset.roles[role_id]
    return {
        **base,
        "role_id": role_id,
        "target_skill_ids": sorted(role.required_skills),
        "confidence": 0.85 if score >= 1.0 else 0.6,
    }


def build_profile_from_goal(goal: GoalSpec) -> StudentProfile:
    max_weeks = goal.constraints.get("max_weeks") or 999
    max_weekly_hours = goal.constraints.get("max_weekly_hours") or 999
    return StudentProfile(
        id="profile_from_query",
        initial_skills=frozenset(goal.initial_skill_ids),
        max_weeks=int(max_weeks),
        max_weekly_hours=int(max_weekly_hours),
        risk_tolerance=0.5,
    )


def run_planner(
    planner_name: str,
    goal: GoalSpec,
    dataset: Dataset,
    profile: StudentProfile,
) -> PlanResult:
    """Alias público para compatibilidad con tests existentes."""
    return run_planner_single(planner_name, goal, dataset, profile)


def run_planner_single(
    planner_name: str,
    goal: GoalSpec,
    dataset: Dataset,
    profile: StudentProfile,
) -> PlanResult:
    planners = {
        "greedy": greedy_plan,
        "ucs": ucs_plan,
        "astar": astar_plan,
    }
    role = resolve_role(goal, dataset)
    rec = role.recommended_skills if role else frozenset()
    if planner_name == "astar":
        return astar_plan(
            profile.initial_skills,
            goal.target_skill_ids,
            dataset.courses,
            profile,
            recommended_skills=rec,
        )
    return planners[planner_name](
        profile.initial_skills,
        goal.target_skill_ids,
        dataset.courses,
        profile,
    )


def run_planner_k(
    planner_name: str,
    k: int,
    goal: GoalSpec,
    dataset: Dataset,
    profile: StudentProfile,
    max_nodes: int = 0,
) -> list[PlanResult]:
    """Devuelve hasta k planes. Solo A* soporta k>1; otros planificadores devuelven 1 plan."""
    if planner_name == "astar":
        role = resolve_role(goal, dataset)
        rec = role.recommended_skills if role else frozenset()
        plans = astar_k_plans(
            profile.initial_skills,
            goal.target_skill_ids,
            dataset.courses,
            profile,
            k=k,
            max_nodes=max_nodes,
            recommended_skills=rec,
        )
        for p in plans:
            p.planner_name = "astar"
        return plans
    return [run_planner_single(planner_name, goal, dataset, profile)]


def _enrich_plan(
    plan: PlanResult,
    goal: GoalSpec,
    dataset: Dataset,
    profile: StudentProfile,
    provider: str,
    settings: Settings,
    monte_carlo_runs: int,
    monte_carlo_seed: int,
    evaluate: bool,
) -> PlanResult:
    """Adjunta Monte Carlo, evaluación LLM y score final al plan."""
    if monte_carlo_runs > 0:
        plan = attach_monte_carlo_to_plan(
            plan,
            dataset.courses,
            profile,
            runs=monte_carlo_runs,
            seed=monte_carlo_seed,
        )

    if evaluate:
        role = resolve_role(goal, dataset)
        if role is not None:
            use_mock = provider == "mock"
            eval_client = LLMClient(settings) if not use_mock else None
            plan.llm_evaluation = evaluate_plan_with_llm(
                plan, role, goal, dataset, eval_client, use_mock=use_mock
            )

    role = resolve_role(goal, dataset)
    if role is not None and plan.valid:
        score, breakdown = compute_score_with_breakdown(plan, role, profile)
        plan.final_score = score
        plan.score_breakdown = breakdown

    return plan


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

    try:
        goal = validate_goal_spec(raw_goal_spec, dataset)
    except ValueError as exc:
        print(_section(
            "Objetivo no planificable",
            f"No se pudo construir un objetivo a partir de la consulta: {exc}\n\n"
            "Reformula el objetivo hacia un rol o unas "
            "habilidades que el dataset cubra.",
        ))
        return 1
    print(_section("GOALSPEC VALIDADO", _pretty_json(asdict(goal))))

    validation_profile = build_profile_from_goal(goal)

    # --- Modo validación manual de trayectoria ---
    course_ids = _parse_course_ids(args.courses)
    if course_ids:
        valid, errors = validate_plan(
            course_ids,
            validation_profile.initial_skills,
            goal.target_skill_ids,
            dataset.courses,
            validation_profile,
        )
        validation_result = {
            "mode": "manual",
            "course_ids": course_ids,
            "initial_skills": sorted(validation_profile.initial_skills),
            "target_skills": sorted(goal.target_skill_ids),
            "max_weeks": validation_profile.max_weeks,
            "max_weekly_hours": validation_profile.max_weekly_hours,
            "valid": valid,
            "errors": errors,
        }
        if valid and args.monte_carlo_runs > 0:
            validation_result["monte_carlo"] = evaluate_plan_monte_carlo(
                course_ids,
                dataset.courses,
                validation_profile,
                runs=args.monte_carlo_runs,
                seed=args.monte_carlo_seed,
            )
        print(_section("VALIDACION FORMAL DE LA TRAYECTORIA", _pretty_json(validation_result)))
        return 0

    # --- Modo planificador automático ---
    k = args.k
    plans = run_planner_k(args.planner, k, goal, dataset, validation_profile, max_nodes=DEFAULT_MAX_NODES)

    if not plans:
        print(_section(f"PLANIFICADOR {args.planner.upper()}", "No se encontro ninguna trayectoria valida."))
        return 1

    enriched: list[PlanResult] = []
    for plan in plans:
        enriched.append(
            _enrich_plan(
                plan,
                goal,
                dataset,
                validation_profile,
                provider,
                settings,
                args.monte_carlo_runs,
                args.monte_carlo_seed,
                args.evaluate,
            )
        )

    ranked = rank_plans(enriched)

    if k == 1 or len(ranked) == 1:
        print(
            _section(
                f"PLAN GENERADO POR {args.planner.upper()}",
                _pretty_json(asdict(ranked[0])),
            )
        )
    else:
        # Mostrar el mejor plan completo y un resumen del ranking
        best = ranked[0]
        print(
            _section(
                f"MEJOR PLAN (1/{len(ranked)}) — {args.planner.upper()}",
                _pretty_json(asdict(best)),
            )
        )

        ranking_summary = [
            {
                "rank": i + 1,
                "course_ids": p.course_ids,
                "total_weeks": p.total_weeks,
                "valid": p.valid,
                "final_score": p.final_score,
                "mc_success": (
                    p.monte_carlo.get("success_probability")
                    if p.monte_carlo and not p.monte_carlo.get("skipped")
                    else None
                ),
                "llm_quality": (
                    p.llm_evaluation.get("global_quality")
                    if p.llm_evaluation and not p.llm_evaluation.get("skipped")
                    else None
                ),
            }
            for i, p in enumerate(ranked)
        ]
        print(_section(f"RANKING DE {len(ranked)} PLANES", _pretty_json(ranking_summary)))

    # --- Métricas CSV ---
    if args.metrics or args.metrics_output:
        rows = [
            extract_metrics_row(p, instance_id="cli", target_skill_ids=goal.target_skill_ids)
            for p in ranked
        ]
        csv_content = rows_to_csv_string(rows)
        if args.metrics:
            print(_section("METRICAS CSV", csv_content))
        if args.metrics_output:
            out_path = pathlib.Path(args.metrics_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(csv_content, encoding="utf-8")
            print(f"\nMetricas guardadas en: {out_path.resolve()}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI del Skill Path Planner.")
    parser.add_argument(
        "--goal",
        required=True,
        help="Objetivo profesional en lenguaje natural.",
    )
    parser.add_argument(
        "--courses",
        default="",
        help="IDs de cursos separados por coma para validar una trayectoria manual. Si se omite, se usa --planner.",
    )
    parser.add_argument(
        "--planner",
        choices=["greedy", "ucs", "astar"],
        default="astar",
        help="Planificador automatico cuando no se pasan cursos manuales.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Numero de trayectorias candidatas a generar (solo A* soporta k>1).",
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
    parser.add_argument(
        "--monte-carlo-runs",
        type=int,
        default=0,
        help="Numero de simulaciones Monte Carlo. 0 lo desactiva.",
    )
    parser.add_argument(
        "--monte-carlo-seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad de Monte Carlo.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluar cualitativamente el plan con el LLM Evaluator.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        default=False,
        help="Imprimir fila CSV de metricas en terminal.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        metavar="ARCHIVO.csv",
        help="Ruta de archivo CSV donde guardar las metricas (crea directorios si no existen).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    raise SystemExit(run_cli(parser.parse_args()))


if __name__ == "__main__":
    main()

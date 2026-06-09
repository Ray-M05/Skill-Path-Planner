"""Ejecuta variantes del experimento sobre las instancias y guarda un CSV de metricas."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset.loader import load_dataset, profile_from_instance
from src.llm.evaluator import evaluate_plan_with_llm
from src.metrics import csv_fields, extract_metrics_row
from src.models import Dataset, GoalSpec, PlanResult, StudentProfile
from src.planners.astar import astar_k_plans
from src.planners.common import make_failure_result
from src.planners.greedy import greedy_plan
from src.planners.ucs import ucs_plan
from src.scoring import compute_score_with_breakdown
from src.simulation.monte_carlo import attach_monte_carlo_to_plan

VARIANTS = ["greedy", "ucs", "astar", "astar_mc", "astar_mc_llm"]


def build_goal_from_instance(instance: dict, dataset: Dataset) -> GoalSpec:
    """Construye GoalSpec directamente desde expected_role_id y el perfil (sin LLM)."""
    role_id = instance["expected_role_id"]
    role = dataset.roles[role_id]
    profile = profile_from_instance(instance)
    return GoalSpec(
        role_id=role_id,
        target_skill_ids=set(role.required_skills),
        initial_skill_ids=set(profile.initial_skills),
        mentioned_skill_ids=set(),
        constraints={},
        ignored_constraints=[],
        unknown_skill_mentions=[],
        confidence=1.0,
    )


def run_variant(
    variant: str,
    goal: GoalSpec,
    dataset: Dataset,
    profile: StudentProfile,
    mc_runs: int,
    seed: int,
    max_nodes: int,
) -> PlanResult:
    target = goal.target_skill_ids
    initial = profile.initial_skills
    courses = dataset.courses

    start_time = time.perf_counter()

    if variant == "greedy":
        plan = greedy_plan(initial, target, courses, profile)
    elif variant == "ucs":
        plan = ucs_plan(initial, target, courses, profile)
    else:
        role = dataset.roles.get(goal.role_id)
        rec = role.recommended_skills if role else frozenset()
        plans = astar_k_plans(initial, target, courses, profile, k=1, max_nodes=max_nodes, recommended_skills=rec)
        if plans:
            plan = plans[0]
            plan.planner_name = variant
        else:
            plan = make_failure_result(
                variant, initial,
                expanded_nodes=0, max_frontier_size=0,
                start_time=start_time,
                message="A* no encontro trayectoria.",
            )

    if variant in ("astar_mc", "astar_mc_llm") and mc_runs > 0:
        plan = attach_monte_carlo_to_plan(plan, courses, profile, runs=mc_runs, seed=seed)

    if variant == "astar_mc_llm":
        role = dataset.roles.get(goal.role_id)
        if role is not None:
            plan.llm_evaluation = evaluate_plan_with_llm(
                plan, role, goal, dataset, llm_client=None, use_mock=True
            )

    role = dataset.roles.get(goal.role_id)
    if role is not None and plan.valid:
        score, breakdown = compute_score_with_breakdown(plan, role, profile)
        plan.final_score = score
        plan.score_breakdown = breakdown

    return plan


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta experimentos comparativos del Skill Path Planner.")
    parser.add_argument("--config", default=None, help="Archivo JSON de configuracion base.")
    parser.add_argument("--instances", default=None, help="Ruta a instances.json.")
    parser.add_argument("--data-dir", default=None, help="Directorio del dataset.")
    parser.add_argument("--variants", nargs="+", default=None, help="Variantes a ejecutar.")
    parser.add_argument("--monte-carlo-runs", type=int, default=None)
    parser.add_argument("--monte-carlo-seed", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--output", default=None, help="Ruta del CSV de salida.")
    parser.add_argument("--append", action="store_true", default=False,
                        help="Agregar filas a un CSV existente sin sobreescribir.")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict:
    """Fusiona config JSON (base) con args CLI (override)."""
    cfg = load_config(args.config)
    if args.instances is not None:
        cfg["instances"] = args.instances
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.variants is not None:
        cfg["variants"] = args.variants
    if args.monte_carlo_runs is not None:
        cfg["monte_carlo_runs"] = args.monte_carlo_runs
    if args.monte_carlo_seed is not None:
        cfg["monte_carlo_seed"] = args.monte_carlo_seed
    if args.max_nodes is not None:
        cfg["max_nodes"] = args.max_nodes
    if args.output is not None:
        cfg["output"] = args.output
    # Defaults
    cfg.setdefault("instances", "data/instances.json")
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("variants", VARIANTS)
    cfg.setdefault("monte_carlo_runs", 500)
    cfg.setdefault("monte_carlo_seed", 42)
    cfg.setdefault("max_nodes", 0)
    cfg.setdefault("output", "experiments/results/raw_results.csv")
    cfg["append"] = args.append
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_args(argv)
    cfg = resolve_config(args)

    dataset = load_dataset(cfg["data_dir"])
    instances_path = Path(cfg["instances"])
    instances: list[dict] = json.loads(instances_path.read_text(encoding="utf-8"))

    output_path = Path(cfg["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if cfg["append"] and output_path.exists() else "w"

    total = len(instances) * len(cfg["variants"])
    done = 0

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields(), extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        for instance in instances:
            if not isinstance(instance.get("student"), dict):
                print(f"[SKIP] {instance['id']}: la instancia no tiene bloque 'student'.")
                done += len(cfg["variants"])
                continue

            profile = profile_from_instance(instance)
            goal = build_goal_from_instance(instance, dataset)

            for variant in cfg["variants"]:
                done += 1
                try:
                    plan = run_variant(
                        variant, goal, dataset, profile,
                        cfg["monte_carlo_runs"],
                        cfg["monte_carlo_seed"],
                        cfg["max_nodes"],
                    )
                    row = extract_metrics_row(
                        plan,
                        instance_id=instance["id"],
                        target_skill_ids=goal.target_skill_ids,
                    )
                    writer.writerow(row)
                    f.flush()
                    status = "OK" if plan.valid else "FAIL"
                    score_str = f"{plan.final_score:.4f}" if plan.final_score is not None else "n/a"
                    print(f"  [{done:3}/{total}] {status} {instance['id']} / {variant}  score={score_str}")
                except Exception:
                    print(f"  [ERROR] {instance['id']} / {variant}")
                    traceback.print_exc()

    print(f"\nResultados guardados en: {output_path.resolve()}")
    print(f"Filas escritas: {done} ({len(instances)} instancias × {len(cfg['variants'])} variantes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
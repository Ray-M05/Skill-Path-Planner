from __future__ import annotations

import csv
import io
from typing import Any
from .models import PlanResult

_CSV_FIELDS = [
    "instance_id",
    "variant",
    "success",
    "valid",
    "total_weeks",
    "total_difficulty",
    "plan_length",
    "coverage",
    "expanded_nodes",
    "max_frontier_size",
    "runtime_seconds",
    "mc_success_probability",
    "mc_expected_weeks",
    "mc_expected_failed_courses",
    "llm_global_quality",
    "final_score",
]


def extract_metrics_row(
    plan: PlanResult,
    instance_id: str = "",
    target_skill_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    mc = plan.monte_carlo or {}
    llm = plan.llm_evaluation or {}

    if target_skill_ids:
        total = len(target_skill_ids)
        reached = len(plan.reached_skills & set(target_skill_ids))
        coverage = round(reached / total, 4) if total > 0 else 0.0
    else:
        coverage = 1.0 if plan.valid else 0.0

    mc_skipped = mc.get("skipped", False)
    llm_skipped = llm.get("skipped", False)

    # Score final: un plan invalido/fallido vale 0 (no se excluye del promedio)
    if plan.final_score is not None:
        final_score: float | str = plan.final_score
    elif not plan.valid:
        final_score = 0.0
    else:
        final_score = ""

    return {
        "instance_id": instance_id,
        "variant": plan.planner_name,
        "success": 1 if plan.course_ids else 0,
        "valid": 1 if plan.valid else 0,
        "total_weeks": plan.total_weeks,
        "total_difficulty": round(plan.total_difficulty, 4),
        "plan_length": len(plan.course_ids),
        "coverage": coverage,
        "expanded_nodes": plan.expanded_nodes,
        "max_frontier_size": plan.max_frontier_size,
        "runtime_seconds": round(plan.runtime_seconds, 6),
        "mc_success_probability": "" if mc_skipped else mc.get("success_probability", ""),
        "mc_expected_weeks": "" if mc_skipped else mc.get("expected_weeks", ""),
        "mc_expected_failed_courses": "" if mc_skipped else mc.get("expected_failed_courses", ""),
        "llm_global_quality": "" if llm_skipped else llm.get("global_quality", ""),
        "final_score": final_score,
    }


def rows_to_csv_string(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def csv_fields() -> list[str]:
    return list(_CSV_FIELDS)
from __future__ import annotations

import csv
import io

from src.metrics import _CSV_FIELDS, csv_fields, extract_metrics_row, rows_to_csv_string
from src.models import PlanResult


def _make_plan(
    course_ids: list[str],
    reached_skills: set[str],
    valid: bool = True,
    monte_carlo: dict | None = None,
    llm_evaluation: dict | None = None,
    final_score: float | None = None,
) -> PlanResult:
    return PlanResult(
        planner_name="astar",
        course_ids=course_ids,
        reached_skills=reached_skills,
        total_weeks=10,
        total_difficulty=0.5,
        expanded_nodes=5,
        max_frontier_size=3,
        runtime_seconds=0.012,
        valid=valid,
        monte_carlo=monte_carlo,
        llm_evaluation=llm_evaluation,
        final_score=final_score,
    )


# ---------------------------------------------------------------------------
# csv_fields
# ---------------------------------------------------------------------------


def test_csv_fields_returns_expected_columns() -> None:
    fields = csv_fields()
    assert "instance_id" in fields
    assert "variant" in fields
    assert "mc_success_probability" in fields
    assert "llm_global_quality" in fields
    assert "final_score" in fields
    assert len(fields) == len(_CSV_FIELDS)


# ---------------------------------------------------------------------------
# extract_metrics_row
# ---------------------------------------------------------------------------


def test_extract_metrics_row_has_all_fields() -> None:
    plan = _make_plan(["c1", "c2"], reached_skills={"skill_a"})

    row = extract_metrics_row(plan)

    for field in _CSV_FIELDS:
        assert field in row, f"Campo faltante: {field}"


def test_extract_metrics_row_basic_values() -> None:
    plan = _make_plan(["c1", "c2"], reached_skills={"skill_a"}, valid=True, final_score=0.75)

    row = extract_metrics_row(plan, instance_id="inst_001")

    assert row["instance_id"] == "inst_001"
    assert row["variant"] == "astar"
    assert row["valid"] == 1
    assert row["success"] == 1
    assert row["plan_length"] == 2
    assert row["total_weeks"] == 10
    assert row["final_score"] == 0.75


def test_extract_metrics_row_invalid_plan() -> None:
    plan = _make_plan([], reached_skills=set(), valid=False)

    row = extract_metrics_row(plan)

    assert row["valid"] == 0
    assert row["success"] == 0
    assert row["coverage"] == 0.0


def test_extract_metrics_row_coverage_with_target_skills() -> None:
    plan = _make_plan(["c1"], reached_skills={"skill_a", "skill_b"})
    target = {"skill_a", "skill_b", "skill_c"}

    row = extract_metrics_row(plan, target_skill_ids=target)

    assert row["coverage"] == pytest.approx(2 / 3, rel=1e-4)


def test_extract_metrics_row_mc_fields_populated() -> None:
    mc = {
        "success_probability": 0.82,
        "expected_weeks": 45.5,
        "expected_failed_courses": 0.3,
    }
    plan = _make_plan(["c1"], reached_skills={"skill_a"}, monte_carlo=mc)

    row = extract_metrics_row(plan)

    assert row["mc_success_probability"] == 0.82
    assert row["mc_expected_weeks"] == 45.5
    assert row["mc_expected_failed_courses"] == 0.3


def test_extract_metrics_row_mc_skipped_gives_empty() -> None:
    plan = _make_plan(
        ["c1"], reached_skills={"skill_a"},
        monte_carlo={"skipped": True, "reason": "invalid"},
    )

    row = extract_metrics_row(plan)

    assert row["mc_success_probability"] == ""
    assert row["mc_expected_weeks"] == ""


def test_extract_metrics_row_llm_field_populated() -> None:
    plan = _make_plan(
        ["c1"], reached_skills={"skill_a"},
        llm_evaluation={"global_quality": 0.91, "goal_alignment": 0.88},
    )

    row = extract_metrics_row(plan)

    assert row["llm_global_quality"] == 0.91


def test_extract_metrics_row_llm_skipped_gives_empty() -> None:
    plan = _make_plan(
        ["c1"], reached_skills={"skill_a"},
        llm_evaluation={"skipped": True, "reason": "no courses"},
    )

    row = extract_metrics_row(plan)

    assert row["llm_global_quality"] == ""


def test_extract_metrics_row_no_final_score_gives_empty() -> None:
    plan = _make_plan(["c1"], reached_skills={"skill_a"}, final_score=None)

    row = extract_metrics_row(plan)

    assert row["final_score"] == ""


# ---------------------------------------------------------------------------
# rows_to_csv_string
# ---------------------------------------------------------------------------


def test_rows_to_csv_string_has_header() -> None:
    plan = _make_plan(["c1"], reached_skills={"skill_a"})
    rows = [extract_metrics_row(plan, instance_id="i1")]

    csv_str = rows_to_csv_string(rows)

    header_line = csv_str.splitlines()[0]
    assert "instance_id" in header_line
    assert "final_score" in header_line


def test_rows_to_csv_string_is_parseable() -> None:
    plans = [
        _make_plan(["c1"], reached_skills={"skill_a"}, final_score=0.7),
        _make_plan(["c2"], reached_skills={"skill_b"}, final_score=0.5),
    ]
    rows = [extract_metrics_row(p, instance_id=f"i{i}") for i, p in enumerate(plans)]

    csv_str = rows_to_csv_string(rows)

    reader = csv.DictReader(io.StringIO(csv_str))
    parsed = list(reader)
    assert len(parsed) == 2
    assert parsed[0]["instance_id"] == "i0"
    assert parsed[1]["instance_id"] == "i1"


def test_rows_to_csv_string_empty_rows() -> None:
    csv_str = rows_to_csv_string([])

    lines = [l for l in csv_str.splitlines() if l]
    assert len(lines) == 1  # solo cabecera


import pytest

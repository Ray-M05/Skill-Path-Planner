from src.dataset.loader import load_dataset
from src.main import (
    _extract_constraints,
    _parse_course_ids,
    _pretty_json,
    build_mock_goal_response,
    build_profile_from_goal,
)
from src.llm.interpreter import validate_goal_spec


def test_parse_course_ids_splits_comma_separated_values() -> None:
    assert _parse_course_ids("course_a, course_b,,course_c") == [
        "course_a",
        "course_b",
        "course_c",
    ]


def test_pretty_json_serializes_sets() -> None:
    rendered = _pretty_json({"skills": {"skill_sql_basic", "skill_python_basic"}})

    assert "skill_python_basic" in rendered
    assert "skill_sql_basic" in rendered


def test_extract_constraints_from_natural_language() -> None:
    constraints = _extract_constraints(
        "Quiero ser ingeniero de machine learning en 18 meses y puedo estudiar 10 horas semanales."
    )

    assert constraints["max_weeks"] == 72
    assert constraints["max_weekly_hours"] == 10
    assert constraints["preferred_pace"] == "18 meses"


def test_mock_goal_response_extracts_role_skills_and_constraints() -> None:
    dataset = load_dataset("data")

    response = build_mock_goal_response(
        "Quiero ser ingeniero de machine learning en 18 meses, se Python basico y quiero cursos gratis.",
        dataset,
    )

    assert response["role_id"] == "role_ml_engineer"
    assert "skill_python_basic" in response["initial_skill_ids"]
    assert response["constraints"]["max_weeks"] == 72
    assert response["ignored_constraints"] == ["gratis"]


def test_profile_from_goal_uses_extracted_constraints() -> None:
    dataset = load_dataset("data")
    raw_goal = build_mock_goal_response(
        "Quiero ser analista de datos en 12 semanas, se SQL basico y puedo dedicar 8 horas semanales.",
        dataset,
    )
    goal = validate_goal_spec(raw_goal, dataset)

    profile = build_profile_from_goal(goal)

    assert profile.id == "profile_from_query"
    assert profile.max_weeks == 12
    assert profile.max_weekly_hours == 8
    assert "skill_sql_basic" in profile.initial_skills

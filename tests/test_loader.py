import json
from pathlib import Path

import pytest

from src.dataset.loader import (
    load_courses,
    load_dataset,
    load_instances,
    load_profiles,
    load_roles,
    load_skills,
)
from src.models import Dataset


def test_load_dataset_from_data_directory() -> None:
    dataset = load_dataset("data")

    assert isinstance(dataset, Dataset)
    assert "skill_python_basic" in dataset.skills
    assert "course_machine_learning_basic" in dataset.courses
    assert "role_ml_engineer" in dataset.roles
    assert "profile_beginner" in dataset.profiles
    assert len(dataset.instances) == 28


def test_load_courses_rejects_unknown_skill(tmp_path: Path) -> None:
    skills = load_skills("data/skills.json")
    courses_path = tmp_path / "courses.json"
    courses_path.write_text(
        json.dumps(
            [
                {
                    "id": "course_bad",
                    "name": "Curso malo",
                    "description": "",
                    "prerequisites": ["skill_missing"],
                    "outcomes": ["skill_python_basic"],
                    "duration_weeks": 4,
                    "weekly_hours": 4,
                    "difficulty": 0.2,
                    "pass_base_probability": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="skill inexistente"):
        load_courses(courses_path, skills)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("duration_weeks", 0),
        ("weekly_hours", 0),
        ("difficulty", 1.5),
        ("pass_base_probability", -0.1),
    ],
)
def test_load_courses_rejects_invalid_numeric_values(
    tmp_path: Path,
    field_name: str,
    bad_value: float,
) -> None:
    skills = load_skills("data/skills.json")
    item = {
        "id": "course_bad",
        "name": "Curso malo",
        "description": "",
        "prerequisites": [],
        "outcomes": ["skill_python_basic"],
        "duration_weeks": 4,
        "weekly_hours": 4,
        "difficulty": 0.2,
        "pass_base_probability": 0.9,
    }
    item[field_name] = bad_value
    courses_path = tmp_path / "courses.json"
    courses_path.write_text(json.dumps([item]), encoding="utf-8")

    with pytest.raises(ValueError, match=field_name):
        load_courses(courses_path, skills)


def test_load_profiles_rejects_invalid_limits(tmp_path: Path) -> None:
    skills = load_skills("data/skills.json")
    profiles_path = tmp_path / "student_profiles.json"
    profiles_path.write_text(
        json.dumps(
            [
                {
                    "id": "profile_bad",
                    "initial_skills": ["skill_python_basic"],
                    "max_weeks": 0,
                    "max_weekly_hours": 10,
                    "risk_tolerance": 0.4,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_weeks"):
        load_profiles(profiles_path, skills)


def test_load_roles_rejects_unknown_skill(tmp_path: Path) -> None:
    skills = load_skills("data/skills.json")
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(
        json.dumps(
            [
                {
                    "id": "role_bad",
                    "name": "Rol malo",
                    "required_skills": ["skill_missing"],
                    "recommended_skills": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="skill inexistente"):
        load_roles(roles_path, skills)


def test_load_instances_rejects_unknown_profile_or_role(tmp_path: Path) -> None:
    skills = load_skills("data/skills.json")
    roles = load_roles("data/roles.json", skills)
    profiles = load_profiles("data/student_profiles.json", skills)
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            [
                {
                    "id": "inst_bad",
                    "profile_id": "profile_missing",
                    "goal_text": "Quiero ser analista de datos.",
                    "expected_role_id": "role_data_analyst",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="perfil inexistente"):
        load_instances(instances_path, profiles, roles)


def test_load_skills_rejects_duplicate_ids(tmp_path: Path) -> None:
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            [
                {
                    "id": "skill_python_basic",
                    "name": "Python basico",
                    "aliases": [],
                    "category": "programming",
                },
                {
                    "id": "skill_python_basic",
                    "name": "Python repetido",
                    "aliases": [],
                    "category": "programming",
                },
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicado"):
        load_skills(skills_path)

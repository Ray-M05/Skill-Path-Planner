from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from ..models import Course, Dataset, Role, Skill, StudentProfile


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _ensure_unique_ids(items: Iterable[dict[str, Any]], source_name: str) -> None:
    seen: set[str] = set()
    for item in items:
        item_id = item.get("id")
        if not item_id:
            raise ValueError(f"{source_name} contiene un registro sin id.")
        if item_id in seen:
            raise ValueError(f"{source_name} contiene un id duplicado: {item_id}.")
        seen.add(item_id)


def _ensure_skill_exists(skill_id: str, skills: dict[str, Skill], context: str) -> None:
    if skill_id not in skills:
        raise ValueError(f"{context} referencia una skill inexistente: {skill_id}.")


def _ensure_positive_int(value: Any, field_name: str, context: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} debe tener {field_name} > 0.")


def _ensure_between_zero_and_one(value: Any, field_name: str, context: str) -> None:
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{context} debe tener {field_name} entre 0 y 1.")


def load_skills(path: str | Path) -> dict[str, Skill]:
    raw_skills = _read_json(path)
    _ensure_unique_ids(raw_skills, "skills.json")
    return {
        item["id"]: Skill(
            id=item["id"],
            name=item["name"],
            aliases=list(item.get("aliases", [])),
            category=item["category"],
        )
        for item in raw_skills
    }


def load_courses(path: str | Path, skills: dict[str, Skill]) -> dict[str, Course]:
    raw_courses = _read_json(path)
    _ensure_unique_ids(raw_courses, "courses.json")
    courses: dict[str, Course] = {}

    for item in raw_courses:
        context = f"El curso {item.get('id')}"
        _ensure_positive_int(item.get("duration_weeks"), "duration_weeks", context)
        _ensure_positive_int(item.get("weekly_hours"), "weekly_hours", context)
        _ensure_between_zero_and_one(item.get("difficulty"), "difficulty", context)
        _ensure_between_zero_and_one(
            item.get("pass_base_probability"),
            "pass_base_probability",
            context,
        )

        prerequisites = frozenset(item.get("prerequisites", []))
        outcomes = frozenset(item.get("outcomes", []))
        for skill_id in prerequisites:
            _ensure_skill_exists(skill_id, skills, context)
        for skill_id in outcomes:
            _ensure_skill_exists(skill_id, skills, context)

        courses[item["id"]] = Course(
            id=item["id"],
            name=item["name"],
            description=item.get("description", ""),
            prerequisites=prerequisites,
            outcomes=outcomes,
            duration_weeks=item["duration_weeks"],
            weekly_hours=item["weekly_hours"],
            difficulty=float(item["difficulty"]),
            pass_base_probability=float(item["pass_base_probability"]),
        )

    return courses


def load_roles(path: str | Path, skills: dict[str, Skill]) -> dict[str, Role]:
    raw_roles = _read_json(path)
    _ensure_unique_ids(raw_roles, "roles.json")
    roles: dict[str, Role] = {}

    for item in raw_roles:
        context = f"El rol {item.get('id')}"
        required_skills = frozenset(item.get("required_skills", []))
        recommended_skills = frozenset(item.get("recommended_skills", []))
        for skill_id in required_skills | recommended_skills:
            _ensure_skill_exists(skill_id, skills, context)

        roles[item["id"]] = Role(
            id=item["id"],
            name=item["name"],
            required_skills=required_skills,
            recommended_skills=recommended_skills,
        )

    return roles


def _build_profile(
    data: dict[str, Any],
    skills: dict[str, Skill],
    context: str,
    profile_id: str,
) -> StudentProfile:
    _ensure_positive_int(data.get("max_weeks"), "max_weeks", context)
    _ensure_positive_int(data.get("max_weekly_hours"), "max_weekly_hours", context)
    _ensure_between_zero_and_one(data.get("risk_tolerance"), "risk_tolerance", context)

    initial_skills = frozenset(data.get("initial_skills", []))
    for skill_id in initial_skills:
        _ensure_skill_exists(skill_id, skills, context)

    return StudentProfile(
        id=profile_id,
        initial_skills=initial_skills,
        max_weeks=data["max_weeks"],
        max_weekly_hours=data["max_weekly_hours"],
        risk_tolerance=float(data["risk_tolerance"]),
    )


def profile_from_instance(
    instance: dict[str, Any],
    skills: dict[str, Skill] | None = None,
) -> StudentProfile:
    instance_id = instance.get("id")
    student = instance.get("student")
    if not isinstance(student, dict):
        raise ValueError(f"La instancia {instance_id} no tiene un bloque 'student'.")
    label = student.get("label") or instance_id
    if skills is not None:
        context = f"El estudiante de la instancia {instance_id}"
        return _build_profile(student, skills, context, label)
    return StudentProfile(
        id=label,
        initial_skills=frozenset(student.get("initial_skills", [])),
        max_weeks=student["max_weeks"],
        max_weekly_hours=student["max_weekly_hours"],
        risk_tolerance=float(student["risk_tolerance"]),
    )


def load_instances(
    path: str | Path,
    skills: dict[str, Skill] | None = None,
    roles: dict[str, Role] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, StudentProfile]]:
    if not Path(path).exists():
        return [], {}
    raw_instances = _read_json(path)
    _ensure_unique_ids(raw_instances, "instances.json")
    profiles: dict[str, StudentProfile] = {}
    for item in raw_instances:
        instance_id = item.get("id")
        role_id = item.get("expected_role_id")
        if roles is not None and role_id not in roles:
            raise ValueError(
                f"La instancia {instance_id} referencia un rol inexistente: {role_id}."
            )
        if skills is not None:
            profile = profile_from_instance(item, skills)
            profiles[profile.id] = profile
    return list(raw_instances), profiles


def load_dataset(data_dir: str | Path) -> Dataset:
    data_path = Path(data_dir)
    skills = load_skills(data_path / "skills.json")
    courses = load_courses(data_path / "courses.json", skills)
    roles = load_roles(data_path / "roles.json", skills)
    instances, profiles = load_instances(data_path / "instances.json", skills, roles)
    return Dataset(
        skills=skills,
        courses=courses,
        roles=roles,
        profiles=profiles,
        instances=instances,
    )

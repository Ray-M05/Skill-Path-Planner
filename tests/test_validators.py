from src.dataset.loader import load_dataset
from src.validators import validate_plan

def test_validate_plan_accepts_valid_ml_engineer_path() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    target_skills = dataset.roles["role_ml_engineer"].required_skills
    course_ids = [
        "course_python_intermediate",
        "course_statistics_basic",
        "course_machine_learning_basic",
        "course_deep_learning_basic",
        "course_mlops_basic",
    ]

    valid, errors = validate_plan(
        course_ids,
        profile.initial_skills,
        target_skills,
        dataset.courses,
        profile,
    )

    assert valid
    assert errors == []


def test_validate_plan_rejects_unknown_course() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]

    valid, errors = validate_plan(
        ["course_missing"],
        profile.initial_skills,
        {"skill_python_intermediate"},
        dataset.courses,
        profile,
    )

    assert not valid
    assert "El curso course_missing no existe." in errors


def test_validate_plan_rejects_repeated_course() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]

    valid, errors = validate_plan(
        ["course_python_intermediate", "course_python_intermediate"],
        profile.initial_skills,
        {"skill_python_intermediate"},
        dataset.courses,
        profile,
    )

    assert not valid
    assert any("repetido" in error for error in errors)


def test_validate_plan_rejects_prerequisite_violation() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]

    valid, errors = validate_plan(
        ["course_machine_learning_basic"],
        profile.initial_skills,
        {"skill_machine_learning_basic"},
        dataset.courses,
        profile,
    )

    assert not valid
    assert any("skill_python_intermediate" in error for error in errors)
    assert any("skill_statistics_basic" in error for error in errors)


def test_validate_plan_rejects_max_weeks_violation() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_data_foundation"]
    limited_profile = type(profile)(
        id="profile_short",
        initial_skills=profile.initial_skills,
        max_weeks=20,
        max_weekly_hours=profile.max_weekly_hours,
        risk_tolerance=profile.risk_tolerance,
    )

    valid, errors = validate_plan(
        [
            "course_data_analysis_basic",
            "course_machine_learning_basic",
            "course_deep_learning_basic",
            "course_mlops_basic",
            "course_statistics_basic",
        ],
        limited_profile.initial_skills,
        set(),
        dataset.courses,
        limited_profile,
    )

    assert not valid
    assert any("semanas" in error for error in errors)


def test_validate_plan_rejects_weekly_hours_violation() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_data_foundation"]
    limited_profile = type(profile)(
        id="profile_limited",
        initial_skills=profile.initial_skills,
        max_weeks=profile.max_weeks,
        max_weekly_hours=5,
        risk_tolerance=profile.risk_tolerance,
    )

    valid, errors = validate_plan(
        ["course_data_analysis_basic"],
        limited_profile.initial_skills,
        {"skill_data_analysis_basic"},
        dataset.courses,
        limited_profile,
    )

    assert not valid
    assert any("horas semanales" in error for error in errors)


def test_validate_plan_rejects_missing_target_skill() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]

    valid, errors = validate_plan(
        ["course_python_intermediate"],
        profile.initial_skills,
        {"skill_mlops_basic"},
        dataset.courses,
        profile,
    )

    assert not valid
    assert "La trayectoria no alcanza skill_mlops_basic." in errors

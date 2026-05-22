from __future__ import annotations

from .models import Course, StudentProfile


def validate_plan(
    course_ids: list[str],
    initial_skills: set[str],
    target_skills: set[str],
    courses: dict[str, Course],
    profile: StudentProfile,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    available_skills = set(initial_skills)
    taken_courses: set[str] = set()
    total_weeks = 0

    for course_id in course_ids:
        if course_id not in courses:
            errors.append(f"El curso {course_id} no existe.")
            continue

        if course_id in taken_courses:
            errors.append(f"El curso {course_id} aparece repetido en la trayectoria.")
            continue

        course = courses[course_id]
        missing_prerequisites = course.prerequisites - available_skills
        for skill_id in sorted(missing_prerequisites):
            errors.append(
                f"{course_id} requiere {skill_id}, pero esa habilidad no esta disponible antes de tomar el curso."
            )

        if course.weekly_hours > profile.max_weekly_hours:
            errors.append(
                f"El curso {course_id} requiere {course.weekly_hours} horas semanales, "
                f"pero el perfil solo permite {profile.max_weekly_hours}."
            )

        total_weeks += course.duration_weeks
        taken_courses.add(course_id)
        available_skills.update(course.outcomes)

    if total_weeks > profile.max_weeks:
        errors.append(
            f"La trayectoria requiere {total_weeks} semanas, pero el perfil solo permite {profile.max_weeks}."
        )

    missing_target_skills = target_skills - available_skills
    for skill_id in sorted(missing_target_skills):
        errors.append(f"La trayectoria no alcanza {skill_id}.")

    return len(errors) == 0, errors

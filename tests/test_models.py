from src.models import Course, Dataset, PlanResult, Role, SearchState, Skill, StudentProfile

def test_dataclasses_can_be_instantiated() -> None:
    skill = Skill(
        id="skill_python_basic",
        name="Python basico",
        aliases=["python"],
        category="programming",
    )
    course = Course(
        id="course_python",
        name="Python",
        description="Intro",
        prerequisites=frozenset(),
        outcomes=frozenset({skill.id}),
        duration_weeks=4,
        weekly_hours=5,
        difficulty=0.2,
        pass_base_probability=0.9,
    )
    role = Role(
        id="role_data_analyst",
        name="Analista de datos",
        required_skills=frozenset({skill.id}),
        recommended_skills=frozenset(),
    )
    profile = StudentProfile(
        id="profile_beginner",
        initial_skills=frozenset(),
        max_weeks=20,
        max_weekly_hours=10,
        risk_tolerance=0.4,
    )
    dataset = Dataset(
        skills={skill.id: skill},
        courses={course.id: course},
        roles={role.id: role},
        profiles={profile.id: profile},
    )
    state = SearchState(
        skills=frozenset({skill.id}),
        taken_courses=(course.id,),
        weeks_used=4,
        difficulty_sum=0.2,
    )
    result = PlanResult(
        planner_name="test",
        course_ids=[course.id],
        reached_skills={skill.id},
        total_weeks=4,
        total_difficulty=0.2,
        expanded_nodes=1,
        max_frontier_size=1,
        runtime_seconds=0.01,
    )

    assert dataset.skills[skill.id] == skill
    assert state.skills == frozenset({skill.id})
    assert result.valid is False
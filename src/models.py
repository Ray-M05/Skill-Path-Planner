from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    aliases: list[str]
    category: str


@dataclass(frozen=True)
class Course:
    id: str
    name: str
    description: str
    prerequisites: frozenset[str]
    outcomes: frozenset[str]
    duration_weeks: int
    weekly_hours: int
    difficulty: float
    pass_base_probability: float


@dataclass(frozen=True)
class Role:
    id: str
    name: str
    required_skills: frozenset[str]
    recommended_skills: frozenset[str]


@dataclass(frozen=True)
class StudentProfile:
    id: str
    initial_skills: frozenset[str]
    max_weeks: int
    max_weekly_hours: int
    risk_tolerance: float


@dataclass(frozen=True)
class Dataset:
    skills: dict[str, Skill]
    courses: dict[str, Course]
    roles: dict[str, Role]
    profiles: dict[str, StudentProfile]
    instances: list[dict[str, Any]] = field(default_factory=list)


@dataclass  # sin frozen=True: los campos dict y list no son hashables
class GoalSpec:
    role_id: str
    target_skill_ids: set[str]
    initial_skill_ids: set[str]
    mentioned_skill_ids: set[str]
    constraints: dict[str, Any]
    ignored_constraints: list[str]
    unknown_skill_mentions: list[str]
    confidence: float


@dataclass(frozen=True)
class SearchState:
    skills: frozenset[str]
    taken_courses: tuple[str, ...]
    weeks_used: int
    difficulty_sum: float


@dataclass
class ScheduleResult:
    course_to_period: dict[str, int]
    period_to_courses: dict[int, list[str]]
    total_periods: int
    feasible: bool
    violations: list[str]


@dataclass
class PlanResult:
    planner_name: str
    course_ids: list[str]
    reached_skills: set[str]
    total_weeks: int
    total_difficulty: float
    expanded_nodes: int
    max_frontier_size: int
    runtime_seconds: float
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    schedule: ScheduleResult | None = None
    monte_carlo: dict[str, Any] | None = None
    llm_evaluation: dict[str, Any] | None = None
    final_score: float | None = None

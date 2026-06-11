from __future__ import annotations

import re
import networkx as nx
import pytest
from src.dataset.loader import load_dataset, profile_from_instance
from src.models import StudentProfile
from src.planners.ucs import ucs_plan
from src.simulation.instance_generator import (
    GenerationConfig,
    _feasible,
    generate_dataset,
    write_dataset,
)
from src.simulation.music_domain import PROFILES, ROLES, SKILLS

_CFG = GenerationConfig(seed=42, n_instances=10)

_GENEROUS = StudentProfile(
    id="consulta", initial_skills=frozenset(), max_weeks=999, max_weekly_hours=999, risk_tolerance=0.5
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(_CFG)


def test_sizes_match_template(dataset):
    assert len(dataset.skills) == len(SKILLS)
    assert len(dataset.courses) == len(SKILLS)  # un curso por skill
    assert len(dataset.roles) == len(ROLES)
    assert len(dataset.profiles) == len(PROFILES)
    assert 0 < len(dataset.instances) <= _CFG.n_instances


def test_names_are_not_abstract(dataset):
    abstract = re.compile(r"^(Rol|Habilidad|Curso)\s+\d+$")
    for skill in dataset.skills.values():
        assert not abstract.match(skill.name)
        assert skill.name in {s.name for s in SKILLS}
    for role in dataset.roles.values():
        assert not abstract.match(role.name)
        assert role.name in {r.name for r in ROLES}


def test_prerequisite_graph_is_acyclic(dataset):
    graph = nx.DiGraph()
    for course in dataset.courses.values():
        for prereq in course.prerequisites:
            for outcome in course.outcomes:
                graph.add_edge(prereq, outcome)
    assert nx.is_directed_acyclic_graph(graph)


def test_graph_is_coherent(dataset):
    producible: set[str] = set()
    for course in dataset.courses.values():
        producible.update(course.outcomes)
        for prereq in course.prerequisites:
            assert prereq in dataset.skills  # sin prereqs colgantes
    for skill_id in dataset.skills:
        assert skill_id in producible  # toda skill tiene curso productor


def test_every_role_reachable_from_zero(dataset):
    for role in dataset.roles.values():
        assert _feasible(set(), set(role.required_skills), dataset.courses, _GENEROUS)


def test_generated_instances_are_solvable(dataset):
    assert dataset.instances
    for instance in dataset.instances:
        profile = profile_from_instance(instance)
        role = dataset.roles[instance["expected_role_id"]]
        plan = ucs_plan(
            set(profile.initial_skills),
            set(role.required_skills),
            dataset.courses,
            profile,
        )
        assert plan.valid


def test_seed_is_reproducible():
    a = generate_dataset(_CFG)
    b = generate_dataset(_CFG)
    assert a.instances == b.instances
    assert [c.difficulty for c in a.courses.values()] == [c.difficulty for c in b.courses.values()]

    other = generate_dataset(GenerationConfig(seed=7, n_instances=_CFG.n_instances))
    assert other.instances != a.instances


def test_generated_dataset_loads_without_errors(dataset, tmp_path):
    out = write_dataset(dataset, tmp_path / "generated")
    loaded = load_dataset(out)
    assert len(loaded.skills) == len(SKILLS)
    assert len(loaded.roles) == len(ROLES)
    assert len(loaded.instances) == len(dataset.instances)

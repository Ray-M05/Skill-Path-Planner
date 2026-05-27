"""Tests para max_nodes, score_breakdown y LLM cache."""
from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.dataset.loader import load_dataset
from src.llm.client import LLMClient
from src.models import PlanResult, StudentProfile
from src.planners.astar import astar_k_plans, astar_plan
from src.scoring import compute_score_with_breakdown


# ---------------------------------------------------------------------------
# A* max_nodes
# ---------------------------------------------------------------------------


def test_astar_max_nodes_zero_means_no_limit() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_data_analyst"]

    plan = astar_plan(profile.initial_skills, role.required_skills, dataset.courses, profile, max_nodes=0)

    assert plan.valid


def test_astar_max_nodes_one_returns_no_plan() -> None:
    """Con max_nodes=1 el árbol colapsa antes de encontrar meta; debe devolver fallo."""
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_ml_engineer"]

    plan = astar_plan(profile.initial_skills, role.required_skills, dataset.courses, profile, max_nodes=1)

    assert not plan.valid
    assert any("nodo" in e.lower() or "trayectoria" in e.lower() for e in plan.validation_errors)


def test_astar_k_plans_max_nodes_limits_expansion() -> None:
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_ml_engineer"]

    plans_limited = astar_k_plans(
        profile.initial_skills, role.required_skills, dataset.courses, profile, k=3, max_nodes=2
    )
    plans_full = astar_k_plans(
        profile.initial_skills, role.required_skills, dataset.courses, profile, k=3, max_nodes=0
    )

    assert len(plans_limited) <= len(plans_full)


def test_astar_max_nodes_sufficient_finds_plan() -> None:
    """Con un límite generoso el planner debe encontrar el plan igual que sin límite."""
    dataset = load_dataset("data")
    profile = dataset.profiles["profile_beginner"]
    role = dataset.roles["role_data_analyst"]

    plan = astar_plan(profile.initial_skills, role.required_skills, dataset.courses, profile, max_nodes=10000)

    assert plan.valid


# ---------------------------------------------------------------------------
# score_breakdown
# ---------------------------------------------------------------------------


def test_compute_score_with_breakdown_returns_tuple() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = dataset.profiles["profile_beginner"]
    plan = PlanResult(
        planner_name="test",
        course_ids=["c1"],
        reached_skills=set(role.required_skills),
        total_weeks=20,
        total_difficulty=0.4,
        expanded_nodes=1,
        max_frontier_size=1,
        runtime_seconds=0.01,
        valid=True,
    )

    score, breakdown = compute_score_with_breakdown(plan, role, profile)

    assert isinstance(score, float)
    assert isinstance(breakdown, dict)


def test_score_breakdown_has_expected_keys() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = dataset.profiles["profile_beginner"]
    plan = PlanResult(
        planner_name="test",
        course_ids=["c1"],
        reached_skills=set(role.required_skills),
        total_weeks=20,
        total_difficulty=0.4,
        expanded_nodes=1,
        max_frontier_size=1,
        runtime_seconds=0.01,
        valid=True,
    )

    _, breakdown = compute_score_with_breakdown(plan, role, profile)

    for key in ("required", "recommended", "time_penalty", "difficulty_penalty", "mc_success", "llm_quality", "signals"):
        assert key in breakdown, f"Falta clave: {key}"


def test_score_breakdown_terms_sum_to_score() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = dataset.profiles["profile_beginner"]
    plan = PlanResult(
        planner_name="test",
        course_ids=["c1"],
        reached_skills=set(role.required_skills),
        total_weeks=20,
        total_difficulty=0.4,
        expanded_nodes=1,
        max_frontier_size=1,
        runtime_seconds=0.01,
        valid=True,
    )

    score, breakdown = compute_score_with_breakdown(plan, role, profile)

    reconstructed = (
        breakdown["required"]
        + breakdown["recommended"]
        + breakdown["time_penalty"]
        + breakdown["difficulty_penalty"]
        + breakdown["mc_success"]
        + breakdown["llm_quality"]
    )
    assert abs(reconstructed - score) < 1e-3


def test_score_breakdown_signals_flags_no_mc_no_llm() -> None:
    dataset = load_dataset("data")
    role = dataset.roles["role_data_analyst"]
    profile = dataset.profiles["profile_beginner"]
    plan = PlanResult(
        planner_name="test",
        course_ids=["c1"],
        reached_skills=set(role.required_skills),
        total_weeks=20,
        total_difficulty=0.4,
        expanded_nodes=1,
        max_frontier_size=1,
        runtime_seconds=0.01,
        valid=True,
    )

    _, breakdown = compute_score_with_breakdown(plan, role, profile)

    assert breakdown["signals"]["has_mc"] is False
    assert breakdown["signals"]["has_llm"] is False
    assert breakdown["signals"]["w_required"] == pytest.approx(0.65, abs=1e-4)


# ---------------------------------------------------------------------------
# LLM cache
# ---------------------------------------------------------------------------


def test_llm_cache_write_and_read(tmp_path) -> None:
    settings = Settings(llm_provider="gemini", llm_cache=True)
    client = LLMClient(settings=settings, cache_dir=tmp_path)

    payload = {"role_id": "role_data_analyst", "confidence": 0.9}
    key = client._cache_key("sys", "user")
    client._write_cache(key, payload)

    result = client._read_cache(key)
    assert result == payload


def test_llm_cache_returns_none_on_miss(tmp_path) -> None:
    settings = Settings(llm_provider="gemini", llm_cache=True)
    client = LLMClient(settings=settings, cache_dir=tmp_path)

    result = client._read_cache("nonexistent_key")
    assert result is None


def test_llm_cache_creates_json_file(tmp_path) -> None:
    settings = Settings(llm_provider="gemini", llm_cache=True)
    client = LLMClient(settings=settings, cache_dir=tmp_path)

    key = client._cache_key("a", "b")
    client._write_cache(key, {"test": 1})

    cache_file = tmp_path / f"{key}.json"
    assert cache_file.exists()
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert stored == {"test": 1}


def test_llm_cache_key_deterministic() -> None:
    settings = Settings(llm_provider="gemini", llm_cache=True)
    client = LLMClient(settings=settings)

    key1 = client._cache_key("system_prompt", "user_prompt")
    key2 = client._cache_key("system_prompt", "user_prompt")
    key3 = client._cache_key("other_system", "user_prompt")

    assert key1 == key2
    assert key1 != key3


def test_llm_cache_different_prompts_different_keys() -> None:
    settings = Settings(llm_provider="gemini", llm_cache=True)
    client = LLMClient(settings=settings)

    k1 = client._cache_key("sys", "prompt_A")
    k2 = client._cache_key("sys", "prompt_B")

    assert k1 != k2

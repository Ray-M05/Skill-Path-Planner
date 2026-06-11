from src.llm.client import LLMClient
from src.config import Settings
from src.dataset.loader import load_dataset
from src.llm.interpreter import interpret_goal, validate_goal_spec
from src.llm.prompts import (
    GOAL_INTERPRETER_SYSTEM_PROMPT,
    build_goal_interpreter_user_prompt,
)


class FakeLLMClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


def test_llm_client_requires_api_key_for_real_calls() -> None:
    client = LLMClient(
        Settings(
            llm_provider="gemini",
            llm_model="gemini-2.5-flash-lite",
            llm_api_key=None,
        )
    )

    try:
        client.complete_json("system", "user")
    except RuntimeError as error:
        assert "GEMINI_API_KEY" in str(error)
    else:
        raise AssertionError("Expected RuntimeError when GEMINI_API_KEY is missing.")


def test_llm_client_mock_provider_returns_mock_response() -> None:
    response = {"role_id": "role_data_analyst"}
    client = LLMClient(
        Settings(llm_provider="mock", llm_model="mock", llm_api_key=None),
        mock_response=response,
    )

    assert client.complete_json("system", "user") == response


def test_goal_interpreter_prompt_contains_controlled_vocabulary() -> None:
    dataset = load_dataset("data")

    prompt = build_goal_interpreter_user_prompt(
        "Quiero ser ingeniero de machine learning.",
        dataset,
    )

    assert "Perfil inicial" not in prompt
    assert "Roles permitidos" in prompt
    assert "Habilidades permitidas" in prompt
    assert "role_ml_engineer" in prompt
    assert "skill_machine_learning_basic" in prompt
    assert "initial_skill_ids" in prompt
    assert "Devuelve este JSON" in prompt
    assert "No inventes IDs" in GOAL_INTERPRETER_SYSTEM_PROMPT


def test_validate_goal_spec_accepts_valid_json() -> None:
    dataset = load_dataset("data")

    goal = validate_goal_spec(
        {
            "role_id": "role_data_analyst",
            "target_skill_ids": ["skill_sql_basic", "skill_data_analysis_basic"],
            "initial_skill_ids": ["skill_python_basic"],
            "mentioned_skill_ids": ["skill_sql_basic"],
            "constraints": {"max_weeks": 40},
            "ignored_constraints": [],
            "unknown_skill_mentions": [],
            "confidence": 0.9,
        },
        dataset,
    )

    assert goal.role_id == "role_data_analyst"
    assert goal.target_skill_ids == {"skill_sql_basic", "skill_data_analysis_basic"}
    assert goal.initial_skill_ids == {"skill_python_basic"}
    assert goal.mentioned_skill_ids == {"skill_sql_basic"}
    assert goal.confidence == 0.9


def test_validate_goal_spec_moves_unknown_skill_ids_to_unknown_mentions() -> None:
    dataset = load_dataset("data")

    goal = validate_goal_spec(
        {
            "role_id": "role_data_analyst",
            "target_skill_ids": ["skill_sql_basic", "skill_missing"],
            "initial_skill_ids": ["skill_python_basic", "skill_initial_missing"],
            "mentioned_skill_ids": ["skill_other_missing"],
            "constraints": {},
            "ignored_constraints": [],
            "unknown_skill_mentions": ["blockchain"],
            "confidence": 0.7,
        },
        dataset,
    )

    assert goal.target_skill_ids == {"skill_sql_basic"}
    assert goal.initial_skill_ids == {"skill_python_basic"}
    assert goal.mentioned_skill_ids == set()
    assert goal.unknown_skill_mentions == [
        "blockchain",
        "skill_missing",
        "skill_initial_missing",
        "skill_other_missing",
    ]


def test_validate_goal_spec_clamps_confidence() -> None:
    dataset = load_dataset("data")

    goal = validate_goal_spec(
        {
            "role_id": "role_data_analyst",
            "target_skill_ids": [],
            "initial_skill_ids": [],
            "mentioned_skill_ids": [],
            "constraints": {},
            "ignored_constraints": [],
            "unknown_skill_mentions": [],
            "confidence": 2.5,
        },
        dataset,
    )

    assert goal.confidence == 1.0


def test_validate_goal_spec_rejects_unknown_role() -> None:
    dataset = load_dataset("data")

    try:
        validate_goal_spec(
            {
                "role_id": "role_missing",
                "target_skill_ids": [],
                "initial_skill_ids": [],
                "mentioned_skill_ids": [],
                "constraints": {},
                "ignored_constraints": [],
                "unknown_skill_mentions": [],
                "confidence": 0.5,
            },
            dataset,
        )
    except ValueError as error:
        assert "role_id" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown role_id.")


def test_interpret_goal_uses_llm_response_and_validates_it() -> None:
    dataset = load_dataset("data")
    fake_client = FakeLLMClient(
        {
            "role_id": "role_ml_engineer",
            "target_skill_ids": ["skill_machine_learning_basic"],
            "initial_skill_ids": ["skill_python_basic"],
            "mentioned_skill_ids": ["skill_python_basic"],
            "constraints": {"max_weeks": 72, "max_weekly_hours": 10},
            "ignored_constraints": ["gratis"],
            "unknown_skill_mentions": [],
            "confidence": 0.85,
        }
    )

    goal = interpret_goal(
        "Quiero ser ingeniero de machine learning gratis.",
        dataset,
        fake_client,
    )

    assert goal.role_id == "role_ml_engineer"
    assert goal.target_skill_ids == {"skill_machine_learning_basic"}
    assert goal.initial_skill_ids == {"skill_python_basic"}
    assert goal.ignored_constraints == ["gratis"]
    assert "Roles permitidos" in fake_client.user_prompt
from src.llm.client import LLMClient
from src.config import Settings


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

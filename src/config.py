from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"
    llm_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 1000
    llm_timeout_seconds: int = 30
    llm_cache: bool = True
    llm_live_tests: bool = False


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"),
        llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or None,
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1000")),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        llm_cache=_as_bool(os.getenv("LLM_CACHE"), True),
        llm_live_tests=_as_bool(os.getenv("LLM_LIVE_TESTS"), False),
    )

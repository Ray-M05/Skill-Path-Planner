from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from ..config import Settings, load_settings

class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        mock_response: dict[str, Any] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.mock_response = mock_response
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(".llm_cache")

    # Cache helpers
    def _cache_key(self, system_prompt: str, user_prompt: str) -> str:
        content = system_prompt + "\n---SEP---\n" + user_prompt
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, key: str, response: dict[str, Any]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

    # Public API
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.settings.llm_provider == "mock":
            if self.mock_response is None:
                raise RuntimeError("LLM_PROVIDER=mock requiere mock_response.")
            return self.mock_response

        if self.settings.llm_provider != "gemini":
            raise ValueError(f"Proveedor LLM no soportado: {self.settings.llm_provider}.")
        if not self.settings.llm_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY o LLM_API_KEY no esta configurada. "
                "Usa mocks en tests o configura .env para llamadas reales."
            )

        if self.settings.llm_cache:
            key = self._cache_key(system_prompt, user_prompt)
            cached = self._read_cache(key)
            if cached is not None:
                return cached

        from google import genai

        client = genai.Client(api_key=self.settings.llm_api_key)
        response = client.models.generate_content(
            model=self.settings.llm_model,
            contents=f"{system_prompt}\n\n{user_prompt}",
            config={
                "temperature": self.settings.llm_temperature,
                "max_output_tokens": self.settings.llm_max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        result = json.loads(response.text)

        if self.settings.llm_cache:
            self._write_cache(key, result)

        return result

from __future__ import annotations

import json
from typing import Any
from ..config import Settings, load_settings


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        mock_response: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.mock_response = mock_response

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.settings.llm_provider == "mock":
            if self.mock_response is None:
                raise RuntimeError("LLM_PROVIDER=mock requiere mock_response.")
            return self.mock_response

        if self.settings.llm_provider != "gemini":
            raise ValueError(f"Proveedor LLM no soportado: {self.settings.llm_provider}.")
        if not self.settings.llm_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY o LLM_API_KEY no esta configurada. Usa mocks en tests o configura .env para llamadas reales."
            )

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
        return json.loads(response.text)

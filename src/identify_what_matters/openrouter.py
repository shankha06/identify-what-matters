"""Synchronous OpenRouter chat-completions client with reasoning continuation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"

# Models that support OpenRouter's reasoning extension and benefit from a
# two-turn self-review continuation.  Everything else gets a single call.
_REASONING_MODELS = frozenset({
    "openai/gpt-6.1-sol",
    "anthropic/claude-4-6-sonnet", "anthropic/claude-sonnet-4-5",
    "google/gemini-3.8-flash",
    "anthropic/claude-haiku-5.5"
})
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@dataclass
class OpenRouterResponse:
    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False


class OpenRouterClient:
    def __init__(self, model: str | None = None, max_tokens: int = 8192,
                 temperature: float = 0.1):
        load_dotenv(ENV_FILE, override=False)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required; set it in the environment or project .env file")
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _uses_reasoning(self) -> bool:
        return any(self.model.startswith(m) or self.model == m for m in _REASONING_MODELS)

    def _complete(self, messages: list[dict[str, Any]], timeout: int,
                  max_tokens: int | None = None, response_schema: dict | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self._uses_reasoning():
            payload["reasoning"] = {"enabled": True}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        cap = max_tokens if max_tokens is not None else self.max_tokens
        if cap is not None:
            payload["max_tokens"] = cap
        if response_schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "response", "strict": True, "schema": response_schema}}
            payload["provider"] = {"require_parameters": True}
            payload["plugins"] = [{"id": "response-healing"}]

        for attempt in range(4):
            response = requests.post(
                url=API_URL,
                headers=self._headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            if response.status_code in (429, 503, 529):
                retry_after = int(response.headers.get("Retry-After", 0) or 0)
                wait = max(retry_after, 2 ** attempt)
                if attempt < 3:
                    time.sleep(wait)
                    continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()  # unreachable, keeps type checker happy
        return response.json()

    def generate(self, user_prompt: str, system_instruction: str = "",
                 request_timeout: int = 150, response_schema: dict | None = None,
                 max_tokens: int | None = None) -> OpenRouterResponse:
        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_prompt})

        first = self._complete(messages, request_timeout, max_tokens, response_schema)
        first_message = first["choices"][0]["message"]
        first_usage = first.get("usage") or {}

        # Two-turn self-review only for reasoning-capable models; skip for fast models.
        if self._uses_reasoning():
            assistant_turn: dict[str, Any] = {"role": "assistant", "content": first_message.get("content")}
            if first_message.get("reasoning_details"):
                assistant_turn["reasoning_details"] = first_message["reasoning_details"]
            messages.extend([
                assistant_turn,
                {"role": "user", "content": "Review your answer carefully. Follow the original instructions and return only the final answer."},
            ])
            second = self._complete(messages, request_timeout, max_tokens, response_schema)
            second_message = second["choices"][0]["message"]
            second_usage = second.get("usage") or {}
            return OpenRouterResponse(
                text=second_message.get("content") or "",
                input_tokens=(first_usage.get("prompt_tokens", 0) or 0) + (second_usage.get("prompt_tokens", 0) or 0),
                output_tokens=(first_usage.get("completion_tokens", 0) or 0) + (second_usage.get("completion_tokens", 0) or 0),
                truncated=second["choices"][0].get("finish_reason") == "length",
            )

        return OpenRouterResponse(
            text=first_message.get("content") or "",
            input_tokens=first_usage.get("prompt_tokens", 0) or 0,
            output_tokens=first_usage.get("completion_tokens", 0) or 0,
            truncated=first["choices"][0].get("finish_reason") == "length",
        )
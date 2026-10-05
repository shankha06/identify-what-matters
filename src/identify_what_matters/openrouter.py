"""Synchronous OpenRouter chat-completions client with reasoning continuation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6.1-sol"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@dataclass
class OpenRouterResponse:
    text: str
    input_tokens: int
    output_tokens: int


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

    def _complete(self, messages: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "reasoning": {"enabled": True},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        response = requests.post(
            url=API_URL,
            headers=self._headers,
            data=json.dumps(payload),
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def generate(self, user_prompt: str, system_instruction: str = "",
                 request_timeout: int = 150) -> OpenRouterResponse:
        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_prompt})

        first = self._complete(messages, request_timeout)
        first_message = first["choices"][0]["message"]
        messages.extend([
            {
                "role": "assistant",
                "content": first_message.get("content"),
                "reasoning_details": first_message.get("reasoning_details"),
            },
            {
                "role": "user",
                "content": "Review your answer carefully. Follow the original instructions and return only the final answer.",
            },
        ])
        second = self._complete(messages, request_timeout)
        second_message = second["choices"][0]["message"]
        first_usage = first.get("usage") or {}
        second_usage = second.get("usage") or {}
        return OpenRouterResponse(
            text=second_message.get("content") or "",
            input_tokens=(first_usage.get("prompt_tokens", 0) or 0)
            + (second_usage.get("prompt_tokens", 0) or 0),
            output_tokens=(first_usage.get("completion_tokens", 0) or 0)
            + (second_usage.get("completion_tokens", 0) or 0),
        )
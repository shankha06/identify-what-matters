"""Cached JSON generation and batching with Element Gateway Claude Sonnet 4.6 and OpenRouter fallback."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from identify_what_matters.openrouter import OpenRouterClient
from identify_what_matters.utils import parse_json_payload

parse_json = parse_json_payload

# Load .env early so LLM_BACKEND and other vars are available before make_backend_client reads them.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

HELPER_DIR = Path(os.environ.get("LLM_HELPER_DIR", "/Users/s0r0qw6/model-py-java"))
DEFAULT_GATEWAY_MODEL = os.environ.get("TRENDS_LLM_MODEL", "claude-sonnet-4-6")
DEFAULT_GATEWAY_VERSION = os.environ.get("TRENDS_LLM_MODEL_VERSION", "default")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class GatewayClientWrapper:
    """Wraps Element Gateway util_llm_client to match the OpenRouterClient interface."""

    def __init__(self, model: str = DEFAULT_GATEWAY_MODEL, model_version: str = DEFAULT_GATEWAY_VERSION,
                 temperature: float = 0.1, max_tokens: int = 8192):
        self.model = model
        self.model_version = model_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        if str(HELPER_DIR) not in sys.path:
            sys.path.insert(0, str(HELPER_DIR))
        with contextlib.suppress(ImportError):
            from dotenv import load_dotenv
            load_dotenv(HELPER_DIR / ".env", override=False)
        import util_llm_client
        util_llm_client.DEFAULT_RETRY_ATTEMPTS = 3
        self.client = util_llm_client.make_llm_client(
            model=model,
            model_version=model_version,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self, user_prompt: str, system_instruction: str = "",
                 request_timeout: int = 150) -> LLMResponse:
        resp = self.client.generate(
            user_prompt=user_prompt,
            system_instruction=system_instruction,
            request_timeout=request_timeout,
        )
        return LLMResponse(
            text=resp.text or "",
            input_tokens=resp.input_tokens or 0,
            output_tokens=resp.output_tokens or 0,
        )


def make_backend_client(model: str | None = None, temperature: float = 0.1,
                        max_tokens: int = 8192) -> Any:
    """Creates the best available LLM client.

    Prefers internal Walmart Element Gateway (Claude Sonnet 4.6).
    Falls back to OpenRouterClient if Gateway credentials/helper are unavailable.
    """
    use_gateway = os.environ.get("LLM_BACKEND", "gateway").lower() != "openrouter"
    if use_gateway:
        try:
            target_model = model or DEFAULT_GATEWAY_MODEL
            return GatewayClientWrapper(model=target_model, temperature=temperature, max_tokens=max_tokens)
        except Exception:
            pass
    return OpenRouterClient(model=model, temperature=temperature, max_tokens=max_tokens)


class LLM:
    def __init__(self, model: str | None = None,
                 cache_dir: str | Path = "output/llm_cache", max_calls: int = 300,
                 temperature: float = 0.1, max_tokens: int = 8192,
                 client: Any | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.calls, self.cached = max_calls, 0, 0
        self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()
        self.client = client or make_backend_client(model=model, temperature=temperature, max_tokens=max_tokens)
        self.model = getattr(self.client, "model", model or DEFAULT_GATEWAY_MODEL)

    def json(self, prompt: str, system: str, validate: Callable[[Any], Any] | None = None,
             timeout: int = 150) -> Any:
        """One cached call returning parsed JSON; one corrective retry on malformed output."""
        key = hashlib.sha256(f"{self.model}\n{system}\n{prompt}".encode()).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            with self._lock:
                self.cached += 1
            return json.loads(cached.read_text())
        last_error = None
        for attempt in range(3):
            with self._lock:
                if self.calls + 2 > self.max_calls:
                    raise RuntimeError(f"LLM call budget {self.max_calls} exhausted")
                self.calls += 2
            suffix = "" if attempt == 0 else "\n\nYour previous answer was not valid JSON for the schema. Return ONLY the JSON."
            response = self.client.generate(prompt + suffix, system_instruction=system, request_timeout=timeout)
            with self._lock:
                self.input_tokens += response.input_tokens or 0
                self.output_tokens += response.output_tokens or 0
            if not response.text.strip():
                raise ValueError("empty response (likely gateway safety filter)")
            try:
                value = parse_json(response.text)
                if validate:
                    value = validate(value)
                cached.write_text(json.dumps(value, ensure_ascii=False))
                return value
            except Exception as error:
                last_error = error
        raise ValueError(f"LLM returned unusable JSON: {last_error}")

    def map(self, function: Callable[[Any], Any], items: list, workers: int = 6) -> list:
        """Run function over items in a thread pool; failures become None, never abort the run."""
        def safe(item):
            try:
                return function(item)
            except Exception as error:
                print(f"[llm] item failed: {type(error).__name__}: {str(error)[:160]}", flush=True)
                return None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(safe, items))

    def usage(self) -> dict:
        return {"model": self.model, "calls": self.calls, "cache_hits": self.cached,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

"""Cached JSON generation and batching on top of OpenRouter."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from identify_what_matters.openrouter import OpenRouterClient


def parse_json(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("no JSON in response")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start:end + 1], strict=False)


class LLM:
    def __init__(self, model: str | None = None,
                 cache_dir: str | Path = "output/llm_cache", max_calls: int = 300,
                 temperature: float = 0.1, max_tokens: int = 8192):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.calls, self.cached = max_calls, 0, 0
        self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()
        self.client = OpenRouterClient(model=model, temperature=temperature, max_tokens=max_tokens)
        self.model = self.client.model

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

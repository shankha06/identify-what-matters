"""Common shared utilities for serialization, date/time handling, and model execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
import threading
from typing import Any, Iterable


class Budget:
    """Thread-safe budget tracker for API calls and token consumption."""
    def __init__(self, state: dict, max_calls: int, max_output_tokens: int, lock: threading.Lock | None = None):
        self.state = state
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self._lock = lock or threading.Lock()

    def reserve(self, expected_output: int, calls: int = 1) -> bool:
        with self._lock:
            used = sum(item.get("output_tokens") or 0 for item in self.state.get("usage", []))
            if self.state.get("calls", 0) + calls > self.max_calls or used + expected_output > self.max_output_tokens:
                return False
            self.state["calls"] = self.state.get("calls", 0) + calls
            return True


try:
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        """HuggingFace Trainer with custom class weights for weighted cross-entropy loss."""
        def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_ = inputs.pop("labels")
            outputs = model(**inputs)
            w = self.class_weights.float() if self.class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels_, weight=w)
            return (loss, outputs) if return_outputs else loss
except ImportError:
    WeightedTrainer = None  # type: ignore[assignment, misc]


def now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Return current timezone-aware UTC datetime formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: Any) -> datetime | None:
    """Safely parse various datetime representations into UTC-aware datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def hours_since(value: Any, reference: datetime | None = None, min_hours: float | None = None) -> float | None:
    """Compute elapsed hours between value and reference time (defaulting to UTC now)."""
    if not value:
        return None
    ref = parse_time(reference) if reference is not None else now()
    if not ref:
        return None
    moment = parse_time(value)
    if not moment:
        return None
    elapsed = (ref - moment).total_seconds() / 3600.0
    return max(min_hours, elapsed) if min_hours is not None else elapsed


def parse_json_payload(text: str) -> Any:
    """Parse JSON from text, tolerating markdown fences and leading/trailing chatter."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
    if not starts:
        raise ValueError("no JSON found in payload")
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise ValueError("no valid JSON object or array found in payload")
    return json.loads(cleaned[start:end + 1], strict=False)


def write_json(target: Path | str, value: Any, indent: int = 2) -> None:
    """Atomically write formatted JSON to a file."""
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=indent, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_rows(target: Path | str, rows: Iterable[dict]) -> None:
    """Write rows of dictionaries as line-delimited JSON (JSONL)."""
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")


def get_torch_device(allow_mps: bool = True) -> str:
    """Detect and return the preferred PyTorch device string ('cuda', 'mps', or 'cpu')."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

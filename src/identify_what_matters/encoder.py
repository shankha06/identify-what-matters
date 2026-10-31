"""Shared access to the fine-tuned multilingual event encoder and clustering."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from identify_what_matters import onnx_backend

MODEL_CANDIDATES = ("artifacts/topic-models/multilingual-e5-base-events-real-v5",
                    "artifacts/topic-models/multilingual-e5-base-events-real-v4",
                    "artifacts/topic-models/multilingual-e5-base-events-real",
                    "artifacts/topic-models/multilingual-e5-base-events")


@lru_cache(maxsize=1)
def load(path: str | None = None):
    model_path = path or next((candidate for candidate in MODEL_CANDIDATES if Path(candidate).exists()), None)
    if model_path is None:
        raise FileNotFoundError("Tuned encoder missing; see docs/TRAINING_RESULTS.md")
    settings = json.loads(Path(model_path, "topic_threshold.json").read_text())
    if onnx_backend.onnx_path(model_path) is not None:
        model = onnx_backend.OnnxEncoder(model_path)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
    return model, settings.get("prefix", ""), float(settings["cosine_threshold"])


def embed(texts: list[str]) -> np.ndarray:
    model, prefix, _ = load()
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return model.encode([prefix + text for text in texts], batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False)


def threshold() -> float:
    return load()[2]


def cluster(vectors: np.ndarray, cosine: float | None = None) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering
    if len(vectors) < 2:
        return [0] * len(vectors)
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=1 - (cosine or threshold())).fit_predict(vectors).tolist()

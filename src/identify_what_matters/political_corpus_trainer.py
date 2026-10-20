"""CPU baseline for the political/civic undercurrent corpus (artifacts/political_corpus/dataset/*.jsonl).

Trains fast TF-IDF + Logistic Regression pipelines per label dimension (is_political, issue_cluster,
sentiment, target, direction, narrative_type), the same recipe as demographics_trainer.py. This is an
immediate, zero-GPU baseline to replace political_pulse.py's keyword/regex scoring with something that
generalizes past its hand-maintained phrase lists - it is NOT the production model. See
docs/TRAINING_RESULTS.md for the planned transformer fine-tune (mDeBERTa/multilingual-e5, A100 VM) and the
promotion gate; this script's accuracy on real held-out comments must be measured before trusting it.

Usage: run `political_corpus.py generate` (+ optionally `label-real`) then `build` first to produce
artifacts/political_corpus/dataset/{train,dev,test}.jsonl, then run this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

DATASET_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "political_corpus" / "dataset"
MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "political_models"
TASKS = ("is_political", "issue_cluster", "sentiment", "target", "direction", "narrative_type")


def _load(split: str) -> list[dict]:
    path = DATASET_DIR / f"{split}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_and_evaluate() -> dict:
    train, dev, test = _load("train"), _load("dev"), _load("test")
    if len(train) < 50:
        raise RuntimeError(
            f"Only {len(train)} train rows under {DATASET_DIR}; run political_corpus.py generate/build first "
            "(and/or label-real against existing output/runs/*/report.json) before training."
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for task in TASKS:
        texts = [row["text"] for row in train]
        y = [str(row[task]) for row in train]
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)),
        ])
        pipe.fit(texts, y)
        joblib.dump(pipe, MODELS_DIR / f"{task}_classifier.joblib")
        task_metrics = {"train_rows": len(texts), "classes": sorted(set(y))}
        for split_name, rows in (("dev", dev), ("test", test)):
            if not rows:
                continue
            split_texts = [row["text"] for row in rows]
            split_y = [str(row[task]) for row in rows]
            pred = pipe.predict(split_texts)
            task_metrics[f"{split_name}_accuracy"] = round(accuracy_score(split_y, pred), 3)
            task_metrics[f"{split_name}_macro_f1"] = round(f1_score(split_y, pred, average="macro", zero_division=0), 3)
            task_metrics[f"{split_name}_rows"] = len(rows)
        metrics[task] = task_metrics
        print(f"{task}: {task_metrics}")
    summary = {"model_version": "v1-tfidf-logreg-baseline", "train_rows": len(train), "dev_rows": len(dev),
              "test_rows": len(test), "metrics": metrics}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_and_evaluate()

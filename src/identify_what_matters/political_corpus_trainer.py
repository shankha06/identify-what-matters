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

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

DATASET_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "political_corpus" / "dataset"
MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "political_models"
TASKS = ("is_political", "issue_cluster", "sentiment", "target", "direction", "narrative_type")


def _load(split: str, dataset_dir: Path = DATASET_DIR) -> list[dict]:
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_and_evaluate(dataset_dir: Path = DATASET_DIR, models_dir: Path = MODELS_DIR,
                       search: bool = False, tasks: tuple[str, ...] = TASKS) -> dict:
    train, dev, test = (_load(split, dataset_dir) for split in ("train", "dev", "test"))
    if len(train) < 50:
        raise RuntimeError(
            f"Only {len(train)} train rows under {DATASET_DIR}; run political_corpus.py generate/build first "
            "(and/or label-real against existing output/runs/*/report.json) before training."
        )
    if search and not dev:
        raise ValueError("Candidate selection requires a nonempty dev split")
    if search and models_dir.exists():
        raise FileExistsError(f"Refusing to overwrite candidate directory {models_dir}")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for task in tasks:
        texts = [row["text"] for row in train]
        y = [str(row[task]) for row in train]
        baseline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)),
        ])
        baseline.fit(texts, y)
        pipe = baseline
        selection = []
        if search:
            dev_texts = [row["text"] for row in dev]
            dev_y = [str(row[task]) for row in dev]
            best_f1 = f1_score(dev_y, baseline.predict(dev_texts), average="macro", zero_division=0)
            selection.append({"features": "word", "C": 2.0, "dev_macro_f1": float(best_f1)})
            for strength in (0.5, 2.0, 8.0):
                features = FeatureUnion([
                    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                            max_features=60000)),
                    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                            sublinear_tf=True, max_features=100000)),
                ])
                candidate = Pipeline([
                    ("tfidf", features),
                    ("clf", LogisticRegression(C=strength, class_weight="balanced", max_iter=1000)),
                ])
                candidate.fit(texts, y)
                candidate_f1 = f1_score(dev_y, candidate.predict(dev_texts), average="macro", zero_division=0)
                selection.append({"features": "word+char", "C": strength,
                                  "dev_macro_f1": float(candidate_f1)})
                if candidate_f1 > best_f1:
                    best_f1, pipe = candidate_f1, candidate
        joblib.dump(pipe, models_dir / f"{task}_classifier.joblib")
        task_metrics = {"train_rows": len(texts), "classes": sorted(set(y))}
        if search:
            task_metrics["selection"] = selection
            task_metrics["selected"] = max(selection, key=lambda result: result["dev_macro_f1"])
        for split_name, rows in (("dev", dev), ("test", test)):
            if not rows:
                continue
            split_texts = [row["text"] for row in rows]
            split_y = [str(row[task]) for row in rows]
            pred = pipe.predict(split_texts)
            task_metrics[f"{split_name}_accuracy"] = round(accuracy_score(split_y, pred), 3)
            task_metrics[f"{split_name}_macro_f1"] = round(f1_score(split_y, pred, average="macro", zero_division=0), 3)
            task_metrics[f"{split_name}_rows"] = len(rows)
            if search and split_name == "test":
                baseline_pred = baseline.predict(split_texts)
                task_metrics["baseline_test_accuracy"] = float(accuracy_score(split_y, baseline_pred))
                task_metrics["baseline_test_macro_f1"] = float(
                    f1_score(split_y, baseline_pred, average="macro", zero_division=0))
        metrics[task] = task_metrics
        print(f"{task}: {task_metrics}")
    summary = {"model_version": "word-char-dev-selected" if search else "v1-tfidf-logreg-baseline",
              "selection_split": "dev" if search else None, "dataset": str(dataset_dir),
              "train_rows": len(train), "dev_rows": len(dev),
              "test_rows": len(test), "metrics": metrics}
    (models_dir / "metadata.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=MODELS_DIR)
    parser.add_argument("--search", action="store_true")
    arguments = parser.parse_args()
    train_and_evaluate(arguments.dataset, arguments.output, arguments.search)

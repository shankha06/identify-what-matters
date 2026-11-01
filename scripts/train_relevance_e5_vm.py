"""Relevance-only bi-encoder fine-tune: anchor (topic text) vs caption, cosine similarity, binary label.

Reports zero-shot baseline and fine-tuned metrics on dev and test:
- ROC-AUC on test
- F1 on test at the threshold chosen on dev (honest), plus best-threshold F1 on test for reference
- P@1 on test: per (run_id, topic_id) group, the top-scored caption is relevant

Run on the GPU VM from the work directory that holds the three jsonl splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

PREFIX = "query: "


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def scores(model: SentenceTransformer, rows: list[dict]) -> np.ndarray:
    anchors = model.encode([PREFIX + r["anchor"] for r in rows], batch_size=128,
                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    captions = model.encode([PREFIX + r["caption"] for r in rows], batch_size=128,
                            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.sum(anchors * captions, axis=1)


def best_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    best_f1, best_th = 0.0, 0.0
    for th in np.arange(0.0, 1.0, 0.005):
        f1 = float(f1_score(y, s >= th, zero_division=0))
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_f1, best_th


def p_at_1(rows: list[dict], s: np.ndarray) -> tuple[float, int]:
    groups: dict[tuple, list[tuple[float, int]]] = {}
    for row, score in zip(rows, s):
        groups.setdefault((row["run_id"], row["topic_id"]), []).append((float(score), int(row["label"])))
    hits = []
    for items in groups.values():
        if any(label == 1 for _, label in items):
            top_label = max(items, key=lambda x: x[0])[1]
            hits.append(top_label == 1)
    return (float(np.mean(hits)) if hits else float("nan")), len(hits)


def metrics(model: SentenceTransformer, dev: list[dict], test: list[dict]) -> dict:
    y_dev = np.array([r["label"] for r in dev])
    y_test = np.array([r["label"] for r in test])
    s_dev = scores(model, dev)
    s_test = scores(model, test)
    _, th = best_threshold(y_dev, s_dev)
    best_f1_test, best_th_test = best_threshold(y_test, s_test)
    p1, n_topics = p_at_1(test, s_test)
    return {
        "dev_threshold": round(th, 3),
        "test_auc": round(float(roc_auc_score(y_test, s_test)), 4),
        "test_f1_at_dev_threshold": round(float(f1_score(y_test, s_test >= th, zero_division=0)), 4),
        "test_best_f1": round(best_f1_test, 4),
        "test_best_threshold": round(best_th_test, 3),
        "test_p_at_1": round(p1, 4),
        "test_topics_scored": n_topics,
        "n_dev": len(dev),
        "n_test": len(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("relevance_v6"))
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--output-dir", type=Path, default=Path("relevance_v6/model"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=Path("relevance_v6/metrics.json"))
    args = parser.parse_args()

    train = load(args.data / "train.jsonl")
    dev = load(args.data / "dev.jsonl")
    test = load(args.data / "test.jsonl")

    model = SentenceTransformer(args.base_model)
    baseline = metrics(model, dev, test)
    print("baseline", json.dumps(baseline), flush=True)

    examples = [InputExample(texts=[PREFIX + r["anchor"], PREFIX + r["caption"]], label=float(r["label"]))
                for r in train]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.CosineSimilarityLoss(model)
    warmup = int(len(loader) * args.epochs * args.warmup_ratio)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, warmup_steps=warmup,
              optimizer_params={"lr": args.lr}, show_progress_bar=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_dir))

    finetuned = metrics(model, dev, test)
    print("finetuned", json.dumps(finetuned), flush=True)

    result = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_train": len(train),
        "baseline": baseline,
        "finetuned": finetuned,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

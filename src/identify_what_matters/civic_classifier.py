"""Train, evaluate, and run inference for the Lightweight Civic/Political Relevance Classifier.

Supports:
- Fine-tuning sequence classification models (e.g. mDeBERTa, mE5, MiniLM) with class reweighting.
- Threshold tuning on the dev split and reporting ROC-AUC, macro-F1, precision, recall, and language-slice metrics.
- Fast local inference pipeline with confidence score and threshold caching.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_split(dataset_dir: Path | str, split: str) -> list[dict]:
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train(args) -> None:
    """Train sequence classification model on the civic relevance training split."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to overwrite")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
    )

    train_rows = load_split(args.dataset, "train")

    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    data = Dataset.from_list([{"text": r["text"], "label": int(r["label"])} for r in train_rows]).shuffle(seed=args.seed)
    data = data.map(encode, batched=True, remove_columns=["text"]).rename_column("label", "labels")

    positives = sum(r["label"] for r in train_rows)
    negatives = len(train_rows) - positives
    pos_weight = negatives / max(positives, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.tensor([1.0, pos_weight], device=device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_ = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.float(),
                labels_,
                weight=weights.float() if weights is not None else None,
            )
            return (loss, outputs) if return_outputs else loss

    total_steps = -(-len(data) // args.batch_size) * args.epochs
    training_args = TrainingArguments(
        output_dir=str(output) + "-work",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=max(1, total_steps // 10),
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        seed=args.seed,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        lr_scheduler_type="linear",
    )

    started = time.time()
    WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=data,
        data_collator=DataCollatorWithPadding(tokenizer),
    ).train()

    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    meta = {
        "task": "civic_relevance",
        "base_model": args.base_model,
        "dataset": str(args.dataset),
        "train_rows": len(train_rows),
        "positives": positives,
        "negatives": negatives,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "seconds": round(time.time() - started, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output / "training_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"status": "trained", "output": str(output), "rows": len(train_rows)}))


def predict_probabilities(model_path: str, texts: list[str], batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """Calculates P(is_civic_political) for a batch of texts."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def evaluate(args) -> list[dict]:
    """Evaluates checkpoints on dev/test splits, computing AUC, F1, and language slices."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    dev_rows = load_split(args.dataset, "dev")
    test_rows = load_split(args.dataset, "test")

    dev_texts = [r["text"] for r in dev_rows]
    dev_labels = np.array([int(r["label"]) for r in dev_rows])
    test_texts = [r["text"] for r in test_rows]
    test_labels = np.array([int(r["label"]) for r in test_rows])

    results = []
    for model_path in args.models:
        t0 = time.time()
        dev_probs = predict_probabilities(model_path, dev_texts, batch_size=args.batch_size, device=args.device)
        test_probs = predict_probabilities(model_path, test_texts, batch_size=args.batch_size, device=args.device)

        # Optimize threshold on dev set
        best_t, best_f1 = 0.5, 0.0
        for cand_t in np.unique(np.round(dev_probs, 3)):
            f1 = f1_score(dev_labels, dev_probs >= cand_t, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(cand_t)

        test_preds = test_probs >= best_t
        overall_auc = float(roc_auc_score(test_labels, test_probs))
        overall_f1 = float(f1_score(test_labels, test_preds, zero_division=0))
        overall_prec = float(precision_score(test_labels, test_preds, zero_division=0))
        overall_rec = float(recall_score(test_labels, test_preds, zero_division=0))

        # Language and origin slice metrics
        slices = defaultdict(lambda: {"labels": [], "preds": [], "probs": []})
        for i, r in enumerate(test_rows):
            lang = r.get("language") or "unknown"
            origin = r.get("origin") or "unknown"
            slices[f"lang:{lang}"]["labels"].append(test_labels[i])
            slices[f"lang:{lang}"]["preds"].append(test_preds[i])
            slices[f"lang:{lang}"]["probs"].append(test_probs[i])
            slices[f"origin:{origin}"]["labels"].append(test_labels[i])
            slices[f"origin:{origin}"]["preds"].append(test_preds[i])
            slices[f"origin:{origin}"]["probs"].append(test_probs[i])

        slice_metrics = {}
        for s_name, s_data in sorted(slices.items()):
            y_true = np.array(s_data["labels"])
            y_pred = np.array(s_data["preds"])
            acc = float((y_true == y_pred).mean())
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            slice_metrics[s_name] = {"n": len(y_true), "accuracy": round(acc, 4), "f1": round(f1, 4)}

        res = {
            "model": str(model_path),
            "test_rows": len(test_rows),
            "test_positives": int(test_labels.sum()),
            "dev_threshold": round(best_t, 3),
            "dev_best_f1": round(best_f1, 4),
            "test_auc": round(overall_auc, 4),
            "test_f1": round(overall_f1, 4),
            "test_precision": round(overall_prec, 4),
            "test_recall": round(overall_rec, 4),
            "seconds": round(time.time() - t0, 2),
            "slices": slice_metrics,
        }
        results.append(res)
        print(json.dumps(res, indent=2))

        if args.write_threshold:
            thresh_file = Path(model_path) / "civic_threshold.json"
            thresh_file.write_text(json.dumps({
                "threshold": round(best_t, 3),
                "dev_f1": round(best_f1, 4),
                "test_f1": round(overall_f1, 4),
                "test_auc": round(overall_auc, 4),
            }, indent=2))

    return results


class CivicClassifier:
    """Lightweight inference wrapper for runtime integration."""

    def __init__(self, model_path: str | Path | None = None,
                 device: str = "cpu", threshold: float | None = None):
        candidates = [
            "artifacts/civic-models/civic-xlmr-large",
            "artifacts/civic-models/civic-me5-base",
            "artifacts/civic-models/civic-mdeberta-xnli",
        ]
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            found = next((c for c in candidates if Path(c).exists()), candidates[1])
            self.model_path = Path(found)
        self.device = device
        thresh_file = self.model_path / "civic_threshold.json"
        if threshold is not None:
            self.threshold = threshold
        elif thresh_file.exists():
            self.threshold = json.loads(thresh_file.read_text()).get("threshold", 0.5)
        else:
            self.threshold = 0.5
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path)).to(self.device)
            self.model.eval()

    def predict(self, texts: list[str], batch_size: int = 64) -> list[dict]:
        """Predicts probability and binary civic/political label for each text."""
        self._ensure_loaded()
        import torch
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                logits = self.model(**inputs).logits
                p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                for prob in p:
                    score = float(prob)
                    results.append({"is_civic": bool(score >= self.threshold), "score": round(score, 4)})
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    tr = commands.add_parser("train")
    tr.add_argument("--dataset", type=Path, default=Path("artifacts/civic_relevance"))
    tr.add_argument("--base-model", required=True)
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=2e-5)
    tr.add_argument("--max-length", type=int, default=256)
    tr.add_argument("--seed", type=int, default=20261003)
    tr.add_argument("--force", action="store_true")

    ev = commands.add_parser("evaluate")
    ev.add_argument("--dataset", type=Path, default=Path("artifacts/civic_relevance"))
    ev.add_argument("--models", nargs="+", required=True)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--write-threshold", action="store_true")
    ev.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        res = evaluate(args)
        if args.output:
            args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

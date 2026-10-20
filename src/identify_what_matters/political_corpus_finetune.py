"""Train and evaluate the political/civic undercurrent classifiers on artifacts/political_corpus (GPU VM).

Single-text multi-class fine-tune, one small model per task (same "one GPU per candidate, 3-5 epochs,
a few minutes each" recipe as stance_finetune.py - simpler here since every task is single-text
classification, not the pair-scoring stance needs):

  train     --task is_political|issue_cluster|sentiment|target|direction|narrative_type
  evaluate  --task ...   compares checkpoints on dev-tuned test, writes per-origin/per-language slices

Saves to a path under artifacts/political_models/<task>/ - copy best_model's config.json-adjacent
`model.safetensors` etc. are NOT compatible with the joblib sklearn pipelines
political_classifier.py loads; see docs/TRAINING_RESULTS.md for the handoff step (export this
transformer to a thin wrapper, or point political_classifier._load_model at a transformers pipeline
instead of joblib - a deliberate decision left for promotion time, not baked in here).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(dataset: str, split: str) -> list[dict]:
    path = Path(dataset, f"{split}.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label_set(task: str, all_rows: list[dict]) -> list[str]:
    return sorted({str(r[task]) for r in all_rows})


def train(args):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite")
    train_rows = rows(args.dataset, "train")
    if not train_rows:
        raise SystemExit(f"No train.jsonl rows under {args.dataset}; run political_corpus.py build first")
    real = [r for r in train_rows if r["origin"] != "synthetic_llm"]
    train_rows = train_rows + real * (args.real_repeat - 1)

    classes = label_set(args.task, rows(args.dataset, "train") + rows(args.dataset, "dev") + rows(args.dataset, "test"))
    label_of = {c: i for i, c in enumerate(classes)}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(classes), ignore_mismatched_sizes=True, dtype=torch.float32)

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    data = Dataset.from_list([{"text": r["text"], "label": label_of[str(r[args.task])]} for r in train_rows])
    data = data.shuffle(seed=args.seed).map(encode, batched=True, remove_columns=["text"]).rename_column("label", "labels")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts = [sum(label_of[str(r[args.task])] == i for r in train_rows) for i in range(len(classes))]
    total = len(train_rows)
    weights = torch.tensor([total / (len(classes) * max(c, 1)) for c in counts], device=device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_ = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels_, weight=weights.float())
            return (loss, outputs) if return_outputs else loss

    total_steps = -(-len(data) // args.batch_size) * args.epochs
    training_args = TrainingArguments(
        output_dir=str(output) + "-work", num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, warmup_steps=max(1, total_steps // 10), weight_decay=0.01, bf16=True, seed=args.seed,
        logging_steps=50, save_strategy="no", report_to="none", lr_scheduler_type="linear")
    started = time.time()
    WeightedTrainer(model=model, args=training_args, train_dataset=data,
                    data_collator=DataCollatorWithPadding(tokenizer)).train()
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "training_run.json").write_text(json.dumps({
        "task": args.task, "base_model": args.base_model, "dataset": args.dataset, "train_rows": len(train_rows),
        "real_rows": len(real), "real_repeat": args.real_repeat, "classes": classes, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "max_length": args.max_length,
        "seconds": round(time.time() - started, 1), "gpu": torch.cuda.get_device_name(0),
        "labels": "LLM judgements (synthetic OpenRouter + real-text relabelled), not human gold",
        "human_gold": False}, indent=2))
    print(json.dumps({"status": "trained", "output": str(output), "rows": len(train_rows), "classes": classes}))


def scores_for(model_path: str, data: list[dict], classes: list[str], device: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    run_file = Path(model_path, "training_run.json")
    tokenizer_source = json.loads(run_file.read_text())["base_model"] if run_file.exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    outputs = []
    for start in range(0, len(data), 128):
        batch_texts = [r["text"] for r in data[start:start + 128]]
        batch = tokenizer(batch_texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            probs = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy()
        outputs.append(probs)
    return np.concatenate(outputs) if outputs else np.zeros((0, len(classes)))


def macro_f1(gold: list, predicted: list) -> float:
    classes = sorted(set(gold) | set(predicted), key=str)
    f1s = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, predicted))
        fp = sum(g != c and p == c for g, p in zip(gold, predicted))
        fn = sum(g == c and p != c for g, p in zip(gold, predicted))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def task_metrics(task: str, data: list[dict], probs: np.ndarray, classes: list[str]) -> dict:
    predicted = [classes[i] for i in probs.argmax(axis=1)]
    gold = [str(r[task]) for r in data]
    slices = defaultdict(lambda: ([], []))
    for r, g, p in zip(data, gold, predicted):
        slices[f"origin:{r.get('origin', 'unknown')}"][0].append(g)
        slices[f"origin:{r.get('origin', 'unknown')}"][1].append(p)
        slices[f"domain:{r.get('domain', 'unknown')}"][0].append(g)
        slices[f"domain:{r.get('domain', 'unknown')}"][1].append(p)
    accuracy = lambda g, p: float(np.mean([a == b for a, b in zip(g, p)])) if g else None
    return {"rows": len(gold), "accuracy": accuracy(gold, predicted), "macro_f1": macro_f1(gold, predicted),
            "slices": {name: {"n": len(g), "accuracy": accuracy(g, p), "macro_f1": macro_f1(g, p)}
                       for name, (g, p) in sorted(slices.items())}}


def evaluate(args):
    dev, test = rows(args.dataset, "dev"), rows(args.dataset, "test")
    classes = label_set(args.task, rows(args.dataset, "train") + dev + test)
    results = []
    for model_path in args.models:
        started = time.time()
        metrics = task_metrics(args.task, test, scores_for(model_path, test, classes, args.device), classes)
        metrics.update({"model": model_path, "seconds": round(time.time() - started, 1)})
        results.append(metrics)
        print(json.dumps({k: metrics[k] for k in ("model", "accuracy", "macro_f1")}), flush=True)
    Path(args.output).write_text(json.dumps({"task": args.task, "dataset": args.dataset, "human_gold": False,
                                             "classes": classes, "results": results}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = ("is_political", "issue_cluster", "sentiment", "target", "direction", "narrative_type")

    training = commands.add_parser("train")
    training.add_argument("--task", choices=tasks, required=True)
    training.add_argument("--dataset", default="artifacts/political_corpus/dataset")
    training.add_argument("--base-model", required=True, help="e.g. microsoft/mdeberta-v3-base or intfloat/multilingual-e5-base")
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--lr", type=float, default=2e-5)
    training.add_argument("--max-length", type=int, default=160)
    training.add_argument("--real-repeat", type=int, default=3)
    training.add_argument("--seed", type=int, default=20261007)
    training.set_defaults(handler=train)

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--task", choices=tasks, required=True)
    evaluation.add_argument("--dataset", default="artifacts/political_corpus/dataset")
    evaluation.add_argument("--models", nargs="*", default=[])
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.set_defaults(handler=evaluate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

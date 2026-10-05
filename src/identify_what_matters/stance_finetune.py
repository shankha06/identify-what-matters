"""Train and evaluate the comment stance (pair) and tone cross-encoders on the stance corpus (GPU VM).

  train     --task stance|tone   fine-tune a multilingual encoder with a classification head
  evaluate  --task stance|tone   compare checkpoints (and a zero-shot NLI baseline for stance) on dev-tuned test

Stance: (context + position, comment) -> does the comment back this position. At inference every side of a
debate is scored and the comment goes to the best side if it clears a dev-selected threshold, else 'no side'.
Tone: (context, comment) -> supportive / critical / humorous / neutral (sarcasm read as intended meaning).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

TONES = ("supportive", "critical", "humorous", "neutral")


def rows(dataset: str, task: str, split: str) -> list[dict]:
    path = Path(dataset, f"{task}_{split}.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pair_of(task: str, row: dict) -> tuple[str, str]:
    return (row["position"], row["comment"]) if task == "stance" else (row["context"], row["comment"])


def train(args):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite")
    labels = 2 if args.task == "stance" else len(TONES)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # fp32 master weights: some hub checkpoints (mDeBERTa-xnli) are stored in fp16 and diverge to NaN otherwise.
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=labels,
                                                               ignore_mismatched_sizes=True, dtype=torch.float32)
    train_rows = rows(args.dataset, args.task, "train")
    real = [r for r in train_rows if r["origin"] != "synthetic"]
    train_rows += real * (args.real_repeat - 1)

    def encode(batch):
        first, second = zip(*(pair_of(args.task, r) for r in batch_rows(batch)))
        return tokenizer(list(first), list(second), truncation=True, max_length=args.max_length)

    def batch_rows(batch):
        keys = list(batch)
        return [dict(zip(keys, values)) for values in zip(*batch.values())]

    columns = ["position", "comment", "label"] if args.task == "stance" else ["context", "comment", "label"]
    data = Dataset.from_list([{k: r[k] for k in columns} for r in train_rows]).shuffle(seed=args.seed)
    data = data.map(encode, batched=True, remove_columns=[c for c in columns if c != "label"]).rename_column("label", "labels")
    weights = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task == "stance":
        positives = sum(r["label"] for r in train_rows)
        weights = torch.tensor([1.0, (len(train_rows) - positives) / max(positives, 1)], device=device)
    elif args.task == "tone":
        # Balanced inverse frequency weights so humorous, supportive, and neutral are not overwhelmed by critical comments
        counts = [sum(r["label"] == i for r in train_rows) for i in range(len(TONES))]
        total = len(train_rows)
        tone_w = [total / (len(TONES) * max(c, 1)) for c in counts]
        weights = torch.tensor(tone_w, device=device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_ = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels_,
                                                     weight=weights.float() if weights is not None else None)
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
        "real_rows": len(real), "real_repeat": args.real_repeat, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "max_length": args.max_length, "seconds": round(time.time() - started, 1),
        "gpu": torch.cuda.get_device_name(0), "labels": "LLM judgements (synthetic OpenRouter + real-comment briefs)",
        "human_gold": False}, indent=2))
    print(json.dumps({"status": "trained", "output": str(output), "rows": len(train_rows)}))


def scores_for(model_path: str, task: str, data: list[dict], device: str, zero_shot: bool = False) -> np.ndarray:
    """Stance: P(backs position). Tone: class probabilities. Zero-shot: NLI entailment of 'agrees that <position>'."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    run_file = Path(model_path, "training_run.json")
    # Saved sentencepiece tokenizers do not always round-trip in transformers 5; the base model's tokenizer is identical.
    tokenizer_source = json.loads(run_file.read_text())["base_model"] if run_file.exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    outputs = []
    if zero_shot:
        entail = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
        pairs = [(r["comment"], "The writer of this comment agrees that " + r["position"].split("|| Position: ")[-1])
                 for r in data]
    else:
        pairs = [pair_of(task, r) for r in data]
    for start in range(0, len(pairs), 128):
        first, second = zip(*pairs[start:start + 128])
        batch = tokenizer(list(first), list(second), truncation=True, max_length=256, padding=True,
                          return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            probs = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy()
        outputs.append(probs[:, entail] if zero_shot else probs[:, 1] if task == "stance" else probs)
    return np.concatenate(outputs)


def macro_f1(gold: list, predicted: list) -> float:
    classes = sorted(set(gold) | set(predicted), key=str)
    f1s = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, predicted))
        fp = sum(g != c and p == c for g, p in zip(gold, predicted))
        fn = sum(g == c and p != c for g, p in zip(gold, predicted))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def comment_decisions(data: list[dict], probs: np.ndarray):
    grouped = defaultdict(dict)
    meta = {}
    for row, p in zip(data, probs):
        key = (row["discussion_id"], row["comment"])
        grouped[key][row["side_index"]] = float(p)
        meta[key] = row
    return grouped, meta


def stance_metrics(data, probs, threshold):
    grouped, meta = comment_decisions(data, probs)
    gold, predicted, slices = [], [], defaultdict(lambda: ([], []))
    for key, sides in grouped.items():
        best = max(sides, key=sides.get)
        guess = best if sides[best] >= threshold else None
        truth = meta[key]["gold_side"]
        gold.append(str(truth))
        predicted.append(str(guess))
        for name in (f"origin:{meta[key]['origin']}", f"lang:{meta[key]['language']}",
                     f"sarcastic:{meta[key]['sarcastic']}"):
            slices[name][0].append(str(truth))
            slices[name][1].append(str(guess))
    accuracy = lambda g, p: float(np.mean([a == b for a, b in zip(g, p)])) if g else None
    return {"comments": len(gold), "accuracy": accuracy(gold, predicted), "macro_f1": macro_f1(gold, predicted),
            "slices": {name: {"n": len(g), "accuracy": accuracy(g, p), "macro_f1": macro_f1(g, p)}
                       for name, (g, p) in sorted(slices.items())}}


def best_threshold(data, probs):
    candidates = np.linspace(0.05, 0.95, 37)
    return float(max(candidates, key=lambda t: stance_metrics(data, probs, t)["macro_f1"]))


def tone_metrics(data, probs):
    predicted = [TONES[i] for i in probs.argmax(axis=1)]
    gold = [TONES[r["label"]] for r in data]
    slices = defaultdict(lambda: ([], []))
    for r, g, p in zip(data, gold, predicted):
        for name in (f"origin:{r['origin']}", f"lang:{r['language']}", f"sarcastic:{r['sarcastic']}"):
            slices[name][0].append(g)
            slices[name][1].append(p)
    accuracy = lambda g, p: float(np.mean([a == b for a, b in zip(g, p)])) if g else None
    return {"rows": len(gold), "accuracy": accuracy(gold, predicted), "macro_f1": macro_f1(gold, predicted),
            "slices": {name: {"n": len(g), "accuracy": accuracy(g, p), "macro_f1": macro_f1(g, p)}
                       for name, (g, p) in sorted(slices.items())}}


def evaluate(args):
    dev, test = rows(args.dataset, args.task, "dev"), rows(args.dataset, args.task, "test")
    results = []
    entries = [(m, False) for m in args.models] + [(m, True) for m in args.zero_shot or []]
    for model_path, zero_shot in entries:
        started = time.time()
        if args.task == "stance":
            threshold = best_threshold(dev, scores_for(model_path, "stance", dev, args.device, zero_shot))
            metrics = stance_metrics(test, scores_for(model_path, "stance", test, args.device, zero_shot), threshold)
            metrics["threshold"] = threshold
        else:
            metrics = tone_metrics(test, scores_for(model_path, "tone", test, args.device))
        metrics.update({"model": model_path, "zero_shot": zero_shot, "seconds": round(time.time() - started, 1)})
        results.append(metrics)
        print(json.dumps({k: metrics[k] for k in ("model", "zero_shot", "accuracy", "macro_f1")}), flush=True)
        if args.write_threshold and args.task == "stance" and not zero_shot:
            Path(model_path, "stance_threshold.json").write_text(json.dumps({"threshold": metrics["threshold"],
                                                                             "selected_on": "dev macro-F1"}))
    Path(args.output).write_text(json.dumps({"task": args.task, "dataset": args.dataset, "human_gold": False,
                                             "results": results}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--task", choices=("stance", "tone"), required=True)
    training.add_argument("--dataset", required=True)
    training.add_argument("--base-model", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--lr", type=float, default=2e-5)
    training.add_argument("--max-length", type=int, default=256)
    training.add_argument("--real-repeat", type=int, default=3)
    training.add_argument("--seed", type=int, default=20261003)
    training.set_defaults(handler=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--task", choices=("stance", "tone"), required=True)
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--models", nargs="*", default=[])
    evaluation.add_argument("--zero-shot", nargs="*", help="NLI models scored zero-shot (stance only)")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--write-threshold", action="store_true")
    evaluation.set_defaults(handler=evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

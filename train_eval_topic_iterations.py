"""Automated 8-Iteration Fine-Tuning and Critical Evaluation Pipeline for Multilingual Event Bi-Encoder on 4x A100 GPU VM."""

import argparse
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from torch.utils.data import BatchSampler
from sklearn.metrics import roc_auc_score, f1_score

PREFIX = "query: "

def load_events(events_path):
    with open(events_path) as f:
        return [json.loads(line) for line in f if line.strip()]

def triplets(events, seed):
    rng = random.Random(seed)
    by_call = defaultdict(list)
    for event in events:
        domain = event.get("domain", event["event_id"].split(".")[0])
        by_call[domain].append(event)
    rows = []
    for event in events:
        domain = event.get("domain", event["event_id"].split(".")[0])
        siblings = [other for other in by_call[domain] if other["event_id"] != event["event_id"]]
        if not siblings:
            siblings = [other for other in events if other["event_id"] != event["event_id"]]
        
        texts = event.get("texts", [])
        if len(texts) < 2:
            continue
        for first, second in itertools.combinations(texts, 2):
            anchor, positive = (first, second) if rng.random() < 0.5 else (second, first)
            negative_event = rng.choice(siblings)
            neg_choice = rng.choice(negative_event["texts"])
            rows.append({
                "anchor": anchor["text"],
                "positive": positive["text"],
                "negative": neg_choice["text"],
                "events": (event["event_id"], negative_event["event_id"])
            })
    rng.shuffle(rows)
    return rows

class EventDisjointBatches(BatchSampler):
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed, self.epoch = size, batch_size, seed, 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pending = list(range(self.size))
        rng.shuffle(pending)
        while pending:
            batch, used, deferred = [], set(), []
            for index in pending:
                evs = self.row_events[index]
                if evs[0] in used or evs[1] in used:
                    deferred.append(index)
                    continue
                batch.append(index)
                used.update(evs)
                if len(batch) >= self.batch_size:
                    break
            yield batch
            pending = deferred

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

def evaluate_on_benchmark(model, benchmark_pairs_path):
    with open(benchmark_pairs_path) as f:
        pairs = [json.loads(line) for line in f]
    
    texts1 = [PREFIX + p["text1"] for p in pairs]
    texts2 = [PREFIX + p["text2"] for p in pairs]
    labels = np.array([p["label"] for p in pairs])
    
    v1 = model.encode(texts1, batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    v2 = model.encode(texts2, batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    sims = np.sum(v1 * v2, axis=1)
    
    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]
    
    auc = float(roc_auc_score(labels, sims))
    best_f1, best_th = 0.0, 0.0
    for th in np.arange(0.3, 0.9, 0.01):
        f1 = float(f1_score(labels, sims >= th))
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
            
    fp_72 = int(np.sum((sims >= 0.72) & (labels == 0)))
    tp_72 = int(np.sum((sims >= 0.72) & (labels == 1)))
    
    return {
        "auc": round(auc, 4),
        "best_f1": round(best_f1, 4),
        "best_th": round(best_th, 3),
        "pos_mean": round(float(pos_sims.mean()), 3),
        "pos_std": round(float(pos_sims.std()), 3),
        "neg_mean": round(float(neg_sims.mean()), 3),
        "neg_std": round(float(neg_sims.std()), 3),
        "margin": round(float(pos_sims.mean() - neg_sims.mean()), 3),
        "fp_at_72": fp_72,
        "fp_rate_72": round(fp_72 / len(neg_sims), 4),
        "tp_at_72": tp_72,
        "tp_rate_72": round(tp_72 / len(pos_sims), 4),
        "n_pos": len(pos_sims),
        "n_neg": len(neg_sims)
    }

def evaluate_on_real_relevance(model, real_test_path):
    with open(real_test_path) as f:
        pairs = [json.loads(line) for line in f]
    anchors = [PREFIX + p["anchor"] for p in pairs]
    captions = [PREFIX + p["caption"] for p in pairs]
    labels = np.array([p["label"] for p in pairs])
    
    v1 = model.encode(anchors, batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    v2 = model.encode(captions, batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    sims = np.sum(v1 * v2, axis=1)
    
    auc = float(roc_auc_score(labels, sims))
    best_f1, best_th = 0.0, 0.0
    for th in np.arange(0.2, 0.8, 0.01):
        f1 = float(f1_score(labels, sims >= th))
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
            
    return {
        "real_test_auc": round(auc, 4),
        "real_test_f1": round(best_f1, 4),
        "real_test_th": round(best_th, 3)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--events", default="artifacts/topic_corpus_v5/dataset/events.jsonl")
    parser.add_argument("--extra-triplets", default="artifacts/relevance_real_v5/train_triplets_v5.jsonl")
    parser.add_argument("--extra-repeat", type=int, default=3)
    parser.add_argument("--benchmark", default="artifacts/topic_corpus_hard_v5/eval_benchmark_pairs.jsonl")
    parser.add_argument("--real-test", default="artifacts/relevance_real_v5/test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"   STARTING ITERATION {args.iteration}: Hyperparameters")
    print(f"   Base: {args.base_model} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | ExtraRepeat: {args.extra_repeat}")
    print(f"=======================================================\n")

    events = [e for e in load_events(args.events) if e.get("split", "train") == "train"]
    print(f"Loaded {len(events)} training events.")
    rows = triplets(events, args.seed)
    print(f"Built {len(rows)} event triplets.")
    
    if args.extra_triplets and Path(args.extra_triplets).exists():
        with open(args.extra_triplets) as f:
            extra = [json.loads(line) for line in f if line.strip()]
        for row in extra:
            for _ in range(args.extra_repeat):
                rows.append({
                    "anchor": row["anchor"],
                    "positive": row["positive"],
                    "negative": row["negative"],
                    "events": tuple(row.get("events", ("extra", "extra:neg")))
                })
        print(f"Appended {len(extra)} extra triplets x {args.extra_repeat} (Total rows: {len(rows)})")
        
    random.Random(args.seed).shuffle(rows)
    row_events = [r.pop("events") for r in rows]
    # Add prefix
    rows = [{k: PREFIX + v for k, v in r.items()} for r in rows]
    dataset = Dataset.from_list(rows)

    sampler = EventDisjointBatches(len(rows), args.batch_size, args.seed)
    sampler.row_events = row_events

    model = SentenceTransformer(args.base_model)
    loss = losses.MultipleNegativesRankingLoss(model)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path / "work"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="no",
        seed=args.seed,
        report_to="none"
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss
    )
    trainer.batch_sampler = sampler

    started = time.time()
    trainer.train()
    elapsed = round(time.time() - started, 1)
    print(f"Iteration {args.iteration} training complete in {elapsed}s.")

    # Save final model
    final_model_dir = output_path / "final"
    model.save_pretrained(str(final_model_dir))

    # CRITICAL EVALUATION
    print(f"--- Running Critical Evaluation for Iteration {args.iteration} ---")
    bench_results = evaluate_on_benchmark(model, args.benchmark)
    real_results = evaluate_on_real_relevance(model, args.real_test)

    eval_summary = {
        "iteration": args.iteration,
        "hyperparameters": {
            "base_model": args.base_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "extra_repeat": args.extra_repeat,
            "elapsed_seconds": elapsed
        },
        "targeted_benchmark": bench_results,
        "real_test": real_results
    }

    print(json.dumps(eval_summary, indent=2))
    with open(output_path / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

if __name__ == "__main__":
    main()

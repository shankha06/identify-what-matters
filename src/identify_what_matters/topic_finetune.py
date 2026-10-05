"""Fine-tune and evaluate same-event topic embeddings on the synthetic event v2 corpus.

Commands:
  evaluate  -- compare any number of models on the event-disjoint dev/test split
  train     -- MultipleNegativesRankingLoss with same-domain hard negatives and
               event-disjoint batches (no false in-batch negatives)
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PREFIXES = {"e5": "query: "}


def load_events(dataset):
    return [json.loads(line) for line in Path(dataset, "events.jsonl").read_text().splitlines() if line.strip()]


def prefix_for(model_path):
    return next((value for key, value in PREFIXES.items() if key in Path(model_path).name.lower()), "")


def encode(model, texts, prefix=""):
    return model.encode([prefix + text for text in texts], batch_size=128, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)


def split_texts(events, split):
    rows = []
    for event in events:
        if event["split"] == split:
            call = event["event_id"].rsplit(".", 1)[0]
            for item in event["texts"]:
                rows.append({**item, "event": event["event_id"], "call": call, "domain": event["domain"]})
    return rows


def pair_scores(rows, vectors):
    similarity = vectors @ vectors.T
    positives, negatives, meta = [], [], []
    for first, second in itertools.combinations(range(len(rows)), 2):
        if rows[first]["call"] != rows[second]["call"]:
            continue
        same = rows[first]["event"] == rows[second]["event"]
        (positives if same else negatives).append(similarity[first, second])
        meta.append((same, rows[first]["language"], rows[second]["language"], similarity[first, second]))
    return np.array(positives), np.array(negatives), meta


def roc_auc(positives, negatives):
    from sklearn.metrics import roc_auc_score
    labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    return float(roc_auc_score(labels, np.r_[positives, negatives]))


def best_f1_threshold(positives, negatives):
    scores = np.r_[positives, negatives]
    labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    best = (0.0, 0.5)
    for threshold in np.unique(np.round(scores, 3)):
        predicted = scores >= threshold
        tp = float((predicted & (labels == 1)).sum())
        precision = tp / max(predicted.sum(), 1)
        recall = tp / max((labels == 1).sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        best = max(best, (f1, float(threshold)))
    return best[1]


def f1_at(positives, negatives, threshold):
    tp = float((positives >= threshold).sum())
    fp = float((negatives >= threshold).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(len(positives), 1)
    return {"f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
            "precision": round(precision, 4), "recall": round(recall, 4)}


def retrieval(rows, vectors, rng, boots=1000):
    """Nearest-neighbour P@1 over all other texts in the split (all domains are distractors)."""
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    hits = np.array([rows[index]["event"] == rows[nearest[index]]["event"] for index in range(len(rows))])
    captions = np.array([row["style"] == "caption" for row in rows])
    comment_hits = []
    for index, row in enumerate(rows):
        if row["style"] == "comment":
            candidates = np.where(captions)[0]
            best = candidates[similarity[index, candidates].argmax()]
            comment_hits.append(rows[best]["event"] == row["event"])
    by_language = defaultdict(list)
    for index, row in enumerate(rows):
        by_language[f"{row['style']}:{row['language']}"].append(bool(hits[index]))
    events = sorted({row["event"] for row in rows})
    per_event = defaultdict(list)
    for index, row in enumerate(rows):
        per_event[row["event"]].append(hits[index])
    samples = [np.mean([value for event in rng.choices(events, k=len(events)) for value in per_event[event]])
               for _ in range(boots)]
    return {"p_at_1": round(float(hits.mean()), 4),
            "p_at_1_ci95": [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)],
            "comment_to_caption_p_at_1": round(float(np.mean(comment_hits)), 4),
            "p_at_1_by_query": {key: round(float(np.mean(value)), 3) for key, value in sorted(by_language.items())}}


def clustering(rows, vectors, threshold):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, v_measure_score
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - threshold).fit_predict(vectors)
    truth = [row["event"] for row in rows]
    return {"ari": round(float(adjusted_rand_score(truth, labels)), 4),
            "v_measure": round(float(v_measure_score(truth, labels)), 4),
            "clusters": int(len(set(labels))), "events": len(set(truth))}


def evaluate_model(model_path, events, device, seed, test_events=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = 128
    prefix = prefix_for(model_path)
    rng = random.Random(seed)
    dev, test = split_texts(events, "dev"), split_texts(test_events or events, "test")
    dev_vectors = encode(model, [row["text"] for row in dev], prefix)
    test_vectors = encode(model, [row["text"] for row in test], prefix)
    dev_pos, dev_neg, _ = pair_scores(dev, dev_vectors)
    test_pos, test_neg, test_meta = pair_scores(test, test_vectors)
    pair_threshold = best_f1_threshold(dev_pos, dev_neg)
    cluster_threshold = max(np.arange(0.20, 0.91, 0.025),
                            key=lambda value: clustering(dev, dev_vectors, value)["ari"])
    cross_lingual = {}
    for language in sorted({row["language"] for row in test} - {"en"}):
        positives = [score for same, first, second, score in test_meta if same and {first, second} == {"en", language}]
        negatives = [score for same, first, second, score in test_meta if not same and {first, second} == {"en", language}]
        if len(positives) >= 5 and len(negatives) >= 5:
            cross_lingual[f"en-{language}"] = {"auc": round(roc_auc(np.array(positives), np.array(negatives)), 4),
                                               "positive_mean_cos": round(float(np.mean(positives)), 3),
                                               "pairs": len(positives) + len(negatives)}
    return {
        "model": model_path, "prefix": prefix,
        "dev": {"pair_auc": round(roc_auc(dev_pos, dev_neg), 4),
                "clustering": clustering(dev, dev_vectors, cluster_threshold)},
        "test": {
            "pair_auc_hard_negatives": round(roc_auc(test_pos, test_neg), 4),
            "pair_threshold_from_dev": round(pair_threshold, 3), **f1_at(test_pos, test_neg, pair_threshold),
            "positive_mean_cos": round(float(test_pos.mean()), 3), "hard_negative_mean_cos": round(float(test_neg.mean()), 3),
            "retrieval": retrieval(test, test_vectors, rng),
            "clustering_threshold_from_dev": round(float(cluster_threshold), 3),
            "clustering": clustering(test, test_vectors, cluster_threshold),
            "cross_lingual_en_vs": cross_lingual,
        },
    }


def evaluate(args):
    events = load_events(args.dataset)
    test_events = load_events(args.test_dataset) if args.test_dataset else None
    results = []
    for model_path in args.models:
        started = time.time()
        result = evaluate_model(model_path, events, args.device, args.seed, test_events)
        result["seconds"] = round(time.time() - started, 1)
        results.append(result)
        print(json.dumps({"model": model_path, "test_auc": result["test"]["pair_auc_hard_negatives"],
                          "p@1": result["test"]["retrieval"]["p_at_1"],
                          "comment->caption": result["test"]["retrieval"]["comment_to_caption_p_at_1"],
                          "ari": result["test"]["clustering"]["ari"]}), flush=True)
        if args.write_threshold and Path(model_path).is_dir():
            Path(model_path, "topic_threshold.json").write_text(json.dumps({
                "cosine_threshold": result["test"]["clustering_threshold_from_dev"], "prefix": result["prefix"],
                "selected_on": "dev split ARI, average-linkage agglomerative", "dataset": args.dataset}, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"dataset": args.dataset, "test_dataset": args.test_dataset or args.dataset,
                                             "origin": "synthetic LLM events; not human gold",
                                             "results": results}, indent=2, ensure_ascii=False) + "\n")


def triplets(events, seed):
    rng = random.Random(seed)
    by_call = defaultdict(list)
    for event in events:
        by_call[event["event_id"].rsplit(".", 1)[0]].append(event)
    rows = []
    for event in events:
        siblings = [other for other in by_call[event["event_id"].rsplit(".", 1)[0]] if other is not event]
        for first, second in itertools.combinations(event["texts"], 2):
            anchor, positive = (first, second) if rng.random() < 0.5 else (second, first)
            negative_event = rng.choice(siblings)
            rows.append({"anchor": anchor["text"], "positive": positive["text"],
                         "negative": rng.choice(negative_event["texts"])["text"],
                         "events": (event["event_id"], negative_event["event_id"])})
    rng.shuffle(rows)
    return rows


def train(args):
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from torch.utils.data import BatchSampler

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite a checkpoint")
    events = [event for event in load_events(args.dataset) if event["split"] == "train"]
    rows = triplets(events, args.seed)
    for path in args.extra_triplets or []:
        extra = [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]
        rows += [{**row, "events": tuple(row["events"])} for row in extra for _ in range(args.extra_repeat)]
    random.Random(args.seed).shuffle(rows)
    row_events = [row.pop("events") for row in rows]
    prefix = prefix_for(args.base_model)
    if prefix:
        rows = [{key: prefix + value for key, value in row.items()} for row in rows]
    dataset = Dataset.from_list(rows)

    class EventDisjointBatches(BatchSampler):
        """Every event appears at most once per batch, as anchor/positive or as hard negative."""

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
                    if len(batch) < self.batch_size and not used & set(row_events[index]):
                        batch.append(index)
                        used.update(row_events[index])
                    else:
                        deferred.append(index)
                pending = deferred
                if len(batch) == self.batch_size:
                    yield batch

        def __len__(self):
            return self.size // self.batch_size

    class Trainer(SentenceTransformerTrainer):
        def get_batch_sampler(self, dataset, batch_size, drop_last, valid_label_columns=None, generator=None, seed=0):
            return EventDisjointBatches(len(dataset), batch_size, args.seed)

    model = SentenceTransformer(args.base_model, device="cuda")
    model.max_seq_length = 128
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output) + "-work", num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, warmup_ratio=0.1, bf16=True, seed=args.seed, logging_steps=20,
        save_strategy="no", report_to="none", dataloader_drop_last=True)
    started = time.time()
    Trainer(model=model, args=training_args, train_dataset=dataset,
            loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(output))
    (output / "training_run.json").write_text(json.dumps({
        "base_model": args.base_model, "dataset": args.dataset, "train_events": len(events),
        "triplets": len(rows), "loss": "MultipleNegativesRankingLoss + same-domain hard negative",
        "batch_sampler": "event-disjoint", "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "seconds": round(time.time() - started, 1), "gpu": torch.cuda.get_device_name(0),
        "origin": "synthetic OpenRouter events", "human_gold": False}, indent=2))
    print(json.dumps({"status": "trained", "output": str(output), "triplets": len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--models", nargs="+", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--seed", type=int, default=20261003)
    evaluation.add_argument("--write-threshold", action="store_true")
    evaluation.add_argument("--test-dataset", help="Score this dataset's test split; thresholds still come from --dataset dev")
    evaluation.set_defaults(handler=evaluate)
    training = commands.add_parser("train")
    training.add_argument("--dataset", required=True)
    training.add_argument("--base-model", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=2e-5)
    training.add_argument("--seed", type=int, default=20261003)
    training.add_argument("--extra-triplets", nargs="*", help="Real-data triplets from relevance.py export")
    training.add_argument("--extra-repeat", type=int, default=2)
    training.set_defaults(handler=train)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

"""Real-data relevance labels distilled from pipeline runs.

Every measured topic's LLM brief marks which Instagram search results are on-topic. Those judgements on real,
noisy, multilingual captions become (topic anchor, caption, label) pairs for evaluating encoders and for
fine-tuning on the VM. Splits are by topic so the same story never appears in train and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

from identify_what_matters.instagram_access import scrub
from identify_what_matters.trends import hours_since


def anchor_text(topic: dict) -> str:
    headline = topic["headlines"][0] if topic.get("headlines") else ""
    return f"{topic['title']}. {headline}".strip()


def export(output_root: Path, destination: Path, seed: int = 20261003) -> dict:
    rows = []
    for run_dir in sorted((output_root / "runs").glob("*")):
        report_path, media_path = run_dir / "report.json", run_dir / "instagram.json"
        if not (report_path.exists() and media_path.exists()):
            continue
        report, media_by_key = json.loads(report_path.read_text()), json.loads(media_path.read_text())
        for topic in report["topics"]:
            brief = topic.get("brief")
            media = (media_by_key.get(topic["topic_key"]) or {}).get("media", [])
            if not brief or not media:
                continue
            on_topic = set(brief.get("on_topic_codes", []))
            reference = datetime.fromisoformat(report["generated_at"])
            judged = sorted(media, key=lambda m: hours_since(m["taken_at"], reference) or 1e9)[:18]
            for item in judged:
                caption = scrub(item["caption"])[:500]
                if len(caption) < 20:
                    continue
                rows.append({"run_id": run_dir.name, "topic_id": topic["topic_id"], "anchor": anchor_text(topic),
                             "caption": caption, "label": int(item["code"] in on_topic), "code": item["code"]})
    topics = sorted({row["topic_id"] for row in rows})
    split_of = {}
    for topic_id in topics:
        bucket = int(hashlib.sha256(f"{seed}:{topic_id}".encode()).hexdigest(), 16) % 10
        split_of[topic_id] = "test" if bucket < 2 else "dev" if bucket < 4 else "train"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "dev", "test"):
        subset = [row for row in rows if split_of[row["topic_id"]] == split]
        (destination / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
        summary[split] = {"pairs": len(subset), "positives": sum(r["label"] for r in subset),
                          "topics": len({r["topic_id"] for r in subset})}
    triplets, rng = [], random.Random(seed)
    for topic_id in [t for t in topics if split_of[t] == "train"]:
        positives = [r for r in rows if r["topic_id"] == topic_id and r["label"] == 1]
        negatives = [r for r in rows if r["topic_id"] == topic_id and r["label"] == 0]
        for positive in positives:
            if negatives:
                triplets.append({"anchor": positive["anchor"], "positive": positive["caption"],
                                 "negative": rng.choice(negatives)["caption"], "events": [topic_id, topic_id + ":neg"]})
    (destination / "train_triplets.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in triplets))
    summary["train_triplets"] = len(triplets)
    summary["runs"] = sorted({row["run_id"] for row in rows})
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def evaluate(dataset: Path, models: list[str], device: str = "cpu") -> list[dict]:
    """AUC of anchor-caption cosine against the LLM on-topic labels; F1 at a dev-chosen threshold."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import f1_score, roc_auc_score

    from identify_what_matters.topic_finetune import prefix_for

    splits = {s: [json.loads(l) for l in (dataset / f"{s}.jsonl").read_text().split("\n") if l.strip()] for s in ("dev", "test")}
    results = []
    for path in models:
        model, prefix = SentenceTransformer(path, device=device), prefix_for(path)
        scores = {}
        for split, rows in splits.items():
            anchors = model.encode([prefix + r["anchor"] for r in rows], normalize_embeddings=True, batch_size=64)
            captions = model.encode([prefix + r["caption"] for r in rows], normalize_embeddings=True, batch_size=64)
            scores[split] = (np.sum(anchors * captions, axis=1), np.array([r["label"] for r in rows]))
        dev_scores, dev_labels = scores["dev"]
        threshold = max(np.unique(np.round(dev_scores, 3)), key=lambda t: f1_score(dev_labels, dev_scores >= t))
        test_scores, test_labels = scores["test"]
        results.append({"model": path, "test_auc": round(float(roc_auc_score(test_labels, test_scores)), 4),
                        "test_f1": round(float(f1_score(test_labels, test_scores >= threshold)), 4),
                        "threshold": float(threshold), "test_pairs": len(test_labels),
                        "test_positive_rate": round(float(test_labels.mean()), 3)})
        print(json.dumps(results[-1]), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--output-root", type=Path, default=Path("output"))
    exporting.add_argument("--destination", type=Path, default=Path("artifacts/relevance_real"))
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--dataset", type=Path, default=Path("artifacts/relevance_real"))
    evaluating.add_argument("--models", nargs="+", required=True)
    evaluating.add_argument("--device", default="cpu")
    evaluating.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export(args.output_root, args.destination), indent=2))
    else:
        results = evaluate(args.dataset, args.models, args.device)
        if args.output:
            args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

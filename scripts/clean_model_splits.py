"""Leakage and duplication cleanup for every model family's train/dev/test splits.

Writes cleaned copies to artifacts/clean_splits/<family>/{train,dev,test}.jsonl and a
split_audit.json report. Originals are never modified.

Rules (applied per family):
- Within a split: drop exact duplicates on the family's within-split key (first occurrence kept).
- Across splits: a text in dev that also occurs in train is dropped from dev; a text in test
  that occurs in train or dev is dropped from test. Train is never dropped.
- Label conflicts (same normalized text, different labels) are counted and reported, not resolved.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def text_of(row: dict, field: str) -> str:
    return norm(row.get(field, ""))


FAMILIES = {
    "political": {
        "files": {"train": "political_corpus/dataset/train.jsonl", "dev": "political_corpus/dataset/dev.jsonl",
                  "test": "political_corpus/dataset/test.jsonl"},
        "text": "text",
        "within": lambda r: (text_of(r, "text"),),
        "label": lambda r: (r.get("is_political"), r.get("direction"), r.get("issue_cluster"),
                            r.get("sentiment"), r.get("target"), r.get("narrative_type")),
    },
    "civic": {
        "files": {"train": "civic_relevance/train.jsonl", "dev": "civic_relevance/dev.jsonl",
                  "test": "civic_relevance/test.jsonl"},
        "text": "text",
        "within": lambda r: (text_of(r, "text"),),
        "label": lambda r: (r.get("label"),),
    },
    "stance": {
        "files": {"train": "stance_corpus/dataset/stance_train.jsonl", "dev": "stance_corpus/dataset/stance_dev.jsonl",
                  "test": "stance_corpus/dataset/stance_test.jsonl"},
        "text": "comment",
        # One comment is labelled once per position (side_index). Key on position, not comment alone.
        "item": lambda r: (str(r.get("discussion_id")), text_of(r, "comment"), str(r.get("position"))),
        "within": lambda r: (str(r.get("discussion_id")), text_of(r, "comment"), str(r.get("position")),
                             str(r.get("side_index")), str(r.get("label"))),
        "label": lambda r: (r.get("label"), r.get("gold_side")),
    },
    "tone": {
        "files": {"train": "stance_corpus/dataset/tone_train.jsonl", "dev": "stance_corpus/dataset/tone_dev.jsonl",
                  "test": "stance_corpus/dataset/tone_test.jsonl"},
        "text": "comment",
        "item": lambda r: (str(r.get("discussion_id")), text_of(r, "comment")),
        "within": lambda r: (str(r.get("discussion_id")), text_of(r, "comment"), str(r.get("label"))),
        "label": lambda r: (r.get("label"),),
    },
    "relevance_v4": {
        "files": {"train": "relevance_real/train.jsonl", "dev": "relevance_real/dev.jsonl",
                  "test": "relevance_real/test.jsonl"},
        "text": "caption",
        "item": lambda r: (text_of(r, "anchor"), text_of(r, "caption")),
        "within": lambda r: (text_of(r, "anchor"), text_of(r, "caption"), str(r.get("label"))),
        "label": lambda r: (r.get("label"),),
    },
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dedupe_within(rows: list[dict], within) -> tuple[list[dict], int]:
    seen, kept = set(), []
    for row in rows:
        key = within(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept, len(rows) - len(kept)


def label_conflicts(rows: list[dict], item, label) -> int:
    """Items (text within its discussion or anchor) that carry more than one label."""
    labels_by_item: dict[tuple, set] = defaultdict(set)
    for row in rows:
        labels_by_item[item(row)].add(label(row))
    return sum(1 for labels in labels_by_item.values() if len(labels) > 1)


def clean_family(root: Path, name: str, spec: dict, out_root: Path, quarantine_conflicts: bool = False) -> dict:
    raw = {s: read_jsonl(root / p) for s, p in spec["files"].items()}
    report = {"raw_rows": {s: len(r) for s, r in raw.items()}, "within_dupes_dropped": {}, "cross_dropped": {}}
    item = spec.get("item", lambda row: (text_of(row, spec["text"]),))
    quarantined = []
    if quarantine_conflicts:
        labels_by_item = defaultdict(set)
        for rows in raw.values():
            for row in rows:
                labels_by_item[item(row)].add(spec["label"](row))
        conflicting = {key for key, labels in labels_by_item.items() if len(labels) > 1}
        report["conflicting_items_quarantined"] = len(conflicting)
        report["conflicting_rows_quarantined"] = {}
        for split, rows in raw.items():
            rejected = [row for row in rows if item(row) in conflicting]
            quarantined.extend({"split": split, "record": row} for row in rejected)
            report["conflicting_rows_quarantined"][split] = len(rejected)
            raw[split] = [row for row in rows if item(row) not in conflicting]
    cleaned = {}
    for split, rows in raw.items():
        cleaned[split], report["within_dupes_dropped"][split] = dedupe_within(rows, spec["within"])
    text_field = spec["text"]
    train_texts = {text_of(r, text_field) for r in cleaned["train"]}
    dev_texts = {text_of(r, text_field) for r in cleaned["dev"] if text_of(r, text_field) not in train_texts}
    before_dev = len(cleaned["dev"])
    cleaned["dev"] = [r for r in cleaned["dev"] if text_of(r, text_field) not in train_texts]
    report["cross_dropped"]["dev_in_train"] = before_dev - len(cleaned["dev"])
    before_test = len(cleaned["test"])
    cleaned["test"] = [r for r in cleaned["test"]
                       if text_of(r, text_field) not in train_texts and text_of(r, text_field) not in dev_texts]
    report["cross_dropped"]["test_in_train_or_dev"] = before_test - len(cleaned["test"])
    report["clean_rows"] = {s: len(r) for s, r in cleaned.items()}
    report["label_conflicts_within_train"] = label_conflicts(cleaned["train"], item, spec["label"])
    report["label_conflicts_within_test"] = label_conflicts(cleaned["test"], item, spec["label"])
    dest = out_root / name
    dest.mkdir(parents=True, exist_ok=True)
    if quarantine_conflicts:
        (dest / "quarantined.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantined), encoding="utf-8",
        )
    for split, rows in cleaned.items():
        (dest / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                             encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/clean_splits"))
    parser.add_argument("--quarantine-conflicts", action="store_true")
    args = parser.parse_args()
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    audit = {name: clean_family(args.artifacts, name, spec, out_root, args.quarantine_conflicts)
             for name, spec in FAMILIES.items()}
    (out_root / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

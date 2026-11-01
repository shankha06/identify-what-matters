"""Dedupe the crawled news pool against every existing party split.

A new headline is dropped if its near-duplicate group (word 3-gram Jaccard, via
scripts/party_clean_splits.group_rows) contains any existing row. Within the
surviving new rows, one row is kept per group.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from party_clean_splits import group_rows  # noqa: E402

ROOT = HERE.parent
EXISTING = [
    ROOT / "artifacts/party_clean_splits/train_clean.jsonl",
    ROOT / "artifacts/party_clean_splits/silver_clean.jsonl",
    ROOT / "artifacts/party_clean_splits/archive/frozen_clean_contaminated.jsonl",
    ROOT / "artifacts/party_v6/heldout_sonnet.jsonl",
    ROOT / "artifacts/party_v6/candidates.jsonl",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    existing_texts: list[str] = []
    for path in EXISTING:
        rows = read_jsonl(path)
        existing_texts.extend(r["text"] for r in rows)
        print(f"existing {path.relative_to(ROOT)}: {len(rows)}")

    new_rows = read_jsonl(args.raw)
    print(f"raw news: {len(new_rows)}")

    texts = existing_texts + [r["text"] for r in new_rows]
    groups = group_rows(texts)
    n_exist = len(existing_texts)
    existing_groups = set(groups[:n_exist])

    kept: list[dict] = []
    seen_groups: set[int] = set()
    dropped_overlap = 0
    dropped_internal = 0
    for i, row in enumerate(new_rows):
        g = groups[n_exist + i]
        if g in existing_groups:
            dropped_overlap += 1
            continue
        if g in seen_groups:
            dropped_internal += 1
            continue
        seen_groups.add(g)
        kept.append(
            {
                "id": row["id"],
                "text": row["text"],
                "source": row["source"],
                "query": row["query"],
                "lang": row["lang"],
                "url": row["url"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"dropped overlapping with existing: {dropped_overlap}")
    print(f"dropped internal near-duplicates: {dropped_internal}")
    print(f"kept {len(kept)} -> {args.output}")
    print("by lang:", dict(Counter(r["lang"] for r in kept)))


if __name__ == "__main__":
    main()

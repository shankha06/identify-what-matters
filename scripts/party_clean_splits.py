"""Exact and near-duplicate cleanup for the 18-label party training pool.

Reads the training pool and the two evaluation sets (silver and frozen). Writes
train_clean.jsonl, silver_clean.jsonl, frozen_clean.jsonl and split_audit.json to --out.
Inputs are never modified.

Rules:
- Text is normalized: lowercase, URLs removed, punctuation and symbols (incl. emoji) removed, whitespace collapsed.
- Near-duplicate: Jaccard >= 0.8 over word 3-gram sets. Connected components form groups.
- Within train: keep the first row per group, drop the rest (reason within_train_dup).
- Train rows whose group contains a silver or frozen row are dropped (reason leaks_eval).
  Eval sets are never modified, so the reported eval numbers stay comparable.
- Label conflicts (one group with more than one party label) are counted and listed, not resolved.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

THRESH = 0.8


def norm(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text.lower())
    chars = [" " if unicodedata.category(c)[0] in "PSZC" else c for c in text]
    return " ".join("".join(chars).split())


def shingles(norm_text: str) -> set[tuple]:
    words = norm_text.split()
    if not words:
        return set()
    if len(words) < 3:
        return {tuple(words)}
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def group_rows(texts: list[str]) -> list[int]:
    sh = [shingles(norm(t)) for t in texts]
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    index: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            index[g].append(i)
    for i, s in enumerate(sh):
        if not s:
            continue
        candidates = set()
        for g in s:
            candidates.update(index[g])
        for j in candidates:
            if j <= i or not sh[j]:
                continue
            if len(s & sh[j]) / len(s | sh[j]) >= THRESH:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("artifacts/party_v4/train_v4.jsonl"))
    parser.add_argument("--silver", type=Path, default=Path("artifacts/party_v4/silver_holdout.jsonl"))
    parser.add_argument("--frozen", type=Path, default=Path("artifacts/party_v3/frozen188.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/party_clean_splits"))
    args = parser.parse_args()

    train, silver, frozen = load(args.train), load(args.silver), load(args.frozen)
    rows = train + silver + frozen
    split = ["train"] * len(train) + ["silver"] * len(silver) + ["frozen"] * len(frozen)
    groups = group_rows([r["text"] for r in rows])

    eval_groups = {g for g, s in zip(groups, split) if s != "train"}
    train_groups_seen: set[int] = set()
    kept_train: list[dict] = []
    drop = Counter()
    for r, g, s in zip(rows, groups, split):
        if s != "train":
            continue
        if g in eval_groups:
            drop["leaks_eval"] += 1
        elif g in train_groups_seen:
            drop["within_train_dup"] += 1
        else:
            train_groups_seen.add(g)
            kept_train.append(r)

    eval_hit = {"silver": 0, "frozen": 0}
    train_groups_by_id = defaultdict(set)
    for g, s in zip(groups, split):
        if s == "train":
            train_groups_by_id[g].add("train")
    for r, g, s in zip(rows, groups, split):
        if s != "train" and g in train_groups_by_id:
            eval_hit[s] += 1

    labels_by_group: dict[int, set] = defaultdict(set)
    for r, g in zip(rows, groups):
        labels_by_group[g].add(r["party"])
    conflicts = [
        {"texts": [r["text"][:120] for r, g in zip(rows, groups) if g == grp],
         "labels": sorted(labs)}
        for grp, labs in labels_by_group.items() if len(labs) > 1
    ]

    before = Counter(r["party"] for r in train)
    after = Counter(r["party"] for r in kept_train)
    audit = {
        "threshold_jaccard": THRESH,
        "raw_rows": {"train": len(train), "silver": len(silver), "frozen": len(frozen)},
        "train_dropped": dict(drop),
        "clean_rows": {"train": len(kept_train), "silver": len(silver), "frozen": len(frozen)},
        "eval_rows_with_train_near_dup": eval_hit,
        "label_conflict_groups": len(conflicts),
        "label_conflict_examples": conflicts[:30],
        "per_label_train_before_after": {k: [before[k], after.get(k, 0)] for k in sorted(before)},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / "train_clean.jsonl", kept_train)
    dump(args.out / "silver_clean.jsonl", silver)
    dump(args.out / "frozen_clean.jsonl", frozen)
    (args.out / "split_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "label_conflict_examples"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

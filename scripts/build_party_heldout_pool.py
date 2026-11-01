"""Build an unlabelled pool of real posts for a clean party held-out set.

Sources (real collected data only; google_instagram_mock_* files are excluded):
- output/reddit_signals.json: post titles and comment bodies
- output/collections/20261004T114827Z.json: post captions
- output/panel_feed_cache.json: panel item captions and titles

Any candidate whose Jaccard >= 0.8 (word 3-grams, see party_clean_splits.py) with a row in
train, silver or frozen is dropped. Within the pool, one row per near-duplicate group is kept.

Output: candidates.jsonl with fields id, text, source, kw_hit. If the pool exceeds --cap,
all keyword-hit rows are kept and the rest is a seeded random sample.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from party_clean_splits import group_rows, load

MIN_LEN = 25
PARTY_KW = re.compile(
    r"\b(bjp|modi|amit shah|congress|rahul gandhi|kharge|gandhi|aap|kejriwal|tmc|mamata|"
    r"samajwadi|akhilesh|dmk|stalin|tdp|naidu|shiv sena|uddhav|shinde|thackeray|ncp|pawar|"
    r"rjd|tejashwi|jdu|nitish|aiadmk|brs|kcr|bsp|mayawati|ysrcp|jagan|cpim|cpi\(m\)|"
    r"cockroach janta party|cockroachjantaparty|cockroachisback)\b",
    re.IGNORECASE,
)


def collect_texts(root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    reddit = json.loads((root / "output/reddit_signals.json").read_text(encoding="utf-8"))
    for post in reddit:
        if post.get("title"):
            out.append((post["title"], "reddit_title"))
        for c in post.get("comments") or []:
            body = (c.get("body") or "").strip()
            if body and body not in ("[deleted]", "[removed]"):
                out.append((body, "reddit_comment"))

    coll = json.loads((root / "output/collections/20261004T114827Z.json").read_text(encoding="utf-8"))
    for post in coll.get("posts") or []:
        cap = (post.get("caption") or "").strip()
        if cap:
            out.append((cap, "instagram_caption"))

    panel = json.loads((root / "output/panel_feed_cache.json").read_text(encoding="utf-8"))
    for account in panel.values():
        for item in account.get("items") or []:
            text = (item.get("caption_text") or item.get("title") or "").strip()
            if text:
                out.append((text, "panel_caption"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train", type=Path, default=Path("artifacts/party_clean_splits/train_clean.jsonl"))
    parser.add_argument("--silver", type=Path, default=Path("artifacts/party_clean_splits/silver_clean.jsonl"))
    parser.add_argument("--frozen", type=Path, default=Path("artifacts/party_clean_splits/archive/frozen_clean_contaminated.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/party_v6/candidates.jsonl"))
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    root = args.root

    existing = load(root / args.train) + load(root / args.silver) + load(root / args.frozen)
    existing_texts = [r["text"] for r in existing]

    seen_text: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for text, source in collect_texts(root):
        if len(text) < MIN_LEN or text in seen_text:
            continue
        seen_text.add(text)
        candidates.append((text, source))

    texts = existing_texts + [t for t, _ in candidates]
    groups = group_rows(texts)
    existing_groups = set(groups[: len(existing_texts)])
    cand_groups = groups[len(existing_texts):]

    kept: list[tuple[str, str]] = []
    drop = Counter()
    used_groups: set[int] = set()
    for (text, source), g in zip(candidates, cand_groups):
        if g in existing_groups:
            drop["near_dup_of_train_silver_frozen"] += 1
        elif g in used_groups:
            drop["within_pool_dup"] += 1
        else:
            used_groups.add(g)
            kept.append((text, source))

    hits = [(t, s) for t, s in kept if PARTY_KW.search(t)]
    rest = [(t, s) for t, s in kept if not PARTY_KW.search(t)]
    if len(hits) + len(rest) > args.cap:
        rng = random.Random(args.seed)
        rng.shuffle(rest)
        rest = rest[: max(0, args.cap - len(hits))]
    chosen = hits + rest
    rng = random.Random(args.seed)
    rng.shuffle(chosen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for i, (text, source) in enumerate(chosen):
            row = {
                "id": f"cand_{i:05d}",
                "text": text,
                "source": source,
                "kw_hit": bool(PARTY_KW.search(text)),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "raw_candidates": len(candidates),
        "dropped": dict(drop),
        "after_dedup": len(kept),
        "kw_hit": len(hits),
        "written": len(chosen),
        "by_source": dict(Counter(s for _, s in chosen)),
    }, indent=2))


if __name__ == "__main__":
    main()

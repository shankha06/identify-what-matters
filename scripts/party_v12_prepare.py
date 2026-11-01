"""Build the v12 party dataset.

Splits (all rows carry `text` and `party`, labels canonical 18-way):
  train  = artifacts/party_clean_splits/train_clean.jsonl
           + 80% of Sonnet-labelled Google News headlines (stratified)
  dev    = 20% of Sonnet-labelled Google News headlines (stratified)
           -> checkpoint selection only
  eval   = artifacts/party_v6/heldout_sonnet.jsonl (300)
           -> scored once, never trained on, never used for selection

Also verifies near-duplicate groups (word 3-gram Jaccard, same helper as the
dedupe step) do not cross split boundaries.
"""
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from party_clean_splits import group_rows  # noqa: E402

LABELS = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "CJP", "None",
]
CANON = {lbl.upper(): lbl for lbl in LABELS}
CANON.update({"CONGRESS": "Congress", "SHIV SENA": "Shiv Sena", "NONE": "None"})

OUT = ROOT / "artifacts" / "party_v12"
TRAIN_CLEAN = ROOT / "artifacts" / "party_clean_splits" / "train_clean.jsonl"
NEWS_LABELS = ROOT / "artifacts" / "party_v7" / "sonnet_labels_news.jsonl"
HELDOUT = ROOT / "artifacts" / "party_v6" / "heldout_sonnet.jsonl"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # News: Sonnet labels joined to candidate text via record.text.
    news, dropped = [], collections.Counter()
    for r in load(NEWS_LABELS):
        lbl = CANON.get(str(r.get("label", "")).strip().upper())
        text = (r.get("record") or {}).get("text", "").strip()
        if lbl is None:
            dropped["invalid_label"] += 1
            continue
        if not text:
            dropped["empty_text"] += 1
            continue
        news.append({"text": text, "party": lbl, "source": "news_v7"})
    print("news kept", len(news), "dropped", dict(dropped))

    # Stratified 80/20 split of news by label.
    rng = random.Random(42)
    by_label = collections.defaultdict(list)
    for r in news:
        by_label[r["party"]].append(r)
    news_train, news_dev = [], []
    for lbl in LABELS:
        rows = by_label.get(lbl, [])
        rng.shuffle(rows)
        cut = int(round(len(rows) * 0.8))
        news_train += rows[:cut]
        news_dev += rows[cut:]

    train_clean = [
        {"text": r["text"], "party": r["party"], "source": "train_clean"}
        for r in load(TRAIN_CLEAN) if r.get("party") in LABELS
    ]

    heldout_raw = load(HELDOUT)
    keys = collections.Counter(k for r in heldout_raw for k in r)
    print("heldout keys", dict(keys))
    label_key = next(k for k in ("party", "label", "expected_label") if k in keys)
    heldout = []
    for r in heldout_raw:
        lbl = CANON.get(str(r.get(label_key, "")).strip().upper())
        if lbl is not None:
            heldout.append({"text": r["text"], "party": lbl, "source": "heldout_sonnet"})
    print("heldout kept", len(heldout), "using label key", label_key)

    train = train_clean + news_train
    rng.shuffle(train)

    # Leakage check: no near-duplicate group may span two splits.
    all_rows = [("train", r) for r in train] + [("dev", r) for r in news_dev] + [("eval", r) for r in heldout]
    groups = group_rows([r["text"] for _, r in all_rows])
    split_of_group = collections.defaultdict(set)
    for (split, _), g in zip(all_rows, groups):
        split_of_group[g].add(split)
    cross = sum(1 for s in split_of_group.values() if len(s) > 1)
    print("cross-split near-duplicate groups:", cross)

    write(OUT / "train.jsonl", train)
    write(OUT / "dev.jsonl", news_dev)
    write(OUT / "eval.jsonl", heldout)

    stats = {
        "train": len(train),
        "train_clean": len(train_clean),
        "news_train": len(news_train),
        "dev": len(news_dev),
        "eval": len(heldout),
        "cross_split_dup_groups": cross,
        "train_label_counts": collections.Counter(r["party"] for r in train),
        "dev_label_counts": collections.Counter(r["party"] for r in news_dev),
        "eval_label_counts": collections.Counter(r["party"] for r in heldout),
        "news_dropped": dict(dropped),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()

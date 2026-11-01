"""Build the v4 party-support training and silver selection sets.

Inputs (under artifacts/ unless noted):
  demographics_models/party_train_17_all.jsonl   existing labelled rows (may include CJP)
  party_v3/pool_teacher18.jsonl                  real pool labelled by the 18-label teacher
  party_v4/cjp_synth.jsonl                       teacher-checked synthetic CJP posts
  party_v3/frozen188.jsonl                       frozen eval rows (never used here)

Outputs (artifacts/party_v4/):
  train_v4.jsonl    training rows: text, party, source
  silver_holdout.jsonl  selection rows (never trained on): text, party, source

Rules:
  - Frozen texts are removed from every source (normalised match).
  - Silver holdout = 15% of the real pool (stratified by teacher label) + 20% of
    CJP synthetic rows. Silver texts are removed from train, including duplicates.
  - Duplicates (normalised text) keep the first occurrence; labelled rows come first.
  - Rows with no label, or a label outside the 18 classes, are dropped.
"""

import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT_DIR = ART / "party_v4"
LABELS_18 = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "CJP", "None",
]
SILVER_REAL_FRAC = 0.15
SILVER_CJP_FRAC = 0.20
SEED = 13


def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratified_split(rows, frac, rng):
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["party"]].append(r)
    silver, train = [], []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        k = round(len(items) * frac) if len(items) >= 4 else 0
        silver.extend(items[:k])
        train.extend(items[k:])
    return train, silver


def main():
    rng = random.Random(SEED)
    frozen_keys = {norm(r["text"]) for r in load(ART / "party_v3" / "frozen188.jsonl")}

    existing = [
        {"text": r.get("text", ""), "party": r.get("party"), "source": "party_train_17_all"}
        for r in load(ART / "demographics_models" / "party_train_17_all.jsonl")
    ]
    real = [
        {"text": r["text"], "party": r.get("teacher_party"), "source": "teacher_real_pool"}
        for r in load(ART / "party_v3" / "pool_teacher18.jsonl")
    ]
    cjp = [
        {"text": r["text"], "party": r.get("party"), "source": "cjp_synth"}
        for r in load(OUT_DIR / "cjp_synth.jsonl")
    ]

    # Clean and de-duplicate each pool first; drop frozen overlap and bad labels.
    dropped = collections.Counter()
    seen = set()

    def clean(rows, source):
        out = []
        for r in rows:
            text = (r.get("text") or "").strip()
            party = r.get("party")
            if not text or party not in LABELS_18:
                dropped[f"{source}:no_label"] += 1
                continue
            key = norm(text)
            if key in frozen_keys:
                dropped[f"{source}:frozen_overlap"] += 1
                continue
            if key in seen:
                dropped[f"{source}:duplicate"] += 1
                continue
            seen.add(key)
            out.append({"text": text, "party": party, "source": source})
        return out

    existing_c = clean(existing, "party_train_17_all")
    real_c = clean(real, "teacher_real_pool")
    cjp_c = clean(cjp, "cjp_synth")

    real_train, real_silver = stratified_split(real_c, SILVER_REAL_FRAC, rng)
    cjp_train, cjp_silver = stratified_split(cjp_c, SILVER_CJP_FRAC, rng)

    silver = real_silver + cjp_silver
    silver_keys = {norm(r["text"]) for r in silver}
    train = [r for r in existing_c + real_train + cjp_train if norm(r["text"]) not in silver_keys]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "train_v4.jsonl", "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "silver_holdout.jsonl", "w", encoding="utf-8") as fh:
        for r in silver:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train={len(train)} silver={len(silver)} -> {OUT_DIR}")
    print("train by label:", collections.Counter(r["party"] for r in train).most_common())
    print("silver by label:", collections.Counter(r["party"] for r in silver).most_common())
    print("train by source:", dict(collections.Counter(r["source"] for r in train)))
    print("dropped:", dict(dropped))


if __name__ == "__main__":
    main()

"""Build the Sonnet-labelled held-out set for the party model.

Selection rule:
- Keep every party-positive row (all non-None labels) so each party has eval support.
- Add a seeded random sample of None rows to bring the set to ~1:1 (party vs None).
- Verify no overlap with train, silver, or frozen splits (Jaccard >= 0.8 over word 3-grams).

Output rows are never used for training or model selection.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from party_clean_splits import group_rows  # noqa: E402

CANON = {
    "AAP": "AAP", "AIADMK": "AIADMK", "BJP": "BJP", "BRS": "BRS", "BSP": "BSP",
    "CJP": "CJP", "CONGRESS": "Congress", "CPIM": "CPIM", "DMK": "DMK", "JDU": "JDU",
    "NCP": "NCP", "NONE": "None", "RJD": "RJD", "SHIV SENA": "Shiv Sena", "SP": "SP",
    "TDP": "TDP", "TMC": "TMC", "YSRCP": "YSRCP",
}
SEED = 17
NONE_SAMPLE = 136
EXCLUDE_IDS = {"cand_00289"}  # Sonnet returned a malformed reply twice; excluded, not hand-labelled.

ART = ROOT / "artifacts"
POOL = ART / "party_v6" / "candidates.jsonl"
LABELS = ART / "party_v6" / "sonnet_labels.jsonl"
OUT = ART / "party_v6" / "heldout_sonnet.jsonl"
REFS = [
    ART / "party_clean_splits" / "train_clean.jsonl",
    ART / "party_clean_splits" / "silver_clean.jsonl",
    ART / "party_clean_splits" / "archive" / "frozen_clean_contaminated.jsonl",
]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    texts = {r["id"]: r["text"] for r in read(POOL)}
    labelled = []
    for r in read(LABELS):
        if r["_labeling_id"] in EXCLUDE_IDS:
            continue
        label = CANON[r["label"].strip().upper()]
        labelled.append({"id": r["_labeling_id"], "text": texts[r["_labeling_id"]], "label": label})

    positives = [r for r in labelled if r["label"] != "None"]
    nones = [r for r in labelled if r["label"] == "None"]
    rng = random.Random(SEED)
    chosen = positives + rng.sample(nones, NONE_SAMPLE)

    # Overlap check against every training-side split.
    ref_texts = [r["text"] for p in REFS for r in read(p)]
    overlap = group_rows([r["text"] for r in chosen] + ref_texts)
    ref_groups = set(overlap[len(chosen):])
    clashes = [i for i in range(len(chosen)) if overlap[i] in ref_groups]
    if clashes:
        chosen = [r for i, r in enumerate(chosen) if i not in set(clashes)]
        print(f"removed {len(clashes)} rows overlapping train/silver/frozen")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = collections.Counter(r["label"] for r in chosen)
    print(f"wrote {len(chosen)} rows to {OUT}")
    print(sorted(counts.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    main()

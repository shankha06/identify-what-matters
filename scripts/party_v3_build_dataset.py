"""Build the v3 party-support training set from teacher-labelled real text and
the existing 17-label training data.

Inputs (all under artifacts/):
  demographics_models/party_train_17_all.jsonl  existing 17-label training rows
  party_v3/pool_teacher.jsonl                   teacher-labelled real pool
  party_v3/frozen188.jsonl                      frozen eval rows (never trained on)

Output:
  party_v3/train_v3.jsonl  rows with text, party, source

Rules:
  - Frozen-eval texts are removed from every source (normalised match).
  - Duplicates (normalised text) keep the first occurrence; existing labelled
    rows come first, so their labels win over teacher labels.
  - Teacher rows with no parsed label are dropped.
  - Only the 17 production labels are kept.
"""

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PARTIES_17 = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "None",
]
OUT = ART / "party_v3" / "train_v3.jsonl"


def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    frozen = load(ART / "party_v3" / "frozen188.jsonl")
    frozen_keys = {norm(r["text"]) for r in frozen}

    sources = [
        ("party_train_17_all", [
            {"text": r.get("text", ""), "party": r.get("party")}
            for r in load(ART / "demographics_models" / "party_train_17_all.jsonl")
        ]),
        ("teacher_real_pool", [
            {"text": r["text"], "party": r.get("teacher_party")}
            for r in load(ART / "party_v3" / "pool_teacher.jsonl")
        ]),
    ]

    seen = set()
    out_rows = []
    dropped = collections.Counter()
    for source, rows in sources:
        for r in rows:
            text = (r.get("text") or "").strip()
            party = r.get("party")
            if not text or party not in PARTIES_17:
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
            out_rows.append({"text": text, "party": party, "source": source})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} rows to {OUT}")
    print("by source:", dict(collections.Counter(r["source"] for r in out_rows)))
    print("by label:", collections.Counter(r["party"] for r in out_rows).most_common())
    print("dropped:", dict(dropped))


if __name__ == "__main__":
    main()

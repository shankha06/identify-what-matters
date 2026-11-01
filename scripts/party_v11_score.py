"""Score a party classifier and write per-class probabilities for each input file.

Output rows keep the input fields and add `probs`, ordered by PARTIES_17.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PARTIES_17 = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD", "JDU",
    "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "None",
]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir)).eval()
    id2label = model.config.id2label
    order = [next(int(k) for k, v in id2label.items() if v == p) for p in PARTIES_17]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        rows = read(path)
        out_path = args.out_dir / f"{path.stem}_probs.jsonl"
        with out_path.open("w", encoding="utf-8") as fh, torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                chunk = rows[i:i + args.batch_size]
                enc = tok([r["text"] for r in chunk], padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
                probs = torch.softmax(model(**enc).logits, dim=-1)[:, order].tolist()
                for r, p in zip(chunk, probs):
                    fh.write(json.dumps({**r, "probs": p}, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

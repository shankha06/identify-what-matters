"""Score party classifiers on JSONL inputs and write full per-label probabilities.

Labels come from each model's id2label (18 labels, including CJP). Max length
matches training (160). Output rows keep the input fields and add `probs`
in the order given by `labels` in each output row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir)).eval()
    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        rows = read(path)
        out_path = args.out_dir / f"{path.stem}_probs.jsonl"
        with out_path.open("w", encoding="utf-8") as fh, torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                chunk = rows[i:i + args.batch_size]
                enc = tok([r["text"] for r in chunk], padding=True, truncation=True,
                          max_length=args.max_len, return_tensors="pt")
                probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
                for r, p in zip(chunk, probs):
                    fh.write(json.dumps({**r, "labels": labels, "probs": p},
                                        ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

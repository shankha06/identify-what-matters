"""Train the v3 17-label party-support candidate (teacher-distilled data).

Runs on the VM (GPU selected via CUDA_VISIBLE_DEVICES). Fixed schedule: no
checkpoint selection on any gold data. The internal 10% validation split is
for monitoring only. The frozen 188-row set is scored once, after training.

    python3 party_v3_train.py --train train_v3.jsonl --frozen frozen188.jsonl \
        --out /home/s0r0qw6/party_bert_training/output_model_v3
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PARTIES_17 = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "None",
]
LABEL2ID = {p: i for i, p in enumerate(PARTIES_17)}
ID2LABEL = {i: p for p, i in LABEL2ID.items()}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TextSet(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collate(tokenizer, max_len):
    def collate(batch):
        enc = tokenizer(
            [r["text"] for r in batch],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([LABEL2ID[r["party"]] for r in batch])
        return enc

    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    model.train()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train_rows = [r for r in load(args.train) if r["party"] in LABEL2ID]
    frozen_rows = [r for r in load(args.frozen) if r["party"] in LABEL2ID]
    fit_rows, mon_rows = train_test_split(
        train_rows,
        test_size=0.1,
        stratify=[r["party"] for r in train_rows],
        random_state=args.seed,
    )
    print(f"fit={len(fit_rows)} monitor={len(mon_rows)} frozen={len(frozen_rows)}", flush=True)
    print("fit label counts:", Counter(r["party"] for r in fit_rows).most_common(), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    # transformers 5.x loads this checkpoint as float16 by default. Pure fp16
    # AdamW produces NaN after the first step, so force float32 weights.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(PARTIES_17),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.float32,
    ).to(device)
    assert next(model.parameters()).dtype == torch.float32
    collate = make_collate(tokenizer, args.max_len)

    fit_loader = DataLoader(TextSet(fit_rows), batch_size=args.batch, shuffle=True, collate_fn=collate)
    mon_loader = DataLoader(TextSet(mon_rows), batch_size=64, shuffle=False, collate_fn=collate)
    frozen_loader = DataLoader(TextSet(frozen_rows), batch_size=64, shuffle=False, collate_fn=collate)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(fit_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)

    model.train()
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step, batch in enumerate(fit_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(fit_loader)} loss {running / 50:.4f}", flush=True)
                running = 0.0

        mon_gold = [LABEL2ID[r["party"]] for r in mon_rows]
        mon_pred = predict(model, mon_loader, device)
        print(
            f"epoch {epoch} done in {time.time() - started:.0f}s | "
            f"monitor acc={accuracy_score(mon_gold, mon_pred):.4f} "
            f"macro_f1={f1_score(mon_gold, mon_pred, average='macro'):.4f}",
            flush=True,
        )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Single evaluation on the frozen set. Not used for any selection.
    gold = [LABEL2ID[r["party"]] for r in frozen_rows]
    pred = predict(model, frozen_loader, device)
    acc = accuracy_score(gold, pred)
    mf1 = f1_score(gold, pred, average="macro")
    labels_present = sorted(set(gold) | set(pred))
    report = classification_report(
        gold, pred, labels=labels_present,
        target_names=[ID2LABEL[i] for i in labels_present], zero_division=0,
    )
    metrics = {
        "frozen_rows": len(frozen_rows),
        "frozen_accuracy": round(acc, 4),
        "frozen_macro_f1": round(mf1, 4),
        "baseline_accuracy": 0.5372,
        "baseline_macro_f1": 0.6056,
        "args": vars(args),
    }
    (out_dir / "frozen_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "frozen_report.txt").write_text(report, encoding="utf-8")
    with open(out_dir / "frozen_predictions.jsonl", "w", encoding="utf-8") as fh:
        for r, p in zip(frozen_rows, pred):
            fh.write(json.dumps({"text": r["text"], "gold": r["party"], "pred": ID2LABEL[p]}, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()

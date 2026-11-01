"""Train an 18-label party-support candidate (v4).

Selection uses the silver holdout only: the checkpoint with the best silver
macro-F1 across epochs is kept. The frozen 188-row set is scored once, after
selection, and is never used for any choice.

Runs on the VM (pick the GPU with CUDA_VISIBLE_DEVICES). Use float32 weights.

    CUDA_VISIBLE_DEVICES=3 python3 party_v4_train.py \
        --train train_v4.jsonl --silver silver_holdout.jsonl --frozen frozen188.jsonl \
        --base intfloat/multilingual-e5-small --out /home/s0r0qw6/party_v4/out_e5s
"""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "CJP", "None",
]
LABEL2ID = {p: i for i, p in enumerate(LABELS)}
ID2LABEL = {i: p for p, i in LABEL2ID.items()}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def expand_classifier(model):
    old_head = model.classifier
    old_labels = {int(index): label for index, label in model.config.id2label.items()}
    if not isinstance(old_head, torch.nn.Linear):
        raise ValueError("Head expansion requires a linear classifier")
    if len(set(old_labels.values())) != len(old_labels) or not set(old_labels.values()) <= set(LABELS):
        raise ValueError("Source checkpoint must have unique, canonical party labels")
    new_head = torch.nn.Linear(old_head.in_features, len(LABELS), bias=old_head.bias is not None)
    new_head.to(device=old_head.weight.device, dtype=old_head.weight.dtype)
    with torch.no_grad():
        new_head.weight.normal_(mean=0.0, std=model.config.initializer_range)
        if new_head.bias is not None:
            new_head.bias.zero_()
        for old_index, label in old_labels.items():
            new_index = LABEL2ID[label]
            new_head.weight[new_index].copy_(old_head.weight[old_index])
            if old_head.bias is not None:
                new_head.bias[new_index].copy_(old_head.bias[old_index])
    model.classifier = new_head
    model.num_labels = len(LABELS)
    model.config.num_labels = len(LABELS)
    model.config.id2label = dict(ID2LABEL)
    model.config.label2id = dict(LABEL2ID)
    print(f"preserved {len(old_labels)} classifier rows; added {set(LABELS) - set(old_labels.values())}", flush=True)
    return model


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
        preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    model.train()
    return preds


def score(rows, preds):
    gold = [LABEL2ID[r["party"]] for r in rows]
    present = sorted(set(gold) | set(preds))
    return {
        "accuracy": round(accuracy_score(gold, preds), 4),
        "macro_f1": round(f1_score(gold, preds, average="macro", labels=present, zero_division=0), 4),
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--silver", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--class_weight", choices=["none", "sqrt"], default="sqrt")
    ap.add_argument("--ignore_mismatched", action="store_true",
                    help="Re-initialise the classifier head when the base checkpoint has a different label count.")
    ap.add_argument("--expand_head", action="store_true",
                    help="Preserve existing classifier rows by label and initialise only new labels.")
    args = ap.parse_args()
    if args.expand_head and args.ignore_mismatched:
        ap.error("--expand_head and --ignore_mismatched are mutually exclusive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train_rows = [r for r in load(args.train) if r["party"] in LABEL2ID]
    silver_rows = [r for r in load(args.silver) if r["party"] in LABEL2ID]
    frozen_rows = [r for r in load(args.frozen) if r["party"] in LABEL2ID]
    print(f"train={len(train_rows)} silver={len(silver_rows)} frozen={len(frozen_rows)}", flush=True)

    counts = collections.Counter(r["party"] for r in train_rows)
    weight = None
    if args.class_weight == "sqrt":
        w = [(max(counts.get(lbl, 1), 1)) ** -0.5 for lbl in LABELS]
        mean = sum(w) / len(w)
        weight = torch.tensor([x / mean for x in w], dtype=torch.float32, device=device)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if args.expand_head:
        model = expand_classifier(AutoModelForSequenceClassification.from_pretrained(
            args.base, dtype=torch.float32,
        )).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            dtype=torch.float32,
            ignore_mismatched_sizes=args.ignore_mismatched,
        ).to(device)
    assert next(model.parameters()).dtype == torch.float32
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params / 1e6:.1f}M", flush=True)

    collate = make_collate(tokenizer, args.max_len)
    fit_loader = DataLoader(TextSet(train_rows), batch_size=args.batch, shuffle=True, collate_fn=collate)
    silver_loader = DataLoader(TextSet(silver_rows), batch_size=64, shuffle=False, collate_fn=collate)
    frozen_loader = DataLoader(TextSet(frozen_rows), batch_size=64, shuffle=False, collate_fn=collate)

    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(fit_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total), total)

    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    model.train()
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        running, steps = 0.0, 0
        for batch in fit_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            steps += 1
        silver_pred = predict(model, silver_loader, device)
        s = score(silver_rows, silver_pred)
        history.append({"epoch": epoch, "train_loss": round(running / max(steps, 1), 4), "silver": s})
        print(
            f"epoch {epoch} loss {running / max(steps, 1):.4f} | silver acc={s['accuracy']} "
            f"macro_f1={s['macro_f1']} | {time.time() - started:.0f}s",
            flush=True,
        )
        if s["macro_f1"] > best_f1:
            best_f1, best_epoch = s["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"selected epoch {best_epoch} (silver macro_f1={best_f1})", flush=True)

    # Inference cost: batch-1 latency over the frozen texts (measurement only).
    model.eval()
    lat_texts = [r["text"] for r in frozen_rows][:200]
    with torch.no_grad():
        enc_warm = tokenizer(lat_texts[0], return_tensors="pt", truncation=True, max_length=args.max_len)
        model(**{k: v.to(device) for k, v in enc_warm.items()})
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for t in lat_texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=args.max_len)
            model(**{k: v.to(device) for k, v in enc.items()})
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / max(len(lat_texts), 1) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    # Frozen set: scored once, after selection.
    frozen_pred = predict(model, frozen_loader, device)
    gold = [LABEL2ID[r["party"]] for r in frozen_rows]
    frozen_score = score(frozen_rows, frozen_pred)
    present = sorted(set(gold) | set(frozen_pred))
    report = classification_report(
        gold, frozen_pred, labels=present,
        target_names=[ID2LABEL[i] for i in present], zero_division=0,
    )
    metrics = {
        "base": args.base,
        "params_m": round(n_params / 1e6, 1),
        "selected_epoch": best_epoch,
        "silver_macro_f1_selected": best_f1,
        "history": history,
        "frozen": frozen_score,
        "baseline_accuracy": 0.5372,
        "baseline_macro_f1": 0.6056,
        "latency_ms_batch1_gpu": round(latency_ms, 2),
        "peak_gpu_mb_train": round(peak_mb, 0) if peak_mb else None,
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "frozen_report.txt").write_text(report, encoding="utf-8")
    with open(out_dir / "frozen_predictions.jsonl", "w", encoding="utf-8") as fh:
        for r, p in zip(frozen_rows, frozen_pred):
            fh.write(json.dumps({"text": r["text"], "gold": r["party"], "pred": ID2LABEL[p]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: metrics[k] for k in metrics if k != "history"}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()

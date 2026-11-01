"""Distillation trainer for the 18-label party-support student (v5).

Loss per row:
    alpha * KL(teacher_soft || student) + (1 - alpha) * CE(original label)
with label smoothing on both targets and an optional sqrt class weight on the
original label. Rows without a usable teacher distribution use a one-hot target
on the original label, so alpha=0 reproduces the hard-label baseline.

Selection uses the silver holdout only (best silver macro-F1 across epochs).
The frozen 188-row set is scored once, after selection, and is never used for any choice.

Runs on the VM (pick the GPU with CUDA_VISIBLE_DEVICES). Use float32 weights.

    CUDA_VISIBLE_DEVICES=3 python3 party_v5_train.py \
        --train train_v5_soft.jsonl --silver silver_holdout.jsonl --frozen frozen188.jsonl \
        --base intfloat/multilingual-e5-small --alpha 0.7 --out /home/s0r0qw6/party_v5/out_e5s_a07
"""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "CJP", "None",
]
LABEL2ID = {p: i for i, p in enumerate(LABELS)}
ID2LABEL = {i: p for p, i in LABEL2ID.items()}
K = len(LABELS)


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
        labels = torch.tensor([LABEL2ID[r["party"]] for r in batch])
        soft = torch.zeros(len(batch), K)
        for i, r in enumerate(batch):
            if r.get("soft_ok") and r.get("soft"):
                soft[i] = torch.tensor(r["soft"], dtype=torch.float32)
            else:
                soft[i, labels[i]] = 1.0
        enc["labels"] = labels
        enc["soft"] = soft
        return enc

    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "soft")}
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


def distill_loss(logits, labels, soft, alpha, ls, sample_w):
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt = soft * (1 - ls) + ls / K
    kl = (torch.special.xlogy(tgt, tgt) - tgt * logp).sum(-1)
    onehot = F.one_hot(labels, K).float()
    ce_tgt = onehot * (1 - ls) + ls / K
    ce = -(ce_tgt * logp).sum(-1)
    per = alpha * kl + (1 - alpha) * ce
    if sample_w is not None:
        w = sample_w[labels]
        return (per * w).sum() / w.sum()
    return per.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--silver", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on teacher KL; 0 = hard labels only")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--class_weight", choices=["none", "sqrt"], default="sqrt")
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
    silver_rows = [r for r in load(args.silver) if r["party"] in LABEL2ID]
    frozen_rows = [r for r in load(args.frozen) if r["party"] in LABEL2ID]
    n_soft = sum(1 for r in train_rows if r.get("soft_ok"))
    agree = [r for r in train_rows if r.get("teacher_party")]
    teacher_agree = sum(1 for r in agree if r["teacher_party"] == r["party"])
    print(
        f"train={len(train_rows)} (teacher soft={n_soft}, teacher agrees with label "
        f"{teacher_agree}/{len(agree)}) silver={len(silver_rows)} frozen={len(frozen_rows)} alpha={args.alpha}",
        flush=True,
    )

    counts = collections.Counter(r["party"] for r in train_rows)
    weight = None
    if args.class_weight == "sqrt":
        w = [(max(counts.get(lbl, 1), 1)) ** -0.5 for lbl in LABELS]
        mean = sum(w) / len(w)
        weight = torch.tensor([x / mean for x in w], dtype=torch.float32, device=device)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=K,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.float32,
    ).to(device)
    assert next(model.parameters()).dtype == torch.float32
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params / 1e6:.1f}M", flush=True)

    collate = make_collate(tokenizer, args.max_len)
    fit_loader = DataLoader(TextSet(train_rows), batch_size=args.batch, shuffle=True, collate_fn=collate)
    silver_loader = DataLoader(TextSet(silver_rows), batch_size=64, shuffle=False, collate_fn=collate)
    frozen_loader = DataLoader(TextSet(frozen_rows), batch_size=64, shuffle=False, collate_fn=collate)

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
            soft = batch.pop("soft").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = distill_loss(logits, labels, soft, args.alpha, args.label_smoothing, weight)
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

    # Inference cost: batch-1 latency over 200 frozen texts (measurement only).
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
        "alpha": args.alpha,
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

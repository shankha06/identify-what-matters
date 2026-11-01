"""Embedding + logistic-regression baseline for the v4 party task.

Frozen embedder (no fine-tuning), then a linear classifier. Selection uses the
silver holdout only (C grid). The frozen set is scored once, after selection.

    CUDA_VISIBLE_DEVICES=3 python3 party_v4_embed_lr.py \
        --train train_v4.jsonl --silver silver_holdout.jsonl --frozen frozen188.jsonl \
        --embedder intfloat/multilingual-e5-small --out /home/s0r0qw6/party_v4/lr_e5s.json
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

LABELS = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "CJP", "None",
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@torch.no_grad()
def embed(texts, tok, model, device, prefix, batch=128):
    out = []
    for i in range(0, len(texts), batch):
        chunk = [prefix + t for t in texts[i:i + batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--silver", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--embedder", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.embedder)
    model = AutoModel.from_pretrained(args.embedder, dtype=torch.float32).to(device).eval()
    # e5 models expect a "query: " prefix for classification-style use.
    prefix = "query: " if "e5" in args.embedder else ""

    train = [r for r in load(args.train) if r["party"] in LABELS]
    silver = [r for r in load(args.silver) if r["party"] in LABELS]
    frozen = [r for r in load(args.frozen) if r["party"] in LABELS]

    t0 = time.time()
    X_tr = embed([r["text"] for r in train], tok, model, device, prefix)
    X_sv = embed([r["text"] for r in silver], tok, model, device, prefix)
    X_fr = embed([r["text"] for r in frozen], tok, model, device, prefix)
    embed_s = time.time() - t0
    y_tr = [r["party"] for r in train]
    y_sv = [r["party"] for r in silver]

    best = None
    grid = {}
    for C in [0.3, 1.0, 3.0, 10.0, 30.0]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        p = clf.predict(X_sv)
        f1 = f1_score(y_sv, p, average="macro", zero_division=0)
        grid[C] = {"silver_macro_f1": round(f1, 4), "silver_acc": round(accuracy_score(y_sv, p), 4)}
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    f1_sel, C_sel, clf = best

    y_fr = [r["party"] for r in frozen]
    p_fr = clf.predict(X_fr)
    result = {
        "embedder": args.embedder,
        "params_m_embedder": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "embed_seconds": round(embed_s, 1),
        "C_grid_silver": grid,
        "selected_C": C_sel,
        "silver_macro_f1_selected": round(f1_sel, 4),
        "frozen": {
            "accuracy": round(accuracy_score(y_fr, p_fr), 4),
            "macro_f1": round(f1_score(y_fr, p_fr, average="macro", zero_division=0), 4),
            "n": len(frozen),
        },
        "baseline_accuracy": 0.5372,
        "baseline_macro_f1": 0.6056,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

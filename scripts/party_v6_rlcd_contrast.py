"""RLCD-style contrast labelling with the Muse teacher on the VM.

Each text gets two teacher queries: a positive prompt (party the author supports)
and a negative prompt (party the author attacks). Both keep the first answer-token
distribution, so contrast scores can be computed offline.

    python3 party_v6_rlcd_contrast.py run IN.jsonl OUT.jsonl
    python3 party_v6_rlcd_contrast.py score OUT.jsonl

run:   IN rows need "text" and "party". Writes both distributions per row.
score: prints accuracy and macro-F1 for the positive prompt alone, the negative
       prompt alone, and the contrast score log(pos) - lam * log(neg).
"""

import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from party_v3_teacher_label import (
    LABELS, MODEL, SYSTEM, URL, final_text, macro_f1, norm_label,
)
from party_v5_soft_label import answer_dist

WORKERS = 64
TOP_K = 20
EPS = 1e-3
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)

NEG_SYSTEM = (
    "You label Indian political social-media text by party criticism.\n"
    "Decide which political party the author attacks, mocks, or criticises.\n"
    "CJP means Cockroach Janta Party, a youth-led anti-establishment movement.\n"
    "Rules:\n"
    "- Label the party the author attacks, mocks, or criticises.\n"
    "- Praise or support for a party is not an attack. If no party is attacked, label None.\n"
    "- If the text is not about party politics, label None.\n"
    "- Choose exactly one label from this list: " + ", ".join(LABELS) + ".\n"
    "Reply with only the label, nothing else."
)


def ask(system, text):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Text:\n" + text},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"reasoning_strength": "low"},
        "logprobs": True,
        "top_logprobs": TOP_K,
    }
    data = json.dumps(body).encode("utf-8")
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                out = json.loads(resp.read())
            choice = out["choices"][0]
            reply = choice["message"]["content"] or ""
            answer = final_text(reply)
            hard = norm_label(answer) if answer is not None else None
            entries = (choice.get("logprobs") or {}).get("content") or []
            return hard, answer_dist(entries, reply, hard)
        except Exception:
            time.sleep(2)
    return None, None


def run(in_path, out_path):
    rows = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()]

    def both(row):
        return ask(SYSTEM, row["text"]), ask(NEG_SYSTEM, row["text"])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(both, rows))

    with open(out_path, "w", encoding="utf-8") as fh:
        for row, ((pos_hard, pos_soft), (neg_hard, neg_soft)) in zip(rows, results):
            out = {
                "text": row["text"],
                "party": row.get("party"),
                "pos_hard": pos_hard,
                "pos_soft": pos_soft,
                "neg_hard": neg_hard,
                "neg_soft": neg_soft,
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"rows={len(rows)} pos_soft={sum(1 for r in results if r[0][1])} "
          f"neg_soft={sum(1 for r in results if r[1][1])}", flush=True)


def argmax_label(dist):
    return LABELS[max(range(len(dist)), key=dist.__getitem__)]


def pos_pred(r):
    return argmax_label(r["pos_soft"]) if r["pos_soft"] else r["pos_hard"]


def neg_pred(r):
    return argmax_label(r["neg_soft"]) if r["neg_soft"] else r["neg_hard"]


def contrast_pred(r, lam):
    if not (r["pos_soft"] and r["neg_soft"]):
        return pos_pred(r)
    scores = [
        math.log(p + EPS) - lam * math.log(n + EPS)
        for p, n in zip(r["pos_soft"], r["neg_soft"])
    ]
    return LABELS[max(range(len(scores)), key=scores.__getitem__)]


def score(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    golds = [r["party"] for r in rows]
    systems = {
        "pos_only": [pos_pred(r) for r in rows],
        "neg_only": [neg_pred(r) for r in rows],
    }
    for lam in LAMBDAS:
        systems[f"contrast_lam{lam}"] = [contrast_pred(r, lam) for r in rows]
    print(f"rows={len(rows)}")
    for name, preds in systems.items():
        correct = sum(1 for g, p in zip(golds, preds) if g == p)
        unparsed = sum(1 for p in preds if p is None)
        print(f"{name:20s} acc={correct / len(rows):.4f} "
              f"macro_f1={macro_f1(golds, preds):.4f} unparsed={unparsed}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        sys.exit(__doc__)

"""Label text with the Muse Glimmer teacher served by vLLM on the VM.

Run on the VM after vLLM is serving on 127.0.0.1:8000 (stdlib only).

    python3 party_v3_teacher_label.py eval  IN.jsonl OUT.jsonl
    python3 party_v3_teacher_label.py label IN.jsonl OUT.jsonl

eval:  IN rows need "text" and "party" (gold). Writes per-row predictions and
       prints accuracy and macro-F1 (same definition as sklearn over the
       union of gold and predicted labels; unparsed replies count as wrong).
label: IN rows need "text". Writes the row with "teacher_party" and "raw".

Label convention (to be confirmed by the user): the party the author takes a
side for. Criticism with no side taken for another party, or text that is not
about party politics, is "None".
"""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "muse-glimmer"
WORKERS = 32

LABELS = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK", "Shiv Sena", "NCP", "RJD",
    "JDU", "AIADMK", "TDP", "YSRCP", "BRS", "BSP", "CPIM", "CJP", "None",
]

SYSTEM = (
    "You label Indian political social-media text by party support.\n"
    "Decide which political party the author supports or takes a side for.\n"
    "CJP means Cockroach Janta Party, a youth-led anti-establishment movement.\n"
    "Rules:\n"
    "- Label the party the author takes a side for (praise, endorsement, defence, "
    "or clear affiliation).\n"
    "- Criticism of a party without taking a side for another party is None.\n"
    "- If the text is not about party politics, or no side is clear, label None.\n"
    "- Choose exactly one label from this list: " + ", ".join(LABELS) + ".\n"
    "Reply with only the label, nothing else."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
# Muse writes its reasoning to a to=self channel and the answer to a to=user
# channel. Special tokens are kept in the request so the channel can be found.
FINAL_MARK = "to=user<|message|>"
SPECIAL_RE = re.compile(r"<\|[a-z_]+\|>")


def final_text(raw):
    if FINAL_MARK not in raw:
        return None
    tail = raw.rsplit(FINAL_MARK, 1)[1]
    return SPECIAL_RE.split(tail)[0]


def norm_label(reply):
    # Strict: the whole cleaned reply must be one label. Free text that merely
    # mentions a party (for example reasoning) is unparsed, not a prediction.
    s = THINK_RE.sub("", reply).strip().strip("\"'*`. ").strip()
    low = s.lower()
    for lab in LABELS:
        if low == lab.lower():
            return lab
    return None


def ask(text):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Text:\n" + text},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"reasoning_strength": "low"},
    }
    data = json.dumps(body).encode("utf-8")
    last_err = ""
    for _ in range(3):
        try:
            req = urllib.request.Request(
                URL, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.loads(resp.read())
            reply = out["choices"][0]["message"]["content"] or ""
            answer = final_text(reply)
            return reply, (norm_label(answer) if answer is not None else None)
        except Exception as exc:  # retry transient failures
            last_err = repr(exc)
            time.sleep(2)
    return "ERROR: " + last_err, None


def macro_f1(golds, preds):
    labs = sorted(set(golds) | {p for p in preds if p})
    scores = []
    for lab in labs:
        tp = sum(1 for g, p in zip(golds, preds) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("eval", "label"):
        sys.exit(__doc__)
    mode, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()]

    texts = [r["text"] for r in rows]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(ask, texts))

    with open(out_path, "w", encoding="utf-8") as fh:
        for row, (raw, pred) in zip(rows, results):
            row = dict(row)
            row["raw"] = raw
            if mode == "eval":
                row["pred"] = pred
            else:
                row["teacher_party"] = pred
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if mode == "eval":
        golds = [r["party"] for r in rows]
        preds = [p for _, p in results]
        correct = sum(1 for g, p in zip(golds, preds) if g == p)
        unparsed = sum(1 for p in preds if p is None)
        errors = sum(1 for raw, _ in results if raw.startswith("ERROR"))
        print(f"rows={len(rows)} accuracy={correct / len(rows):.4f} "
              f"macro_f1={macro_f1(golds, preds):.4f} "
              f"unparsed={unparsed} request_errors={errors}")
    else:
        unparsed = sum(1 for _, p in results if p is None)
        print(f"labeled={len(rows)} unparsed={unparsed}")


if __name__ == "__main__":
    main()

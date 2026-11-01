"""Soft teacher labels for distillation: Muse first-answer-token distribution over the 18 labels.

Run on the VM next to party_v3_teacher_label.py (copy both to one folder):

    python3 party_v5_soft_label.py IN.jsonl OUT.jsonl

Output rows: text, party (original label), teacher_party (strict hard label or None),
soft (list of 18 probabilities in LABELS order, or None), soft_ok.

Approximation: the label is read from the first answer token only. Ambiguous first
tokens (for example B for BJP, BRS, BSP) go to the teacher's strict answer when it
is in the group, so within-group confusion is not captured. Mass outside the 18
labels is dropped and the rest is renormalised.
"""

import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from party_v3_teacher_label import FINAL_MARK, LABELS, MODEL, SYSTEM, URL, final_text, norm_label

WORKERS = 64
TOP_K = 20


def first_token_dist(top, hard):
    """Distribution over LABELS from the first answer token's top-k candidates.

    A candidate token that is a prefix of several labels (for example B for BJP,
    BRS, BSP) is resolved to the teacher's strict answer `hard` when that label is
    in the group. Otherwise its mass is split equally across the group.
    """
    probs = {}
    for cand in top:
        tok = cand["token"].strip()
        if not tok:
            continue
        match = [lab for lab in LABELS if lab.startswith(tok)]
        if not match:
            continue
        p = math.exp(cand["logprob"])
        if len(match) > 1 and hard in match:
            match = [hard]
        for lab in match:
            probs[lab] = probs.get(lab, 0.0) + p / len(match)
    total = sum(probs.values())
    if total <= 0:
        return None
    return [probs.get(lab, 0.0) / total for lab in LABELS]


def answer_dist(token_entries, reply, hard):
    if FINAL_MARK not in reply:
        return None
    cut = reply.rfind(FINAL_MARK) + len(FINAL_MARK)
    pos = 0
    for entry in token_entries:
        if pos >= cut:
            return first_token_dist(entry.get("top_logprobs") or [], hard)
        pos += len(entry["token"])
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
        "logprobs": True,
        "top_logprobs": TOP_K,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            out = json.loads(resp.read())
    except Exception:
        return None, None, None
    choice = out["choices"][0]
    reply = choice["message"]["content"] or ""
    entries = (choice.get("logprobs") or {}).get("content") or []
    answer = final_text(reply)
    hard = norm_label(answer) if answer is not None else None
    soft = answer_dist(entries, reply, hard)
    return hard, soft, reply


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    in_path, out_path = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda r: ask(r["text"]), rows))

    n_soft = n_hard = n_agree = n_err = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for row, (hard, soft, reply) in zip(rows, results):
            if reply is None:
                n_err += 1
            if hard is not None:
                n_hard += 1
            if soft is not None:
                n_soft += 1
                if hard is not None and LABELS[max(range(len(soft)), key=soft.__getitem__)] == hard:
                    n_agree += 1
            out = {
                "text": row["text"],
                "party": row.get("party"),
                "source": row.get("source"),
                "teacher_party": hard,
                "soft": soft,
                "soft_ok": soft is not None,
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        f"rows={len(rows)} request_errors={n_err} hard_parsed={n_hard} soft_parsed={n_soft} "
        f"argmax_soft_equals_hard={n_agree}/{max(n_hard, 1)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""GEPA-style reflective search over the teacher's labelling rules (VM, stdlib only).

The dev set is sampled from train rows and excludes any text in silver or frozen.
The teacher reflects on sampled errors and proposes new rules. A candidate is kept
only if macro-F1 rises by ACCEPT_GAIN and the None false-positive rate stays under FP_CAP.
Silver and frozen are scored once, after the search.

    python3 party_v9_gepa_prompt.py TRAIN.jsonl SILVER.jsonl FROZEN.jsonl OUT_DIR
"""

import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from party_v3_teacher_label import LABELS, MODEL, SYSTEM, URL, final_text, macro_f1, norm_label

WORKERS = 64
ROUNDS = 5
CANDIDATES = 3
MINIBATCH_ERRORS = 25
FP_CAP = 0.10
ACCEPT_GAIN = 0.005
N_PARTY, N_NONE = 120, 80
SEED = 17

HEADER = (
    "You label Indian political social-media text by party support.\n"
    "Decide which political party the author supports or takes a side for.\n"
    "CJP means Cockroach Janta Party, a youth-led anti-establishment movement.\n"
    "Rules:\n"
)
FOOTER = (
    "- Choose exactly one label from this list: " + ", ".join(LABELS) + ".\n"
    "Reply with only the label, nothing else."
)
BASE_RULES = (
    "- Label the party the author takes a side for (praise, endorsement, defence, or clear affiliation).\n"
    "- Criticism of a party without taking a side for another party is None.\n"
    "- If the text is not about party politics, or no side is clear, label None.\n"
)

REFLECT_SYSTEM = (
    "You are improving the labelling rules for a party-support annotator.\n"
    "Below are the current rules and examples the annotator got wrong.\n"
    "Write improved rules in the same bullet style, each line starting with '- '.\n"
    "Do not add a rule that labels plain criticism or unrelated text as a party.\n"
    "Output only the rules, no preface."
)


def chat(system, user, max_tokens):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"reasoning_strength": "low"},
    }
    data = json.dumps(body).encode("utf-8")
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
            reply = out["choices"][0]["message"]["content"] or ""
            return final_text(reply)
        except Exception:
            time.sleep(2)
    return None


def classify(system, text):
    answer = chat(system, "Text:\n" + text, 1024)
    return norm_label(answer) if answer is not None else None


def evaluate(rules, rows):
    system = HEADER + rules + FOOTER
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        preds = list(pool.map(lambda r: classify(system, r["text"]), rows))
    golds = [r["party"] for r in rows]
    none_idx = [i for i, g in enumerate(golds) if g == "None"]
    none_fp = sum(1 for i in none_idx if preds[i] not in (None, "None")) / max(len(none_idx), 1)
    acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(rows)
    return {"f1": macro_f1(golds, preds), "acc": acc, "none_fp": none_fp, "preds": preds, "rules": rules}


def reflect(rules, errors):
    lines = "\n".join(f"- gold={e['party']} predicted={e['pred']} text: {e['text'][:300]}" for e in errors)
    user = "Current rules:\n" + rules + "\nErrors:\n" + lines
    out = chat(REFLECT_SYSTEM, user, 2048)
    if not out:
        return None
    kept = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("- ")]
    if not kept or any(lab in " ".join(kept) for lab in ("Reply with only", "Choose exactly")):
        return None
    return "\n".join(kept) + "\n"


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def build_dev(train_path, excluded):
    rng = random.Random(SEED)
    rows = [r for r in load(train_path) if r["text"] not in excluded]
    party = [r for r in rows if r["party"] != "None"]
    none = [r for r in rows if r["party"] == "None"]
    return rng.sample(party, N_PARTY) + rng.sample(none, N_NONE)


def search(dev, log):
    rng = random.Random(SEED)
    best = evaluate(BASE_RULES, dev)
    log.write(json.dumps({"round": 0, "f1": best["f1"], "none_fp": best["none_fp"], "rules": BASE_RULES}) + "\n")
    print(f"round=0 dev_f1={best['f1']:.4f} dev_acc={best['acc']:.4f} none_fp={best['none_fp']:.4f}", flush=True)
    for rnd in range(1, ROUNDS + 1):
        errors = [
            {"text": r["text"], "party": r["party"], "pred": p}
            for r, p in zip(dev, best["preds"]) if r["party"] != p
        ]
        cands = []
        for _ in range(CANDIDATES):
            sample = rng.sample(errors, min(MINIBATCH_ERRORS, len(errors)))
            new = reflect(best["rules"], sample)
            if new:
                cands.append(new)
        results = [evaluate(c, dev) for c in cands]
        for res in results:
            log.write(json.dumps({"round": rnd, "f1": res["f1"], "none_fp": res["none_fp"],
                                  "rules": res["rules"]}) + "\n")
        ok = [res for res in results if res["none_fp"] <= FP_CAP]
        winner = max(ok, key=lambda res: res["f1"], default=None)
        if winner and winner["f1"] >= best["f1"] + ACCEPT_GAIN:
            best = winner
            print(f"round={rnd} accepted dev_f1={best['f1']:.4f} none_fp={best['none_fp']:.4f}", flush=True)
        else:
            print(f"round={rnd} no accept (candidates={len(results)})", flush=True)
    return best["rules"]


def main(train_path, silver_path, frozen_path, out_dir):
    assert HEADER + BASE_RULES + FOOTER == SYSTEM, "baseline rules do not reproduce SYSTEM"
    os.makedirs(out_dir, exist_ok=True)
    silver = load(silver_path)
    frozen = load(frozen_path)
    excluded = {r["text"] for r in silver} | {r["text"] for r in frozen}
    dev = build_dev(train_path, excluded)
    print(f"dev rows={len(dev)} party={sum(1 for r in dev if r['party'] != 'None')}", flush=True)

    with open(os.path.join(out_dir, "v9_log.jsonl"), "w", encoding="utf-8") as log:
        best_rules = search(dev, log)
    with open(os.path.join(out_dir, "v9_best_rules.txt"), "w", encoding="utf-8") as fh:
        fh.write(best_rules)
    print("best rules written", flush=True)

    for name, rows in (("silver", silver), ("frozen_reporting_only", frozen)):
        base = evaluate(BASE_RULES, rows)
        best = evaluate(best_rules, rows)
        print(f"{name} baseline acc={base['acc']:.4f} f1={base['f1']:.4f} none_fp={base['none_fp']:.4f}")
        print(f"{name} searched acc={best['acc']:.4f} f1={best['f1']:.4f} none_fp={best['none_fp']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

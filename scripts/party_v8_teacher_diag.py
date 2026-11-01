"""Diagnostics for the Muse teacher on the VM. Not a labeller.

    python3 party_v8_teacher_diag.py effort IN.jsonl OUT.jsonl
        greedy label at reasoning_strength low, medium and high
    python3 party_v8_teacher_diag.py sample IN.jsonl OUT.jsonl K TEMP
        K samples per row at temperature TEMP, reasoning low
    python3 party_v8_teacher_diag.py report OUT.jsonl

report prints recall, None false positives, pass@K and majority-vote metrics.
"""

import json
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from party_v3_teacher_label import LABELS, MODEL, SYSTEM, URL, final_text, macro_f1, norm_label

WORKERS = 64
EFFORTS = ("low", "medium", "high")


def call(text, effort, temperature, n):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Text:\n" + text},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
        "n": n,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"reasoning_strength": effort},
    }
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
            labels = []
            for choice in out["choices"]:
                reply = choice["message"]["content"] or ""
                answer = final_text(reply)
                labels.append(norm_label(answer) if answer is not None else None)
            return labels, None
        except Exception as exc:
            last_err = repr(exc)
            time.sleep(2)
    return [], last_err


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def run_effort(in_path, out_path):
    rows = load(in_path)

    def one(row):
        out = {}
        for effort in EFFORTS:
            labels, err = call(row["text"], effort, 0, 1)
            out[effort] = {"label": labels[0] if labels else None, "error": err}
        return out

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        for row, res in zip(rows, results):
            fh.write(json.dumps({"text": row["text"], "party": row.get("party"), "effort": res},
                                ensure_ascii=False) + "\n")
    print(f"rows={len(rows)}", flush=True)


def run_sample(in_path, out_path, k, temp):
    rows = load(in_path)

    def one(row):
        return call(row["text"], "low", temp, k)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        for row, (labels, err) in zip(rows, results):
            fh.write(json.dumps({"text": row["text"], "party": row.get("party"),
                                 "samples": labels, "error": err},
                                ensure_ascii=False) + "\n")
    print(f"rows={len(rows)} k={k} temp={temp}", flush=True)


def report_effort(path):
    rows = load(path)
    golds = [r["party"] for r in rows]
    none_gold = sum(1 for g in golds if g == "None")
    party_gold = len(golds) - none_gold
    for effort in EFFORTS:
        preds = [r["effort"][effort]["label"] for r in rows]
        errs = sum(1 for r in rows if r["effort"][effort]["error"])
        acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(rows)
        recall = sum(1 for g, p in zip(golds, preds) if g != "None" and g == p) / max(party_gold, 1)
        none_fp = sum(1 for g, p in zip(golds, preds) if g == "None" and p not in (None, "None"))
        print(f"effort={effort:<7} errors={errs} acc={acc:.4f} "
              f"macro_f1={macro_f1(golds, preds):.4f} party_recall={recall:.4f} "
              f"gold_None_predicted_party={none_fp}/{none_gold}")


def report_sample(path):
    rows = load(path)
    golds = [r["party"] for r in rows]
    party_idx = [i for i, g in enumerate(golds) if g != "None"]
    none_idx = [i for i, g in enumerate(golds) if g == "None"]
    majority = []
    for r in rows:
        votes = Counter(s for s in r["samples"] if s)
        majority.append(votes.most_common(1)[0][0] if votes else None)
    errs = sum(1 for r in rows if r["error"])
    total = sum(len(r["samples"]) for r in rows)
    none_share = sum(1 for r in rows for s in r["samples"] if s == "None") / max(total, 1)
    unanimous = sum(1 for r in rows if r["samples"] and len(set(r["samples"])) == 1) / len(rows)
    maj_recall = sum(1 for i in party_idx if majority[i] == golds[i]) / max(len(party_idx), 1)
    pass_at_k = sum(1 for i in party_idx if golds[i] in rows[i]["samples"]) / max(len(party_idx), 1)
    headroom = sum(1 for i in party_idx if golds[i] in rows[i]["samples"] and majority[i] != golds[i])
    maj_none_fp = sum(1 for i in none_idx if majority[i] not in (None, "None"))
    print(f"rows={len(rows)} errors={errs} none_share_of_samples={none_share:.4f} unanimous_rows={unanimous:.4f}")
    print(f"majority_acc={sum(1 for g, p in zip(golds, majority) if g == p) / len(rows):.4f} "
          f"majority_party_recall={maj_recall:.4f}")
    print(f"pass_at_k_party_recall={pass_at_k:.4f} "
          f"party_rows_gold_sampled_but_not_majority={headroom}/{len(party_idx)}")
    print(f"gold_None_majority_predicted_party={maj_none_fp}/{len(none_idx)}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "effort":
        run_effort(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 6 and sys.argv[1] == "sample":
        run_sample(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]))
    elif len(sys.argv) == 3 and sys.argv[1] == "report":
        if "effort" in load(sys.argv[2])[0]:
            report_effort(sys.argv[2])
        else:
            report_sample(sys.argv[2])
    else:
        sys.exit(__doc__)

"""Out-of-fold check of reflective rule search on the gold-convention frozen set (VM).

For each of 5 stratified folds of frozen:
  - search rules on the other 4 folds (dev); the teacher reflects on dev errors;
  - accept a candidate only if dev macro-F1 gains, the None false-positive rate stays under FP_CAP,
    and silver macro-F1 does not drop below the current best;
  - score the held-out fold once with the chosen rules.
Out-of-fold predictions are the estimate. Silver is used only as a guard inside the search.

    python3 party_v10_cv_search.py FROZEN.jsonl SILVER.jsonl OUT_DIR
"""

import json
import os
import random
import sys

from party_v3_teacher_label import SYSTEM, macro_f1
from party_v9_gepa_prompt import BASE_RULES, HEADER, FOOTER, evaluate, load, reflect

assert HEADER + BASE_RULES + FOOTER == SYSTEM, "baseline rules do not reproduce SYSTEM"

K_FOLDS = 5
ROUNDS = 3
CANDIDATES = 3
MINIBATCH_ERRORS = 20
FP_CAP = 0.10
ACCEPT_GAIN = 0.01
SILVER_TOL = 0.0
SEED = 17


def make_folds(rows, k=K_FOLDS, seed=SEED):
    rng = random.Random(seed)
    by_label = {}
    for i, r in enumerate(rows):
        by_label.setdefault(r["party"], []).append(i)
    fold = [0] * len(rows)
    for idx in by_label.values():
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % k
    return fold


def search_fold(dev, silver, log):
    rng = random.Random(SEED)
    best = evaluate(BASE_RULES, dev)
    best_sil = evaluate(BASE_RULES, silver)
    log.write(json.dumps({"event": "start", "dev_f1": best["f1"], "silver_f1": best_sil["f1"]}) + "\n")
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
        accepted = []
        for c in cands:
            res = evaluate(c, dev)
            if res["none_fp"] > FP_CAP or res["f1"] < best["f1"] + ACCEPT_GAIN:
                log.write(json.dumps({"round": rnd, "dev_f1": res["f1"], "none_fp": res["none_fp"],
                                      "status": "dev_reject"}) + "\n")
                continue
            sil = evaluate(c, silver)
            if sil["f1"] < best_sil["f1"] - SILVER_TOL:
                log.write(json.dumps({"round": rnd, "dev_f1": res["f1"], "silver_f1": sil["f1"],
                                      "status": "silver_reject"}) + "\n")
                continue
            accepted.append((res, sil))
        if accepted:
            res, sil = max(accepted, key=lambda pair: pair[0]["f1"])
            best, best_sil = res, sil
            log.write(json.dumps({"round": rnd, "status": "accepted", "dev_f1": best["f1"],
                                  "silver_f1": best_sil["f1"], "rules": best["rules"]}) + "\n")
        else:
            log.write(json.dumps({"round": rnd, "status": "none_accepted"}) + "\n")
        log.flush()
    return best["rules"]


def party_recall(golds, preds):
    party = [(g, p) for g, p in zip(golds, preds) if g != "None"]
    return sum(1 for g, p in party if g == p) / max(len(party), 1)


def summarize(name, golds, preds):
    none_idx = [i for i, g in enumerate(golds) if g == "None"]
    fp = sum(1 for i in none_idx if preds[i] not in (None, "None")) / max(len(none_idx), 1)
    acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(golds)
    print(f"{name}: acc={acc:.4f} f1={macro_f1(golds, preds):.4f} "
          f"party_recall={party_recall(golds, preds):.4f} none_fp={fp:.4f}", flush=True)


def main(frozen_path, silver_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frozen = load(frozen_path)
    silver = load(silver_path)
    fold = make_folds(frozen)
    golds = [r["party"] for r in frozen]

    oof = [None] * len(frozen)
    fold_rules = {}
    with open(os.path.join(out_dir, "v10_log.jsonl"), "w", encoding="utf-8") as log:
        for k in range(K_FOLDS):
            dev = [r for r, f in zip(frozen, fold) if f != k]
            held_idx = [i for i, f in enumerate(fold) if f == k]
            held = [frozen[i] for i in held_idx]
            log.write(json.dumps({"fold": k, "dev_n": len(dev), "held_n": len(held)}) + "\n")
            rules = search_fold(dev, silver, log)
            fold_rules[k] = rules
            res = evaluate(rules, held)
            for j, i in enumerate(held_idx):
                oof[i] = res["preds"][j]
            print(f"fold {k} done (held={len(held)})", flush=True)

    base = evaluate(BASE_RULES, frozen)
    summarize("frozen baseline (all rows)", golds, base["preds"])
    summarize("frozen searched out-of-fold", golds, oof)
    with open(os.path.join(out_dir, "v10_fold_rules.json"), "w", encoding="utf-8") as fh:
        json.dump(fold_rules, fh, ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

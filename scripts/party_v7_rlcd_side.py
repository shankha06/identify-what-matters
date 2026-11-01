"""RLCD with a convention-axis contrast: lenient 'takes a side' versus strict 'explicit side only'.

The positive prompt counts implicit side-taking (sarcasm at rivals, affiliation, slogans).
The negative prompt counts explicit side-taking only. Where both agree (explicit praise),
the contrast cancels, so the signal sits on implicit cases only.

    python3 party_v7_rlcd_side.py run IN.jsonl OUT.jsonl
    python3 party_v7_rlcd_side.py score OUT.jsonl

Reuses ask(), the score table and the output format of party_v6_rlcd_contrast.py.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import party_v6_rlcd_contrast as v6
from party_v3_teacher_label import LABELS

CJP = "CJP means Cockroach Janta Party, a youth-led anti-establishment movement.\n"
LABEL_LINE = "- Choose exactly one label from this list: " + ", ".join(LABELS) + ".\n"
REPLY = "Reply with only the label, nothing else."

POS_SIDE = (
    "You label Indian political social-media text by the party the author takes a side for.\n"
    "A side can be explicit (praise, endorsement, defence) or implicit: sarcasm or mockery "
    "aimed at a rival party, insider or affiliated framing, or a slogan or promise that "
    "favours one party.\n"
    + CJP
    + "Rules:\n"
    "- If the author's stance points to one party, label that party, even when the side is implicit.\n"
    "- If the text is not about party politics, or no side is taken for any party, label None.\n"
    + LABEL_LINE
    + REPLY
)

NEG_STRICT = (
    "You label Indian political social-media text. Decide only whether the author explicitly "
    "takes a party's side.\n"
    + CJP
    + "Rules:\n"
    "- Label a party only when the text explicitly praises, endorses or defends that party.\n"
    "- Mentions of parties, leaders or government without explicit support are None.\n"
    "- Criticism without explicit support for a party is None.\n"
    "- If the text is not about party politics, label None.\n"
    + LABEL_LINE
    + REPLY
)


def run(in_path, out_path):
    rows = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()]

    def both(row):
        return v6.ask(POS_SIDE, row["text"]), v6.ask(NEG_STRICT, row["text"])

    with ThreadPoolExecutor(max_workers=v6.WORKERS) as pool:
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


def report(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    golds = [r["party"] for r in rows]
    v6.score(path)
    party_gold = sum(1 for g in golds if g != "None")
    none_gold = len(golds) - party_gold
    for lam in v6.LAMBDAS:
        preds = [v6.contrast_pred(r, lam) for r in rows]
        recall = sum(1 for g, p in zip(golds, preds) if g != "None" and g == p) / max(party_gold, 1)
        none_fp = sum(1 for g, p in zip(golds, preds) if g == "None" and p != "None")
        print(f"lam={lam:<5} party_recall={recall:.4f} "
              f"gold_None_predicted_party={none_fp}/{none_gold}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        report(sys.argv[2])
    else:
        sys.exit(__doc__)

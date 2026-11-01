"""Generate synthetic CJP (Cockroach Janta Party) posts with the Muse teacher and
keep only posts the teacher also labels CJP.

Run on the VM next to party_v3_teacher_label.py (copy both to one folder):

    python3 party_v4_generate_cjp.py OUT.jsonl

Output rows: text, party="CJP", source="cjp_synth_teacher_checked", teacher_party.
The scenario list is copied from PARTY_SCENARIOS["CJP"] in
src/identify_what_matters/party_lean_generator.py; keep the two in sync.
"""

import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from party_v3_teacher_label import LABELS, SYSTEM, final_text, norm_label, URL, MODEL

SCENARIOS = [
    "Youth-led anti-establishment satire and memes mocking politicians and the Cockroach Janta Party movement",
    "Protest mobilisation against electoral roll revision (SIR) and voter-rights concerns, Bengaluru and Mumbai rallies",
    "Support for CJP's call to hold leaders accountable for unemployment, paper leaks and rising costs",
    "Criticizing mainstream parties (BJP, Congress, AAP) as the same old system; CJP as a fresh youth alternative",
    "Civic participation: sharing CJP events, volunteer sign-ups, and calls to join the movement",
]
STYLES = [
    "tweet under 280 characters",
    "Reddit comment, casual, one or two sentences",
    "Instagram caption with emoji",
    "short Hinglish (Hindi-English mix) post",
    "news-style headline and one line of context",
]
PROMPT = (
    "Write {n} different realistic social-media posts for this scenario: {scenario}.\n"
    "Style: {style}. Vary tone and wording. Each post should sound like a real person.\n"
    "Output one post per line, no numbering, no quotes, no commentary."
)


def request(messages, temperature):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1500,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"reasoning_strength": "low"},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"] or ""


def generate_job(job):
    scenario, style, seed = job
    raw = request(
        [{"role": "user", "content": PROMPT.format(n=10, scenario=scenario, style=style)}],
        temperature=0.9,
    )
    answer = final_text(raw) or ""
    lines = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip().strip('"')
        if len(line) >= 15:
            lines.append(line)
    return lines


def teacher_label(text):
    raw = request(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Text:\n" + text},
        ],
        temperature=0,
    )
    answer = final_text(raw)
    return norm_label(answer) if answer is not None else None


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_path = sys.argv[1]
    random.seed(7)

    jobs = []
    for scenario in SCENARIOS:
        for style in STYLES:
            for rep in range(3):
                jobs.append((scenario, style, rep))

    with ThreadPoolExecutor(max_workers=16) as pool:
        batches = list(pool.map(generate_job, jobs))

    seen, texts = set(), []
    for batch in batches:
        for text in batch:
            key = re.sub(r"\s+", " ", text.lower())
            if key not in seen:
                seen.add(key)
                texts.append(text)
    print(f"generated unique={len(texts)}", flush=True)

    with ThreadPoolExecutor(max_workers=32) as pool:
        labels = list(pool.map(teacher_label, texts))

    kept = 0
    agree = sum(1 for lab in labels if lab == "CJP")
    with open(out_path, "w", encoding="utf-8") as fh:
        for text, lab in zip(texts, labels):
            if lab != "CJP":
                continue
            kept += 1
            row = {"text": text, "party": "CJP", "source": "cjp_synth_teacher_checked", "teacher_party": lab}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"teacher agrees CJP on {agree}/{len(texts)}; kept {kept}", flush=True)
    print("teacher label counts:", {lab: labels.count(lab) for lab in set(labels) if lab in LABELS}, flush=True)


if __name__ == "__main__":
    main()

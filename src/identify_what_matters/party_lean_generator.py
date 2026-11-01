"""Dedicated corpus generator for party_support / political lean classification.

Problem: The existing demographics_trainer.py seed has only ~130 samples with ~20 party-labeled
examples — heavily biased toward 'None', so the model always predicts non-partisan.

This generator produces 600+ realistic Indian social media comments with known party lean,
covering both EXPLICIT signals (vote BJP, Modi zindabad) and IMPLICIT signals (governance
critique framing, policy issue positions, issue salience patterns).

The output JSONL is compatible with demographics_trainer.py's SYNTHETIC_PATH format.

Commands:
  generate   -- Call OpenRouter to generate labelled party-lean samples
  merge      -- Merge generated data with any existing synthetic_dataset.jsonl

Usage:
  uv run python -m identify_what_matters.party_lean_generator generate
  uv run python -m identify_what_matters.party_lean_generator merge
  uv run python -m identify_what_matters.party_lean_generator generate --rounds 6 --workers 8
  uv run python -m identify_what_matters.party_lean_generator generate --dry-run  # print prompts only

After generation + merge:
  uv run python src/identify_what_matters/demographics_trainer.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from identify_what_matters.openrouter import DEFAULT_MODEL, OpenRouterClient
from identify_what_matters.utils import now_iso as now, parse_json_payload as parse_payload, write_json

MODEL = DEFAULT_MODEL

DEMO_DATA_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models"
OUTPUT_DIR = DEMO_DATA_DIR / "party_lean_corpus"
SYNTHETIC_PATH = DEMO_DATA_DIR / "synthetic_dataset.jsonl"

# Party classes. CJP = Cockroach Janta Party (emerging youth-led movement; see docs/MODEL_AUDIT_RCA.md).
PARTIES = ["BJP", "Congress", "AAP", "TMC", "SP", "DMK", "CJP", "None"]

# Per-class generation scenarios designed to produce IMPLICIT signals, not just explicit mentions.
# Each scenario describes context + examples to guide the LLM to produce diverse, realistic samples.
PARTY_SCENARIOS = {
    "BJP": [
        "Celebrating India's economic growth, ISRO success, infrastructure achievements, G20 presidency under current government",
        "Expressing Hindu pride, Ram Mandir construction, Kashi Vishwanath corridor, Kumbh Mela",
        "Praising Double Engine government, Vande Bharat trains, expressways, new airports, Viksit Bharat",
        "Support for surgical strikes, national security measures, strong foreign policy under current PM",
        "Defending the current PM's development agenda, atmanirbhar bharat, digital india, UPI global",
        "Anti-opposition narrative: criticizing congress dynasty, pappu politics, 60 saal loota, UPA scams",
    ],
    "Congress": [
        "Criticizing paper leak scandals, NEET exam fraud, student welfare failures under current government",
        "Advocating for NYAY minimum income guarantee, caste census, social justice, OBC rights",
        "Expressing concerns about inflation, petrol prices, LPG costs, government economic mismanagement",
        "Defending Rahul Gandhi's Bharat Jodo Yatra, constitutional values, save democracy narrative",
        "Criticizing electoral bond scam, crony capitalism, Adani-Modi nexus, corporate favoritism",
        "Supporting minority rights, secular values, anti-CAA, anti-NRC, constitution protection",
    ],
    "AAP": [
        "Praising free electricity, free water, mohalla clinics in Delhi, best education system",
        "Supporting Arvind Kejriwal's governance model, free bus rides for women, education revolution",
        "Criticizing BJP's false promises vs AAP's real delivery on ground in Delhi",
        "Discussing AAP's expansion to Punjab, Gujarat, national politics, alternative to BJP-Congress",
        "Supporting AAP's fight against corruption and old political parties",
    ],
    "TMC": [
        "Expressing pride in Didi's governance, Bengal development, social welfare schemes",
        "Criticizing BJP's interference in Bengal politics, central agency raids on Bengal",
        "Supporting Mamata's fight for federalism and state rights against BJP central control",
        "Regional pride: Bengal culture, language, Durga Puja, Rabindranath Tagore",
        "Supporting TMC's welfare schemes, Lakshmir Bhandar, Swasthya Sathi health scheme",
    ],
    "SP": [
        "Supporting Akhilesh Yadav's vision for UP, PDA politics (backwards, Dalits, minorities)",
        "Criticizing BJP's governance in UP, Yogi Adityanath's failures, law and order",
        "OBC/backward class solidarity, social justice demands, reservation protection",
        "Youth unemployment in UP, lack of government jobs, competitive exam failures",
    ],
    "DMK": [
        "Tamil pride, Dravidian movement, opposing Hindi imposition, protecting Tamil language",
        "Supporting MK Stalin's governance, welfare schemes in Tamil Nadu, Dravidian model",
        "Criticizing BJP's anti-federalism, NEP education policy, NEET in Tamil Nadu",
        "Religious minority support, anti-Hindutva narrative in Tamil Nadu context",
    ],
    "CJP": [
        "Youth-led anti-establishment satire and memes mocking politicians and the Cockroach Janta Party movement",
        "Protest mobilisation against electoral roll revision (SIR) and voter-rights concerns, Bengaluru and Mumbai rallies",
        "Support for CJP's call to hold leaders accountable for unemployment, paper leaks and rising costs",
        "Criticizing mainstream parties (BJP, Congress, AAP) as the same old system; CJP as a fresh youth alternative",
        "Civic participation: sharing CJP events, volunteer sign-ups, and calls to join the movement",
    ],
    "None": [
        "Non-partisan civic complaint: pothole, water supply, pollution, traffic, local issue",
        "Personal issue: job search, family, health, education cost, entertainment, food",
        "Non-political everyday: cricket match reaction, Bollywood movie, celebrity gossip",
        "Fact-based neutral: sharing news, government notification, statistical data",
        "Anti-all-parties: disgusted with all politicians, don't trust any party, aapolitical",
        "Issue-focused but no party: demand for better hospitals, roads, farmer support, generic",
    ],
}

# Age brackets and their distribution per party (to ensure realistic demographic spread)
AGE_DIST = {
    "BJP": ["25-34", "35-54", "55+", "25-34", "35-54"],
    "Congress": ["18-24", "25-34", "35-54", "25-34", "18-24"],
    "AAP": ["18-24", "25-34", "18-24", "25-34"],
    "TMC": ["25-34", "35-54", "55+", "35-54"],
    "SP": ["18-24", "25-34", "18-24", "25-34"],
    "DMK": ["18-24", "25-34", "35-54", "25-34"],
    "CJP": ["18-24", "18-24", "25-34", "25-34"],
    "None": ["18-24", "25-34", "35-54", "18-24", "25-34", "35-54"],
}

MENTAL_STATES = [
    "Civic Anger & Institutional Skepticism",
    "Cultural Pride & National Solidarity",
    "Financial Anxiety & Economic Strain",
    "Ironic Detachment & Memetic Cynicism",
    "Objective Deliberation & Fact-Checking",
    "Social Empathy & Grassroots Care",
]

_lock = threading.Lock()


def _prompt_for_party(party: str, scenario: str, round_idx: int) -> str:
    party_context = {
        "BJP": "supporter of BJP/NDA/PM Modi",
        "Congress": "supporter of Congress/INC/Rahul Gandhi/India Alliance",
        "AAP": "supporter of Aam Aadmi Party/Arvind Kejriwal",
        "TMC": "supporter of Trinamool Congress/Mamata Banerjee",
        "SP": "supporter of Samajwadi Party/Akhilesh Yadav",
        "DMK": "supporter of DMK/MK Stalin/Dravidian politics",
        "CJP": "supporter of the Cockroach Janta Party (CJP), a youth-led anti-establishment movement",
        "None": "non-partisan citizen who avoids party affiliation",
    }
    lean = party_context[party]
    language_hint = ["Hindi/English mix (Hinglish)", "Tamil/English mix", "Bengali/English", "pure English", "Hindi"][round_idx % 5]

    return (
        "You are generating training data for an Indian political lean classifier. "
        "Your task: produce 8 diverse, REALISTIC Indian Instagram or Reddit comments from a " + lean + ".\n\n"
        "Scenario context: " + scenario + "\n\n"
        "Requirements:\n"
        "- Language: " + language_hint + ". Mix styles — short punchy comment, longer opinion, question, sarcastic remark\n"
        "- Each comment is 20-220 characters\n"
        "- CRITICAL: Comments should reflect the party lean IMPLICITLY through framing and issue salience, not just by typing 'vote BJP' explicitly. "
        "Real social media users express party lean through: issue choice, framing of events, emotional tone, what they criticize vs praise.\n"
        "- For 'None' party: comment must be entirely non-partisan — no party name, no political direction\n"
        "- Do NOT use @handles or #hashtags. No URLs. No fabricated specific criminal allegations against named individuals.\n"
        "- Vary: age implication (young slang vs mature writing), gender (mix male/female/neutral voice), emotional state\n\n"
        "Return ONLY a JSON object:\n"
        '{"comments": [\n'
        '  {"text": "...", "age_bracket": "18-24|25-34|35-54|55+", "gender": "male|female|neutral",\n'
        '   "mental_state": "one of: Civic Anger & Institutional Skepticism | Cultural Pride & National Solidarity | '
        'Financial Anxiety & Economic Strain | Ironic Detachment & Memetic Cynicism | '
        'Objective Deliberation & Fact-Checking | Social Empathy & Grassroots Care",\n'
        '   "impatience": "impatient|patient", "irrationality": "irrational|rational",\n'
        '   "party_support": "' + party + '"},\n'
        '  ...\n'
        "]}\n\n"
        "Produce exactly 8 comments."
    )


def _validate_item(item: dict, party: str) -> dict | None:
    text = str(item.get("text", "")).strip()
    if not (15 <= len(text) <= 250):
        return None
    age = item.get("age_bracket", "")
    if age not in ("13-17", "18-24", "25-34", "35-54", "55+"):
        return None
    gender = item.get("gender", "")
    if gender not in ("male", "female", "neutral"):
        gender = "neutral"
    mental = item.get("mental_state", "")
    if mental not in MENTAL_STATES:
        return None
    impatience = item.get("impatience", "")
    if impatience not in ("impatient", "patient"):
        impatience = "patient"
    irrationality = item.get("irrationality", "")
    if irrationality not in ("irrational", "rational"):
        irrationality = "rational"
    # Ensure party_support matches what we asked for
    if item.get("party_support") not in PARTIES:
        return None
    return {
        "text": text,
        "age_bracket": age,
        "gender": gender,
        "mental_state": mental,
        "impatience": impatience,
        "irrationality": irrationality,
        "party_support": party,  # force correct label regardless of LLM drift
    }


def generate(args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient(model=args.model)
    state_path = OUTPUT_DIR / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"calls": 0, "rows": 0, "failures": []}

    def run_scenario(party: str, scenario_idx: int, scenario: str) -> None:
        for round_idx in range(args.rounds):
            out_path = OUTPUT_DIR / f"{party}.s{scenario_idx}.r{round_idx}.jsonl"
            if out_path.exists():
                continue

            prompt = _prompt_for_party(party, scenario, round_idx)
            if args.dry_run:
                print(f"\n=== {party} s{scenario_idx} r{round_idx} ===\n{prompt[:300]}...")
                return

            for attempt in range(3):
                try:
                    response = client.generate(
                        user_prompt=prompt, request_timeout=120,
                        system_instruction="You produce strict JSON datasets. Output only the JSON object."
                    )
                except Exception as e:
                    with _lock:
                        state["failures"].append({"call": f"{party}.s{scenario_idx}.r{round_idx}", "err": str(e)[:60]})
                    continue

                try:
                    payload = parse_payload(response.text)
                    comments = payload.get("comments", [])
                    rows = [_validate_item(c, party) for c in comments]
                    rows = [r for r in rows if r is not None]
                    if len(rows) < 3:
                        raise ValueError(f"only {len(rows)} valid rows")
                except Exception as e:
                    with _lock:
                        state["failures"].append({"call": f"{party}.s{scenario_idx}.r{round_idx}.a{attempt}", "err": str(e)[:60]})
                    continue

                with out_path.open("w") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                with _lock:
                    state["calls"] += 1
                    state["rows"] += len(rows)
                    write_json(state_path, state)
                print(f"[party_lean] {party}.s{scenario_idx}.r{round_idx}: {len(rows)} rows", flush=True)
                break

    tasks = []
    for party in (args.only or PARTIES):
        scenarios = PARTY_SCENARIOS.get(party, [])
        for si, scenario in enumerate(scenarios):
            tasks.append((party, si, scenario))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pool.map(lambda t: run_scenario(*t), tasks)

    write_json(state_path, state)
    print(f"\nDone. Total calls: {state['calls']}, rows: {state['rows']}, failures: {len(state['failures'])}")


def merge(args) -> None:
    """Merge all generated JSONL files into the demographics_trainer synthetic_dataset.jsonl."""
    parts = list(OUTPUT_DIR.glob("*.jsonl"))
    if not parts:
        print("No generated files found. Run 'generate' first.")
        return

    existing = []
    if SYNTHETIC_PATH.exists():
        with SYNTHETIC_PATH.open() as f:
            existing = [line.strip() for line in f if line.strip()]
        print(f"Existing synthetic_dataset.jsonl: {len(existing)} rows")

    seen_hashes = {hashlib.sha256(r.encode()).hexdigest()[:12] for r in existing}
    new_rows = []
    for part in sorted(parts):
        with part.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                h = hashlib.sha256(line.encode()).hexdigest()[:12]
                if h not in seen_hashes:
                    new_rows.append(line)
                    seen_hashes.add(h)

    print(f"New rows to add: {len(new_rows)}")
    all_rows = existing + new_rows
    with SYNTHETIC_PATH.open("w") as f:
        for row in all_rows:
            f.write(row + "\n")
    print(f"Merged → {SYNTHETIC_PATH}: {len(all_rows)} total rows")
    print("Next step: uv run python src/identify_what_matters/demographics_trainer.py")


def main():
    ap = argparse.ArgumentParser(description="Party lean corpus generator for demographics classifier")
    sub = ap.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Call OpenRouter to generate labelled party-lean samples")
    gen.add_argument("--rounds", type=int, default=4, help="Rounds per scenario (default: 4, ~8 comments each)")
    gen.add_argument("--workers", type=int, default=6, help="Concurrent API calls")
    gen.add_argument("--model", default=MODEL)
    gen.add_argument("--only", nargs="+", choices=PARTIES, help="Only generate for specific parties")
    gen.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")

    mrg = sub.add_parser("merge", help="Merge generated JSONL into demographics_trainer synthetic_dataset.jsonl")

    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args)
    elif args.cmd == "merge":
        merge(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

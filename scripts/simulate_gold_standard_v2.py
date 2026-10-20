#!/usr/bin/env python3
"""
Simulate an expanded real-world Gold Standard Dataset (150+ diverse samples)
for rigorous cross-validation across all demographic dimensions:
- mental_state (all 6 classes balanced)
- age_bracket (all 5 classes balanced)
- gender (male, female, neutral)
- party_support (all 17 parties + None)
- real-world messiness: Hinglish, slang, code-switching, civic issues, personal stories, cynicism, optimism, grief.
"""

import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from identify_what_matters.llm import make_backend_client
from identify_what_matters.utils import parse_json_payload

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "artifacts" / "demographics_models" / "real_world_gold_standard_v2.jsonl"
EXISTING_GOLD = Path(__file__).resolve().parents[1] / "artifacts" / "demographics_models" / "gold_standard_benchmark.jsonl"

MENTAL_STATES = [
    "Civic Anger & Institutional Skepticism",
    "Cultural Pride & National Solidarity",
    "Financial Anxiety & Economic Strain",
    "Ironic Detachment & Memetic Cynicism",
    "Objective Deliberation & Fact-Checking",
    "Social Empathy & Grassroots Care"
]

AGE_BRACKETS = ["13-17", "18-24", "25-34", "35-54", "55+"]

SYSTEM_PROMPT = """You are an expert sociologist and computational linguist simulating highly authentic, ground-truth benchmark data of Indian social media comments across Twitter/X, Reddit India, Facebook groups, and YouTube comments.
You output strictly valid JSON."""

PROMPT_CONFIGS = [
    # 1. Social Empathy & Grassroots Care
    {
        "mental_state": "Social Empathy & Grassroots Care",
        "ages": ["18-24", "25-34", "35-54", "55+"],
        "desc": "Deep human compassion, helping community members during crises, organizing blood drives, stray animal care, pooling funds for househelp/maid's child school fees, food distribution to homeless, mental health peer support, heartfelt solidarity.",
        "count": 25
    },
    # 2. Ironic Detachment & Memetic Cynicism
    {
        "mental_state": "Ironic Detachment & Memetic Cynicism",
        "ages": ["13-17", "18-24", "25-34"],
        "desc": "Gen-Z / millennial nihilistic dark humor, corporate rat race fatigue, startup buzzwords satire, Bangalore tech park traffic / water crisis sarcasm, Delhi AQI 999 air purifier memes, 'delulu is the solulu', poking fun at election manifestos and astrologers.",
        "count": 25
    },
    # 3. Financial Anxiety & Economic Strain
    {
        "mental_state": "Financial Anxiety & Economic Strain",
        "ages": ["25-34", "35-54", "55+"],
        "desc": "Middle class squeeze, soaring private school tuition, medical hospital bills wiping out savings, home loan EMI spikes, tech layoffs, rent deposit robbery in Bengaluru/Mumbai, vegetable and milk price doubling, fear of retirement insolvency.",
        "count": 25
    },
    # 4. Objective Deliberation & Fact-Checking
    {
        "mental_state": "Objective Deliberation & Fact-Checking",
        "ages": ["25-34", "35-54", "55+"],
        "desc": "Analytical, data-backed discourse citing official data (RBI reports, CAG audits, NFHS-5, MOSPI, economic survey), examining railway budget outlays vs safety Kavach implementation, calm policy comparison without partisan shouting.",
        "count": 25
    },
    # 5. Cultural Pride & National Solidarity
    {
        "mental_state": "Cultural Pride & National Solidarity",
        "ages": ["18-24", "25-34", "35-54", "55+"],
        "desc": "Celebration of Indian heritage, ISRO space missions, UPI digital infrastructure triumph, Indian sports victories, classical art/festivals revival, emotional patriotic unity, border security respect.",
        "count": 25
    },
    # 6. Civic Anger & Institutional Skepticism
    {
        "mental_state": "Civic Anger & Institutional Skepticism",
        "ages": ["18-24", "25-34", "35-54", "55+"],
        "desc": "Righteous indignation over potholes, bridge collapses, paper leaks (NEET/UGC-NET), police corruption, delayed justice in courts, tax harassment without civic infrastructure, unpunished VIP road rage.",
        "count": 25
    },
    # 7. Teenagers (13-17 cohort focus)
    {
        "mental_state": "Ironic Detachment & Memetic Cynicism",
        "ages": ["13-17"],
        "desc": "High school board exam stress, coaching class pressure (Kota culture), Instagram reel addiction, gaming slang, generational disconnect with boomer parents, subtle existential dread.",
        "count": 20
    },
    # 8. Senior Citizens (55+ cohort focus)
    {
        "mental_state": "Civic Anger & Institutional Skepticism",
        "ages": ["55+"],
        "desc": "Pension delays, EPFO digital portal glitches, nostalgia for honest old days, fear of cyber/digital banking scams targeting seniors, neighborhood parks maintenance, rising cost of medicines.",
        "count": 20
    }
]

TEMPLATE = """Generate {count} distinct, hyper-realistic, raw Indian social media comments for the Mental State: '{mental_state}'.
Context/Themes: {desc}
Allowed Age brackets for this batch: {ages}

Requirements:
- Authentic language: mix of Hinglish (Hindi written in Roman script), natural Indian English, colloquial expressions.
- Authentic length: 15 to 60 words per comment.
- Varied gender: 'male', 'female', 'neutral'
- Impatience: 'impatient', 'neutral', 'patient'
- Irrationality: 'irrational', 'neutral', 'rational'
- Party support: Realistic mix of parties ('BJP', 'Congress', 'AAP', 'TMC', 'DMK', 'SP', 'Shiv Sena', 'TDP', 'None', etc.)

Output strictly a JSON object:
{{
  "samples": [
    {{
      "text": "<comment text>",
      "mental_state": "{mental_state}",
      "age_bracket": "<one of the allowed ages>",
      "gender": "<male|female|neutral>",
      "impatience": "<impatient|neutral|patient>",
      "irrationality": "<irrational|neutral|rational>",
      "party": "<party name or None>"
    }}
  ]
}}
"""

def generate_batch(cfg, batch_num):
    client = make_backend_client(max_tokens=8192)
    prompt = TEMPLATE.format(
        count=cfg["count"],
        mental_state=cfg["mental_state"],
        desc=cfg["desc"],
        ages=json.dumps(cfg["ages"])
    )
    try:
        resp = client.generate(user_prompt=prompt, request_timeout=60, system_instruction=SYSTEM_PROMPT)
        data = parse_json_payload(resp.text)
        samples = data.get("samples", [])
        cleaned = []
        for s in samples:
            txt = str(s.get("text", "")).strip()
            if len(txt) >= 20:
                s["text"] = txt
                cleaned.append(s)
        return cfg["mental_state"], batch_num, cleaned
    except Exception as e:
        print(f"Error in batch {batch_num} for {cfg['mental_state']}: {e}")
        return cfg["mental_state"], batch_num, []

def main():
    print("Loading existing 52-sample gold benchmark...")
    existing = []
    if EXISTING_GOLD.exists():
        with open(EXISTING_GOLD, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line.strip()))
    print(f"Loaded {len(existing)} existing gold samples.")

    print(f"Launching generation for ~190 additional gold-standard samples across {len(PROMPT_CONFIGS)} categories...")
    generated = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(generate_batch, cfg, i) for i, cfg in enumerate(PROMPT_CONFIGS)]
        for fut in as_completed(futures):
            st, idx, batch = fut.result()
            generated.extend(batch)
            print(f"[{st} #{idx}] Got {len(batch)} samples. Total so far: {len(generated)}")

    total_gold = existing + generated
    print(f"\nFinal Gold Standard Benchmark size: {len(total_gold)} samples ({len(existing)} existing + {len(generated)} generated)")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in total_gold:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved expanded gold benchmark to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

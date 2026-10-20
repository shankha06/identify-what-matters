#!/usr/bin/env python3
"""
Extensively Enrich Strategic Intelligence Dataset & Gold Standard Benchmark.
Adds targeted, class-balanced, high-friction, and hard edge cases:
1. Under-represented Trajectory Stages:
   - pre_breakout (early signals, niche forum leaks, hyper-local agitation)
   - settling_quiet (post-flashpoint inquiries, cooling down, procedural aftermath)
2. Hard Ordinal Criticality Distinctions:
   - low vs medium (routine administrative policy vs contentious public friction)
   - high vs critical (widespread debate vs imminent violence / street confrontation)
3. Subtle Astroturf vs Organic Phrasing:
   - covert IT cell templates with slight word variations
   - genuine impassioned citizen vernacular vs hashtag bombing
4. Cross-Axis Disambiguation:
   - youth agitation vs public safety (student safety protests)
   - economic distress vs infrastructure failure (power/water tariff hikes vs blackout collapses)
   - institutional integrity vs communal faultlines (citizenship/electoral roll disputes)
"""

import json
import random
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/Users/s0r0qw6/identify-what-matters-main/src')
sys.path.insert(0, '/Users/s0r0qw6/identify-what-matters-main')
from identify_what_matters.llm import make_backend_client
from identify_what_matters.utils import parse_json_payload

OUTPUT_DIR = Path("data/strategic_intelligence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = OUTPUT_DIR / "strategic_train.jsonl"
GOLD_PATH = OUTPUT_DIR / "strategic_gold_benchmark.jsonl"

TARGETED_GENERATION_TASKS = [
    # --- FOCUS 1: PRE-BREAKOUT & SIMMERING EARLY SIGNALS (Stage under-representation) ---
    {
        "threat_axis": "youth_campus_agitation",
        "scenario": "Undergraduate students in 2 state universities quietly circulating Telegram links about arbitrary 40% examination revaluation fee hike before semester exams; no press coverage yet.",
        "forced_stage": "pre_breakout",
        "forced_crit": "medium",
        "count": 12
    },
    {
        "threat_axis": "infrastructure_governance_failure",
        "scenario": "Residents in newly built peri-urban housing board societies sharing photos of expanding foundation cracks and untreated sewage seepage in basement parking on local X/WhatsApp groups.",
        "forced_stage": "pre_breakout",
        "forced_crit": "medium",
        "count": 12
    },
    {
        "threat_axis": "economic_distress",
        "scenario": "Cotton and soybean farmers in northern Karnataka/Vidarbha reporting widespread private trader cartelization paying 25% below MSP at local mandis with subtle calls for tractor rallies.",
        "forced_stage": "pre_breakout",
        "forced_crit": "high",
        "count": 12
    },
    {
        "threat_axis": "disinformation_rumor_panic",
        "scenario": "Closed regional WhatsApp groups spreading whispered audio claims of artificial salt/edible oil shortage starting next week, beginning minor grocery store panic buying.",
        "forced_stage": "pre_breakout",
        "forced_crit": "high",
        "count": 12
    },
    {
        "threat_axis": "institutional_integrity",
        "scenario": "Booth-level volunteer networks in municipal wards quietly cross-referencing voter lists and noticing identical surname clusters marked as shifted without notice.",
        "forced_stage": "pre_breakout",
        "forced_crit": "high",
        "count": 12
    },

    # --- FOCUS 2: SETTLING QUIET (Procedural aftermath, commission reports, cooling down) ---
    {
        "threat_axis": "public_safety_crisis",
        "scenario": "Three months after a major highway protest over highway robbery, police have chargesheeted 4 suspects, fast-track court begins routine hearings, traffic has completely normalized.",
        "forced_stage": "settling_quiet",
        "forced_crit": "low",
        "count": 12
    },
    {
        "threat_axis": "infrastructure_governance_failure",
        "scenario": "Six months after foot overbridge slab collapse, structural inquiry committee submits final 200-page engineering report, compensation disbursed, replacement tender floated.",
        "forced_stage": "settling_quiet",
        "forced_crit": "low",
        "count": 12
    },
    {
        "threat_axis": "youth_campus_agitation",
        "scenario": "University administration agreed to rollback mess fee hike by 15%, student union calls off hunger strike, regular semester classes resume normally across all departments.",
        "forced_stage": "settling_quiet",
        "forced_crit": "low",
        "count": 12
    },
    {
        "threat_axis": "communal_regional_faultline",
        "scenario": "Following festival procession route friction, local peace committee meeting of elders signs joint harmony declaration, police pickets withdraw, market reopens.",
        "forced_stage": "settling_quiet",
        "forced_crit": "low",
        "count": 12
    },

    # --- FOCUS 3: CRITICAL FLASHPOINTS & STREET MOBILIZATION (Imminent violence/destabilization) ---
    {
        "threat_axis": "public_safety_crisis",
        "scenario": "Hospital premises stormed by thousands of angry protesters following alleged assault on junior doctor; emergency services halted statewide, calls for nationwide medical strike.",
        "forced_stage": "breaking_out",
        "forced_crit": "critical",
        "count": 15
    },
    {
        "threat_axis": "institutional_integrity",
        "scenario": "Violent confrontations outside central election commission office; multiple political youth wings marching with barricade breaking, tear gas shells fired by riot police.",
        "forced_stage": "breaking_out",
        "forced_crit": "critical",
        "count": 15
    },
    {
        "threat_axis": "communal_regional_faultline",
        "scenario": "Mob violence and vehicle arson following provocative social media broadcast regarding disputed religious structure; section 144 imposed, internet suspended in 4 districts.",
        "forced_stage": "breaking_out",
        "forced_crit": "critical",
        "count": 15
    },
    {
        "threat_axis": "youth_campus_agitation",
        "scenario": "Massive youth assembly occupying main railway junction over cancelled recruitment exam results; rail tracks blocked, police baton charge, statewide bandh declared.",
        "forced_stage": "peak_saturation",
        "forced_crit": "critical",
        "count": 15
    },

    # --- FOCUS 4: SUBTLE ASTROTURF VS ORGANIC & BOT SPAM DISTINCTION ---
    {
        "threat_axis": "institutional_integrity",
        "scenario": "Coordinated party IT-cell narrative pushing identical phrasing claiming 'Foreign conspiracy to defame democratic institutions' with subtle syntactic alterations across 20 accounts.",
        "forced_stage": "peak_saturation",
        "forced_crit": "medium",
        "forced_auth": "astroturf",
        "count": 14
    },
    {
        "threat_axis": "economic_distress",
        "scenario": "Automated bot spam accounts spamming finance ministry replies with crypto investment links and generic anti-tax hashtag floods: #TaxTerrorism #ModiGovtLoot #SalariedSlavery.",
        "forced_stage": "simmering_entropy",
        "forced_crit": "medium",
        "forced_auth": "bot_spam",
        "count": 14
    },
    {
        "threat_axis": "economic_distress",
        "scenario": "Genuine middle-class IT employee sharing itemized monthly salary slip breakdown showing PF, TDS, professional tax, GST on health insurance vs zero municipal benefits.",
        "forced_stage": "simmering_entropy",
        "forced_crit": "high",
        "forced_auth": "organic",
        "count": 14
    },

    # --- FOCUS 5: ROUTINE CIVIC DELIBERATION (True Negative / Low Criticality Baseline) ---
    {
        "threat_axis": "general_civic_deliberation",
        "scenario": "Detailed discussion on new circular bus routes launched by city transport corporation, comparing ticket fares with metro line connections.",
        "forced_stage": "settling_quiet",
        "forced_crit": "low",
        "forced_auth": "organic",
        "count": 14
    },
    {
        "threat_axis": "general_civic_deliberation",
        "scenario": "Citizen debate on urban tree canopy census guidelines, comparing native neem/peepal planting with ornamental avenue trees along ring road.",
        "forced_stage": "simmering_entropy",
        "forced_crit": "low",
        "forced_auth": "organic",
        "count": 14
    },
    {
        "threat_axis": "general_civic_deliberation",
        "scenario": "Evaluation of rooftop rainwater harvesting compliance rules issued by state pollution control board for apartments exceeding 20 units.",
        "forced_stage": "settling_quiet",
        "forced_crit": "low",
        "forced_auth": "organic",
        "count": 14
    },
    {
        "threat_axis": "general_civic_deliberation",
        "scenario": "Parliamentary committee review of postal department digital banking services and rural post office insurance penetration figures.",
        "forced_stage": "settling_quiet",
        "forced_crit": "low",
        "forced_auth": "organic",
        "count": 14
    }
]

ENRICH_SYSTEM_PROMPT = """You are a senior social data scientist specializing in nuanced Indian civic discourse, behavioral dynamics, and narrative forensics.
You generate realistic, varied texts capturing authentic Hinglish, English, Indian regional contexts, slang, and rhetorical styles.
Your task is to generate precise training and gold benchmark data for a multi-task neural classifier with strict labels.
You must output STRICTLY valid JSON conforming to the requested schema."""

ENRICH_PROMPT_TEMPLATE = """Generate {count} highly realistic social discourse samples (social posts, comments, or news reaction snippets) for:
Threat Axis: '{axis}'
Context: {scenario}

Specifications for these samples:
- Text should sound genuine (20 to 80 words), reflecting real Indian social media discourse (mix of English and natural Hinglish).
- Each sample MUST be annotated with:
  1. "threat_axis": '{axis}'
  2. "criticality": one of ["critical", "high", "medium", "low"] (Target: {target_crit})
  3. "authenticity": one of ["organic", "astroturf", "bot_spam"] (Target: {target_auth})
  4. "trajectory_stage": one of ["pre_breakout", "breaking_out", "peak_saturation", "simmering_entropy", "settling_quiet"] (Target: {target_stage})

Output STRICTLY a JSON object with this format:
{{
  "samples": [
    {{
      "text": "<realistic discourse text with vivid specific context>",
      "threat_axis": "{axis}",
      "criticality": "<critical|high|medium|low>",
      "authenticity": "<organic|astroturf|bot_spam>",
      "trajectory_stage": "<pre_breakout|breaking_out|peak_saturation|simmering_entropy|settling_quiet>"
    }}
  ]
}}"""

def run_enrich_task(task_def, task_id):
    client = make_backend_client(max_tokens=8192)
    axis = task_def["threat_axis"]
    sc = task_def["scenario"]
    cnt = task_def.get("count", 12)
    target_stage = task_def.get("forced_stage", "pre_breakout or breaking_out")
    target_crit = task_def.get("forced_crit", "as appropriate")
    target_auth = task_def.get("forced_auth", "organic or astroturf")

    prompt = ENRICH_PROMPT_TEMPLATE.format(
        count=cnt,
        axis=axis,
        scenario=sc,
        target_stage=target_stage,
        target_crit=target_crit,
        target_auth=target_auth
    )

    try:
        resp = client.generate(user_prompt=prompt, request_timeout=90, system_instruction=ENRICH_SYSTEM_PROMPT)
        data = parse_json_payload(resp.text)
        samples = data.get("samples", [])
        valid = []
        for s in samples:
            txt = str(s.get("text", "")).strip()
            if len(txt) >= 20 and s.get("threat_axis") and s.get("criticality") and s.get("authenticity") and s.get("trajectory_stage"):
                s["text"] = txt
                valid.append(s)
        return task_id, axis, valid
    except Exception as e:
        print(f"Error in task {task_id} ({axis}): {e}")
        return task_id, axis, []

def main():
    print(f"🚀 Running targeted enrichment across {len(TARGETED_GENERATION_TASKS)} specialized scenario clusters...")
    all_new_samples = []

    # Run 2 iterations of each targeted task to obtain rich diversity
    expanded_tasks = []
    for i, t in enumerate(TARGETED_GENERATION_TASKS):
        expanded_tasks.append((t, f"{i}_a"))
        expanded_tasks.append((t, f"{i}_b"))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(run_enrich_task, t, tid) for t, tid in expanded_tasks]
        for fut in as_completed(futures):
            tid, axis, items = fut.result()
            all_new_samples.extend(items)
            print(f"  ✓ [{tid}] ({axis[:20]}): {len(items)} samples collected (Total: {len(all_new_samples)})")

    # Read existing train and gold sets
    existing_train = []
    if TRAIN_PATH.exists():
        with open(TRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): existing_train.append(json.loads(line))

    existing_gold = []
    if GOLD_PATH.exists():
        with open(GOLD_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): existing_gold.append(json.loads(line))

    existing_texts = {s["text"].strip().lower() for s in existing_train + existing_gold}

    new_unique = []
    for s in all_new_samples:
        t = s["text"].strip().lower()
        if t not in existing_texts:
            existing_texts.add(t)
            new_unique.append(s)

    print(f"\nCollected {len(new_unique)} genuinely new unique samples from enrichment!")
    random.shuffle(new_unique)

    # Allocate 40% of newly enriched hard edge-cases to Gold Benchmark to make it substantially larger and tougher!
    # Allocate 60% to Train set.
    gold_extra_count = int(len(new_unique) * 0.40)
    extra_gold = new_unique[:gold_extra_count]
    extra_train = new_unique[gold_extra_count:]

    final_train = existing_train + extra_train
    final_gold = existing_gold + extra_gold

    print(f"Updated Train Set: {len(existing_train)} -> {len(final_train)} samples")
    print(f"Updated Gold Benchmark: {len(existing_gold)} -> {len(final_gold)} samples")

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for s in final_train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        for s in final_gold:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Enrichment complete! Datasets saved to {TRAIN_PATH} and {GOLD_PATH}")

if __name__ == "__main__":
    main()

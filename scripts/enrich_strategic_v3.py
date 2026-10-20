#!/usr/bin/env python3
"""
Targeted Contrastive Generator for Strategic Intelligence v3.
Focused squarely on resolving:
1. Youth Campus Agitation vs Public Safety Crisis vs Institutional Integrity
2. Infrastructure Governance Failure vs Economic Distress vs General Civic Deliberation
3. Communal Regional Faultlines vs Disinformation Rumor Panic vs Public Safety Crisis
4. Hard boundary examples for Trajectory Stages (pre_breakout signals vs simmering entropy)
5. Criticality calibrations (medium policy tension vs high confrontation vs critical violence)
"""

import json
import random
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from identify_what_matters.llm import make_backend_client
from identify_what_matters.utils import parse_json_payload

OUTPUT_DIR = Path("data/strategic_intelligence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = OUTPUT_DIR / "strategic_train.jsonl"
GOLD_PATH = OUTPUT_DIR / "strategic_gold_benchmark.jsonl"

THREAT_AXES = [
    "institutional_integrity",
    "public_safety_crisis",
    "economic_distress",
    "youth_campus_agitation",
    "communal_regional_faultline",
    "infrastructure_governance_failure",
    "disinformation_rumor_panic",
    "general_civic_deliberation"
]

HIGH_DISAMBIGUATION_SCENARIOS = [
    # --- BATCH 1: YOUTH CAMPUS AGITATION (PURE STUDENT / RECRUITMENT / CAMPUS FOCUS) ---
    {
        "axis": "youth_campus_agitation",
        "scenario": "Paper leaks in state SSC / police constable exams, aspirants protesting outside exam board office demanding CBI probe, hall ticket cancellations, age relaxations",
        "nuance": "Must clearly reflect student/aspirant perspective, coaching hub culture (Mukherjee Nagar/Prayagraj/Kota), exam transparency, university library/hostel issues.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["medium", "high", "critical"]
    },
    {
        "axis": "youth_campus_agitation",
        "scenario": "Central university students protesting UGC grant cuts, PhD fellowship disbursement delays, campus curfew timings, student union election bans",
        "nuance": "Campus-centric, student rights, academic autonomy, hostel fee disputes, academic strike.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy", "settling_quiet"],
        "target_crits": ["low", "medium", "high"]
    },
    {
        "axis": "youth_campus_agitation",
        "scenario": "Armed forces short-term recruitment scheme pushback, state railway station sit-ins by rural youth who trained for years for army recruitment rallies",
        "nuance": "Rural youth physical training ground grievances, age limit cutoff disqualification, deep emotional distress over career uncertainty.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation"],
        "target_crits": ["high", "critical"]
    },

    # --- BATCH 2: PUBLIC SAFETY CRISIS (LAW ENFORCEMENT, VIOLENT CRIME, CIVIC DANGER) ---
    {
        "axis": "public_safety_crisis",
        "scenario": "Violent crime wave against women or children in metropolitan transit / neighborhoods; citizen night vigils, public outrage at delayed police response",
        "nuance": "Focus on personal physical vulnerability, street safety, law & order breakdown, emergency policing failure. NOT students or university politics.",
        "target_stages": ["breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["high", "critical"]
    },
    {
        "axis": "public_safety_crisis",
        "scenario": "Highway robbery gangs, extortion syndicates targeting night trucks and suburban commuters, terrorizing residents",
        "nuance": "Physical crime, fear of stepping out at night, lack of highway patrols, urgent community safety alarm.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy"],
        "target_crits": ["medium", "high", "critical"]
    },
    {
        "axis": "public_safety_crisis",
        "scenario": "Toxic spurious liquor (hooch) tragedy leading to multiple hospitalizations and fatalities in rural district, illicit distillery nexus",
        "nuance": "Public health emergency caused by criminal contraband, immediate life-threatening casualty crisis, police dereliction.",
        "target_stages": ["breaking_out", "peak_saturation", "settling_quiet"],
        "target_crits": ["critical", "high"]
    },

    # --- BATCH 3: INFRASTRUCTURE & GOVERNANCE FAILURE (SYSTEMIC PUBLIC UTILITY BREAKDOWN) ---
    {
        "axis": "infrastructure_governance_failure",
        "scenario": "Urban water supply contamination with sewer lines, cholera outbreak in three municipal wards, total civic apathy",
        "nuance": "Municipal utility neglect, water pipeline corrosion, city corporation negligence, physical engineering collapse.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["medium", "high", "critical"]
    },
    {
        "axis": "infrastructure_governance_failure",
        "scenario": "Severe urban flooding caused by storm water drain encroachment and delayed desilting; residential basements submerged, power transformer explosions",
        "nuance": "Urban town planning, stormwater infrastructure collapse, municipal corporation audit failure.",
        "target_stages": ["breaking_out", "peak_saturation", "settling_quiet"],
        "target_crits": ["high", "critical", "medium"]
    },
    {
        "axis": "infrastructure_governance_failure",
        "scenario": "Newly opened expressway bridge structural cracks, concrete spalling within 6 months of inauguration, emergency vehicular diversion",
        "nuance": "Public works department corruption, civil engineering quality, contractor audit failure.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy"],
        "target_crits": ["medium", "high"]
    },

    # --- BATCH 4: COMMUNAL & REGIONAL FAULTLINES (ETHNIC, SECTARIAN, LINGUISTIC) ---
    {
        "axis": "communal_regional_faultline",
        "scenario": "Clash over religious procession routes through sensitive localities; stone pelting, mutual curfew imposition, peace committee mediation",
        "nuance": "Inter-community religious friction, inflammatory sloganeering, delicate social fabric balance, section 144.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation", "settling_quiet"],
        "target_crits": ["medium", "high", "critical"]
    },
    {
        "axis": "communal_regional_faultline",
        "scenario": "Interstate river water sharing dispute triggering border blockades, commercial vehicle vandalism bearing neighboring state plates",
        "nuance": "Regional chauvinism, linguistic/nativist identity mobilization, border transit shutdown.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation"],
        "target_crits": ["medium", "high", "critical"]
    },
    {
        "axis": "communal_regional_faultline",
        "scenario": "Demographic change allegations in tribal belt lands, traditional village council boycotting migrant commercial traders",
        "nuance": "Indigenous land rights vs influx of outsiders, indigenous customary law vs commercial settlement.",
        "target_stages": ["pre_breakout", "simmering_entropy", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },

    # --- BATCH 5: DISINFORMATION, RUMOR & PANIC (DEEPFAKES, PHANTOM SCARE, MANUFACTURED PANIC) ---
    {
        "axis": "disinformation_rumor_panic",
        "scenario": "Viral manipulated audio/video clip claiming school children are being kidnapped by organized van gangs, triggering paranoid lynch mobs",
        "nuance": "Pure false rumor vector, unverified forwarded voice notes, hysteria, child-lifter paranoia spreading across messaging apps.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation"],
        "target_crits": ["high", "critical"]
    },
    {
        "axis": "disinformation_rumor_panic",
        "scenario": "Doctored circular using fake government ministry letterhead announcing nationwide bank locker confiscation or savings tax freeze",
        "nuance": "Financial rumor mongering, fabricated official notification, WhatsApp screenshot contagion causing bank branch panic.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy"],
        "target_crits": ["medium", "high"]
    },

    # --- BATCH 6: INSTITUTIONAL INTEGRITY (CONSTITUTIONAL BODIES, JUDICIARY, REGULATORY CORRUPTION) ---
    {
        "axis": "institutional_integrity",
        "scenario": "Electoral roll mass deletions without verification notices, allegations of selective voter suppression in opposition strongholds",
        "nuance": "Election management, constitutional norms, democratic process legitimacy, judicial writ petitions.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy"],
        "target_crits": ["medium", "high", "critical"]
    },
    {
        "axis": "institutional_integrity",
        "scenario": "Supreme Court collegium appointments impasse, central government returning recommended judicial elevation files repeatedly",
        "nuance": "Judiciary vs Executive separation of powers, constitutional appointment stalemate, rule of law debate.",
        "target_stages": ["simmering_entropy", "settling_quiet", "breaking_out"],
        "target_crits": ["low", "medium", "high"]
    },

    # --- BATCH 7: ECONOMIC DISTRESS (INFLATION, LIVELIHOODS, TRADE CARTELS, JOBS) ---
    {
        "axis": "economic_distress",
        "scenario": "Sharp spike in retail vegetable prices (onion, tomato) combined with retail milk price hike squeezing middle class household budgets",
        "nuance": "Kitchen budget inflation, real wage stagnation, purchasing power erosion, daily consumption stress.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy"],
        "target_crits": ["low", "medium", "high"]
    },
    {
        "axis": "economic_distress",
        "scenario": "Textile and weaving MSME units shutting power looms due to yarn price volatility and delayed input tax credit refunds, laying off daily wage workers",
        "nuance": "Small business solvency, working capital crisis, factory closures, informal labor distress.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },

    # --- BATCH 8: GENERAL CIVIC DELIBERATION (ROUTINE CITIZEN POLICY, CONSTRUCTIVE CIVIC DEBATE) ---
    {
        "axis": "general_civic_deliberation",
        "scenario": "Citizens debating the pros and cons of proposed electric vehicle charging tariff policies and solar rooftop net-metering slabs",
        "nuance": "Constructive civic policy evaluation, polite analytical discourse, technical comparisons, non-crisis governance.",
        "target_stages": ["simmering_entropy", "settling_quiet", "pre_breakout"],
        "target_crits": ["low", "medium"]
    },
    {
        "axis": "general_civic_deliberation",
        "scenario": "Neighborhood residents discussing municipal park timings, walking track maintenance, and pet dog leashing bylaws",
        "nuance": "Community association governance, civil civic discussion, peaceful administrative resolution.",
        "target_stages": ["simmering_entropy", "settling_quiet"],
        "target_crits": ["low", "medium"]
    }
]

SYSTEM_PROMPT = """You are an elite data engineer specializing in strategic governance NLP and multi-task classifiers.
Your job is to generate diverse, hyper-realistic, Indian political/civic posts, commentary, and news snippets in English, Hinglish, and regional vernacular style.

CRITICAL DISAMBIGUATION RULES:
1. "youth_campus_agitation": MUST focus on students, exams (NEET, UPSC, SSC, state PSC), coaching institutes, universities, fee hikes, youth employment quotas.
2. "public_safety_crisis": MUST focus on physical violence, street crime, safety of women/children, emergency policing, law & order breakdown. Do not confuse with student rallies!
3. "infrastructure_governance_failure": Focus on civil engineering, roads, bridges, drainage, water supply, electricity grids, structural collapse.
4. "communal_regional_faultline": Focus on religious, caste, linguistic, or interstate border tensions.
5. "disinformation_rumor_panic": Focus on fake circulars, viral child abduction rumors, deepfakes, panic bank runs.
6. "institutional_integrity": Focus on Election Commission, Judiciary, Parliament, CAG, federal investigation agencies.
7. "economic_distress": Focus on inflation, unemployment, farm crisis, MSME factory closures, tax burdens.
8. "general_civic_deliberation": Focus on calm, constructive civic debates, public policy analysis, municipal rules.

You must output valid JSON only in the following schema:
{
  "samples": [
    {
      "text": "<realistic civic commentary, post, or snippet>",
      "threat_axis": "<exact axis>",
      "criticality": "<low|medium|high|critical>",
      "authenticity": "<organic|astroturf|bot_spam>",
      "trajectory_stage": "<pre_breakout|breaking_out|peak_saturation|simmering_entropy|settling_quiet>"
    }
  ]
}
"""

def generate_cluster(scenario_def, index):
    client = make_backend_client()
    axis = scenario_def["axis"]
    sc = scenario_def["scenario"]
    nu = scenario_def["nuance"]
    stages = scenario_def["target_stages"]
    crits = scenario_def["target_crits"]

    prompt = f"""Generate 14 distinct, realistic training samples for Indian civic discourse.
Threat Axis: "{axis}"
Scenario Context: {sc}
Nuance requirement: {nu}

Target Stage variations to cover: {', '.join(stages)}
Target Criticality variations to cover: {', '.join(crits)}

Ensure a realistic mix of authenticities:
- 8 Organic citizen posts (conversational, emotional, personal experience, Hinglish/English mix)
- 4 Astroturf posts (coordinated party talking points, standardized attack/defense templates)
- 2 Bot spam posts (repetitive hashtag dumps, links, copy-paste flooding)

Make every single text distinct, rich in real-world Indian terminology (e.g., DM office, SP, FIR, dharna, chakka jam, mandis, coaching hubs, patwari, tehsildar, high court petition).
Output ONLY valid JSON.
"""
    try:
        resp = client.generate(user_prompt=prompt, request_timeout=90, system_instruction=SYSTEM_PROMPT)
        data = parse_json_payload(resp.text)
        samples = data.get("samples", [])
        valid = []
        for s in samples:
            txt = str(s.get("text", "")).strip()
            if len(txt) >= 25 and s.get("threat_axis") == axis and s.get("criticality") in ["low", "medium", "high", "critical"] and s.get("authenticity") in ["organic", "astroturf", "bot_spam"] and s.get("trajectory_stage") in ["pre_breakout", "breaking_out", "peak_saturation", "simmering_entropy", "settling_quiet"]:
                s["text"] = txt
                valid.append(s)
        return index, axis, valid
    except Exception as e:
        print(f"Error generating cluster {index} ({axis}): {e}")
        return index, axis, []

def main():
    print(f"🚀 Generating high-precision contrastive samples across {len(HIGH_DISAMBIGUATION_SCENARIOS)} targeted clusters...")
    
    # We will run 2 parallel calls per scenario definition for rich diversity
    tasks = []
    for i, scen in enumerate(HIGH_DISAMBIGUATION_SCENARIOS):
        tasks.append((scen, f"{i}_pass1"))
        tasks.append((scen, f"{i}_pass2"))

    all_generated = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(generate_cluster, t[0], t[1]) for t in tasks]
        for fut in as_completed(futures):
            idx, axis, items = fut.result()
            all_generated.extend(items)
            print(f"  ✓ [{idx}] {axis}: +{len(items)} samples (Running total: {len(all_generated)})")

    # Load existing datasets
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

    seen = {x["text"].strip().lower() for x in existing_train + existing_gold}
    unique_new = []
    for s in all_generated:
        txt = s["text"].strip().lower()
        if txt not in seen:
            seen.add(txt)
            unique_new.append(s)

    print(f"\nGenerated {len(unique_new)} new unique, verified contrastive samples.")
    random.seed(42)
    random.shuffle(unique_new)

    # Split 35% into Gold Benchmark, 65% into Train
    gold_count = int(len(unique_new) * 0.35)
    new_gold = unique_new[:gold_count]
    new_train = unique_new[gold_count:]

    final_train = existing_train + new_train
    final_gold = existing_gold + new_gold

    print(f"Final Train size: {len(existing_train)} -> {len(final_train)}")
    print(f"Final Gold size: {len(existing_gold)} -> {len(final_gold)}")

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for s in final_train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        for s in final_gold:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Enriched datasets written successfully to {TRAIN_PATH} and {GOLD_PATH}!")

if __name__ == "__main__":
    main()

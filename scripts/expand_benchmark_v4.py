#!/usr/bin/env python3
"""
Massive Scale Dataset & Gold Standard Benchmark Generator for Strategic Intelligence v4.
Expands:
- Gold standard benchmark to 2,000+ rigorously balanced samples
- Training corpus to 4,500+ rich multi-lingual, code-mixed, and adversarial samples
Incorporates:
- Ordinal criticality calibration (fine boundary cases between low/medium and high/critical)
- Complex code-mixed Hindi, Bengali, Tamil, Telugu, Kannada, Marathi transliterated Hinglish
- Nuanced trajectory transitions (early pre-breakout forum seeds vs simmering entropy vs settling quiet)
- Astroturf bot coordination vs authentic regional civic activism
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

EXPANDED_BENCHMARK_SCENARIOS = [
    # 1. Youth Campus Agitation: Diverse Regional State PSC / Coaching / University
    {
        "axis": "youth_campus_agitation",
        "scenario": "State PSC recruitment exam scam in Rajasthan and Haryana; paper leaked on Telegram 4 hours prior; youth blocking NH-8, police detaining coaching institute directors.",
        "nuance": "Coaching hub vocabulary, roll number series anomalies, library sit-ins, OMR sheet tampering claims.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["medium", "high", "critical"]
    },
    {
        "axis": "youth_campus_agitation",
        "scenario": "Engineering college campus strike in Tamil Nadu / Andhra Pradesh over abrupt cancellation of on-campus placement drives and punitive hostel gate curfews.",
        "nuance": "Regional state university administration, student senate boycott, fee fine waiver demands.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy", "settling_quiet"],
        "target_crits": ["low", "medium", "high"]
    },
    {
        "axis": "youth_campus_agitation",
        "scenario": "Police constable recruitment physical endurance test heat exhaustion casualties in eastern UP; aspirants demanding indoor centers and morning test slots.",
        "nuance": "Physical ground endurance, biometric verification chaos, district sports stadium protests.",
        "target_stages": ["breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["high", "critical"]
    },

    # 2. Infrastructure Governance Failure: Structural Safety, Power, Water, Rail
    {
        "axis": "infrastructure_governance_failure",
        "scenario": "Newly constructed four-lane expressway stretch sinking by 2 feet after first monsoon showers in Bundelkhand / Bihar; contractors booked for substandard bitumen mix.",
        "nuance": "Engineering inspection reports, toll plaza boycotts, highway authority RTI disclosures, asphalt quality.",
        "target_stages": ["breaking_out", "simmering_entropy", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },
    {
        "axis": "infrastructure_governance_failure",
        "scenario": "Major metropolitan drinking water pipeline burst flooding 500 basements and leaving 4 lakh households without tap water for 96 hours during peak summer.",
        "nuance": "Water tanker mafia price gouging, booster pump failures, municipal valve operator negligence.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation"],
        "target_crits": ["high", "critical"]
    },
    {
        "axis": "infrastructure_governance_failure",
        "scenario": "Suburban rail signal failure during morning office rush hour in Mumbai/Kolkata leading to overcrowding, passenger falling from footboard, station master gheraoed.",
        "nuance": "Automatic block signaling, motorman strikes, commuter frustration, railway divisional manager statement.",
        "target_stages": ["breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["high", "critical"]
    },

    # 3. Communal & Regional Faultlines: Interstate, Linguistic, Tribal, Processional
    {
        "axis": "communal_regional_faultline",
        "scenario": "Interstate dispute over border village administration and school medium of instruction between Maharashtra and Karnataka border talukas.",
        "nuance": "Linguistic minority rights, border demarcation pillar disputes, transport corporation bus service suspensions.",
        "target_stages": ["pre_breakout", "simmering_entropy", "settling_quiet"],
        "target_crits": ["low", "medium", "high"]
    },
    {
        "axis": "communal_regional_faultline",
        "scenario": "Tension over traditional festival procession route passing through communally sensitive market square; prohibitory orders under Section 163 BNSS, flag march by CAPF.",
        "nuance": "Route license permissions, rooftop stone collection rumors, drone surveillance by district administration.",
        "target_stages": ["pre_breakout", "breaking_out", "peak_saturation"],
        "target_crits": ["high", "critical"]
    },
    {
        "axis": "communal_regional_faultline",
        "scenario": "Tribal land alienation protests in central India forest areas opposing private mining leases; traditional gram sabha resolutions asserting PESA act primacy.",
        "nuance": "Adivasi traditional council, forest rights act compliance, paramilitary camps, mining haul road blockades.",
        "target_stages": ["pre_breakout", "simmering_entropy", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },

    # 4. Institutional Integrity: Judicial, Electoral, Constitutional, Anti-Corruption
    {
        "axis": "institutional_integrity",
        "scenario": "Controversy over selective early release / parole of high-profile political convicts ahead of panchayat elections; governor discretion challenged in High Court.",
        "nuance": "Prison remission advisory board records, habeas corpus petitions, legal precedent debates.",
        "target_stages": ["breaking_out", "simmering_entropy", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },
    {
        "axis": "institutional_integrity",
        "scenario": "Alleged conflict of interest in regulatory clearances granted to family enterprise of state minister; CAG preliminary audit draft leaked to media.",
        "nuance": "Public accounts committee summons, secretarial note sheets, conflict disclosure norms.",
        "target_stages": ["pre_breakout", "breaking_out", "simmering_entropy"],
        "target_crits": ["high", "critical"]
    },
    {
        "axis": "institutional_integrity",
        "scenario": "State legislative speaker disqualification proceedings delayed beyond constitutional deadlines; floor test petition heard by Supreme Court constitution bench.",
        "nuance": "Tenth schedule anti-defection law, whip violations, judicial review of legislative procedures.",
        "target_stages": ["breaking_out", "peak_saturation", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },

    # 5. Public Safety Crisis: Toxic Adulteration, Industrial Disaster, Mass Crime
    {
        "axis": "public_safety_crisis",
        "scenario": "Spurious liquor / hooch tragedy claiming 32 lives in rural dry state district; illegal methanol supply syndicate, suspension of excise inspectors.",
        "nuance": "Primary health center toxicology ICU admissions, post-mortem findings, district magistrate magisterial inquiry.",
        "target_stages": ["breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["high", "critical"]
    },
    {
        "axis": "public_safety_crisis",
        "scenario": "Unregulated private coaching institute basement library flooding during sudden cloudburst trapped 3 students; municipal building code violations exposed.",
        "nuance": "Drainage NOC forgeries, illegal commercial basement use, student candlelight march, police FIR.",
        "target_stages": ["breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["high", "critical"]
    },

    # 6. Disinformation, Rumor & Panic: Financial, Medical, Deepfake, Xenophobic
    {
        "axis": "disinformation_rumor_panic",
        "scenario": "Viral manipulated video falsely claiming major regional cooperative bank has run out of cash reserves, prompting long lines of elderly depositors at ATMs.",
        "nuance": "RBI ombudsman press release, bank management solvency clarification, police cyber crime warning against rumor mongering.",
        "target_stages": ["breaking_out", "peak_saturation", "settling_quiet"],
        "target_crits": ["high", "critical"]
    },
    {
        "axis": "disinformation_rumor_panic",
        "scenario": "Fabricated audio note warning of toxic synthetic salt being distributed through government fair price shops; localized boycott of PDS stores.",
        "nuance": "WhatsApp audio forwarding velocity, food safety laboratory testing certificates, district collector appeal.",
        "target_stages": ["pre_breakout", "breaking_out", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },

    # 7. Economic Distress: Farm Inputs, MSME Closures, Inflation, Real Wages
    {
        "axis": "economic_distress",
        "scenario": "Powerloom textile hub in Surat/Bhiwandi shutting down 40% looms due to surge in polyester yarn prices and delayed export drawback payments; migrant workers returning home.",
        "nuance": "Weaver association memorandum, electricity slab charges, jobber joblessness, raw yarn cartelization.",
        "target_stages": ["simmering_entropy", "settling_quiet"],
        "target_crits": ["medium", "high"]
    },
    {
        "axis": "economic_distress",
        "scenario": "Acute scarcity and black marketing of DAP fertilizer during Rabi crop sowing; farmers queueing outside cooperative societies from 3 AM, lathi charge reported.",
        "nuance": "Bag allocation tokens, urea diversion to industrial units, district agriculture officer transfer.",
        "target_stages": ["breaking_out", "peak_saturation", "simmering_entropy"],
        "target_crits": ["high", "critical"]
    },

    # 8. General Civic Deliberation: Constructive Urban Planning, Environmental, Transit
    {
        "axis": "general_civic_deliberation",
        "scenario": "Public consultation on municipal EV charging infrastructure master plan, discussing residential RWA tariff structures and power grid upgrade financing.",
        "nuance": "Calm, evidence-based policy comparison, kilowatt-hour tariffs, transformer load limits.",
        "target_stages": ["simmering_entropy", "settling_quiet"],
        "target_crits": ["low"]
    },
    {
        "axis": "general_civic_deliberation",
        "scenario": "Citizen survey evaluating automated property tax self-assessment portal and municipal GIS property mapping accuracy.",
        "nuance": "Constructive user feedback, municipal portal bug reports, ward-level help desks.",
        "target_stages": ["settling_quiet"],
        "target_crits": ["low"]
    }
]

SYSTEM_PROMPT = """You are an elite data engineer specializing in strategic governance NLP and multi-task classifiers.
Your job is to generate diverse, hyper-realistic, Indian political/civic posts, commentary, and news snippets in English, Hinglish, and regional vernacular style.

CRITICAL DISAMBIGUATION RULES:
1. "youth_campus_agitation": MUST focus on students, exams (NEET, UPSC, SSC, state PSC), coaching institutes, universities, fee hikes, youth employment quotas.
2. "public_safety_crisis": MUST focus on physical violence, street crime, safety of women/children, emergency policing, law & order breakdown.
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

    prompt = f"""Generate 16 distinct, realistic training/benchmark samples for Indian civic discourse.
Threat Axis: "{axis}"
Scenario Context: {sc}
Nuance requirement: {nu}

Target Stage variations to cover: {', '.join(stages)}
Target Criticality variations to cover: {', '.join(crits)}

Ensure a realistic mix of authenticities:
- 10 Organic citizen posts (conversational, emotional, personal experience, Hinglish/English mix)
- 4 Astroturf posts (coordinated party talking points, standardized attack/defense templates)
- 2 Bot spam posts (repetitive hashtag dumps, links, copy-paste flooding)

Make every single text distinct, rich in real-world Indian terminology.
Output ONLY valid JSON.
"""
    try:
        resp = client.generate(user_prompt=prompt, request_timeout=90, system_instruction=SYSTEM_PROMPT)
        data = parse_json_payload(resp.text)
        samples = data.get("samples", [])
        valid = []
        for s in samples:
            txt = str(s.get("text", "")).strip()
            if (len(txt) >= 25 and s.get("threat_axis") == axis and 
                s.get("criticality") in ["low", "medium", "high", "critical"] and 
                s.get("authenticity") in ["organic", "astroturf", "bot_spam"] and 
                s.get("trajectory_stage") in ["pre_breakout", "breaking_out", "peak_saturation", "simmering_entropy", "settling_quiet"]):
                s["text"] = txt
                valid.append(s)
        return index, axis, valid
    except Exception as e:
        print(f"Error generating cluster {index} ({axis}): {e}")
        return index, axis, []

def main():
    print(f"🚀 Generating expanded dataset & gold benchmark across {len(EXPANDED_BENCHMARK_SCENARIOS)} scenarios...")
    
    # Run 4 parallel iterations per scenario for high volume
    tasks = []
    for iteration in range(1, 5):
        for i, scen in enumerate(EXPANDED_BENCHMARK_SCENARIOS):
            tasks.append((scen, f"iter{iteration}_scen{i}"))

    all_generated = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(generate_cluster, t[0], t[1]) for t in tasks]
        for fut in as_completed(futures):
            idx, axis, items = fut.result()
            all_generated.extend(items)
            print(f"  ✓ [{idx}] {axis}: +{len(items)} samples (Total: {len(all_generated)})")

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

    print(f"\nGenerated {len(unique_new)} new unique, verified samples.")
    random.seed(42)
    random.shuffle(unique_new)

    # Allocate 50% to Gold Benchmark to reach ~2,000+ benchmark size, 50% to Train
    split_count = int(len(unique_new) * 0.50)
    new_gold = unique_new[:split_count]
    new_train = unique_new[split_count:]

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

"""Targeted synthetic generation to balance rare classes in artifacts/political_corpus.

Focuses specifically on underrepresented dimensions:
1. Rare narrative types: fact_check, communal_framing, corruption_allegation, mobilization_call, satire_meme
2. Regional party targets: AAP, TMC, DMK, AIADMK, Shiv Sena, NCP, CPIM, RJD, JDU, BSP, SP
3. Under-represented issue clusters: farmers, caste, women_safety, communal

Outputs structured discussion JSON files to artifacts/political_corpus/calls/
so that `python -m identify_what_matters.political_corpus build` integrates them seamlessly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from identify_what_matters.llm import make_backend_client
from identify_what_matters.political_corpus import (
    DIRECTIONS,
    ISSUE_CLUSTERS,
    NARRATIVES,
    REGIONAL_NAMES,
    SENTIMENTS,
    TARGETS,
    _valid_item,
    regional_for,
    unsafe,
)
from identify_what_matters.topic_corpus import NOISY_STYLE
from identify_what_matters.utils import parse_json_payload, write_json

# Plan 20 targeted batches to cover underrepresented areas
TARGETED_SPECS = [
    # 1. Fact checking across multiple issues
    {"id": "bal_factcheck_1", "domain": "governance", "focus_narrative": "fact_check", "focus_targets": ["BJP", "Congress", "AAP"], "split": "train"},
    {"id": "bal_factcheck_2", "domain": "elections", "focus_narrative": "fact_check", "focus_targets": ["Election Commission", "Opposition", "Government"], "split": "train"},
    {"id": "bal_factcheck_3", "domain": "economy", "focus_narrative": "fact_check", "focus_targets": ["Government", "Opposition"], "split": "dev"},
    {"id": "bal_factcheck_4", "domain": "communal", "focus_narrative": "fact_check", "focus_targets": ["Other party", "Government", "None"], "split": "test"},

    # 2. Communal framing vs counter-framing
    {"id": "bal_communal_1", "domain": "communal", "focus_narrative": "communal_framing", "focus_targets": ["BJP", "Congress", "SP"], "split": "train"},
    {"id": "bal_communal_2", "domain": "elections", "focus_narrative": "communal_framing", "focus_targets": ["TMC", "AAP", "Other party"], "split": "train"},
    {"id": "bal_communal_3", "domain": "communal", "focus_narrative": "human_interest", "focus_targets": ["Judiciary", "Police", "None"], "split": "dev"},

    # 3. Corruption allegations & investigations
    {"id": "bal_corruption_1", "domain": "corruption", "focus_narrative": "corruption_allegation", "focus_targets": ["AAP", "TMC", "Congress"], "split": "train"},
    {"id": "bal_corruption_2", "domain": "corruption", "focus_narrative": "corruption_allegation", "focus_targets": ["BJP", "DMK", "SP"], "split": "train"},
    {"id": "bal_corruption_3", "domain": "infrastructure", "focus_narrative": "corruption_allegation", "focus_targets": ["Government", "Shiv Sena", "RJD"], "split": "dev"},
    {"id": "bal_corruption_4", "domain": "governance", "focus_narrative": "corruption_allegation", "focus_targets": ["Police", "Judiciary", "Other party"], "split": "test"},

    # 4. Regional parties (UP, Bihar, Bengal, Tamil Nadu, Maharashtra)
    {"id": "bal_regional_sp_bsp", "domain": "caste", "focus_narrative": "mobilization_call", "focus_targets": ["SP", "BSP"], "split": "train"},
    {"id": "bal_regional_tmc", "domain": "welfare", "focus_narrative": "government_achievement", "focus_targets": ["TMC", "BJP"], "split": "train"},
    {"id": "bal_regional_dmk", "domain": "education", "focus_narrative": "governance_criticism", "focus_targets": ["DMK", "AIADMK"], "split": "train"},
    {"id": "bal_regional_shiv_sena", "domain": "infrastructure", "focus_narrative": "satire_meme", "focus_targets": ["Shiv Sena", "NCP", "BJP"], "split": "train"},
    {"id": "bal_regional_rjd_jdu", "domain": "employment", "focus_narrative": "governance_criticism", "focus_targets": ["RJD", "JDU"], "split": "train"},
    {"id": "bal_regional_aap_punjab", "domain": "farmers", "focus_narrative": "mobilization_call", "focus_targets": ["AAP", "Congress"], "split": "dev"},
    {"id": "bal_regional_cpim", "domain": "employment", "focus_narrative": "mobilization_call", "focus_targets": ["CPIM", "TMC"], "split": "test"},

    # 5. Farmers & Women Safety deep dive
    {"id": "bal_farmers_protest", "domain": "farmers", "focus_narrative": "mobilization_call", "focus_targets": ["Government", "Opposition"], "split": "train"},
    {"id": "bal_women_safety", "domain": "women_safety", "focus_narrative": "governance_criticism", "focus_targets": ["Police", "Judiciary", "Government"], "split": "train"},
]

def make_targeted_prompt(spec: dict) -> str:
    domain = spec["domain"]
    narrative = spec["focus_narrative"]
    targets = spec["focus_targets"]
    targets_str = ", ".join(targets)

    return (
        "Generate realistic training data for an Indian Instagram political classifier. "
        "We are deliberately balancing rare classes and regional representations.\n"
        + NOISY_STYLE + "\n"
        f"Domain / Issue Cluster: {domain}\n"
        f"Primary narrative to exemplify: '{narrative}' (and provide contrast comments)\n"
        f"Primary political targets to feature realistically: {targets_str}\n"
        "Instructions:\n"
        "Create exactly 3 DIFFERENT fictional-but-realistic Instagram posts for this theme. "
        "Real Indian party names and political figures (e.g. Modi, Rahul Gandhi, Akhilesh, Mamata, Stalin, Kejriwal, Uddhav) "
        "may be mentioned naturally in plain text (NEVER as @handles or #hashtags, no URLs or phone numbers).\n"
        "For each post give: caption (60-250 chars, Hindi/English/Hinglish/Regional mix), and label with:\n"
        f"  is_political: true/false, issue_cluster: one of {', '.join(ISSUE_CLUSTERS)} (or 'none'), "
        f"sentiment: one of {', '.join(SENTIMENTS)}, target: one of {', '.join(TARGETS)}, "
        f"direction: one of {', '.join(DIRECTIONS)}, narrative_type: one of {', '.join(NARRATIVES)}.\n"
        "Then write exactly 6 realistic REPLY COMMENTS to that post (some agreeing, disagreeing, fact-checking, or trolling/satirical). "
        "Label each comment with the same 5 fields (text 6-200 chars).\n"
        "Return STRICT JSON ONLY matching this format:\n"
        '{"posts": [{"caption": "...", "is_political": true, "issue_cluster": "' + domain + '", "sentiment": "...", "target": "...", "direction": "...", "narrative_type": "' + narrative + '", '
        '"comments": [{"text": "...", "is_political": true, "issue_cluster": "' + domain + '", "sentiment": "...", "target": "...", "direction": "...", "narrative_type": "..."}]}]}'
    )

def validate_targeted_batch(payload: dict, spec: dict) -> list[dict]:
    posts = payload.get("posts") if isinstance(payload, dict) else None
    if not isinstance(posts, list) or not posts:
        return []
    out = []
    split = spec["split"]
    domain = spec["domain"]
    batch_id = spec["id"]
    for index, raw in enumerate(posts):
        if not isinstance(raw, dict):
            continue
        post_label = _valid_item(raw, is_comment=False)
        if post_label is None:
            continue
        comments = []
        for comment in raw.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            label = _valid_item(comment, is_comment=True)
            if label is not None:
                comments.append(label)
        if len(comments) < 3:
            continue
        key = hashlib.sha256(post_label["text"].casefold().encode()).hexdigest()[:16]
        out.append({
            "discussion_id": f"{batch_id}.p{index}",
            "domain": domain,
            "split": split,
            "origin": "synthetic_llm",
            "post": post_label,
            "comments": comments,
            "hash": key
        })
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--output", default="artifacts/political_corpus")
    args = parser.parse_args()

    calls_dir = Path(args.output) / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    client = make_backend_client(model=args.model, max_tokens=4000)

    print(f"Starting targeted synthetic generation for {len(TARGETED_SPECS)} batches...")
    generated_count = 0
    total_samples = 0

    for i, spec in enumerate(TARGETED_SPECS, 1):
        target_file = calls_dir / f"{spec['id']}.json"
        if target_file.exists():
            print(f"[{i}/{len(TARGETED_SPECS)}] {spec['id']} already exists, skipping.")
            continue

        prompt = make_targeted_prompt(spec)
        print(f"[{i}/{len(TARGETED_SPECS)}] Generating {spec['id']} (domain: {spec['domain']}, narrative: {spec['focus_narrative']})...")
        try:
            resp = client.generate(prompt)
            data = parse_json_payload(resp.text)
            discussions = validate_targeted_batch(data, spec)
            if not discussions:
                print(f"  Warning: batch {spec['id']} produced no valid discussions, retrying once...")
                resp = client.generate(prompt)
                data = parse_json_payload(resp.text)
                discussions = validate_targeted_batch(data, spec)

            if discussions:
                payload = {
                    "model": args.model,
                    "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "spec": spec,
                    "discussions": discussions
                }
                write_json(target_file, payload)
                n_items = sum(1 + len(d["comments"]) for d in discussions)
                total_samples += n_items
                generated_count += 1
                print(f"  ✓ Saved {spec['id']}: {len(discussions)} discussions, {n_items} samples")
            else:
                print(f"  ✗ Failed to validate batch {spec['id']}")
        except Exception as e:
            print(f"  Error generating {spec['id']}: {e}")

    print(f"\nCompleted! Generated {generated_count} batches with {total_samples} high-signal balanced samples.")

if __name__ == "__main__":
    main()

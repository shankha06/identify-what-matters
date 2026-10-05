"""Augment the stance and tone corpus with targeted regional Indic dialect debates (pa, gu, ta, mr, te, kn, bn)
and sarcastic/ironic expressions, then rebuild the gold dataset with balanced class weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from identify_what_matters.llm import LLM
from identify_what_matters.stance_corpus import TONES, context_text

TARGET_REGIONS = [
    {"lang": "pa", "region": "Punjab", "topic": "Punjab tube well electricity subsidy and crop diversification dispute", "script_regex": r"[\u0a00-\u0a7f]"},
    {"lang": "pa", "region": "Punjab", "topic": "Punjab Panchayat election reservation and sarpanch power row", "script_regex": r"[\u0a00-\u0a7f]"},
    {"lang": "gu", "region": "Gujarat", "topic": "Gujarat GIFT city liquor permit relaxation debate", "script_regex": r"[\u0a80-\u0aff]"},
    {"lang": "gu", "region": "Gujarat", "topic": "Gujarat potato and onion MSP procurement price dispute", "script_regex": r"[\u0a80-\u0aff]"},
    {"lang": "ta", "region": "Tamil Nadu", "topic": "Tamil Nadu governor vs elected govt bills deadlock and university VC appointment", "script_regex": r"[\u0b80-\u0bff]"},
    {"lang": "ta", "region": "Tamil Nadu", "topic": "Chennai metro phase 2 central funding allocation controversy", "script_regex": r"[\u0b80-\u0bff]"},
    {"lang": "mr", "region": "Maharashtra", "topic": "Maratha reservation quota vs OBC ceiling in local elections", "script_regex": r"[\u0900-\u097f]"},
    {"lang": "te", "region": "Telangana/AP", "topic": "Hyderabad real estate HYDRA demolition drive controversy", "script_regex": r"[\u0c00-\u0c7f]"},
    {"lang": "kn", "region": "Karnataka", "topic": "Bengaluru water tanker pricing regulation and Cauvery water sharing", "script_regex": r"[\u0c80-\u0cff]"},
    {"lang": "bn", "region": "West Bengal", "topic": "Kolkata civic infrastructure and primary teacher recruitment scam trial", "script_regex": r"[\u0980-\u09ff]"},
]


def generate_targeted_regional(llm: LLM, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    discussions = []
    
    for i, target in enumerate(TARGET_REGIONS):
        checkpoint = output_dir / f"regional_aug_{target['lang']}_{i}.json"
        if checkpoint.exists():
            try:
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
                discussions.append(data)
                continue
            except Exception:
                pass
        
        system = "You are an expert dataset creator for Indian social media debates in strict JSON format."
        prompt = f"""Generate exactly 1 realistic Indian Instagram debate about: {target['topic']} ({target['region']}).
Requirements:
1. Caption: 50-250 characters, capturing an Indian news/meme post.
2. Question: 1 neutral sentence describing the contested debate question.
3. Positions: exactly 3 mutually exclusive positions (Position 0, Position 1, Position 2 compromise/third view).
4. Exactly 16 comments:
   - At least 6 comments in native {target['lang']} script.
   - 4 comments in Hinglish (Roman script Hindi/English mix).
   - 4 comments in English.
   - 2 comments in Hindi (Devanagari script).
   - Stance distribution: at least 4 for side 0, at least 4 for side 1, at least 2 for side 2, at least 2 neutral/no side (side: null).
   - Tone distribution: must use ONLY one of: "supportive", "critical", "humorous", "neutral". Label the INTENDED meaning of sarcasm.
   - At least 4 comments MUST be sarcastic/ironic (e.g. praise that is actually mockery, mocking emojis). Set "sarcastic": true.
   - Exactly 1 off-topic comment ("off_topic": true). Others false.

JSON Schema:
{{
  "caption": "...",
  "question": "...",
  "positions": ["...", "...", "..."],
  "comments": [
    {{
      "text": "...",
      "language": "{target['lang']}",
      "side": 0,
      "tone": "critical",
      "sarcastic": false,
      "off_topic": false
    }}
  ]
}}
Return ONLY the JSON object.
"""
        print(f"Generating regional debate for {target['lang']} ({target['region']})...")
        try:
            res = llm.json(prompt, system)
            if not isinstance(res, dict) or "comments" not in res or len(res["comments"]) < 12:
                continue
            
            # Validate and clean
            valid_comments = []
            for c in res["comments"]:
                txt = " ".join(str(c.get("text", "")).split())
                if len(txt) < 5:
                    continue
                lang = c.get("language")
                if lang == target["lang"] and not re.search(target["script_regex"], txt):
                    lang = f"{target['lang']}_latn"
                side = c.get("side")
                if side not in (0, 1, 2, None):
                    side = None
                tone = c.get("tone")
                if tone not in TONES:
                    tone = "neutral"
                valid_comments.append({
                    "text": txt,
                    "language": lang,
                    "side": side,
                    "tone": tone,
                    "sarcastic": bool(c.get("sarcastic")),
                    "off_topic": bool(c.get("off_topic"))
                })
            
            disc = {
                "discussion_id": f"aug_regional_{target['lang']}_{i}",
                "domain": "regional_governance_debates",
                "round": 99,
                "split": "train",
                "origin": "synthetic_regional_aug",
                "caption": res.get("caption", target["topic"]),
                "question": res.get("question", target["topic"]),
                "positions": res.get("positions", ["Side A", "Side B", "Compromise"]),
                "comments": valid_comments
            }
            checkpoint.write_text(json.dumps(disc, ensure_ascii=False, indent=2), encoding="utf-8")
            discussions.append(disc)
        except Exception as e:
            print(f"Error generating {target['lang']}: {e}")
            
    return discussions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stance-corpus", type=Path, default=Path("artifacts/stance_corpus"))
    parser.add_argument("--max-calls", type=int, default=15)
    args = parser.parse_args()

    llm = LLM(max_calls=args.max_calls)
    aug_dir = args.stance_corpus / "regional_aug"
    new_discs = generate_targeted_regional(llm, aug_dir)
    print(f"Successfully generated {len(new_discs)} targeted regional debate discussions.")


if __name__ == "__main__":
    main()

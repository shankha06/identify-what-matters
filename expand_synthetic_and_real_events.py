import json, random, re
from pathlib import Path
from collections import defaultdict

# 1. Load base synthetic events
with open("artifacts/topic_corpus_v2/dataset/events.jsonl") as f:
    events = [json.loads(line) for line in f]
print(f"Loaded {len(events)} existing base events.")

# 2. Load the targeted hard event clusters we created
from generate_hard_event_corpus import DISTINCT_EVENTS

# Convert DISTINCT_EVENTS into the same schema:
# {"event_id": ..., "domain": ..., "split": ..., "texts": [{"style": "caption", "language": "en", "text": ...}, ...]}
new_events = []
for ev in DISTINCT_EVENTS:
    domain = ev["event_id"].split(".")[0]
    t_items = []
    for txt in ev["texts"]:
        lang = "en"
        if re.search(r"[\u0900-\u097f]", txt):
            lang = "hi"
        elif any(w in txt.lower().split() for w in ["ke", "ki", "ka", "par", "se", "ne", "aur", "me", "hai", "hain"]):
            lang = "hinglish"
        t_items.append({"style": "caption", "language": lang, "text": txt})
    
    new_events.append({
        "event_id": ev["event_id"],
        "domain": domain,
        "split": "train",
        "texts": t_items
    })

print(f"Added {len(new_events)} new targeted real-political event clusters.")

# 3. Also mine high-confidence on-topic clusters from recent pipeline runs (real Instagram data!)
runs = [
    '20261003T003208Z', '20261003T011645Z', '20261003T031411Z', '20261003T065657Z',
    '20261003T114148Z', '20261004T131104Z', '20261007T160056Z', '20261008T083208Z', '20261008T175913Z'
]

run_events = 0
for r in runs:
    rep_p = Path(f"output/runs/{r}/report.json")
    ig_p = Path(f"output/runs/{r}/instagram.json")
    if not (rep_p.exists() and ig_p.exists()):
        continue
    rep = json.loads(rep_p.read_text())
    ig_data = json.loads(ig_p.read_text())
    
    for t in rep.get("topics", []):
        brief = t.get("brief")
        if not brief:
            continue
        on_codes = set(brief.get("on_topic_codes", []))
        if len(on_codes) < 2:
            continue
        media = (ig_data.get(t["topic_key"]) or {}).get("media", [])
        on_captions = [m.get("caption", "").strip() for m in media if m.get("code") in on_codes and len(m.get("caption", "").strip()) >= 25]
        headlines = [h.strip() for h in t.get("headlines", []) if len(h.strip()) >= 20]
        
        all_event_texts = [t["title"]] + headlines[:3] + on_captions[:6]
        if len(all_event_texts) >= 3:
            run_events += 1
            t_items = []
            for txt in all_event_texts:
                lang = "en"
                if re.search(r"[\u0900-\u097f]", txt):
                    lang = "hi"
                elif any(w in txt.lower().split() for w in ["ke", "ki", "ka", "par", "se", "ne", "aur", "me", "hai", "hain"]):
                    lang = "hinglish"
                t_items.append({"style": "caption", "language": lang, "text": txt[:300]})
            
            new_events.append({
                "event_id": f"real_run_{r}_{t['topic_id']}",
                "domain": t.get("category", "civic"),
                "split": "train",
                "texts": t_items
            })

print(f"Extracted {run_events} high-fidelity real topic clusters from production runs.")

all_combined_events = events + new_events
print(f"Total unified event corpus: {len(all_combined_events)} event clusters.")

out_dir = Path("artifacts/topic_corpus_v5/dataset")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "events.jsonl", "w") as f:
    for ev in all_combined_events:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

print(f"Successfully wrote combined event corpus to {out_dir / 'events.jsonl'}")

import json
import random
from collections import defaultdict
from pathlib import Path

# Paths
SRC_DATA = Path("artifacts/demographics_models/synthetic_dataset.jsonl")
OUT_BALANCED = Path("artifacts/demographics_models/synthetic_demographics_balanced.jsonl")
OUT_PAIRS = Path("artifacts/demographics_models/demographics_contrastive_pairs.jsonl")

samples = []
with open(SRC_DATA, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                samples.append(json.loads(line.strip()))
            except Exception:
                pass

print(f"Total raw synthetic samples: {len(samples)}")

# Cap overrepresented classes to maintain balance:
# Max 2000 per mental_state to prevent Civic Anger dominance
by_mental = defaultdict(list)
for s in samples:
    m = s.get("mental_state")
    if m:
        by_mental[m].append(s)

balanced_samples = []
random.seed(42)
for m, items in by_mental.items():
    random.shuffle(items)
    # If over 2000, keep 2000
    chosen = items[:2200]
    balanced_samples.extend(chosen)
    print(f"Mental state '{m}': {len(items)} raw -> {len(chosen)} kept")

print(f"Balanced synthetic dataset total: {len(balanced_samples)}")

with open(OUT_BALANCED, "w", encoding="utf-8") as f:
    for s in balanced_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Create contrastive pairs for metric learning / representation fine-tuning
# Format: {"anchor": text_a, "positive": text_b, "negative": text_c, "task": "mental_state"}
pairs = []
for m, items in by_mental.items():
    if len(items) < 2:
        continue
    # Sample up to 1000 pairs per mental state
    other_states = [k for k in by_mental.keys() if k != m]
    num_pairs = min(800, len(items))
    for i in range(num_pairs):
        a = random.choice(items)["text"]
        p = random.choice(items)["text"]
        while p == a and len(items) > 1:
            p = random.choice(items)["text"]
        neg_state = random.choice(other_states)
        n = random.choice(by_mental[neg_state])["text"]
        pairs.append({
            "anchor": a,
            "positive": p,
            "negative": n,
            "task": "mental_state",
            "label": m
        })

# Do the same for age brackets
by_age = defaultdict(list)
for s in balanced_samples:
    a = s.get("age_bracket")
    if a:
        by_age[a].append(s)

for a_group, items in by_age.items():
    if len(items) < 2:
        continue
    other_ages = [k for k in by_age.keys() if k != a_group]
    num_pairs = min(600, len(items))
    for i in range(num_pairs):
        a = random.choice(items)["text"]
        p = random.choice(items)["text"]
        while p == a and len(items) > 1:
            p = random.choice(items)["text"]
        neg_age = random.choice(other_ages)
        n = random.choice(by_age[neg_age])["text"]
        pairs.append({
            "anchor": a,
            "positive": p,
            "negative": n,
            "task": "age_bracket",
            "label": a_group
        })

print(f"Generated {len(pairs)} contrastive triplets across mental_state and age_bracket tasks.")
with open(OUT_PAIRS, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Saved contrastive triplets to: {OUT_PAIRS}")

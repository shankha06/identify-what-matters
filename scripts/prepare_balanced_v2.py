import json
import random
from collections import defaultdict
from pathlib import Path

# Paths
SRC_DATA = Path("artifacts/demographics_models/synthetic_dataset.jsonl")
OUT_BALANCED = Path("artifacts/demographics_models/synthetic_demographics_balanced_v2.jsonl")
OUT_PAIRS = Path("artifacts/demographics_models/demographics_contrastive_pairs_v2.jsonl")

samples = []
with open(SRC_DATA, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                samples.append(json.loads(line.strip()))
            except Exception:
                pass

print(f"Total raw synthetic samples: {len(samples)}")

# Filter out low quality / ambiguous samples:
# Keep samples where text length >= 20 chars
samples = [s for s in samples if len(s.get("text", "").strip()) >= 20]

by_mental = defaultdict(list)
for s in samples:
    m = s.get("mental_state")
    if m:
        by_mental[m].append(s)

# Filter noisy Civic Anger samples (e.g. random chatter that got tagged as Civic Anger)
cleaned_by_mental = {}
for m, items in by_mental.items():
    if m == "Civic Anger & Institutional Skepticism":
        # Remove clear false positives: Ram mandir, saree, movie slaps, cricket victory
        bad_words = ["saree", "movie was pure", "cricket or javelin", "team india"]
        filtered = [s for s in items if not any(w in s["text"].lower() for w in bad_words)]
        cleaned_by_mental[m] = filtered
        print(f"Cleaned Civic Anger: {len(items)} -> {len(filtered)}")
    else:
        cleaned_by_mental[m] = items

# Cap each mental state at 1500 to achieve true representation balance:
# Classes:
# Social Empathy: ~950 (take all)
# Financial Anxiety: ~1600 (take 1400)
# Ironic Detachment: ~1600 (take 1400)
# Objective Deliberation: ~1700 (take 1400)
# Cultural Pride: ~3400 (take 1400)
# Civic Anger: ~13000 (take 1400 high-quality)

random.seed(42)
balanced_samples = []

for m, items in cleaned_by_mental.items():
    random.shuffle(items)
    cap = 1400
    chosen = items[:cap]
    balanced_samples.extend(chosen)
    print(f"Mental state '{m}': kept {len(chosen)} samples")

print(f"\nTotal balanced dataset size: {len(balanced_samples)}")

with open(OUT_BALANCED, "w", encoding="utf-8") as f:
    for s in balanced_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Saved balanced dataset to: {OUT_BALANCED}")

# Generate contrastive triplets (anchor, positive, negative):
# Intra-class representations are pulled together; inter-class representations are pushed apart.
pairs = []
for m, items in cleaned_by_mental.items():
    if len(items) < 2:
        continue
    other_states = [k for k in cleaned_by_mental.keys() if k != m]
    # Sample up to 1000 triplets per mental state
    num_triplets = min(1000, len(items))
    for i in range(num_triplets):
        a = random.choice(items)["text"]
        p = random.choice(items)["text"]
        while p == a and len(items) > 1:
            p = random.choice(items)["text"]
        neg_state = random.choice(other_states)
        n = random.choice(cleaned_by_mental[neg_state])["text"]
        pairs.append({
            "anchor": a,
            "positive": p,
            "negative": n,
            "task": "mental_state",
            "label": m
        })

print(f"Generated {len(pairs)} contrastive triplets for mental state representation learning.")
with open(OUT_PAIRS, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Saved contrastive triplets to: {OUT_PAIRS}")

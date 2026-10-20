import json, random
from pathlib import Path

# Combine artifacts/relevance_real_v5/train_triplets.jsonl with artifacts/topic_corpus_hard_v5/hard_triplets.jsonl
triplets = []
with open("artifacts/relevance_real_v5/train_triplets.jsonl") as f:
    for line in f:
        triplets.append(json.loads(line))

print(f"Base real triplets: {len(triplets)}")

with open("artifacts/topic_corpus_hard_v5/hard_triplets.jsonl") as f:
    for line in f:
        # Boost weight of these critical near-misses
        d = json.loads(line)
        for _ in range(5):
            triplets.append(d)

print(f"Total merged training triplets (with 5x hard near-miss boosting): {len(triplets)}")

out_path = Path("artifacts/relevance_real_v5/train_triplets_v5.jsonl")
with open(out_path, "w") as f:
    for tr in triplets:
        f.write(json.dumps(tr, ensure_ascii=False) + "\n")

print("Saved combined triplets to", out_path)

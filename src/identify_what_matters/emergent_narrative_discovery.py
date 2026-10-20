"""Discovers emergent issue/narrative clusters that ISSUE_CLUSTER_KEYWORDS doesn't cover yet.

Mirrors hiker_client.py's hashtag-discovery pattern (`HikerCollector._proposals()`: find repeated
high-engagement patterns, rank, write a reviewable proposal - never auto-apply) but for SEMANTIC
clusters of political text instead of hashtags. Fixed keyword taxonomies go stale the moment a new
controversy invents its own vocabulary; this is the system noticing "a lot of distinct political text
is clustering together around one theme, and none of our 14 issue clusters match it" on its own,
using the same fine-tuned multilingual encoder emergence.py already relies on for topic clustering.

Output is a proposals file (output/discovery/issue_cluster_candidates.json) for a human to read and,
if it's a real recurring issue, fold into political_classifier.ISSUE_CLUSTER_KEYWORDS or the next
political_corpus.py generation round - never silently added to the live taxonomy.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from identify_what_matters import political_classifier as pc

PROPOSALS_PATH = Path("output/discovery/issue_cluster_candidates.json")
MIN_CLUSTER_SIZE = 3
MIN_TEXT_LEN = 20
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "is", "are", "was", "were", "this", "that",
    "with", "has", "have", "had", "not", "but", "it", "its", "as", "at", "by", "from", "or", "be", "will",
    "hai", "hain", "ka", "ki", "ke", "ko", "ke", "aur", "se", "me", "mein", "ye", "yeh", "ho", "toh",
}


def _keywords(texts: list[str], top_n: int = 6) -> list[str]:
    """Cheap representative-keyword extraction for a discovered cluster - frequency over the cluster's
    own texts, not a trained model (this is diagnostic labelling for the proposal, not a classification)."""
    counts = Counter()
    for text in texts:
        for word in text.lower().split():
            word = "".join(ch for ch in word if ch.isalnum())
            if len(word) >= 4 and word not in STOPWORDS:
                counts[word] += 1
    return [w for w, _ in counts.most_common(top_n)]


def discover_emergent_clusters(items: list[dict], min_cluster_size: int = MIN_CLUSTER_SIZE) -> list[dict]:
    """items: [{"text": str, "engagement": float, "source": str (e.g. topic_key/username)}, ...] - already
    filtered by the caller to political, issue_cluster == 'none' text. Returns ranked emergent clusters."""
    candidates = [item for item in items if len(item.get("text", "")) >= MIN_TEXT_LEN]
    if len(candidates) < min_cluster_size:
        return []

    from identify_what_matters import encoder
    vectors = encoder.embed([item["text"] for item in candidates])
    labels = encoder.cluster(vectors)

    by_label: dict[int, list[dict]] = defaultdict(list)
    for item, label in zip(candidates, labels):
        by_label[label].append(item)

    clusters = []
    for label, members in by_label.items():
        if len(members) < min_cluster_size:
            continue
        texts = [m["text"] for m in members]
        total_engagement = sum(m.get("engagement", 0) or 0 for m in members)
        clusters.append({
            "candidate_cluster_id": f"emergent_{label}",
            "size": len(members),
            "sources": sorted({m.get("source", "") for m in members if m.get("source")})[:10],
            "total_engagement": round(total_engagement, 1),
            "representative_keywords": _keywords(texts),
            "sample_texts": texts[:4],
            "status": "proposal_requires_human_review",
        })
    return sorted(clusters, key=lambda c: -c["total_engagement"])


def scan_topics_for_emergent_clusters(topics: list[dict]) -> list[dict]:
    """Convenience entry point for dashboard/report builders: scans a run's topics for political text
    that the fixed issue-cluster taxonomy didn't match, and discovers clusters among it."""
    items = []
    for topic in topics:
        brief = topic.get("brief") or {}
        candidates = [(topic.get("title", ""), topic.get("topic_key", ""), 0)]
        candidates += [(c.get("text", ""), topic.get("topic_key", ""), c.get("like_count") or 0)
                      for c in brief.get("comment_pool") or []]
        for text, source, engagement in candidates:
            if not text.strip():
                continue
            result = pc.classify(text)
            if result["is_political"]["value"] and result["issue_cluster"]["value"] == "none":
                items.append({"text": text, "engagement": engagement, "source": source})
    return discover_emergent_clusters(items)


def write_proposals(topics: list[dict], output_dir: str | Path = "output/discovery") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = scan_topics_for_emergent_clusters(topics)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scanned_topics": len(topics),
        "emergent_clusters": clusters[:30],
        "policy": "Candidate issue/narrative clusters the fixed taxonomy doesn't match. Review before "
                 "adding to political_classifier.ISSUE_CLUSTER_KEYWORDS or the next political_corpus.py "
                 "generation round - not auto-applied.",
    }
    (output_dir / "issue_cluster_candidates.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

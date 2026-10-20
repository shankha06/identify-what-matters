"""Account-by-topic graph and bot / astroturf summaries for measured topics."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def on_topic_media(topic: dict) -> list[dict]:
    media = topic.get("instagram", {}).get("media", [])
    if topic.get("brief"):
        on_topic = set(topic["brief"].get("on_topic_codes", []))
        return [m for m in media if m.get("code") in on_topic]
    return media


def topic_accounts(topic: dict) -> dict[str, dict]:
    """Account id -> label and post / comment counts for one topic's on-topic posts."""
    accounts: dict[str, dict] = {}

    def add(account_id: str, label: str | None, kind: str) -> None:
        if not account_id:
            return
        entry = accounts.setdefault(account_id, {"label": label or account_id, "posts": 0, "comments": 0})
        entry[kind] += 1
        if label:
            entry["label"] = label

    for media in on_topic_media(topic):
        add(str(media.get("author_id") or ""), media.get("author"), "posts")
        for comment in media.get("comments", []):
            add(str(comment.get("author_id") or ""), comment.get("author"), "comments")
    return accounts


def build_account_graph(topics: list[dict], min_topics: int = 2, min_shared: int = 3,
                        min_jaccard: float = 0.8, min_link_jaccard: float = 0.2, limit: int = 25,
                        max_accounts: int = 300) -> dict:
    """Accounts active in several topics, topics linked by shared commenters, and coordination candidates."""
    membership: dict[str, dict[str, dict]] = defaultdict(dict)
    labels: dict[str, str] = {}
    titles: dict[str, str] = {}
    for index, topic in enumerate(topics):
        topic_id = topic.get("topic_id") or topic.get("topic_key") or f"topic-{index}"
        titles[topic_id] = topic.get("title", topic_id)
        for account, entry in topic_accounts(topic).items():
            membership[account][topic_id] = entry
            labels[account] = entry["label"]

    cross = {account: tids for account, tids in membership.items() if len(tids) >= min_topics}
    rows = sorted(
        ({"account": account, "label": labels[account], "topic_count": len(tids),
          "topics": [titles[t] for t in tids], "posts": sum(e["posts"] for e in tids.values()),
          "comments": sum(e["comments"] for e in tids.values())} for account, tids in cross.items()),
        key=lambda r: (-r["topic_count"], -r["comments"]))

    capped = sorted(cross, key=lambda a: -len(cross[a]))[:max_accounts]
    pairs = []
    for a, b in combinations(capped, 2):
        shared = set(cross[a]) & set(cross[b])
        union = set(cross[a]) | set(cross[b])
        if len(shared) >= min_shared and len(shared) / len(union) >= min_jaccard:
            pairs.append({"accounts": [labels[a], labels[b]], "shared_topics": len(shared),
                          "jaccard": round(len(shared) / len(union), 2),
                          "topics": [titles[t] for t in sorted(shared)]})

    topic_sets: dict[str, set] = defaultdict(set)
    for account, tids in membership.items():
        for tid in tids:
            topic_sets[tid].add(account)
    links = []
    for x, y in combinations(sorted(topic_sets), 2):
        shared = topic_sets[x] & topic_sets[y]
        union = topic_sets[x] | topic_sets[y]
        if len(shared) >= min_shared and len(shared) / len(union) >= min_link_jaccard:
            links.append({"topics": [titles[x], titles[y]], "shared_accounts": len(shared),
                          "jaccard": round(len(shared) / len(union), 2)})

    return {"cross_topic_accounts": len(cross), "accounts": rows[:limit],
            "coordination_pairs": sorted(pairs, key=lambda p: -p["jaccard"])[:limit],
            "topic_links": sorted(links, key=lambda l: -l["jaccard"])[:limit]}


def astroturf_summary(topic: dict, detector=None, min_comments: int = 20, max_comments: int = 300) -> dict | None:
    """Bot and template-coordination check on a topic's on-topic comments (heuristic, indicative only)."""
    comments = [dict(c) for m in on_topic_media(topic) for c in m.get("comments", []) if c.get("text")]
    if len(comments) < min_comments:
        return None
    if detector is None:
        from identify_what_matters.bot_detector import bot_detector as detector
    debate = (topic.get("brief") or {}).get("debate") or {}
    sides = [s for s in debate.get("sides", []) if isinstance(s, dict) and s.get("position")]
    audit = detector.analyze_topic_astroturf(topic.get("title", ""), comments[:max_comments],
                                             debate_sides=sides or None)
    return {"astroturf_index": audit["astroturf_index"], "bot_share_pct": audit["bot_share_pct"],
            "bot_comment_count": audit["bot_comment_count"],
            "total_comments_audited": audit["total_comments_audited"],
            "template_clusters": len(audit["template_clusters"]),
            "coordination_status": audit["coordination_status"], "side_bot_bias": audit["side_bot_bias"],
            "verdict_summary": audit["verdict_summary"]}

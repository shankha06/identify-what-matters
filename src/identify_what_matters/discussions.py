"""Discussion-first discovery: what Indian Instagram is arguing about, independent of news keywords.

Three lanes feed one pool of fresh posts, clustered into discussions with the fine-tuned E5 encoder:
  * panel   - latest posts of a fixed panel of Indian meme/news/celebrity/sports accounts, each scored against
              that account's own baseline (overperformance, comment-to-like "ratio" lift)
  * trends  - X (Twitter) India and Google Trends terms searched directly on Instagram (already 1-3 words)
  * seeds   - recent posts on broad Indian hashtags
Comments on the hottest posts carry the debate signal (comment likes = endorsement, replies = argument).

`collect_instagram` performs every Instagram request and nothing else, persisting a raw capture
that can be replayed deterministically offline for analysis.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from identify_what_matters import encoder
from identify_what_matters.instagram_access import scrub
from identify_what_matters.trends import CATEGORIES, SEED_HASHTAGS, SYSTEM, hours_since, probe

PANEL_PATH = Path(__file__).resolve().parents[2] / "config" / "panel.json"
KINDS = ("news_event", "debate_controversy", "viral_moment", "meme_format", "celebrity_gossip", "sports_moment",
         "announcement_release", "festival_culture", "other")
# Instagram keyword search returns nothing for long phrases; trending phrases are cut to 3 content words.
FILLER = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "case", "news", "latest"}
# Devotional/greeting posts draw floods of ritual replies ("Jai Shani Dev") that look like overperformance but are not discussion.
RITUAL = re.compile(r"दर्शन|आरती|darshan|aarti|bhakti|#jai|jai shri|suprabhat|good morning|सुप्रभात|happy birthday|#bhakti", re.I)


def load_panel(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or PANEL_PATH).read_text())


def select_trend_terms(signals: list[dict], exclude: set[str], limit: int = 20) -> list[dict]:
    """Short trending terms worth searching on Instagram: X trends (current, persistent first), then Google Trends."""
    x = sorted((s for s in signals if s["source"] == "x_trends"),
               key=lambda s: (not s.get("x_latest"), -s.get("x_hours", 0), s.get("x_rank", 99)))
    google = sorted((s for s in signals if s["source"] == "google_trends"), key=lambda s: -(s.get("traffic") or 0))
    terms, seen = [], {e.casefold() for e in exclude}
    for signal in [item for pair in zip(x, google) for item in pair] + x[len(google):] + google[len(x):]:
        words = [w for w in signal["title"].lstrip("#").split() if w.casefold() not in FILLER]
        term = " ".join(words[:3])
        if len(term) < 3 or term.casefold() in seen:
            continue
        seen.add(term.casefold())
        terms.append({"term": term, "source": signal["source"], "x_hours": signal.get("x_hours"),
                      "x_rank": signal.get("x_rank"), "traffic": signal.get("traffic")})
        if len(terms) >= limit:
            break
    return terms


def panel_scores(feeds: dict[str, list[dict]], reference, max_age_hours: float = 48,
                 baselines: dict[str, dict] | None = None) -> list[dict]:
    """Score fresh panel posts against their own account's typical post (CrowdTangle-style overperformance).

    Engagement accumulates over roughly a day, so the account baseline is scaled by min(1, age/24).
    If historical baselines are supplied, they are used directly or blended when feeds have few settled posts."""
    hot = []
    baselines = baselines or {}
    for account, items in feeds.items():
        usable = [m for m in items if m["taken_at"] and not (m["pinned"] or m["paid"] or m["counts_hidden"])
                  and m.get("like_count") is not None]
        if not usable:
            continue
        settled = [m for m in usable if (hours_since(m["taken_at"], reference) or 0) >= 6]
        hist = baselines.get(account)
        if len(settled) >= 3:
            cur_comments = median(m["comment_count"] or 0 for m in settled)
            cur_likes = median(m["like_count"] or 0 for m in settled)
            cur_ratio = median((m["comment_count"] or 0) / (m["like_count"] + 1) for m in settled)
            if hist:
                base_comments = 0.5 * hist["median_comments"] + 0.5 * cur_comments
                base_likes = 0.5 * hist["median_likes"] + 0.5 * cur_likes
                base_ratio = 0.5 * hist["median_ratio"] + 0.5 * cur_ratio
            else:
                base_comments, base_likes, base_ratio = cur_comments, cur_likes, cur_ratio
        elif hist:
            base_comments = hist["median_comments"]
            base_likes = hist["median_likes"]
            base_ratio = hist["median_ratio"]
        elif usable:
            base_comments = median(m["comment_count"] or 0 for m in usable)
            base_likes = median(m["like_count"] or 0 for m in usable)
            base_ratio = median((m["comment_count"] or 0) / (m["like_count"] + 1) for m in usable)
        else:
            continue

        for item in usable:
            age = hours_since(item["taken_at"], reference)
            if age is None or age > max_age_hours:
                continue
            weight = max(0.25, min(1.0, age / 24))
            comments, likes = item["comment_count"] or 0, item["like_count"] or 0
            over_comments = (comments + 5) / (base_comments * weight + 5)
            over_likes = (likes + 20) / (base_likes * weight + 20)
            ratio_lift = (comments / (likes + 1)) / base_ratio if base_ratio > 0 else 1.0
            cap_val = item.get("caption")
            cap_str = cap_val.get("text", "") if isinstance(cap_val, dict) else str(cap_val or "")
            item["panel"] = {"account": account, "group": item.get("panel_group"),
                             "ritual": bool(RITUAL.search(cap_str)),
                             "over_comments": round(over_comments, 2), "over_likes": round(over_likes, 2),
                             "ratio_lift": round(ratio_lift, 2),
                             "heat": round(math.log(over_comments) + 0.5 * math.log(over_likes)
                                           + 0.5 * math.log(max(ratio_lift, 0.1)), 3)}
            hot.append(item)
    return sorted(hot, key=lambda m: m["panel"]["heat"], reverse=True)


def collect_instagram(plan: dict, transport, reference, panel: dict, terms: list[dict],
                      panel_comment_posts: int = 25, term_comment_posts: int = 10, seed_hashtags: int = 8,
                      baselines: dict[str, dict] | None = None, early_comment_posts: int = 12,
                      comment_posts: int = 3) -> dict:
    """Every Instagram request of a run, in priority order (budget may end the run early). No LLM calls."""
    bundle = {"feeds": {}, "missing_accounts": [], "terms": {}, "probes": {}, "seeds": {}, "comments": {},
              "stopped": None}

    def call(method, *args):
        if bundle["stopped"]:
            return None
        try:
            return method(*args)
        except Exception as error:
            text = str(error).lower()
            if "budget" in text or "blocked" in text or "challenge" in text or "login" in text:
                bundle["stopped"] = f"{type(error).__name__}: {str(error)[:120]}"
                print(f"[collect] stopping Instagram requests: {bundle['stopped']}", flush=True)
            return None

    def fetch_comments(item):
        if item["code"] in bundle["comments"]:
            return
        sampled = call(transport.comments, item["id"])
        if sampled:
            bundle["comments"][item["code"]] = sampled

    for account in panel["accounts"]:
        items = call(transport.user_feed, account["username"], panel.get("doc_id"))
        if items is None:
            bundle["missing_accounts"].append(account["username"])
            continue
        for item in items:
            item["panel_group"] = account["group"]
        bundle["feeds"][account["username"]] = items
    print(f"[collect] panel: {len(bundle['feeds'])} accounts read, {len(bundle['missing_accounts'])} missing", flush=True)
    scored = [m for m in panel_scores(bundle["feeds"], reference, baselines=baselines) if not m["panel"]["ritual"]]
    # Young over-performers get comment threads first: their opinions are the earliest breakout evidence.
    early = sorted((m for m in scored if (hours_since(m["taken_at"], reference) or 1e9) <= 12
                    and (m["comment_count"] or 0) >= 15
                    and (m["panel"]["over_comments"] >= 2 or m["panel"]["ratio_lift"] >= 1.5)),
                   key=lambda m: m["panel"]["heat"] + math.log(12 / max(hours_since(m["taken_at"], reference), 0.5)),
                   reverse=True)
    hot = [m for m in scored if m["panel"]["over_comments"] >= 1.5 or m["panel"]["ratio_lift"] >= 2]
    for item in list({m["code"]: m for m in early[:early_comment_posts] + hot}.values())[:panel_comment_posts]:
        fetch_comments(item)

    # Proactive Autonomous Radar: identify sudden flame posts / velocity shocks
    # and extract comment entities to proactively expand search queries before external news
    from identify_what_matters.proactive_radar import identify_proactive_candidates, generate_proactive_probe_queries
    proactive_cands = identify_proactive_candidates(bundle, reference, baselines=baselines)
    proactive_queries = generate_proactive_probe_queries(proactive_cands, max_queries=8)
    if proactive_queries:
        print(f"[collect] proactive radar discovered {len(proactive_queries)} emerging entity seeds: {proactive_queries}", flush=True)
    bundle["proactive_candidates"] = proactive_cands
    bundle["proactive_queries"] = proactive_queries

    # Search trending terms + autonomous proactive entity queries
    all_search_terms = terms + [{"term": q, "source": "proactive_radar"} for q in proactive_queries]
    for entry in all_search_terms:
        if entry["term"] in bundle["terms"]:
            continue
        results = call(transport.search, entry["term"])
        if results is not None:
            bundle["terms"][entry["term"]] = results

    probe_keys = set(plan.get("probe_keys", []))
    for topic in [t for t in plan.get("topics", []) if t["topic_key"] in probe_keys]:
        if not bundle["stopped"]:
            bundle["probes"][topic["topic_key"]] = probe(topic, transport, reference, comment_posts=comment_posts)
    for tag in SEED_HASHTAGS[:seed_hashtags]:
        result = call(transport.hashtag, tag)
        if result:
            bundle["seeds"][tag] = result["media"]
    fresh_term_posts = []
    for results in bundle["terms"].values():
        fresh = [m for m in results if (hours_since(m["taken_at"], reference) or 1e9) <= 72]
        if fresh:
            fresh_term_posts.append(max(fresh, key=lambda m: m.get("comment_count") or 0))
    for item in sorted(fresh_term_posts, key=lambda m: m.get("comment_count") or 0, reverse=True)[:term_comment_posts]:
        fetch_comments(item)
    for media in _all_media(bundle):
        sampled = bundle["comments"].get(media["code"])
        if sampled and not media.get("comments"):
            media["comments"] = sampled["comments"]
            if media.get("comment_count") is None:
                media["comment_count"] = sampled["total"]
    return bundle


def _all_media(bundle: dict):
    for items in bundle["feeds"].values():
        yield from items
    for items in bundle["terms"].values():
        yield from items
    for items in bundle["seeds"].values():
        yield from items
    for result in bundle["probes"].values():
        yield from result["media"]


def discover_discussions(bundle: dict, terms: list[dict], llm, reference, max_clusters: int = 45) -> list[dict]:
    """Cluster fresh panel/trend/seed posts across accounts; the LLM names real discussions and drops noise."""
    meta = {t["term"]: t for t in terms}
    pool, index = [], {}
    sources = ([(m, None) for items in bundle["feeds"].values() for m in items if m.get("panel") and not m["panel"]["ritual"]]
               + [(m, term) for term, items in bundle["terms"].items() for m in items]
               + [(m, None) for items in bundle["seeds"].values() for m in items])
    for item, term in sources:
        age = hours_since(item["taken_at"], reference)
        if age is None or age > 72 or (len(scrub(item["caption"])) <= 15 and not item.get("comments")):
            continue
        if item["code"] in index:
            if term:
                index[item["code"]].setdefault("x_terms", []).append(term)
            continue
        if term:
            item["x_terms"] = [term]
        index[item["code"]] = item
        pool.append(item)
    if len(pool) < 3:
        return []

    def post_text(m: dict) -> str:
        clean_cap = scrub(m["caption"])
        top_comms = " ".join(scrub(c["text"])[:80] for c in _top_comments(m, 3))
        # If caption is very short or just emojis/tags, emphasize top comments
        if len(clean_cap) < 20 and top_comms:
            return f"{clean_cap} | Discussion: {top_comms}"[:400]
        return f"{clean_cap[:350]} | {top_comms}"[:400]

    texts = [post_text(m) for m in pool]
    groups = defaultdict(list)
    for item, label in zip(pool, encoder.cluster(encoder.embed(texts))):
        groups[label].append(item)

    def strength(group):
        heat = sum(max(m.get("panel", {}).get("heat", 0), 0) for m in group)
        raw_comments = sum(m.get("comment_count") or 0 for m in group)
        multi_author_bonus = len({m["author_id"] for m in group}) * 2.0
        
        # Political/civic undercurrent boost
        combined_text = " ".join(m.get("caption") or "" for m in group)
        from identify_what_matters.proactive_radar import POLITICAL_CIVIC_PATTERNS
        is_political = bool(POLITICAL_CIVIC_PATTERNS.search(combined_text))
        political_multiplier = 2.0 if is_political else 1.0

        return (heat + math.log1p(raw_comments) * 1.5 + multi_author_bonus) * political_multiplier

    candidates = [g for g in groups.values() if len({m["author_id"] for m in g}) >= 2
                  or any(m.get("panel", {}).get("heat", 0) >= 1.5 and (m.get("comment_count") or 0) >= 100 for m in g)
                  or any((m.get("comment_count") or 0) >= 150 for m in g)
                  or any(_early_spark(m, reference) for m in g)]
    
    # Pre-filter obvious spam/promotional/giveaway/devotional candidate clusters locally
    clean_candidates = []
    spam_pattern = re.compile(r"(?i)\b(giveaway|dm for promo|paid promotion|click link in bio|follow back|100% genuine profit|telegram channel|crypto trading)\b")

    # Optional Stage 0 Civic Classifier score enrichment
    civic_clf = None
    try:
        from identify_what_matters.civic_classifier import CivicClassifier
        civic_clf = CivicClassifier(device="cpu")
    except Exception:
        civic_clf = None

    for g in candidates:
        caps = " ".join(m.get("caption") or "" for m in g)
        if spam_pattern.search(caps) and not any(m.get("panel", {}).get("group") in ("news", "political") for m in g):
            continue

        # If civic classifier is active, tag group with max civic relevance score
        if civic_clf is not None:
            group_caps = [scrub(m.get("caption") or "") for m in g if len(scrub(m.get("caption") or "")) >= 25]
            if group_caps:
                try:
                    preds = civic_clf.predict(group_caps[:5])
                    max_civic = max(p.get("score", 0.0) for p in preds)
                    for m in g:
                        m["civic_score"] = max_civic
                except Exception:
                    pass

        clean_candidates.append(g)
    candidates = sorted(clean_candidates, key=strength, reverse=True)[:max_clusters]
    if not candidates:
        return []

    def label(batch):
        listing = "\n".join(json.dumps({
            "i": i, "posts": [scrub(m["caption"])[:200] for m in group[:5]],
            "top_comments": [scrub(c["text"])[:120] for m in group for c in _top_comments(m, 3)][:6],
            "accounts": sorted({m["panel"]["group"] for m in group if m.get("panel")}),
            "x_trend_terms": sorted({t for m in group for t in m.get("x_terms", [])})}, ensure_ascii=False)
            for i, group in batch)
        return llm.json(
            "Clusters of fresh Instagram posts (with top comments) from Indian meme, news, celebrity and sports pages "
            "and from searches of terms trending in India. For each cluster decide if it is ONE coherent, specific "
            "thing people are discussing (is_topic=true): a news event, a controversy or debate, a viral moment, a "
            "meme format, celebrity gossip, a sports moment, a release/announcement, or a festival/cultural moment. "
            "is_topic=false for ads, giveaways, generic motivational/devotional content, unrelated mixes, or posts "
            "that only share a hashtag.\n"
            f"category from: {', '.join(CATEGORIES)}. kind from: {', '.join(KINDS)}.\n"
            'Return {"clusters": [{"i": 0, "is_topic": true, "title": "<= 10 words, English, specific", '
            '"category": "...", "kind": "...", "entities": ["..."]}]}\n\n' + listing,
            SYSTEM, validate=lambda v: v["clusters"])

    batches = [list(enumerate(candidates))[i:i + 15] for i in range(0, len(candidates), 15)]
    topics = []
    for result in llm.map(label, batches):
        for item in result or []:
            position = item.get("i")
            if not (isinstance(position, int) and 0 <= position < len(candidates) and item.get("is_topic") and item.get("title")):
                continue
            group = candidates[position]
            used = sorted({t for m in group for t in m.get("x_terms", [])})
            topics.append({
                "topic_key": f"d{position}", "origin": "instagram_discussion", "title": item["title"].strip(),
                "category": item.get("category") if item.get("category") in CATEGORIES else "viral_memes_social",
                "kind": item.get("kind") if item.get("kind") in KINDS else "other",
                "entities": (item.get("entities") or [])[:6], "instagram_queries": used[:3], "hashtags": [],
                "headlines": [], "publishers": 0, "languages": [], "trends_traffic": sum(meta[t].get("traffic") or 0 for t in used if t in meta),
                "trend_queries": used, "x_trends": [{"term": t, "hours": meta[t].get("x_hours"), "rank": meta[t].get("x_rank")}
                                                    for t in used if t in meta and meta[t]["source"] == "x_trends"],
                "social_feed_hits": 0, "reddit_engagement": 0, "earliest": None, "latest": None,
                "instagram": {"media": group}})
    return fold_duplicates(topics, llm)


def _early_spark(item: dict, reference) -> bool:
    """A single young post already far above its account's usual comments: too early to have spread yet."""
    panel, age = item.get("panel") or {}, hours_since(item.get("taken_at"), reference)
    return (age is not None and age <= 6 and (item.get("comment_count") or 0) >= 30
            and (panel.get("over_comments", 0) >= 3 or panel.get("ratio_lift", 0) >= 2))


def _top_comments(item: dict, k: int) -> list[dict]:
    return sorted((c for c in item.get("comments", []) if c.get("text")),
                  key=lambda c: c.get("like_count") or 0, reverse=True)[:k]


def fold_duplicates(topics: list[dict], llm) -> list[dict]:
    """Merge discussion clusters that are the same story using local encoder embeddings first, with LLM fallback."""
    if len(topics) < 2:
        return topics
    # Local heavy lifting: compute pairwise embeddings with fine-tuned multilingual encoder
    try:
        texts = [f"{t['title']}. {' '.join(t.get('entities', []))}" for t in topics]
        vecs = encoder.embed(texts)
        sim = vecs @ vecs.T
        local_groups = []
        visited = set()
        for i in range(len(topics)):
            if i in visited:
                continue
            grp = [i]
            for j in range(i + 1, len(topics)):
                if j not in visited and sim[i, j] >= 0.70:  # High confidence threshold for automatic local merge
                    grp.append(j)
                    visited.add(j)
            if len(grp) > 1:
                local_groups.append(grp)
                visited.add(i)
        
        absorbed = set()
        for group in local_groups:
            members = [i for i in group if i not in absorbed]
            for other in members[1:]:
                target, source = topics[members[0]], topics[other]
                known = {m["code"] for m in target["instagram"]["media"]}
                target["instagram"]["media"] += [m for m in source["instagram"]["media"] if m["code"] not in known]
                target["trend_queries"] = sorted(set(target["trend_queries"]) | set(source["trend_queries"]))
                target["x_trends"] = list({x["term"]: x for x in target["x_trends"] + source["x_trends"]}.values())
                target["trends_traffic"] += source["trends_traffic"]
                absorbed.add(other)
        topics = [t for i, t in enumerate(topics) if i not in absorbed]
        if len(topics) < 2:
            return topics
    except Exception as e:
        print(f"[fold_duplicates] local encoder pre-pass skipped: {e}", flush=True)

    listing = "\n".join(json.dumps({"i": i, "title": t["title"], "entities": t["entities"]}, ensure_ascii=False)
                        for i, t in enumerate(topics))
    try:
        groups = llm.json("Group entries that are the same specific story, event or discussion (not merely related). "
                          'Return {"groups": [[i, j], ...]} with only groups of 2 or more.\n\n' + listing,
                          SYSTEM, validate=lambda v: v["groups"])
    except Exception:
        return topics
    absorbed = set()
    for group in groups:
        members = [i for i in group if isinstance(i, int) and 0 <= i < len(topics) and i not in absorbed]
        for other in members[1:]:
            target, source = topics[members[0]], topics[other]
            known = {m["code"] for m in target["instagram"]["media"]}
            target["instagram"]["media"] += [m for m in source["instagram"]["media"] if m["code"] not in known]
            target["trend_queries"] = sorted(set(target["trend_queries"]) | set(source["trend_queries"]))
            target["x_trends"] = list({x["term"]: x for x in target["x_trends"] + source["x_trends"]}.values())
            target["trends_traffic"] += source["trends_traffic"]
            absorbed.add(other)
    return [t for i, t in enumerate(topics) if i not in absorbed]

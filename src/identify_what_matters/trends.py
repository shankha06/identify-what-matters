"""India trend pipeline: what India is discussing and debating, measured on Instagram.

  plan     -> Google Trends + Google News in 10 languages -> E5 story clusters -> LLM topics + Instagram queries
  collect  -> X trends + Reddit; Instagram: account-panel feeds, trending-term searches, planned
              topic probes, seed hashtags, comments on the hottest posts -> instagram_capture.json
  measure  -> replay the capture: panel overperformance, discussion clusters (discussions.py), LLM briefs
              with debate structure, scores, cross-run status, audit -> report.md / report.json / trends.sqlite
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import numpy as np

from identify_what_matters import encoder
from identify_what_matters.instagram_access import scrub
from identify_what_matters.utils import hours_since, now, parse_time

CATEGORIES = (
    "politics_governance", "civic_issues", "economy_business", "markets_finance", "jobs_education",
    "technology", "science_space", "sports", "cinema_tv_ott", "music_celebrity", "health",
    "environment_weather_disaster", "crime_law", "religion_festivals_culture", "food_lifestyle",
    "fashion_beauty", "viral_memes_social", "travel_transport", "international", "other",
)
POLITICS_ECONOMY_CATEGORIES = {"politics_governance", "civic_issues", "economy_business", "markets_finance",
                               "crime_law", "international"}
SEED_HASHTAGS = ("trendingindia", "viralindia", "indiannews", "indianmemes", "desimemes", "reelsindia",
                 "bollywood", "indiancricket", "mumbai", "delhi", "bengaluru", "hyderabad", "chennai",
                 "kolkata", "trendingreels", "news")
# Political and economy-focused hashtags for targeted collection
POLITICS_HASHTAGS = ("indianpolitics", "bjp", "congress", "narendramodi", "rahulgandhi", "amitshah",
                     "arvindkejriwal", "mamatabanerjee", "loksabha", "vidhansabha", "electionindia",
                     "indiaeconomy", "rupee", "inflation", "budget2025", "makeinindia", "gst",
                     "farmersprotest", "indiaopposition", "rahul", "modi", "bjp4india")
GENERIC_HASHTAGS = {
    "india", "reels", "reel", "viral", "trending", "explore", "explorepage", "fyp", "foryou",
    "foryoupage", "love", "instagood", "photooftheday", "fashion", "beautiful", "happy",
    "cute", "tbt", "like4like", "followme", "picoftheday", "follow", "me", "selfie",
    "summer", "art", "instadaily", "friends", "repost", "nature", "girl", "fun", "style",
    "smile", "food", "instalike", "family", "travel", "fitness", "igers", "tagsforlikes",
    "follow4follow", "nofilter", "life", "beauty", "amazing", "instagram", "photography",
    "photo", "vibe", "reelsindia", "reelsinstagram", "trendingreels", "viralreels",
    "trendingindia", "viralindia", "indiannews", "indianmemes", "desimemes", "news",
}
SYSTEM = ("You are a senior analyst of Indian public discussion and social media. Be precise and conservative. "
          "Never invent facts that are not in the provided material. Output only strict JSON.")


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _require(*keys):
    def check(value):
        if not isinstance(value, dict) or any(key not in value for key in keys):
            raise ValueError(f"missing keys {keys}")
        return value
    return check


# ---------------------------------------------------------------- stories

def publisher_first(members: list[dict]) -> dict[str, str]:
    """Earliest publish time per outlet among a story's members."""
    first: dict[str, datetime] = {}
    for member in members:
        publisher, published = member.get("publisher"), parse_time(member.get("published_at"))
        if publisher and published and (publisher not in first or published < first[publisher]):
            first[publisher] = published
    return {name: moment.isoformat() for name, moment in first.items()}


def publisher_pickup(first: dict[str, str]) -> str | None:
    """When the third distinct outlet carried the story; None with fewer than three."""
    times = sorted(parse_time(value) for value in first.values())
    return times[2].isoformat() if len(times) >= 3 else None


def build_stories(signals: list[dict], reference: datetime, max_age_hours: float = 48) -> list[dict]:
    fresh = [s for s in signals if s.get("published_at") is None
             or (hours_since(s["published_at"], reference) or 0) <= max_age_hours]
    texts = [s["title"] + (" | " + s["related"][0] if s.get("related") else "") for s in fresh]
    labels = encoder.cluster(encoder.embed(texts))
    groups = defaultdict(list)
    for signal, label in zip(fresh, labels):
        groups[label].append(signal)
    stories = []
    for index, members in enumerate(sorted(groups.values(), key=len, reverse=True)):
        languages = Counter(m.get("language", "en") for m in members if m["source"] == "google_news")
        titles, seen_languages = [], set()
        for member in sorted(members, key=lambda m: m["source"] != "google_trends"):
            language = member.get("language", "trend")
            if language not in seen_languages or len(titles) < 2:
                titles.append(member["title"])
                seen_languages.add(language)
            if len(titles) >= 4:
                break
        times = [m["published_at"] for m in members if m.get("published_at")]
        first_by_publisher = publisher_first(members)
        stories.append({
            "story_id": f"c{index}", "titles": titles, "signals": [m["signal_id"] for m in members],
            "publishers": len({m.get("publisher") for m in members if m.get("publisher")}),
            "languages": sorted(languages), "feeds": sorted({m["feed"] for m in members}),
            "trends_traffic": sum(m.get("traffic") or 0 for m in members if m["source"] == "google_trends"),
            "trend_queries": [m["title"] for m in members if m["source"] == "google_trends"],
            "reddit_engagement": sum((m.get("engagement") or {}).get("comments") or 0 for m in members),
            "social_feed_hits": sum(1 for m in members if m["feed"].startswith("news_search:")),
            "earliest": min(times) if times else None, "latest": max(times) if times else None,
            "publisher_first": first_by_publisher, "pickup_at": publisher_pickup(first_by_publisher),
        })
    return stories


# ---------------------------------------------------------------- plan

def _plan_prompt(batch: list[dict]) -> str:
    lines = [json.dumps({"id": s["story_id"], "headlines": s["titles"], "google_searches": s["trend_queries"],
                         "publishers": s["publishers"], "languages": s["languages"]}, ensure_ascii=False)
             for s in batch]
    return (
        "Below are clusters of news headlines and Google search trends collected in India in the last 48 hours.\n"
        "For EACH cluster decide whether it is a specific, current topic that people in India are likely to be "
        "discussing on social media.\n"
        "keep=false for: routine listings (daily gold/fuel rates, horoscopes, routine weather), market tickers, "
        "evergreen explainers, product deal posts, and foreign stories without a clear Indian audience angle. "
        "International stories with a strong Indian angle (cricket, Indian diaspora, global events affecting India) "
        "may be kept with category international or their subject category.\n"
        f"category must be one of: {', '.join(CATEGORIES)}.\n"
        "instagram_queries: 2-3 Instagram searches, each ONLY 1-2 words (Instagram keyword search returns nothing for "
        "long phrases). Use the most distinctive names: person, film, team, place, organisation, event or a common "
        "hashtag word (e.g. 'Drishyam 3', 'Gyanesh Kumar', 'Flydubai'). Most specific first. No '#'.\n"
        "hashtags: 1-2 hashtags Indian users would actually put on posts about it, lowercase, no spaces, no '#' "
        "(e.g. 'drishyam3', 'gyaneshkumar').\n"
        "same_as: ids of other clusters in this list that are the same topic.\n"
        'Return {"topics": [{"id": "...", "keep": true, "title": "<= 10 words, English, specific", '
        '"category": "...", "entities": ["..."], "instagram_queries": ["..."], "hashtags": ["..."], "same_as": []}]} '
        "with one entry per input cluster.\n\nClusters:\n" + "\n".join(lines)
    )


def plan_topics(stories: list[dict], llm, batch_size: int = 30) -> list[dict]:
    batches = [stories[i:i + batch_size] for i in range(0, len(stories), batch_size)]

    def run(batch):
        # Gateway safety filters return empty text for some crime headlines; bisect to drop only those.
        try:
            return llm.json(_plan_prompt(batch), SYSTEM, validate=lambda v: v["topics"])
        except ValueError:
            if len(batch) == 1:
                print(f"[plan] dropped cluster the LLM would not process: {batch[0]['titles'][0][:80]}", flush=True)
                return []
            middle = len(batch) // 2
            return run(batch[:middle]) + run(batch[middle:])

    by_story = {s["story_id"]: s for s in stories}
    kept = []
    for result in llm.map(run, batches):
        for item in result or []:
            story = by_story.get(item.get("id"))
            if story and item.get("keep") and item.get("title") and item.get("instagram_queries"):
                category = item.get("category") if item.get("category") in CATEGORIES else "other"
                raw_tags = [re.sub(r"[^\w]", "", h.lower()) for h in item.get("hashtags", [])[:3] if isinstance(h, str) and h]
                tags = [h for h in raw_tags if h and h not in GENERIC_HASHTAGS][:2]
                same_as_ids = [i for i in item.get("same_as", []) if isinstance(i, str) and i in by_story]
                kept.append({"story_ids": [story["story_id"]] + same_as_ids,
                             "title": item["title"].strip(), "category": category,
                             "entities": item.get("entities", [])[:6],
                             "hashtags": tags,
                             "instagram_queries": [q.strip().lstrip("#") for q in item["instagram_queries"][:3] if isinstance(q, str) and q.strip()]})
    return merge_topics(kept, by_story, llm)


def merge_topics(topics: list[dict], by_story: dict, llm) -> list[dict]:
    if not topics:
        return []
    listing = "\n".join(json.dumps({"i": i, "title": t["title"], "entities": t["entities"]}, ensure_ascii=False)
                        for i, t in enumerate(topics))
    groups = llm.json(
        "These topic titles were extracted independently and may contain duplicates (same underlying story, "
        "event or discussion). Group duplicates. Do not group merely related topics.\n"
        'Return {"groups": [[i, j, ...], ...]} listing only groups with 2 or more members.\n\n' + listing,
        SYSTEM, validate=lambda v: v["groups"])
    parent = list(range(len(topics)))
    for group in groups:
        members = [i for i in group if isinstance(i, int) and 0 <= i < len(topics)]
        for member in members[1:]:
            parent[member] = members[0]
    merged = {}
    for index, topic in enumerate(topics):
        root = index
        while parent[root] != root:
            root = parent[root]
        if root not in merged:
            merged[root] = {**topic, "story_ids": list(topic["story_ids"])}
        else:
            target = merged[root]
            target["story_ids"] += topic["story_ids"]
            target["instagram_queries"] = list(dict.fromkeys(target["instagram_queries"] + topic["instagram_queries"]))[:4]
            target["hashtags"] = list(dict.fromkeys(target.get("hashtags", []) + topic.get("hashtags", [])))[:3]
            target["entities"] = list(dict.fromkeys(target["entities"] + topic["entities"]))[:8]
    result = []
    for index, topic in enumerate(merged.values()):
        stories = [by_story[s] for s in dict.fromkeys(topic["story_ids"]) if s in by_story]
        first_by_publisher: dict[str, str] = {}
        for story in stories:
            for publisher, when in story.get("publisher_first", {}).items():
                first_by_publisher[publisher] = min(first_by_publisher.get(publisher, when), when, key=parse_time)
        topic.update({
            "topic_key": f"t{index}", "origin": "news_and_search",
            "headlines": [title for story in stories for title in story["titles"]][:6],
            "publishers": sum(s["publishers"] for s in stories),
            "languages": sorted({lang for s in stories for lang in s["languages"]}),
            "trends_traffic": sum(s["trends_traffic"] for s in stories),
            "trend_queries": [q for s in stories for q in s["trend_queries"]],
            "social_feed_hits": sum(s["social_feed_hits"] for s in stories),
            "reddit_engagement": sum(s["reddit_engagement"] for s in stories),
            "earliest": min((s["earliest"] for s in stories if s["earliest"]), default=None),
            "latest": max((s["latest"] for s in stories if s["latest"]), default=None),
            "publisher_first": first_by_publisher, "pickup_at": publisher_pickup(first_by_publisher),
        })
        result.append(topic)
    return result


def plan_sentinel_topics(output_root: Path, llm, reference: datetime, max_clusters: int = 20) -> list[dict]:
    """Phase 1 native Sentinel Radar: scan recent panel feed captures for flame posts,

    cluster emerging comment/caption entities, and formulate proactive Instagram-native topics.
    """
    from identify_what_matters.proactive_radar import identify_proactive_candidates, POLITICAL_CIVIC_PATTERNS
    captures = sorted(output_root.glob("runs/*/instagram_capture.json"))
    bundle = {"feeds": {}, "comments": {}}
    for c_path in reversed(captures):
        try:
            data = json.loads(c_path.read_text())
            feeds = data.get("feeds", {})
            total_items = sum(len(v.get("items", [])) if isinstance(v, dict) else len(v) for v in feeds.values())
            if total_items > 0:
                bundle["feeds"] = {k: v.get("items", []) if isinstance(v, dict) else v for k, v in feeds.items()}
                bundle["comments"] = {k: {"comments": v.get("comments", []) if isinstance(v, dict) else v}
                                      for k, v in data.get("comments", {}).items()}
                break
        except Exception:
            continue

    if not bundle["feeds"]:
        return []

    candidates = identify_proactive_candidates(bundle, reference)
    if not candidates:
        return []

    # Rank proactive flame posts: prioritize political undercurrents and high velocity
    top_candidates = sorted(
        candidates,
        key=lambda c: (1 if POLITICAL_CIVIC_PATTERNS.search(c.get("caption", "")) else 0, c.get("proactive_score", 0)),
        reverse=True
    )[:35]

    texts = [c["caption"] + " " + " ".join(e["entity"].lstrip("#") for e in c.get("surging_entities", []))
             for c in top_candidates]
    labels = encoder.cluster(encoder.embed(texts))
    groups = defaultdict(list)
    for c, lbl in zip(top_candidates, labels):
        groups[lbl].append(c)

    prompt_clusters = []
    cluster_order = []
    for i, (lbl, grp) in enumerate(groups.items()):
        all_ents = list(dict.fromkeys(e["entity"].lstrip("#") for c in grp for e in c.get("surging_entities", [])))
        prompt_clusters.append({
            "i": i,
            "captions": [c["caption"][:140] for c in grp[:3]],
            "surging_entities": all_ents[:5],
            "proactive_score": round(max(c["proactive_score"] for c in grp), 1)
        })
        cluster_order.append(grp)
        if len(prompt_clusters) >= max_clusters:
            break

    if not prompt_clusters:
        return []

    listing = "\n".join(json.dumps(p, ensure_ascii=False) for p in prompt_clusters)
    prompt = (
        "Clusters of surging Instagram flame posts and controversial comments detected by real-time Sentinel Radar in India.\n"
        "For each cluster, formulate an Instagram-native emerging topic title (<= 10 words, English, specific), "
        "identify category from: politics_governance, crime_law, civic_issues, viral_memes_social, sports, music_celebrity, international, other.\n"
        "Provide 2-3 specific 1-2 word search queries for Instagram probing (no '#').\n"
        'Return {"topics": [{"i": 0, "title": "...", "category": "...", "entities": ["..."], "instagram_queries": ["..."]}]}\n\n'
        + listing
    )

    try:
        res = llm.json(prompt, SYSTEM, validate=lambda v: v["topics"] if isinstance(v, dict) else v)
        items = res if isinstance(res, list) else res.get("topics", [])
    except Exception as e:
        print(f"[plan] sentinel radar LLM labeling error: {e}", flush=True)
        return []

    sentinel_topics = []
    for t in items:
        pos = t.get("i")
        if not (isinstance(pos, int) and 0 <= pos < len(cluster_order) and t.get("title")):
            continue
        grp = cluster_order[pos]
        max_score = max(c["proactive_score"] for c in grp)
        ents = [e for e in t.get("entities", []) if isinstance(e, str)]
        queries = [q.strip().lstrip("#") for q in t.get("instagram_queries", []) if isinstance(q, str) and q.strip()][:3]
        if not queries:
            queries = [e["entity"].lstrip("#") for c in grp for e in c.get("surging_entities", [])][:2]
        cat = t.get("category") if t.get("category") in CATEGORIES else "politics_governance"
        sentinel_topics.append({
            "topic_key": f"s{pos}",
            "origin": "sentinel_radar",
            "title": t["title"].strip(),
            "category": cat,
            "entities": ents[:6],
            "instagram_queries": queries[:3],
            "hashtags": [],
            "headlines": [c["caption"][:100] for c in grp[:3]],
            "publishers": 0,
            "languages": [],
            "trends_traffic": 0,
            "trend_queries": [],
            "social_feed_hits": 0,
            "reddit_engagement": 0,
            "proactive_score": max_score,
            "trajectory": "spark",
            "earliest": reference.isoformat(),
            "latest": reference.isoformat(),
        })

    print(f"[plan] Sentinel Radar generated {len(sentinel_topics)} native emerging topics", flush=True)
    return sentinel_topics


def prior_score(topic: dict, reference: datetime) -> float:
    age = hours_since(topic.get("latest"), reference)
    base = (math.log1p(topic.get("publishers", 0)) + 0.7 * math.log1p(max(len(topic.get("languages", [])) - 1, 0))
            + 0.6 * math.log1p(topic.get("trends_traffic", 0) / 1000) + 0.5 * math.log1p(topic.get("social_feed_hits", 0))
            + 0.3 * math.log1p(topic.get("reddit_engagement", 0) / 50) + (0.5 if age is not None and age <= 12 else 0))
    if topic.get("origin") == "sentinel_radar":
        # Substantial boost for native sentinel flame topics to prioritize them in probing
        base += 4.0 + math.log1p(topic.get("proactive_score", 0))
    return base


def select_for_probe(topics: list[dict], reference: datetime, limit: int, per_category: int = 6) -> list[dict]:
    if limit <= 0:
        return []
    ranked = sorted(topics, key=lambda t: prior_score(t, reference), reverse=True)
    chosen, counts, selected_ids = [], Counter(), set()
    political_categories = {"politics_governance", "civic_issues"}
    reserved = min((limit + 1) // 2, sum(t["category"] in political_categories for t in ranked))
    for topic in ranked:
        if topic["category"] in political_categories and len(chosen) < reserved:
            chosen.append(topic)
            selected_ids.add(id(topic))
            counts[topic["category"]] += 1
    for topic in ranked:
        if id(topic) not in selected_ids and counts[topic["category"]] < (max(per_category, reserved) if topic["category"] in political_categories else per_category):
            chosen.append(topic)
            selected_ids.add(id(topic))
            counts[topic["category"]] += 1
        if len(chosen) >= limit:
            break
    for topic in ranked:
        if len(chosen) >= limit:
            break
        if id(topic) not in selected_ids:
            chosen.append(topic)
            selected_ids.add(id(topic))
    return chosen


# ---------------------------------------------------------------- Instagram evidence

async def probe(topic: dict, transport, reference: datetime | None = None, max_queries: int = 2, enough_recent: int = 6,
                comment_posts: int = 3, comment_pages: int = 2, reply_threads: int = 5, call_timeout: int = 150) -> dict:
    """Search queries in order until enough recent posts; then sample comments on the top recent posts."""
    import asyncio

    async def _safe(method, *args, **kwargs):
        try:
            result = method(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await asyncio.wait_for(result, timeout=call_timeout)
            return result
        except asyncio.TimeoutError:
            return None
        except Exception as error:
            raise

    media, seen, used = [], set(), []
    reference = reference or now()
    for query in topic["instagram_queries"][:3]:
        if len(used) >= max_queries:
            break
        try:
            results = await _safe(transport.search, query)
        except Exception as error:
            topic.setdefault("instagram_errors", []).append(f"{query}: {type(error).__name__}")
            if "budget" in str(error).lower() or "blocked" in str(error).lower():
                break
            continue
        if results is None:
            continue
        used.append(query)
        for item in results:
            if item["code"] not in seen:
                seen.add(item["code"])
                media.append(item)
        if sum((hours_since(m["taken_at"], reference) or 1e9) <= 168 for m in media) >= enough_recent:
            break
    hashtag_counts = {}
    valid_tags = [t for t in topic.get("hashtags", []) if t not in GENERIC_HASHTAGS]
    for tag in valid_tags[:1]:
        try:
            result = await _safe(transport.hashtag, tag)
        except Exception as error:
            topic.setdefault("instagram_errors", []).append(f"#{tag}: {type(error).__name__}")
            continue
        if result:
            hashtag_counts[tag] = result["media_count"]
            for item in result["media"]:
                if item["code"] not in seen:
                    seen.add(item["code"])
                    media.append(item)
    recent = [m for m in media if (hours_since(m["taken_at"], reference) or 1e9) <= 24 * 30]
    commentable = [m for m in recent if (m.get("comment_count") or 0) > 0]
    for item in sorted(commentable, key=lambda m: m["comment_count"], reverse=True)[:comment_posts]:
        try:
            sampled = await _safe(transport.comments, item["id"], max_pages=comment_pages, max_reply_threads=reply_threads)
        except Exception as error:
            topic.setdefault("instagram_errors", []).append(f"comments: {type(error).__name__}")
            continue
        if sampled:
            item["comments"] = sampled["comments"]
            if item.get("comment_count") is None:
                item["comment_count"] = sampled["total"]
    return {"media": media, "queries_used": used, "hashtag_counts": hashtag_counts}


# Max absolute gap between like-weighted and unweighted shares that marks the opinion as exposure-sensitive.
EXPOSURE_GAP_FLAG = 0.15
_TONES = {"type": "string", "enum": ["supportive", "critical", "humorous", "neutral"]}
_FRACTION = {"type": "number"}
BRIEF_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["on_topic", "discussion_level", "summary", "what_people_say", "stance", "why_trending",
                 "caution", "debate", "comment_labels", "political_framing"],
    "properties": {
        "on_topic": {"type": "array", "items": {"type": "integer"}},
        "discussion_level": {"type": "string", "enum": ["none", "low", "medium", "high"]},
        "summary": {"type": "string"},
        "what_people_say": {"type": "array", "items": {"type": "string"}},
        "stance": {"type": "object", "additionalProperties": False,
                   "required": ["supportive", "critical", "humorous", "neutral"],
                   "properties": {k: _FRACTION for k in ["supportive", "critical", "humorous", "neutral"]}},
        "why_trending": {"type": "string"},
        "caution": {"type": ["string", "null"]},
        "debate": {"type": "object", "additionalProperties": False,
                   "required": ["is_debate", "question", "sides", "intensity"],
                   "properties": {
                       "is_debate": {"type": "boolean"},
                       "question": {"type": "string"},
                       "sides": {"type": "array", "items": {
                           "type": "object", "additionalProperties": False,
                           "required": ["position", "share", "example"],
                           "properties": {"position": {"type": "string"}, "share": _FRACTION,
                                          "example": {"type": "string"}}}},
                       "intensity": {"type": "string", "enum": ["low", "medium", "high"]}}},
        "comment_labels": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["c", "tone", "sarcastic", "side", "off_topic"],
            "properties": {"c": {"type": "integer"}, "tone": _TONES, "sarcastic": {"type": "boolean"},
                           "side": {"type": ["integer", "null"]}, "off_topic": {"type": "boolean"}}}},
        "political_framing": {"type": "object", "additionalProperties": False,
                              "required": ["direction", "confidence", "basis"],
                              "properties": {
                                  "direction": {"type": "string", "enum": [
                                      "pro_incumbent", "anti_incumbent", "pro_opposition",
                                      "anti_opposition", "neutral"]},
                                  "confidence": _FRACTION, "basis": {"type": "string"}}},
    },
}


def brief(topic: dict, llm, reference: datetime, comments_per_post: int = 25, comment_pool_size: int = 120) -> dict:
    by_age = sorted(topic.get("instagram", {}).get("media", []),
                    key=lambda m: hours_since(m["taken_at"], reference) or 1e9)
    media = list({m["code"]: m for m in by_age[:18] + [m for m in by_age if m.get("comments")][:18]}.values())
    posts = [{"i": i, "age_hours": round(hours_since(m["taken_at"], reference) or -1), "likes": m["like_count"],
              "comments": m["comment_count"], "reel": m["is_reel"],
              "source": (m.get("panel") or {}).get("group") or ("trend search" if m.get("x_terms") else "search/hashtag"),
              "caption": scrub(m["caption"])[:350]}
             for i, m in enumerate(media)]
    comments, pool = [], []
    from identify_what_matters import stance
    for i, m in sorted(enumerate(media), key=lambda pair: pair[1].get("comment_count") or 0, reverse=True):
        raw_ranked = sorted((c for c in m.get("comments", []) if c.get("text")), key=lambda c: c.get("like_count") or 0, reverse=True)
        # Pre-filter noise comments (spam, bots, emoji-only) before feeding to prompt and pool
        noise_evals = stance.filter_comment_noise([c["text"] for c in raw_ranked])
        ranked = [c for c, ne in zip(raw_ranked, noise_evals) if not ne["is_noise"]]
        for c in ranked[:comments_per_post]:
            if len(pool) >= comment_pool_size:
                break
            comments.append(f"[c{len(pool)}; post {i}; {c.get('like_count') or 0} likes; {c.get('reply_count') or 0} replies] "
                            f"{scrub(c['text'])[:200]}")
            pool.append(c)
    prompt = (
        f"Topic: {topic['title']} (category {topic['category']})\n"
        f"News headlines: {json.dumps(topic['headlines'][:6], ensure_ascii=False)}\n"
        f"Search / X trend terms: {json.dumps(topic['trend_queries'][:5], ensure_ascii=False)}\n"
        f"Instagram posts: {json.dumps(posts, ensure_ascii=False)}\n"
        f"Top Instagram comments (comment likes = how many people endorse it; replies = arguments under it): "
        f"{json.dumps(comments, ensure_ascii=False)}\n\n"
        "Tasks:\n"
        "1. on_topic: indices of posts clearly about THIS specific story/event (search and hashtags return many "
        "loosely related or unrelated posts; exclude those, and exclude posts about a different event).\n"
        "2. discussion_level on Instagram: none, low, medium or high, judged from on-topic posts, recency and engagement.\n"
        "3. summary: 1-2 sentences on what the topic is, using only the material.\n"
        "4. what_people_say: up to 4 short bullets on the main reactions/opinions/jokes in captions and comments, "
        "weighting highly liked comments more.\n"
        "5. stance: fractions (sum 1) of supportive, critical, humorous, neutral among on-topic posts and comments.\n"
        "6. why_trending: one sentence.\n"
        "7. caution: null, or a short note if claims are unverified, communal, graphic or likely misinformation.\n"
        "8. debate: are commenters actually disagreeing with each other (not just reacting)? If yes give the contested "
        "question in one neutral line, 2-3 sides with their approximate share of comment support (use comment likes) "
        "and a short neutral paraphrase of a representative comment (no slurs, no names of private people), and "
        "intensity low/medium/high. If no, is_debate=false.\n"
        "9. comment_labels: for EVERY comment c0, c1, ... above give tone (supportive, critical, humorous or neutral "
        "toward the post's subject; read sarcasm as its intended meaning), sarcastic true/false, and side = index of "
        "the debate side it backs (0-based, null if none or no debate). Off-topic or spam comments: tone neutral, "
        "side null, off_topic true.\n"
        "10. political_framing: classify the POST's OWN CAPTION/NARRATIVE (not the comments) as one of: "
        "\"pro_incumbent\" (frames BJP/NDA positively — achievement, defence, development), "
        "\"anti_incumbent\" (frames BJP/NDA critically — failure, corruption, attack), "
        "\"pro_opposition\" (frames Opposition/INDIA bloc positively), "
        "\"anti_opposition\" (frames Opposition critically), or \"neutral\" (factual, no partisan lean). "
        "Give confidence 0-1 and a basis phrase (≤10 words from the caption that signals the frame). "
        "If there are no posts or the topic is non-political, use neutral with confidence 0.\n"
        'Return {"on_topic": [..], "discussion_level": "...", "summary": "...", "what_people_say": ["..."], '
        '"stance": {"supportive": 0, "critical": 0, "humorous": 0, "neutral": 0}, "why_trending": "...", "caution": null, '
        '"debate": {"is_debate": true, "question": "...", "sides": [{"position": "...", "share": 0.5, "example": "..."}], '
        '"intensity": "medium"}, "comment_labels": [{"c": 0, "tone": "critical", "sarcastic": false, "side": 1, '
        '"off_topic": false}], '
        '"political_framing": {"direction": "neutral", "confidence": 0.0, "basis": ""}}'
    )
    result = llm.json(prompt, SYSTEM, validate=_require("summary", "on_topic", "discussion_level"),
                      schema=BRIEF_SCHEMA)
    result["on_topic_codes"] = [media[i]["code"] for i in result.get("on_topic", []) if isinstance(i, int) and 0 <= i < len(media)]
    opinion = measured_opinion(result, pool)
    if opinion:
        opinion["source"] = "LLM brief labels"
    result["measured_opinion"] = opinion
    result["comment_pool"] = [{"text": scrub(c["text"])[:200], "likes": c.get("like_count") or c.get("comment_like_count") or 0,
                               "replies": c.get("reply_count") or c.get("child_comment_count") or 0,
                               "author_id": str((c.get("user") or {}).get("pk") or c.get("author_id") or ""),
                               "created_at": c.get("created_at")} for c in pool]
    return result


def _effective_n(weights: list[float]) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2; equals n for equal weights."""
    squares = sum(w * w for w in weights)
    return sum(weights) ** 2 / squares if squares else 0.0


def _wilson_interval(p: float, n: float, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval; Wald intervals overshoot 0-1 and collapse near the bounds."""
    if n <= 0:
        return None
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def measured_opinion(result: dict, pool: list[dict]) -> dict | None:
    """Like-weighted shares from per-comment labels: what commenters actually say, not the LLM's overall impression."""
    labels = {item["c"]: item for item in result.get("comment_labels") or []
              if isinstance(item, dict) and isinstance(item.get("c"), int) and 0 <= item["c"] < len(pool)}
    if not labels:
        return None
    sides = [s for s in ((result.get("debate") or {}).get("sides") or []) if isinstance(s, dict) and s.get("position")]
    tones, side_weight, quotes, sarcastic, used = Counter(), Counter(), {}, 0, 0
    all_weights, aligned_weights = [], []
    for index, label in labels.items():
        if label.get("off_topic"):
            continue
        comment = pool[index]
        weight = 1 + math.log1p(comment.get("like_count") or 0)
        used += 1
        tone = label.get("tone") if label.get("tone") in ("supportive", "critical", "humorous", "neutral") else "neutral"
        tones[tone] += weight
        all_weights.append(weight)
        sarcastic += bool(label.get("sarcastic"))
        side = label.get("side")
        if isinstance(side, int) and 0 <= side < len(sides):
            side_weight[side] += weight
            aligned_weights.append(weight)
            best = quotes.get(side)
            if best is None or (comment.get("like_count") or 0) > (best.get("like_count") or 0):
                quotes[side] = comment
    if not used:
        return None
    tone_total, side_total = sum(tones.values()), sum(side_weight.values())
    tone_keys = ("supportive", "critical", "humorous", "neutral")

    # Exposure-bias sensitivity: the same labels counted once per comment, without like weighting.
    # Reported only; the weighted shares above remain the default.
    plain_tones, plain_sides = _tone_and_side_totals(labels, pool, sides, lambda comment: 1.0)
    plain_tone_total, plain_side_total = sum(plain_tones.values()), sum(plain_sides.values())
    tone_gap = max(abs(tones[k] / tone_total - plain_tones[k] / plain_tone_total) for k in tone_keys)
    side_gap = max((abs(side_weight[i] / side_total - plain_sides[i] / plain_side_total)
                    for i in range(len(sides))), default=0.0) if side_total and plain_side_total else 0.0
    gap = round(max(tone_gap, side_gap), 2)

    # Intervals use the effective n of the weighted labels, so heavy-liked comments do not overstate precision.
    n_eff, side_n_eff = _effective_n(all_weights), _effective_n(aligned_weights)
    return {
        "comments": used, "sarcastic": sarcastic, "effective_n": round(n_eff, 1),
        "tone": {k: round(tones[k] / tone_total, 2) for k in tone_keys},
        "tone_ci": {k: _wilson_interval(tones[k] / tone_total, n_eff) for k in tone_keys},
        "sides": [{"position": sides[i]["position"], "share": round(side_weight[i] / side_total, 2) if side_total else 0.0,
                   "share_ci": _wilson_interval(side_weight[i] / side_total, side_n_eff) if side_total else None,
                   "comments": sum(1 for l in labels.values() if l.get("side") == i and not l.get("off_topic")),
                   "top_comment": scrub(quotes[i]["text"])[:180] if i in quotes else None,
                   "top_comment_likes": quotes[i].get("like_count") if i in quotes else None}
                  for i in range(len(sides))],
        "unaligned_share": round(1 - side_total / tone_total, 2) if sides and tone_total else None,
        "weighting_sensitivity": {
            "max_gap": gap,
            "flag": gap >= EXPOSURE_GAP_FLAG,
            "unweighted_tone": {k: round(plain_tones[k] / plain_tone_total, 2) for k in tone_keys},
            "unweighted_sides": [round(plain_sides[i] / plain_side_total, 2) if plain_side_total else 0.0
                                 for i in range(len(sides))],
        },
    }


def _tone_and_side_totals(labels: dict, pool: list[dict], sides: list[dict], weight_of) -> tuple[Counter, Counter]:
    """Tone and side totals over non-off-topic labels, using the supplied per-comment weight."""
    tones, side_weight = Counter(), Counter()
    for index, label in labels.items():
        if label.get("off_topic"):
            continue
        weight = weight_of(pool[index])
        tone = label.get("tone") if label.get("tone") in ("supportive", "critical", "humorous", "neutral") else "neutral"
        tones[tone] += weight
        side = label.get("side")
        if isinstance(side, int) and 0 <= side < len(sides):
            side_weight[side] += weight
    return tones, side_weight


# ---------------------------------------------------------------- metrics, history, status

def instagram_metrics(topic: dict, reference: datetime) -> dict:
    media = topic.get("instagram", {}).get("media", [])
    on_topic = set((topic.get("brief") or {}).get("on_topic_codes", []))
    relevant = [m for m in media if m["code"] in on_topic] if topic.get("brief") else media
    ages = [hours_since(m["taken_at"], reference) for m in relevant]
    ages = [a for a in ages if a is not None]
    engagement = sum(((m["like_count"] or 0) + 3 * (m["comment_count"] or 0)) / max(hours_since(m["taken_at"], reference) or 1, 1)
                     for m in relevant if m["taken_at"] and (hours_since(m["taken_at"], reference) or 1e9) <= 168)
    last_24 = sum(a <= 24 for a in ages)
    last_week = sum(a <= 168 for a in ages)
    tag_media = [m for m in media if m["query"].startswith("#")]
    metrics = {
        "search_results": len(media), "on_topic_posts": len(relevant), "posts_24h": last_24, "posts_72h": sum(a <= 72 for a in ages),
        "posts_7d": last_week, "authors_7d": len({m["author_id"] for m in relevant if (hours_since(m["taken_at"], reference) or 1e9) <= 168}),
        "recency_ratio": round(last_24 / last_week, 2) if last_week else 0.0,
        "engagement_per_hour": round(engagement, 1),
        "comments_7d": sum(m["comment_count"] or 0 for m in relevant if (hours_since(m["taken_at"], reference) or 1e9) <= 168),
        "plays_7d": sum(m["play_count"] or 0 for m in relevant if (hours_since(m["taken_at"], reference) or 1e9) <= 168),
        "sampled_comments": sum(len(m.get("comments", [])) for m in media),
        "hashtag_counts": topic.get("instagram", {}).get("hashtag_counts", {}),
        "hashtag_on_topic_share": round(sum(m["code"] in on_topic for m in tag_media) / len(tag_media), 2) if tag_media and topic.get("brief") else None,
        "likes_7d": sum(m["like_count"] or 0 for m in relevant if (hours_since(m["taken_at"], reference) or 1e9) <= 168),
        "top_posts": [m["url"] for m in sorted(relevant, key=lambda m: (m["comment_count"] or 0) + (m["like_count"] or 0) / 20, reverse=True)[:3]],
    }
    panel = [m["panel"] for m in relevant if m.get("panel")]
    sampled = [c for m in relevant for c in m.get("comments", [])]
    metrics.update({
        "panel_accounts": len({p["account"] for p in panel}),
        "panel_max_over": max(((p["over_comments"] or 0.0) for p in panel), default=0.0),
        "panel_max_ratio_lift": max(((p["ratio_lift"] or 0.0) for p in panel), default=0.0),
        "comment_like_ratio": round(metrics["comments_7d"] / max(metrics["likes_7d"], 1), 3),
        "replies_per_comment": round(sum(c.get("reply_count") or 0 for c in sampled) / len(sampled), 2) if sampled else 0.0,
        "comments_scored": sum(len(m.get("comments", [])) for m in relevant),
        "comments_platform": sum(m["comment_count"] or 0 for m in relevant),
    })
    metrics["comment_coverage"] = (round(min(metrics["comments_scored"] / metrics["comments_platform"], 1.0), 3)
                                   if metrics["comments_platform"] else None)
    metrics["discussion_level"] = discussion_level(metrics)
    return metrics


def lead_lag(topic: dict) -> dict:
    """Instagram versus news timing; a positive lead_vs_news_hours means Instagram posted first."""
    media = topic.get("instagram", {}).get("media", [])
    if topic.get("brief"):
        on_topic = set(topic["brief"].get("on_topic_codes", []))
        media = [m for m in media if m.get("code") in on_topic]
    first_post = min((parse_time(m.get("taken_at")) for m in media if m.get("taken_at")), default=None)
    news, pickup = parse_time(topic.get("earliest")), parse_time(topic.get("pickup_at"))
    result = {"instagram_first_at": first_post.isoformat() if first_post else None,
              "news_first_at": news.isoformat() if news else None,
              "pickup_at": pickup.isoformat() if pickup else None,
              "lead_vs_news_hours": None, "lead_vs_pickup_hours": None}
    if first_post is None:
        result["verdict"] = "no_instagram_evidence"
    elif news is None:
        result["verdict"] = "instagram_only"
    else:
        lead = round((news - first_post).total_seconds() / 3600, 1)
        result["lead_vs_news_hours"] = lead
        if pickup:
            result["lead_vs_pickup_hours"] = round((pickup - first_post).total_seconds() / 3600, 1)
        result["verdict"] = "instagram_first" if lead >= 1 else "news_first" if lead <= -1 else "simultaneous"
    return result


def lead_lag_summary(topics: list[dict]) -> dict:
    leads = [t["lead"]["lead_vs_news_hours"] for t in topics
             if (t.get("lead") or {}).get("lead_vs_news_hours") is not None]
    verdicts = Counter((t.get("lead") or {}).get("verdict", "not_measured") for t in topics)
    return {"verdicts": dict(verdicts), "compared_with_news": len(leads),
            "median_lead_vs_news_hours": round(median(leads), 1) if leads else None}


def stage_history(timeline: list[dict], stage: str | None, reference: datetime) -> dict:
    """Breakout-stage transitions across earlier runs, and hours spent in the current stage."""
    stages = [(p["at"], p["stage"]) for p in timeline if p.get("stage")]
    transitions = [{"from": before, "to": after, "at": at}
                   for (_, before), (at, after) in zip(stages, stages[1:]) if before != after]
    if stage and stages and stages[-1][1] != stage:
        transitions.append({"from": stages[-1][1], "to": stage, "at": reference.isoformat()})
    started = reference
    for at, value in reversed(stages):
        if value != stage:
            break
        started = parse_time(at)
    return {"stage": stage, "runs_observed": len(timeline) + 1,
            "stage_hours": round((reference - started).total_seconds() / 3600, 1),
            "transitions": transitions[-5:]}


def discussion_level(metrics: dict) -> str:
    """Evidence-based level from measured posts, authors, panel breadth and engagement (not the LLM's impression)."""
    if (metrics["authors_7d"] >= 10 and metrics["posts_72h"] >= 10 and metrics["engagement_per_hour"] >= 2000) \
            or (metrics.get("panel_accounts", 0) >= 4 and metrics["comments_7d"] >= 2000):
        return "high"
    if (metrics["authors_7d"] >= 5 and metrics["posts_72h"] >= 5) or metrics.get("panel_accounts", 0) >= 2 \
            or (metrics.get("panel_max_over", 0) >= 4 and metrics["comments_7d"] >= 500):
        return "medium"
    if metrics["posts_7d"] >= 1:
        return "low"
    return "none"


def scores(topic: dict) -> dict:
    ig = topic["instagram_metrics"]
    level = {"none": 0, "low": 0.5, "medium": 1.0, "high": 1.5}[ig["discussion_level"]]
    instagram_heat = (math.log1p(ig["posts_72h"]) + 0.5 * math.log1p(ig["engagement_per_hour"])
                      + 0.3 * math.log1p(ig["comments_7d"]) + level
                      + 0.4 * math.log1p(ig["panel_max_over"]) + 0.25 * min(ig["panel_accounts"], 6))
    news_heat = math.log1p(topic["publishers"]) + 0.5 * math.log1p(len(topic["languages"]))
    x_hours = sum(x.get("hours") or 0 for x in topic.get("x_trends", []))
    search_heat = (math.log1p(topic["trends_traffic"] / 100) + 0.3 * math.log1p(topic["reddit_engagement"] / 50)
                   + 0.4 * math.log1p(x_hours))
    debate = (topic.get("brief") or {}).get("debate") or {}
    shares = sorted((float(s.get("share") or 0) for s in debate.get("sides", []) if isinstance(s, dict)), reverse=True)
    contested = bool(debate.get("is_debate"))
    balance = 1 - (shares[0] - shares[1]) if contested and len(shares) >= 2 else 0.0
    debate_score = (math.log1p(20 * ig["comment_like_ratio"]) + math.log1p(2 * ig["replies_per_comment"])
                    + 0.5 * math.log1p(ig["panel_max_ratio_lift"]) + 0.3 * math.log1p(ig["comments_7d"] / 100)
                    + (1.5 * balance + {"low": 0.3, "medium": 0.7, "high": 1.0}.get(debate.get("intensity"), 0) if contested else 0))
    return {"instagram_heat": round(instagram_heat, 3), "news_heat": round(news_heat, 3),
            "search_heat": round(search_heat, 3), "debate_score": round(debate_score, 3),
            "trend_score": round(0.55 * instagram_heat + 0.30 * news_heat + 0.15 * search_heat, 3)}


def topic_text(topic: dict) -> str:
    extra = (topic.get("brief") or {}).get("summary") or (topic["headlines"][0] if topic.get("headlines") else "")
    return f"{topic['title']}. {', '.join(topic.get('entities', []))}. {extra}"[:400]


def link(queries: list[dict], candidates: list[list[dict]], llm, batch_size: int = 30, extra: str = "") -> list[str | None]:
    """For each query topic pick the candidate that is the same story/event, or None (LLM-verified shortlist)."""
    answers: list[str | None] = [None] * len(queries)
    pending = [i for i, options in enumerate(candidates) if options]
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        items = [{"i": i, "topic": queries[i]["text"], "candidates": [{"id": c["id"], "topic": c["text"]} for c in candidates[i]]}
                 for i in chunk]
        try:
            result = llm.json(
                "For each topic, decide which candidate (if any) is the SAME ongoing story, event or discussion "
                "(wording, language and new developments of the same story count as the same; merely related topics "
                "do not: different matches or medals in one tournament, different incidents in one city, or different "
                'cases about one institution are NOT the same). ' + extra + 'Return {"links": [{"i": 0, "same_as": "<candidate id or null>"}]}\n\n'
                + json.dumps(items, ensure_ascii=False), SYSTEM, validate=lambda v: v["links"])
        except Exception as error:
            print(f"[link] failed: {type(error).__name__}", flush=True)
            continue
        for item in result:
            index, chosen = item.get("i"), item.get("same_as")
            if index in chunk and chosen in {c["id"] for c in candidates[index]}:
                answers[index] = chosen
    return answers


def shortlist(query_vectors: np.ndarray, pool_vectors: np.ndarray, pool: list[dict], k: int = 3, floor: float = 0.35):
    if not len(pool) or not len(query_vectors):
        return [[] for _ in range(len(query_vectors))]
    similarity = query_vectors @ pool_vectors.T
    return [[pool[j] for j in np.argsort(-row)[:k] if row[j] >= floor] for row in similarity]


def dedupe_native(native: list[dict], probed: list[dict], llm) -> list[dict]:
    """Fold Instagram-native clusters that duplicate a news/search topic into that topic's Instagram evidence."""
    if not native or not probed:
        return native
    pool = [{"id": t["topic_key"], "text": topic_text(t), "topic": t} for t in probed]
    queries = [{"text": topic_text(t)} for t in native]
    q_vecs, p_vecs = encoder.embed([q["text"] for q in queries]), encoder.embed([p["text"] for p in pool])
    sim = q_vecs @ p_vecs.T
    
    # Fast local deduplication: if cosine similarity >= 0.72, they are the same specific story
    local_links = [None] * len(queries)
    unresolved_queries = []
    unresolved_options = []
    unresolved_indices = []
    
    for i, q in enumerate(queries):
        best_j = int(np.argmax(sim[i]))
        if sim[i, best_j] >= 0.72:
            local_links[i] = pool[best_j]["id"]
            print(f"[dedupe_native] local encoder high-confidence match ({sim[i, best_j]:.3f}): '{q['text'][:40]}' -> '{pool[best_j]['text'][:40]}'", flush=True)
        elif sim[i, best_j] >= 0.35:
            # Ambiguous: pass to LLM shortlist
            candidates = [pool[j] for j in np.argsort(-sim[i])[:3] if sim[i, j] >= 0.35]
            unresolved_queries.append(q)
            unresolved_options.append(candidates)
            unresolved_indices.append(i)
            
    if unresolved_queries:
        decisions = link(unresolved_queries, unresolved_options, llm, extra="Answer null unless the topic is about the candidate's main event itself; "
                         "a different athlete, medal, match, person or incident is NOT the same even if the candidate's "
                         "headlines mention it. ")
        for idx, decision in zip(unresolved_indices, decisions):
            local_links[idx] = decision
            
    decisions = local_links
    by_key = {t["topic_key"]: t for t in probed}
    kept = []
    for topic, target in zip(native, decisions):
        if target:
            news_topic = by_key[target]
            media = news_topic.setdefault("instagram", {"media": []})["media"]
            known = {m["code"] for m in media}
            media += [m for m in topic["instagram"]["media"] if m["code"] not in known]
            news_topic["trend_queries"] = list(dict.fromkeys(news_topic["trend_queries"] + topic.get("trend_queries", [])))
            news_topic["x_trends"] = list({x["term"]: x for x in news_topic.get("x_trends", []) + topic.get("x_trends", [])}.values())
            news_topic.setdefault("kind", topic.get("kind"))
            print(f"[dedupe] discussion '{topic['title'][:50]}' -> '{news_topic['title'][:50]}'", flush=True)
        else:
            kept.append(topic)
    return kept


class History:
    """Topic identity across runs: embedding shortlist, then LLM decides whether it is the same story."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started_at TEXT, meta TEXT);
            CREATE TABLE IF NOT EXISTS topics (topic_id TEXT PRIMARY KEY, title TEXT, category TEXT,
                first_seen TEXT, last_seen TEXT, vector BLOB);
            CREATE TABLE IF NOT EXISTS observations (run_id TEXT, topic_id TEXT, at TEXT, trend_score REAL,
                instagram_heat REAL, news_heat REAL, payload TEXT, PRIMARY KEY (run_id, topic_id));
            CREATE TABLE IF NOT EXISTS account_baselines (account TEXT PRIMARY KEY, median_comments REAL,
                median_likes REAL, median_ratio REAL, sample_count INTEGER, last_updated TEXT);
        """)

    def load_baselines(self) -> dict[str, dict]:
        """Load stored engagement baselines per panel account."""
        rows = self.db.execute("SELECT account, median_comments, median_likes, median_ratio, sample_count, last_updated "
                               "FROM account_baselines").fetchall()
        return {r[0]: {"median_comments": r[1], "median_likes": r[2], "median_ratio": r[3],
                       "sample_count": r[4], "last_updated": r[5]} for r in rows}

    def update_baselines(self, feeds: dict[str, list[dict]], reference: datetime) -> None:
        """Update running engagement baselines per account using settled posts from feeds."""
        baselines = self.load_baselines()
        with self.db:
            for account, items in feeds.items():
                usable = [m for m in items if m.get("taken_at") and not (m.get("pinned") or m.get("paid") or m.get("counts_hidden"))
                          and m.get("like_count") is not None and m.get("comment_verified")]
                settled = [m for m in usable if (hours_since(m["taken_at"], reference) or 0) >= 6] or usable
                if len(settled) < 3:
                    continue
                cur_comments = median(m.get("comment_count") or 0 for m in settled)
                cur_likes = median(m.get("like_count") or 0 for m in settled)
                cur_ratio = median((m.get("comment_count") or 0) / ((m.get("like_count") or 0) + 1) for m in settled)
                count = len(settled)
                if account in baselines:
                    old = baselines[account]
                    old_n = old["sample_count"]
                    new_n = old_n + count
                    # Running weighted blend toward newer settled observations
                    blend = min(0.5, count / new_n)
                    new_comments = round((1 - blend) * old["median_comments"] + blend * cur_comments, 2)
                    new_likes = round((1 - blend) * old["median_likes"] + blend * cur_likes, 2)
                    new_ratio = round((1 - blend) * old["median_ratio"] + blend * cur_ratio, 4)
                    self.db.execute("UPDATE account_baselines SET median_comments=?, median_likes=?, median_ratio=?, "
                                    "sample_count=?, last_updated=? WHERE account=?",
                                    (new_comments, new_likes, new_ratio, new_n, reference.isoformat(), account))
                else:
                    self.db.execute("INSERT INTO account_baselines VALUES (?,?,?,?,?,?)",
                                    (account, round(cur_comments, 2), round(cur_likes, 2), round(cur_ratio, 4),
                                     count, reference.isoformat()))

    def match(self, topics: list[dict], reference: datetime, llm, lookback_days: int = 7) -> None:
        cutoff = (reference - timedelta(days=lookback_days)).isoformat()
        rows = self.db.execute("SELECT topic_id, title, first_seen, vector FROM topics WHERE last_seen >= ?", (cutoff,)).fetchall()
        pool = [{"id": r[0], "text": r[1], "first_seen": r[2]} for r in rows]
        pool_vectors = np.array([np.frombuffer(r[3], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1))
        vectors = encoder.embed([topic_text(t) for t in topics])
        decisions = link([{"text": topic_text(t)} for t in topics], shortlist(vectors, pool_vectors, pool), llm) if rows else [None] * len(topics)
        first_seen = {p["id"]: p["first_seen"] for p in pool}
        claimed = set()
        for topic, vector, best in zip(topics, vectors, decisions):
            topic["_vector"] = vector.astype(np.float32)
            topic["timeline"] = []
            if best and best not in claimed:
                claimed.add(best)
                topic["topic_id"], topic["first_seen"] = best, first_seen[best]
                topic["timeline"] = self.timeline(best)
                latest = topic["timeline"][-1] if topic["timeline"] else None
                topic["previous"] = ({"trend_score": latest["trend_score"], "instagram_heat": latest["instagram_heat"],
                                      "at": latest["at"], "hashtag_counts": latest["hashtag_counts"]} if latest else None)
            else:
                topic["topic_id"] = f"T{reference.strftime('%Y%m%d%H%M')}-{topic['topic_key']}"
                topic["first_seen"], topic["previous"] = reference.isoformat(), None

    def save(self, run_id: str, reference: datetime, topics: list[dict], meta: dict) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?)", (run_id, reference.isoformat(), json.dumps(meta)))
            for topic in topics:
                self.db.execute("INSERT INTO topics VALUES (?,?,?,?,?,?) ON CONFLICT(topic_id) DO UPDATE SET "
                                "last_seen=excluded.last_seen, title=excluded.title",
                                (topic["topic_id"], topic["title"], topic["category"], topic["first_seen"],
                                 reference.isoformat(), topic["_vector"].tobytes()))
                self.db.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?)",
                                (run_id, topic["topic_id"], reference.isoformat(), topic["scores"]["trend_score"],
                                 topic["scores"]["instagram_heat"], topic["scores"]["news_heat"],
                                 json.dumps({k: v for k, v in topic.items() if not k.startswith("_") and k != "instagram"},
                                            ensure_ascii=False, default=str)))

    def timeline(self, topic_id: str, limit: int = 200) -> list[dict]:
        """Earlier observations of one topic, oldest first: breakout stage, heat and press per run."""
        rows = self.db.execute("SELECT at, trend_score, instagram_heat, news_heat, payload FROM observations "
                               "WHERE topic_id=? ORDER BY at DESC LIMIT ?", (topic_id, limit)).fetchall()
        points = []
        for at, trend_score, instagram_heat, news_heat, payload in reversed(rows):
            data = json.loads(payload)
            points.append({"at": at, "trend_score": trend_score, "instagram_heat": instagram_heat,
                           "news_heat": news_heat, "stage": (data.get("breakout") or {}).get("stage"),
                           "publishers": data.get("publishers"),
                           "hashtag_counts": (data.get("instagram_metrics") or {}).get("hashtag_counts", {})})
        return points

    def runs(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def forget(self, run_id: str) -> None:
        """Drop a run's own rows so re-measuring it does not compare the run with itself."""
        with self.db:
            self.db.execute("DELETE FROM observations WHERE run_id=?", (run_id,))
            self.db.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
            self.db.execute("DELETE FROM topics WHERE topic_id NOT IN (SELECT DISTINCT topic_id FROM observations)")


def hashtag_growth(topic: dict, reference: datetime) -> float | None:
    """New posts per hour on the topic hashtag since the previous run (Instagram's own total counter)."""
    previous = topic.get("previous") or {}
    hours = hours_since(previous.get("at"), reference)
    share = topic["instagram_metrics"].get("hashtag_on_topic_share")
    if share is None or share < 0.5:
        return None
    rates = [(count - previous["hashtag_counts"][tag]) / hours
             for tag, count in topic["instagram_metrics"]["hashtag_counts"].items()
             if hours and hours >= 1 and isinstance(count, int) and isinstance(previous.get("hashtag_counts", {}).get(tag), int)]
    return round(max(rates), 2) if rates else None


def status(topic: dict, reference: datetime) -> str:
    previous = topic.get("previous")
    ig = topic["instagram_metrics"]
    first_age = hours_since(topic["first_seen"], reference) or 0
    news_age = hours_since(topic.get("earliest"), reference)
    topic["hashtag_posts_per_hour"] = hashtag_growth(topic, reference)
    fresh_on_instagram = ig["posts_24h"] >= 5 and ig["recency_ratio"] >= 0.6
    fresh_in_news = news_age is None or news_age <= 36
    if (topic.get("breakout") or {}).get("stage") == "settled":
        before = (previous or {}).get("trend_score") or 0.01
        return "fading" if previous and topic["scores"]["trend_score"] <= 0.7 * before else "settled"
    if previous:
        before, current = previous["trend_score"] or 0.01, topic["scores"]["trend_score"]
        growing_tag = (topic["hashtag_posts_per_hour"] or 0) >= 5
        substantial = ig["discussion_level"] != "none" and current >= 3.0
        if substantial and (current >= 1.3 * before or (growing_tag and current >= before)):
            return "rising"
        if current <= 0.7 * before:
            return "fading"
        return "emerging" if first_age <= 24 and fresh_on_instagram and fresh_in_news else "sustained"
    return "emerging" if fresh_on_instagram and fresh_in_news else "new"


# ---------------------------------------------------------------- report

def render(report: dict) -> str:
    lines = [f"# India Trend Report - {report['generated_at'][:16].replace('T', ' ')} UTC", "",
             f"Run `{report['run_id']}`. Signals: {report['signals']} items from {report['sources_ok']} feeds "
             f"({report['sources_failed']} unavailable). Topics planned from news/search: {report['topics_planned']}; "
             f"measured on Instagram: {report['topics_probed']} via `{report['instagram_transport']}`. "
             f"Instagram panel: {report.get('panel_accounts_read', 0)} accounts read; trending terms searched: "
             f"{report.get('trend_terms_searched', 0)}; Instagram-first discussions found: {report['native_topics']}. "
             f"Prior runs in history: {report['history_runs'] - 1}. "
             f"LLM: {report['llm']['calls']} calls, {report['llm']['output_tokens']:,} output tokens. "
             "Reddit discussion is shown as a separate proxy and is not Instagram measurement.", ""]
    if report["history_runs"] <= 1:
        lines += ["> First run: 'emerging' uses within-run evidence only (5+ on-topic Instagram posts in 24 h making up "
                  "60%+ of the past week's posts, and news no older than 36 h). Run every 3-6 h to get rising/fading.", ""]
    lines += ["Status: **emerging** = new and concentrated in the last 24 h on Instagram; **rising** = trend score up 30%+ "
              "or topic hashtag growing 5+ posts/h since the last run; **sustained** / **fading** = flat / down 30%+; "
              "**new** = first seen this run without a 24 h concentration; **settled** = its posts have gone quiet, so it is no longer a live trend.", "",
              "Stage (from the topic's own posts): **pre breakout** = posts under 12 h over-performing their accounts, "
              "1-2 accounts, no press; **breaking out** = 3+ accounts posted in the last 6 h; **peak** = 8+ accounts or "
              "3+ news publishers; **saturated** = 5+ publishers; **settled** = no new post in 24 h; **simmering** = "
              "active without early-acceleration evidence; **unobserved** = insufficient Instagram evidence. "
              "Reddit proxy activity does not change these stages.", ""]

    def block(topic):
        ig, brief_ = topic["instagram_metrics"], topic.get("brief") or {}
        traj = f" | stage: **{topic['trajectory'].replace('_', ' ')}**" if topic.get("trajectory") else ""
        out = [f"### {topic['title']}",
               f"`{topic['category']}`" + (f" `{topic['kind']}`" if topic.get("kind") else "")
               + f" | **{topic['status']}**{traj} | trend score {topic['scores']['trend_score']} "
               f"(Instagram {topic['scores']['instagram_heat']}, news {topic['scores']['news_heat']}, search {topic['scores']['search_heat']}) "
               f"| Instagram discussion: {ig['discussion_level']}"]
        if brief_.get("summary"):
            out.append(f"\n{brief_['summary']}")
        if brief_.get("why_trending"):
            out.append(f"\n*Why now:* {brief_['why_trending']}")
        reddit_proxy = topic.get("reddit_proxy") or {}
        if reddit_proxy.get("thread_count"):
            out.append("\n*Reddit proxy (observed Reddit data; not Instagram):* "
                       f"{reddit_proxy['thread_count']} threads across {reddit_proxy['subreddit_count']} subreddits, "
                       f"{reddit_proxy['upvotes']:,} post upvotes, {reddit_proxy['post_comments']:,} post comments, "
                       f"and {reddit_proxy['sampled_comments']} sampled comments.")
            for excerpt in reddit_proxy.get("comment_excerpts", [])[:3]:
                out.append(f"> {excerpt['text']} ({excerpt['subreddit']}, score {excerpt['score']})")
        if brief_.get("what_people_say"):
            out.append("\n*What people are saying:*")
            out += [f"- {point}" for point in brief_["what_people_say"]]
        stance = brief_.get("stance") or {}
        measured = brief_.get("measured_opinion")
        breakout = topic.get("breakout") or {}
        if breakout.get("reasons") and breakout.get("stage") in ("pre_breakout", "breaking_out"):
            out.append(f"\n*Breakout signals (score {breakout['score']}):* " + "; ".join(breakout["reasons"]) + ".")
        stage_log = topic.get("stage_history") or {}
        if stage_log.get("runs_observed", 0) > 1 or stage_log.get("transitions"):
            steps = [f"{t['from'].replace('_', ' ')} to {t['to'].replace('_', ' ')} at {t['at'][:16].replace('T', ' ')} UTC"
                     for t in stage_log.get("transitions", [])]
            current = (stage_log.get("stage") or "unknown").replace("_", " ")
            out.append(f"\n*Stage history:* {'; '.join(steps) or 'no stage change'}; in {current} for "
                       f"{stage_log.get('stage_hours')} h across {stage_log.get('runs_observed')} observations.")
        if measured:
            tones = measured["tone"]
            # Visual ASCII sentiment breakdown bar
            w = 16
            tot = sum(tones.values()) or 1.0
            s_bar = ("█" * round((tones.get("supportive", 0) / tot) * w) +
                     "░" * round((tones.get("critical", 0) / tot) * w) +
                     "▒" * round((tones.get("humorous", 0) / tot) * w) +
                     "·" * round((tones.get("neutral", 0) / tot) * w))[:w]
            if len(s_bar) < w:
                s_bar += "·" * (w - len(s_bar))
            coverage = ig.get("comment_coverage")
            cover_text = (f"; {coverage:.0%} of {ig['comments_platform']:,} posted comments fetched"
                          if coverage is not None else "")
            out.append(f"\n*Opinion in comments ({measured['comments']} comments labelled{cover_text}, like-weighted"
                       + (f", {measured['sarcastic']} sarcastic" if measured["sarcastic"] else "")
                       + f", labelled by {measured.get('source', 'the LLM brief')}):* "
                       + ", ".join(f"{k} {round(v * 100)}%" for k, v in measured["tone"].items())
                       + f"\n  Sentiment balance: `[{s_bar}]` (█ Sup / ░ Crit / ▒ Hum / · Neut)")
        elif stance:
            basis = ("overall LLM estimate" if ig.get("sampled_comments")
                     else "estimated from captions only; no comments were sampled")
            out.append(f"\n*Stance ({basis}):* "
                       + ", ".join(f"{k} {round(float(v) * 100)}%" for k, v in stance.items() if isinstance(v, (int, float))))
        debate = brief_.get("debate") or {}
        if debate.get("is_debate") and debate.get("question"):
            out.append(f"\n*Debate ({debate.get('intensity', '?')} intensity, debate score {topic['scores']['debate_score']}):* "
                       f"{debate['question']}")
            if measured and any(s["comments"] for s in measured["sides"]):
                for side in measured["sides"][:3]:
                    sh = round(side['share'] * 100)
                    side_bar = "■" * round(sh / 10)
                    out.append(f"- {sh}% `[{side_bar:<10}]` ({side['comments']} comments) {side['position']}"
                               + (f' - top comment: "{side["top_comment"]}" ({side["top_comment_likes"] or 0:,} likes)'
                                  if side.get("top_comment") else ""))
                if measured.get("unaligned_share"):
                    out.append(f"- {round(measured['unaligned_share'] * 100)}% of comment weight takes no side")
            else:
                for side in debate.get("sides", [])[:3]:
                    if isinstance(side, dict) and side.get("position"):
                        share = side.get("share")
                        out.append(f"- ~{round(float(share) * 100)}% (estimate) " if isinstance(share, (int, float)) else "- ")
                        out[-1] += side["position"] + (f' - e.g. "{side["example"]}"' if side.get("example") else "")
        lead = topic.get("lead") or {}
        pickup_note = (f"; third outlet picked it up {lead['lead_vs_pickup_hours']} h after the first post"
                       if lead.get("lead_vs_pickup_hours") is not None else "")
        timing = {"instagram_first": f"Instagram first, {lead.get('lead_vs_news_hours')} h before the earliest news item{pickup_note}",
                  "news_first": f"news first, {abs(lead.get('lead_vs_news_hours') or 0)} h before the first Instagram post",
                  "simultaneous": "news and Instagram within an hour of each other",
                  "instagram_only": "Instagram only; no news item in the 48 h window"}.get(lead.get("verdict"))
        if timing:
            out.append(f"\n*Timing:* {timing}.")
        bot = topic.get("astroturf")
        if bot:
            out.append(f"\n*Bot check:* {bot['verdict_summary']} Astroturf index {bot['astroturf_index']}/100 over "
                       f"{bot['total_comments_audited']} comments.")
        if brief_.get("caution"):
            out.append(f"\n> Caution: {brief_['caution']}")
        panel = (f"; Instagram panel: {ig['panel_accounts']} accounts, top post {ig['panel_max_over']}x its account's usual "
                 f"comments, comment:like {ig['comment_like_ratio']}, {ig['replies_per_comment']} replies per sampled comment"
                 if ig.get("panel_accounts") else "")
        x_terms = ", ".join(f"{x['term']} ({x['hours']} h)" for x in topic.get("x_trends", [])[:3])
        out.append(f"\n*Evidence:* Instagram {ig['on_topic_posts']} on-topic posts ({ig['posts_24h']} in 24 h, {ig['posts_72h']} in 72 h, "
                   f"{ig['authors_7d']} authors/7 d), {ig['comments_7d']:,} comments, {ig['engagement_per_hour']:,} engagement/h"
                   + panel
                   + "".join(f"; #{tag} {count:,} posts total" for tag, count in ig.get("hashtag_counts", {}).items() if isinstance(count, int))
                   + (f" (+{topic['hashtag_posts_per_hour']}/h since last run)" if topic.get("hashtag_posts_per_hour") is not None else "")
                   + (f"; trending on X: {x_terms}" if x_terms else "")
                   + f"; news {topic['publishers']} publishers in {len(topic['languages'])} languages"
                   + (f"; Google searches {topic['trends_traffic']:,}+" if topic["trends_traffic"] else "") + ".")
        if (topic.get("judge") or {}).get("verdict") == "fail":
            out.append(f"*Audit:* flagged - {topic['judge'].get('reason', '')}")
        if topic["headlines"]:
            out.append(f"*Headlines:* " + " / ".join(topic["headlines"][:3]))
        if ig["top_posts"]:
            out.append("*Top posts:* " + " ".join(ig["top_posts"]))
        return out + [""]

    def flagged(topic):
        audit = topic.get("judge") or {}
        return audit.get("coherent") is False or audit.get("india_relevant") is False

    level = lambda t: t["instagram_metrics"]["discussion_level"]
    measured = [t for t in report["topics"] if t.get("brief")]
    clean = [t for t in measured if not flagged(t)]
    debated = sorted([t for t in clean if (t["brief"].get("debate") or {}).get("is_debate") and level(t) != "none"],
                     key=lambda t: t["scores"]["debate_score"], reverse=True)
    early = sorted([t for t in clean if (t.get("breakout") or {}).get("stage") in ("pre_breakout", "breaking_out")],
                   key=lambda t: t["breakout"]["score"], reverse=True)
    sections = [
        ("About to break out (young, accelerating, little or no press yet)", early[:10]),
        ("Most debated right now (people disagreeing in comments)", debated[:10]),
        ("Emerging and rising", [t for t in clean if t["status"] in ("emerging", "rising") and level(t) != "none"][:15]),
        ("Trending now on Instagram India", [t for t in clean if level(t) in ("medium", "high")][:15]),
        ("Instagram-first discussions (found via the account panel / X trends, not news)",
         [t for t in clean if t["origin"] in ("instagram_discussion", "instagram_native")][:15]),
        ("In the news, little Instagram discussion yet (watchlist)", [t for t in clean if level(t) in ("none", "low")
                                                                       and t["origin"] == "news_and_search"][:10]),
        ("Flagged by audit (incoherent or not India-relevant)", [t for t in measured if flagged(t)][:15]),
    ]
    shown = set()
    for heading, topics in sections:
        topics = [t for t in topics if t["topic_id"] not in shown]
        if not topics:
            continue
        lines += [f"## {heading}", ""]
        for topic in topics:
            shown.add(topic["topic_id"])
            lines += block(topic)

    # Sentiment & Opinion Summary Table with visual bars across all measured topics
    measured_opinions = [t for t in measured if (t.get("brief") or {}).get("measured_opinion")]
    if measured_opinions:
        lines += ["## Public Opinion & Sentiment Landscape", "",
                  "Summary of like-weighted commenter sentiment across all topics measured with neural stance and tone models.",
                  "",
                  "| Topic | Category | Sample | Sup | Crit | Hum | Neut | Sentiment Distribution | Debate / Stance |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"]
        for t in measured_opinions:
            op = t["brief"]["measured_opinion"]
            tones = op.get("tone", {})
            sup = round(tones.get("supportive", 0) * 100)
            crit = round(tones.get("critical", 0) * 100)
            hum = round(tones.get("humorous", 0) * 100)
            neut = round(tones.get("neutral", 0) * 100)
            w = 14
            tot = (sup + crit + hum + neut) or 100
            bar = ("█" * round(sup / tot * w) +
                   "░" * round(crit / tot * w) +
                   "▒" * round(hum / tot * w) +
                   "·" * round(neut / tot * w))[:w]
            if len(bar) < w:
                bar += "·" * (w - len(bar))
            debate = t["brief"].get("debate") or {}
            d_desc = f"{debate.get('intensity', '').title()} debate" if debate.get("is_debate") else "Consensus"
            topic_title = t['title'].replace("|", "/").strip()[:34]
            lines.append(f"| {topic_title} | `{t.get('category', 'other')}` | {op['comments']} | "
                         f"{sup}% | {crit}% | {hum}% | {neut}% | `[{bar}]` | {d_desc} |")
        lines += ["", "*Legend: `█` Supportive | `░` Critical | `▒` Humorous / Sarcastic | `·` Neutral*", ""]

    graph = report.get("account_graph") or {}
    if graph.get("accounts"):
        lines += ["## Accounts active across topics (possible amplifiers)", "",
                  f"{graph['cross_topic_accounts']} accounts posted or commented in 2+ measured topics. Coordination "
                  "candidates share 3+ topics with at least 80% overlap of their topic sets.", "",
                  "| Account | Topics | Posts | Comments | Topic titles |", "| --- | ---: | ---: | ---: | --- |"]
        for row in graph["accounts"][:15]:
            titles = " / ".join(title.replace("|", "/")[:50] for title in row["topics"][:3])
            lines.append(f"| {row['label']} | {row['topic_count']} | {row['posts']} | {row['comments']} | {titles} |")
        if graph.get("coordination_pairs"):
            lines += ["", "Coordination candidates:", ""]
            lines += [f"- {p['accounts'][0]} and {p['accounts'][1]} share {p['shared_topics']} topics (Jaccard {p['jaccard']})"
                      for p in graph["coordination_pairs"][:10]]
        lines.append("")
    unprobed = [t for t in report["topics"] if not t.get("brief")]
    if report.get("panel_pulse"):
        lines += ["## Panel pulse: posts drawing unusual attention for their account", "",
                  "| Page type | Age | Comments (x usual) | Comment:like lift | Post | Most-liked comment |", "| --- | --- | --- | --- | --- | --- |"]
        for row in report["panel_pulse"]:
            caption = row["caption"].replace("|", "/")
            comment = (row["top_comment"] or "").replace("|", "/")
            lines.append(f"| {row['group']} | {row['age_hours']} h | {row['comments'] or 0:,} ({row['over_comments']}x) | "
                         f"{row['ratio_lift']}x | [{caption[:90]}]({row['url']}) | "
                         + (f"{comment[:90]} ({row['top_comment_likes'] or 0:,} likes)" if comment else "-") + " |")
        lines.append("")
    if unprobed:
        lines += ["## Other India topics in news/search (not measured on Instagram this run)", ""]
        lines += [f"- {t['title']} (`{t['category']}`, {t['publishers']} publishers)" for t in unprobed[:40]]
    categories = Counter(t["category"] for t in measured)
    lines += ["", "## Coverage", "", "| Category | Measured topics |", "| --- | --- |"]
    lines += [f"| {c} | {n} |" for c, n in categories.most_common()]
    if report.get("quality"):
        q = report["quality"]
        lines += ["", "## Automated audit (LLM judge, same model family - indicative only)", "",
                  f"Audited {q['audited']} reported topics: coherent {q['coherent_rate']:.0%}, India-relevant {q['india_relevant_rate']:.0%}, "
                  f"summary faithful to evidence {q['faithful_rate']:.0%}, pass {q['pass_rate']:.0%}. "
                  "Topics failing coherence or India relevance are listed under 'Flagged by audit', not in the main sections."]
    lines += ["", "Method and caveats: Instagram counts come from keyword-search samples, not a census; 'emerging' is relative to "
              "this system's observations. Summaries are LLM-generated from the listed posts/comments and are not fact checks. "
              "Timing compares the first sampled Instagram post with the earliest news item in the 48 h window: the sample "
              "can understate Instagram's lead, and the window can cap the news start. Bot checks are heuristic and indicative only."]
    return "\n".join(lines) + "\n"


def reddit_proxy_for_topic(topic: dict, posts: list[dict]) -> dict:
    """Summarize directly matched Reddit evidence without imputing Instagram metrics."""
    stopwords = {
        "about", "after", "again", "also", "among", "been", "before", "being", "between", "could",
        "from", "have", "into", "just", "more", "most", "other", "over", "said", "says", "such",
        "than", "that", "their", "there", "these", "they", "this", "those", "through", "under",
        "were", "what", "when", "where", "which", "while", "with", "would", "your", "india", "news",
        "today", "delhi", "police", "people", "case", "years", "first", "time",
    }

    def normalized(value: str) -> str:
        return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))

    def tokens(value: str) -> set[str]:
        return {word for word in re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
                if len(word) >= 4 and word not in stopwords and not word.isdigit()}

    reference_texts = list(dict.fromkeys(
        value.strip() for value in [topic.get("title", ""), *(topic.get("headlines") or []),
                                    *(topic.get("entities") or [])]
        if isinstance(value, str) and value.strip()
    ))
    reference_normalized = {normalized(value) for value in reference_texts}
    reference_tokens = [tokens(value) for value in reference_texts]
    matched = []
    for post in posts:
        title = str(post.get("title") or "")
        post_normalized = normalized(title)
        post_tokens = tokens(title)
        exact = bool(post_normalized and post_normalized in reference_normalized)
        best_overlap = 0
        best_ratio = 0.0
        if not exact and post_tokens:
            for words in reference_tokens:
                overlap = len(words & post_tokens)
                ratio = overlap / min(len(words), len(post_tokens)) if words else 0.0
                if overlap > best_overlap or (overlap == best_overlap and ratio > best_ratio):
                    best_overlap, best_ratio = overlap, ratio
        if not exact and (best_overlap < 3 or best_ratio < 0.5):
            continue
        engagement = post.get("engagement") or {}
        feed = str(post.get("feed") or "unknown")
        subreddit = feed.rsplit("/", 1)[0] if "/" in feed else feed
        comments = [comment for comment in post.get("comments", [])
                    if isinstance(comment, dict) and str(comment.get("body") or "").strip()]
        matched.append({
            "title": title,
            "feed": feed,
            "subreddit": subreddit,
            "score": int(engagement.get("score") or 0),
            "comments": int(engagement.get("comments") or 0),
            "sampled_comments": len(comments),
            "published_at": post.get("published_at"),
            "url": post.get("url"),
            "match": "exact_headline" if exact else "strong_title_overlap",
            "comment_excerpts": [
                {"text": re.sub(r"\s+", " ", str(comment.get("body")))[:280],
                 "score": int(comment.get("score") or 0), "subreddit": subreddit}
                for comment in sorted(comments, key=lambda item: int(item.get("score") or 0), reverse=True)[:2]
            ],
        })
    matched.sort(key=lambda item: (item["score"] + 2 * item["comments"], item["published_at"] or ""), reverse=True)
    excerpts = [excerpt for post in matched for excerpt in post["comment_excerpts"]][:5]
    return {
        "source": "saved_reddit_signals",
        "label": "Reddit proxy; not Instagram measurement",
        "thread_count": len(matched),
        "subreddit_count": len({item["subreddit"] for item in matched}),
        "upvotes": sum(item["score"] for item in matched),
        "post_comments": sum(item["comments"] for item in matched),
        "sampled_comments": sum(item["sampled_comments"] for item in matched),
        "threads": matched[:8],
        "comment_excerpts": excerpts,
    }


# ---------------------------------------------------------------- orchestration

JUDGE_SYSTEM = ("You are a strict auditor of automated trend reports. You did not write the report. Judge only from the "
                "evidence shown. Output only strict JSON.")


def judge(topics: list[dict], llm, batch_size: int = 10) -> dict | None:
    """Independent audit pass: is each reported topic coherent, India-relevant and backed by its Instagram evidence?"""
    audited = [t for t in topics if t.get("brief")]
    if not audited:
        return None

    def run(batch):
        items = []
        for index, topic in enumerate(batch):
            media = {m["code"]: m for m in topic.get("instagram", {}).get("media", [])}
            captions = [scrub(media[c]["caption"])[:200] for c in topic["brief"].get("on_topic_codes", [])[:4] if c in media]
            items.append({"i": index, "title": topic["title"], "summary": topic["brief"].get("summary"),
                          "what_people_say": topic["brief"].get("what_people_say"),
                          "on_topic_captions": captions, "headlines": topic["headlines"][:2]})
        result = llm.json(
            "Audit each reported trend. coherent: the summary and the on-topic captions are about one specific story or "
            "event (not a mix of different events). Headlines come from automatic news clustering and may include a stray "
            "unrelated one: ignore a single stray headline for coherence and mention it in the reason instead. "
            "india_relevant: people in India are plausibly discussing it "
            "(Indian subject, audience or angle). faithful: the summary and 'what people say' are supported by the "
            "captions/headlines shown. verdict pass only if all three are true.\n"
            'Return {"audits": [{"i": 0, "coherent": true, "india_relevant": true, "faithful": true, '
            '"verdict": "pass", "reason": "<= 20 words"}]}\n\n' + json.dumps(items, ensure_ascii=False),
            JUDGE_SYSTEM, validate=lambda v: v["audits"])
        return list(zip(batch, result))

    rows = []
    for pairs in llm.map(run, [audited[i:i + batch_size] for i in range(0, len(audited), batch_size)]):
        for topic, audit in pairs or []:
            topic["judge"] = audit
            rows.append(audit)
    if not rows:
        return None
    rate = lambda key, value=True: sum(r.get(key) == value for r in rows) / len(rows)
    return {"audited": len(rows), "coherent_rate": rate("coherent"), "india_relevant_rate": rate("india_relevant"),
            "faithful_rate": rate("faithful"), "pass_rate": rate("verdict", "pass")}

def make_transport(kind: str, budget: int):
    if kind != "hiker":
        raise ValueError(f"Unsupported Instagram transport: {kind}")
    from identify_what_matters.instagram_access import HikerTransport
    return HikerTransport(request_budget=budget)


def plan_from_signals(signals: list[dict], availability: dict, run_dir: Path, llm, reference: datetime,
                      max_topics: int, output_root: Path | None = None) -> dict:
    stories = build_stories(signals, reference)
    news_topics = plan_topics(stories, llm)

    # Phase 1: Native Sentinel Radar execution
    sentinel_topics = []
    if output_root:
        sentinel_topics = plan_sentinel_topics(output_root, llm, reference)
    elif run_dir.parent.exists():
        sentinel_topics = plan_sentinel_topics(run_dir.parent.parent, llm, reference)

    # Combine news topics and native sentinel flame topics
    all_topics = sentinel_topics + news_topics
    probe_set = select_for_probe(all_topics, reference, max_topics)
    plan = {"run_id": run_dir.name, "planned_at": reference.isoformat(), "availability": availability,
            "signals": len(signals), "stories": len(stories),
            "sentinel_topics": len(sentinel_topics),
            "topics": all_topics,
            "probe_keys": [t["topic_key"] for t in probe_set],
            "probe_queries": [{"key": t["topic_key"], "queries": t["instagram_queries"][:3], "hashtags": t.get("hashtags", [])[:1]}
                              for t in probe_set]}
    write(run_dir / "signals.json", signals)
    write(run_dir / "stories.json", stories)
    write(run_dir / "plan.json", plan)
    return plan


def plan_from_discovery_shortlist(discovery: dict, run_dir: Path, reference: datetime, max_topics: int) -> dict:
    """Build Instagram probes from ranked discovery candidates without LLM topic synthesis."""
    signals = discovery.get("signals", [])
    candidates = discovery.get("candidates", [])
    candidate_by_id = {str(item.get("story_id")): item for item in candidates if item.get("story_id")}
    shortlist = discovery.get("probe_candidate_ids", [])
    if not shortlist:
        shortlist = [item.get("story_id") for item in candidates
                     if item.get("prefilter", {}).get("recommended_for_instagram_probe")]
    signal_by_id = {str(item.get("id")): item for item in signals if item.get("id")}
    stopwords = {"about", "after", "against", "also", "and", "are", "been", "before", "between", "from",
                 "have", "into", "just", "more", "most", "news", "over", "said", "says", "that", "their",
                 "there", "these", "they", "this", "those", "through", "under", "were", "what", "when",
                 "where", "which", "while", "with", "would", "your"}

    def concise_query(value: str) -> str:
        words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
        salient = [word for word in words if len(word) >= 3 and not word.isdigit() and word.casefold() not in stopwords]
        return " ".join(salient[:5])

    topics = []
    for story_id in shortlist:
        candidate = candidate_by_id.get(str(story_id))
        if not candidate:
            continue
        titles = [title.strip() for title in candidate.get("titles", []) if isinstance(title, str) and title.strip()]
        evidence = [signal_by_id.get(str(signal_id), {}) for signal_id in candidate.get("signals", [])]
        reddit_titles = [item.get("title", "") for item in evidence if item.get("source") == "reddit"]
        queries = [query.strip().lstrip("#") for query in candidate.get("trend_queries", [])
                   if isinstance(query, str) and query.strip()]
        queries.extend(concise_query(title) for title in reddit_titles + titles)
        queries = list(dict.fromkeys(query for query in queries if len(query) >= 3))[:3]
        if not titles or not queries:
            continue
        topic_key = f"discovery_{candidate['story_id']}"
        tags = list(dict.fromkeys(re.findall(r"#([\w]{3,30})", " ".join(titles))))[:2]
        entities = list(dict.fromkeys(
            match.strip() for title in titles[:2]
            for match in re.findall(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){1,2}", title)
        ))[:6]
        topics.append({
            "topic_key": topic_key,
            "origin": "discovery_shortlist_local_queries",
            "title": titles[0][:160],
            "category": "other",
            "entities": entities,
            "hashtags": tags,
            "instagram_queries": queries,
            "headlines": titles[:6],
            "publishers": candidate.get("publishers", 0),
            "languages": candidate.get("languages", []),
            "trends_traffic": candidate.get("trends_traffic", 0),
            "trend_queries": candidate.get("trend_queries", []),
            "social_feed_hits": candidate.get("social_feed_hits", 0),
            "reddit_engagement": candidate.get("reddit_engagement", 0),
            "earliest": candidate.get("earliest"),
            "latest": candidate.get("latest"),
            "discovery_prefilter": candidate.get("prefilter", {}),
        })
        if len(topics) >= max_topics:
            break

    probe_keys = [topic["topic_key"] for topic in topics]
    plan = {
        "run_id": run_dir.name,
        "planned_at": reference.isoformat(),
        "availability": discovery.get("availability", {}),
        "signals": len(signals),
        "stories": len(candidates),
        "topics": topics,
        "probe_keys": probe_keys,
        "probe_queries": [{"key": topic["topic_key"], "queries": topic["instagram_queries"],
                           "hashtags": topic["hashtags"][:1]} for topic in topics],
        "planning_mode": "ranked_discovery_shortlist_local_queries",
    }
    write(run_dir / "signals.json", signals)
    write(run_dir / "stories.json", candidates)
    write(run_dir / "plan.json", plan)
    return plan


def run_plan(args, run_dir: Path, llm) -> dict:
    if args.discovery_file:
        discovery_path = Path(args.discovery_file)
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        signals = discovery.get("signals")
        availability = discovery.get("availability", {})
        if not isinstance(signals, list) or not isinstance(availability, dict):
            raise ValueError(f"Invalid discovery artifact: expected signals list and availability object in {discovery_path}")
        if args.use_discovery_shortlist:
            return plan_from_discovery_shortlist(discovery, run_dir, now(), args.max_topics)
    else:
        if args.use_discovery_shortlist:
            raise ValueError("--use-discovery-shortlist requires --discovery-file")
        from identify_what_matters.sources import collect_signals
        signals, availability = collect_signals(include_reddit=not args.no_reddit)
    output_root = Path(args.output_root) if hasattr(args, "output_root") else run_dir.parent
    return plan_from_signals(signals, availability, run_dir, llm, now(), args.max_topics, output_root=output_root)


def run_collect(args, run_dir: Path) -> dict:
    """All network collection that needs no LLM: public signals (X trends, Reddit) and every Instagram request."""
    from identify_what_matters.discussions import collect_instagram, load_panel, select_trend_terms
    from identify_what_matters.sources import collect_signals
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"topics": [], "probe_keys": [], "probe_queries": []}
    if plan_path.exists():
        plan["probe_keys"] = [t["topic_key"] for t in plan["topics"]
                              if t.get("category") in POLITICS_ECONOMY_CATEGORIES][:args.max_topics]
        write(plan_path, plan)
    if getattr(args, "skip_public_sources", False):
        signals, availability = [], {}
    else:
        signals, availability = collect_signals(include_reddit=not args.no_reddit, include_google=not plan_path.exists())
    planned = json.loads((run_dir / "signals.json").read_text()) if (run_dir / "signals.json").exists() else []
    exclude = {q for p in plan.get("probe_queries", []) for q in p["queries"]}
    terms = select_trend_terms(planned + signals, exclude, args.trend_terms)
    print(f"[collect] {len(signals)} signals ({sum(not isinstance(v, int) for v in availability.values())} sources unavailable); "
          f"{len(terms)} trending terms; {len(plan.get('probe_keys', []))} planned topics to probe", flush=True)
    transport = None if args.no_instagram else make_transport(args.instagram, args.request_budget)
    if transport:
        panel = load_panel(args.panel)
        try:
            bundle = asyncio.run(collect_instagram(plan, transport, parse_time(transport.captured_at), panel, terms,
                                                  comment_posts=getattr(args, "comment_posts", 8),
                                                  panel_total_budget=getattr(args, "panel_total_budget", 800),
                                                  deep_comment_pages=getattr(args, "comment_pages", 4),
                                                  deep_reply_threads=getattr(args, "reply_threads", 6),
                                                  score_topics=getattr(args, "score_topics", 5),
                                                  score_posts=getattr(args, "score_posts", 8),
                                                  score_comment_pages=getattr(args, "score_comment_pages", 10)))
        finally:
            # Persist the request cache even when the run is interrupted, so paid responses are not lost.
            transport.close(run_dir / "instagram_capture.json")
        from identify_what_matters.account_discovery import record_run_observations, write_proposals as write_account_proposals
        record_run_observations(bundle, panel, run_dir.name)
        write_account_proposals()
    else:
        bundle = {"feeds": {}, "terms": {}, "probes": {}, "comments": {}, "stopped": False, "missing_accounts": []}
    collected = {"captured_at": (transport.captured_at if transport else now().isoformat()),
                 "transport": args.instagram if transport else "none",
                 "instagram_requests": transport.used if transport else 0,
                 "stopped": bundle["stopped"], "missing_accounts": bundle["missing_accounts"],
                 "availability": availability, "trend_terms": terms, "signals": signals}
    write(run_dir / "collect.json", collected)
    instagram_status = "Instagram skipped" if transport is None else f"{transport.used} Instagram requests"
    print(f"[collect] {instagram_status}; panel {len(bundle['feeds'])} accounts "
          f"(missing: {', '.join(bundle['missing_accounts']) or 'none'}); {len(bundle['terms'])} terms; "
          f"{len(bundle['probes'])} probes; {len(bundle['comments'])} comment threads -> {run_dir}", flush=True)
    return collected


def run_measure(args, run_dir: Path, llm) -> dict:
    """Analysis on a collected run (replayed from its capture, no Instagram requests): discussions, briefs, report."""
    from identify_what_matters.discussions import collect_instagram, discover_discussions, load_panel
    from identify_what_matters.instagram_access import ReplayTransport
    import time
    started = time.monotonic()

    def step(message: str) -> None:
        print(f"[measure] {message} (+{time.monotonic() - started:.0f}s)", flush=True)

    step(f"start: {run_dir.name}")
    collect_path = run_dir / "collect.json"
    collected = json.loads(collect_path.read_text()) if collect_path.exists() else {"availability": {}, "signals": [], "trend_terms": []}
    if collected.get("transport") == "none":
        raise RuntimeError("This run is public-sources-only; Instagram measurement requires a real capture.")
    if (run_dir / "plan.json").exists():
        plan = json.loads((run_dir / "plan.json").read_text())
    else:
        reference = parse_time(collected.get("captured_at")) or now()
        plan = plan_from_signals(collected["signals"], collected["availability"], run_dir, llm, reference, args.max_topics)
        plan["probe_keys"], plan["probe_queries"] = [], []
        write(run_dir / "plan.json", plan)
    capture = Path(args.capture or run_dir / "instagram_capture.json")
    transport = ReplayTransport(capture) if capture.exists() else None
    reference = parse_time(getattr(transport, "captured_at", None)) or now()
    topics = plan["topics"]
    probed = [t for t in topics if t["topic_key"] in set(plan["probe_keys"])]
    metrics_only = getattr(args, "metrics_only", False)
    native, bundle = [], {"feeds": {}, "terms": {}, "comments": {}}
    history = History(Path(args.output_root) / "trends.sqlite")
    if not transport and (run_dir / "instagram.json").exists():
        stored_instagram = json.loads((run_dir / "instagram.json").read_text())
        for topic in probed:
            topic["instagram"] = stored_instagram.get(topic["topic_key"]) or {"media": [], "queries_used": [], "hashtag_counts": {}}
    if transport:
        step(f"replaying saved capture: {len(transport.feeds)} feeds, {len(transport.comment_map)} comment threads")
        baselines = history.load_baselines()
        bundle = asyncio.run(collect_instagram(plan, transport, reference, load_panel(args.panel), collected["trend_terms"],
                                              baselines=baselines, comment_posts=getattr(args, "comment_posts", 4)))
        step(f"replay done: {len(bundle['feeds'])} feeds, {len(bundle['comments'])} comment threads, "
             f"{len(bundle['probes'])} probes, {len(bundle['terms'])} terms, {len(bundle.get('commenter_overlap', {}))} overlapping commenters")
        if bundle.get("feeds"):
            history.update_baselines(bundle["feeds"], reference)
        for topic in probed:
            if not topic.get("instagram", {}).get("media"):
                topic["instagram"] = bundle["probes"].get(topic["topic_key"]) or {"media": [], "queries_used": [], "hashtag_counts": {}}
        panel_by_code = {m["code"]: m["panel"] for items in bundle["feeds"].values() for m in items if m.get("panel")}
        for topic in probed:
            for item in topic["instagram"]["media"]:
                if item["code"] in panel_by_code and not item.get("panel"):
                    item["panel"] = panel_by_code[item["code"]]
        if not metrics_only:
            step("discovering native discussions from panel, trend and seed posts")
            native = dedupe_native(discover_discussions(bundle, collected["trend_terms"], llm, reference), probed, llm)
            step(f"native discussions: {len(native)}; LLM {llm.usage()}")
    measured = probed + native
    # Ensure all media across probed and native discussions have their captured comments attached
    comment_lookup = {}
    if transport:
        for k, v in transport.comment_map.items():
            if isinstance(v, dict):
                comment_lookup[str(k)] = v.get("comments", [])
    for k, v in bundle.get("comments", {}).items():
        if isinstance(v, dict):
            comment_lookup[str(k)] = v.get("comments", [])
    if comment_lookup:
        for topic in measured:
            for item in topic.get("instagram", {}).get("media", []):
                if not item.get("comments"):
                    mid = str(item.get("id"))
                    code = str(item.get("code"))
                    c_list = comment_lookup.get(mid) or comment_lookup.get(code)
                    if c_list:
                        item["comments"] = c_list
                        if item.get("comment_count") is None:
                            item["comment_count"] = len(c_list)
    comments_per_post = getattr(args, "comments_per_post", 25)
    comment_pool_size = getattr(args, "comment_pool_size", 120)
    brief_topics = [t for t in measured if t.get("instagram", {}).get("media")]
    step(f"briefing {len(brief_topics)} topics with posts and comments")
    briefs = [] if metrics_only else llm.map(
        lambda t: brief(t, llm, reference, comments_per_post=comments_per_post, comment_pool_size=comment_pool_size),
        brief_topics)
    for topic, result in zip(brief_topics, briefs):
        topic["brief"] = result
    step(f"briefs done; LLM {llm.usage()}")
    from identify_what_matters import stance
    if stance.available():
        targets = [t for t in measured if t.get("brief")]
        for topic, opinion in zip(targets, stance.model_opinions(targets)):
            if opinion:
                topic["brief"]["measured_opinion"] = opinion

    all_topics = topics + native
    step(f"computing metrics for {len(all_topics)} topics")
    for topic in all_topics:
        if metrics_only:
            topic.setdefault("first_seen", topic.get("earliest") or reference.isoformat())
        topic["instagram_metrics"] = instagram_metrics(topic, reference)
        topic["scores"] = scores(topic)
    if metrics_only:
        history_runs = history.runs()
    else:
        history.forget(run_dir.name)
        step("matching topics against history")
        history.match(all_topics, reference, llm)
        history_runs = history.runs() + 1
        step(f"history matched; LLM {llm.usage()}")
    from identify_what_matters.proactive_radar import breakout_assessment
    for topic in all_topics:
        topic["breakout"] = breakout_assessment(topic, reference)
        if metrics_only and not topic.get("instagram", {}).get("media"):
            topic["breakout"] = {
                "stage": "unobserved",
                "score": 0.0,
                "reasons": ["No topic-level Instagram posts were captured before collection stopped."],
                "signals": {},
            }
        topic["trajectory"] = topic["breakout"]["stage"]
        topic["status"] = status(topic, reference)
        topic["stage_history"] = stage_history(topic.pop("timeline", []), topic["trajectory"], reference)
        topic["lead"] = lead_lag(topic)
    from identify_what_matters.account_graph import astroturf_summary, build_account_graph
    for topic in all_topics:
        if topic.get("brief"):
            try:
                topic["astroturf"] = astroturf_summary(topic)
            except Exception as error:
                print(f"[measure] bot check skipped for '{topic['title'][:40]}': {type(error).__name__}: {str(error)[:120]}", flush=True)
    reddit_path = Path(args.output_root) / "reddit_signals.json"
    reddit_posts = json.loads(reddit_path.read_text(encoding="utf-8")) if reddit_path.exists() else []
    for topic in all_topics:
        topic["reddit_proxy"] = reddit_proxy_for_topic(topic, reddit_posts)
    all_topics.sort(key=lambda t: (t.get("brief") is None, -t["scores"]["trend_score"]))
    step("quality judge" if not args.no_judge and not metrics_only else "quality judge skipped")
    quality = judge(all_topics, llm) if not args.no_judge and not metrics_only else None
    availability = {**plan["availability"], **collected["availability"]}
    report = {
        "run_id": run_dir.name, "generated_at": reference.isoformat(), "signals": plan["signals"] + len(collected["signals"]),
        "sources_ok": sum(isinstance(v, int) for v in availability.values()),
        "sources_failed": sum(not isinstance(v, int) for v in availability.values()),
        "availability": availability, "topics_planned": len(topics), "topics_requested": len(probed),
        "topics_probed": sum(bool((t.get("instagram") or {}).get("queries_used")) for t in probed),
        "topics_with_instagram_evidence": sum(bool(t.get("instagram", {}).get("media")) for t in probed),
        "native_topics": len(native), "instagram_transport": collected.get("transport", "replay") if transport else "none",
        "reddit_proxy_topics": sum(bool(t["reddit_proxy"]["thread_count"]) for t in all_topics),
        "reddit_proxy_threads": sum(t["reddit_proxy"]["thread_count"] for t in all_topics),
        "instagram_requests": collected.get("instagram_requests"), "panel_accounts_read": len(bundle["feeds"]),
        "missing_accounts": collected.get("missing_accounts", []), "trend_terms_searched": len(bundle["terms"]),
        "comment_threads": len(bundle["comments"]), "commenter_overlap": bundle.get("commenter_overlap", {}),
        "commenter_personas": commenter_personas(bundle, measured),
        "history_runs": history_runs, "llm": llm.usage(), "quality": quality,
        "analysis_mode": "instagram_metrics_only" if metrics_only else "full_llm_analysis",
        "panel_pulse": panel_pulse(bundle, reference, baselines=history.load_baselines()),
        "account_graph": build_account_graph([t for t in all_topics if t.get("brief")]),
        "lead_lag": lead_lag_summary(all_topics),
        "topics": [{k: v for k, v in t.items() if not k.startswith("_") and k != "instagram"} for t in all_topics],
    }
    if not metrics_only:
        history.save(run_dir.name, reference, all_topics, {k: v for k, v in report.items() if k != "topics"})
    write(run_dir / "instagram.json", {t["topic_key"]: t.get("instagram") for t in measured})
    write(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(render(report))
    step(f"report written: {run_dir / 'report.md'}")
    try:
        from identify_what_matters.emergent_narrative_discovery import write_proposals as write_cluster_proposals
        write_cluster_proposals(all_topics)
    except Exception as error:
        print(f"[measure] emergent issue-cluster discovery skipped: {type(error).__name__}: {str(error)[:120]}", flush=True)
    return report


def pulse_labels(caption: str, comment_texts: list[str]) -> dict:
    from collections import Counter
    from identify_what_matters import political_classifier, stance
    from identify_what_matters.demographics import _get_models, _ml_predict
    from identify_what_matters.political_pulse import political_pulse_engine
    caption = caption or ""
    texts = [t for t in comment_texts if t][:10] or [caption]
    labels = political_classifier.classify(caption)
    tones = stance.label_comments(caption, [], texts) if stance.available() else []
    tone_counts = Counter(t["tone"] for t in tones)
    state_model = _get_models()[2]
    archetypes = Counter()
    for text in texts:
        archetype, _ = _ml_predict(text, state_model, sorted(state_model.classes_))
        if archetype:
            archetypes[archetype] += 1
    archetype = archetypes.most_common(1)[0][0] if archetypes else None
    return {
        "lean": political_pulse_engine.classify_text_favorability(caption)["leaning"],
        "direction": labels["direction"]["value"], "target": labels["target"]["value"],
        "narrative": labels["narrative_type"]["value"], "issue": labels["issue_cluster"]["value"],
        "tone": tone_counts.most_common(1)[0][0] if tone_counts else None,
        "sarcasm_share": round(sum(t["sarcastic"] for t in tones) / len(tones), 2) if tones else None,
        "archetype": archetype,
        "archetype_share": round(archetypes[archetype] / sum(archetypes.values()), 2) if archetype else None,
    }


def commenter_personas(bundle: dict, measured: list[dict], limit: int = 50) -> list[dict]:
    from collections import Counter, defaultdict
    from identify_what_matters import stance
    from identify_what_matters.discussions import _all_media
    from identify_what_matters.political_pulse import political_pulse_engine
    from identify_what_matters.demographics import _get_models, _ml_predict, infer_profile
    overlap = bundle.get("commenter_overlap") or {}
    if not overlap:
        return []
    code_topic = {}
    for topic in measured:
        for m in topic.get("instagram", {}).get("media", []):
            code_topic.setdefault(m["code"], topic["title"])
    per_user = defaultdict(list)
    for media in _all_media(bundle):
        for c in media.get("comments") or []:
            uid = c.get("author") or c.get("author_id") or ""
            if uid in overlap and c.get("text"):
                per_user[uid].append({"code": media["code"], "text": c["text"], "reply": bool(c.get("parent_id")),
                                      "likes": c.get("like_count") or 0})
    state_model = _get_models()[2]
    classes = sorted(state_model.classes_)
    personas = []
    for uid, items in sorted(per_user.items(), key=lambda kv: -len({i["code"] for i in kv[1]}))[:limit]:
        posts = {i["code"] for i in items}
        texts = [i["text"] for i in items][:30]
        leans = Counter(political_pulse_engine.classify_text_favorability(t)["leaning"] for t in texts)
        tones = Counter(x["tone"] for x in stance.label_comments("", [], texts)) if stance.available() else Counter()
        archetypes = Counter()
        for t in texts:
            archetype, _ = _ml_predict(t, state_model, classes)
            if archetype:
                archetypes[archetype] += 1
        topics = Counter(code_topic[c] for c in posts if c in code_topic)
        profile = bundle.get("profiles", {}).get(uid) or {}
        inferred = {}
        if profile:
            demo = infer_profile(uid, profile.get("biography") or "", profile.get("city_name") or "")
            inferred = {"gender": demo.gender, "age_bracket": demo.age_bracket,
                        "location": demo.location_city or demo.location_state, "party_support": demo.party_support}
        personas.append({
            "username": uid, "posts": len(posts), "activity": len(items),
            "profile": {"followers": profile.get("follower_count"), "following": profile.get("following_count"),
                        "total_posts": profile.get("media_count"), "verified": profile.get("is_verified"),
                        "private": profile.get("is_private"), "business": profile.get("is_business"),
                        "category": profile.get("category"), "bio": (profile.get("biography") or "")[:160]} if profile else None,
            "inferred": inferred or None,
            "comments": sum(not i["reply"] for i in items), "replies": sum(i["reply"] for i in items),
            "likes_received": sum(i["likes"] for i in items),
            "topics": [t for t, _ in topics.most_common(3)],
            "lean": leans.most_common(1)[0][0] if leans else None,
            "tone": tones.most_common(1)[0][0] if tones else None,
            "archetype": archetypes.most_common(1)[0][0] if archetypes else None,
        })
    return personas


def panel_pulse(bundle: dict, reference: datetime, limit: int = 200,
                baselines: dict[str, dict] | None = None) -> list[dict]:
    """Panel posts drawing the most unusual attention for their own account, shown unclustered in the report."""
    from identify_what_matters.discussions import panel_scores
    rows = []
    for item in [m for m in panel_scores(bundle.get("feeds", {}), reference, baselines=baselines) if not m["panel"]["ritual"]][:limit]:
        top = max((c for c in item.get("comments", []) if c.get("text")), key=lambda c: c.get("like_count") or 0, default=None)
        labels = pulse_labels(item["caption"], [c.get("text", "") for c in item.get("comments", [])])
        rows.append({**labels, "group": item["panel"]["group"], "age_hours": round(hours_since(item["taken_at"], reference) or 0),
                     "comments": item["comment_count"], "likes": item["like_count"], **item["panel"],
                     "caption": scrub(item["caption"])[:140].replace("\n", " "), "url": item["url"],
                     "top_comment": scrub(top["text"])[:120].replace("\n", " ") if top else None,
                     "top_comment_likes": top.get("like_count") if top else None})
    return rows


def latest_planned_run(output_root: str) -> str | None:
    """Most recent run that has a plan but no Instagram capture yet (the next one to collect)."""
    runs = sorted(Path(output_root, "runs").glob("*/plan.json"), reverse=True)
    pending = [p.parent.name for p in runs if not (p.parent / "instagram_capture.json").exists()]
    return pending[0] if pending else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "collect", "measure", "run", "watch"),
                        help="plan: news/search signals + LLM topics; collect: X trends, Reddit and all Instagram "
                             "requests, no LLM; measure: analysis + report from the capture; "
                             "run: all three in a single execution; watch: run every --every-hours")
    parser.add_argument("--every-hours", type=float, default=4.0)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--metrics-only", action="store_true",
                        help="skip LLM summaries, discussion synthesis, history matching, and audit during measurement")
    parser.add_argument("--run-id", help="Run to collect/measure; 'latest' = newest planned run not yet collected")
    parser.add_argument("--discovery-file", help="plan from a saved discovery JSON instead of fetching public sources")
    parser.add_argument("--use-discovery-shortlist", action="store_true",
                        help="use saved ranked candidates directly when OpenRouter topic synthesis is unavailable")
    parser.add_argument("--instagram", choices=("hiker",), default="hiker",
                        help="collect Instagram data via the hosted Hiker API (HIKERAPI_TOKEN)")
    parser.add_argument("--no-instagram", action="store_true",
                        help="Collect public trend sources only; do not access Instagram or create an Instagram measurement report")
    parser.add_argument("--capture", help="Instagram capture to analyse (default: the run's instagram_capture.json)")
    parser.add_argument("--panel", help="Account panel JSON (default config/panel.json)")
    parser.add_argument("--trend-terms", type=int, default=20, help="X/Google trending terms searched on Instagram")
    parser.add_argument("--max-topics", type=int, default=150, help="News/search topics measured on Instagram per run")
    parser.add_argument("--request-budget", type=int, default=1500, help="Instagram requests per run")
    parser.add_argument("--panel-total-budget", type=int, default=800,
                        help="Requests per run spent on verified comment totals for panel posts (cached across runs)")
    parser.add_argument("--comment-posts", type=int, default=8, help="Top posts per probed topic to sample deep comments from")
    parser.add_argument("--comment-pages", type=int, default=4, help="Comment pages fetched per deep-sampled post")
    parser.add_argument("--reply-threads", type=int, default=6, help="Reply threads expanded per deep-sampled post")
    parser.add_argument("--score-topics", type=int, default=5, help="Top-volume topics whose posts get full comment depth")
    parser.add_argument("--score-posts", type=int, default=8, help="Posts per scored topic to fetch comments for")
    parser.add_argument("--score-comment-pages", type=int, default=10, help="Comment pages per post for scored topics")
    parser.add_argument("--comments-per-post", type=int, default=25, help="Max ranked comments per post to include in pool")
    parser.add_argument("--comment-pool-size", type=int, default=120, help="Max total comments per topic brief pool")
    parser.add_argument("--llm-max-calls", type=int, default=400)
    parser.add_argument("--no-reddit", action="store_true")
    parser.add_argument("--skip-public-sources", action="store_true",
                        help="during collect, skip Reddit, X, Google Trends, and News; use planned signals for Instagram search terms")
    parser.add_argument("--output-root", default="output")
    args = parser.parse_args()
    if args.command == "watch":
        import random
        import subprocess
        import time
        while True:
            passthrough, skip = [], False
            for value in sys.argv[2:]:
                if skip:
                    skip = False
                elif value == "--every-hours":
                    skip = True
                elif not value.startswith("--every-hours="):
                    passthrough.append(value)
            command = [sys.executable, "-m", "identify_what_matters.trends", "run"] + passthrough
            print(f"[watch] {now().isoformat()} starting run", flush=True)
            subprocess.run(command, check=False)
            time.sleep(args.every_hours * 3600 * random.uniform(0.9, 1.1))
    run_id = latest_planned_run(args.output_root) if args.run_id == "latest" else args.run_id
    if args.run_id == "latest" and not run_id:
        parser.error("no planned run is waiting for collection")
    run_dir = Path(args.output_root, "runs", run_id or now().strftime("%Y%m%dT%H%M%SZ"))
    if args.command == "collect":
        run_collect(args, run_dir)
        if args.no_instagram:
            print("Public-source collection complete; this run has no Instagram measurement step.")
        else:
            print(f"Next step: identify-what-matters-trends measure --run-id {run_dir.name}")
        return
    from identify_what_matters.llm import LLM
    llm = LLM(max_calls=args.llm_max_calls, cache_dir=Path(args.output_root) / "llm_cache")
    if args.command in ("plan", "run"):
        plan = run_plan(args, run_dir, llm)
        print(json.dumps({"run_id": run_dir.name, "signals": plan["signals"], "stories": plan["stories"],
                          "topics": len(plan["topics"]), "to_probe": len(plan["probe_keys"])}), flush=True)
        if args.command == "plan":
            print(f"Next step: identify-what-matters-trends collect --run-id {run_dir.name}")
            return
        run_collect(args, run_dir)
    elif not run_id:
        parser.error("measure needs --run-id")
    run_measure(args, run_dir, llm)
    print(f"Report: {run_dir / 'report.md'}  (LLM {llm.usage()})")


if __name__ == "__main__":
    main()

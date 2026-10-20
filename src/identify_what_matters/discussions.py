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

import asyncio
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

from identify_what_matters import encoder
from identify_what_matters.instagram_access import scrub
from identify_what_matters.trends import CATEGORIES, POLITICS_HASHTAGS, SEED_HASHTAGS, SYSTEM, hours_since, probe

PANEL_PATH = Path(__file__).resolve().parents[2] / "config" / "panel.json"
KINDS = ("news_event", "debate_controversy", "viral_moment", "meme_format", "celebrity_gossip", "sports_moment",
         "announcement_release", "festival_culture", "other")
# Instagram keyword search returns nothing for long phrases; trending phrases are cut to 3 content words.
FILLER = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "case", "news", "latest"}
# Devotional/greeting posts draw floods of ritual replies ("Jai Shani Dev") that look like overperformance but are not discussion.
RITUAL = re.compile(r"दर्शन|आरती|darshan|aarti|bhakti|#jai|jai shri|suprabhat|good morning|सुप्रभात|happy birthday|#bhakti", re.I)


def load_panel(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or PANEL_PATH).read_text())


def panel_accounts_for_run(panel: dict, plan: dict, reference) -> list[dict]:
    accounts = panel["accounts"]
    accounts_per_run = panel.get("accounts_per_run", len(accounts))
    if not isinstance(accounts_per_run, int) or accounts_per_run < 1:
        raise ValueError("accounts_per_run must be a positive integer")
    if len(accounts) <= accounts_per_run:
        return accounts

    cohort_count = math.ceil(len(accounts) / accounts_per_run)
    run_id = str(plan.get("run_id") or "")
    try:
        run_day = datetime.strptime(run_id[:8], "%Y%m%d").date()
    except ValueError:
        run_day = reference.date()
    cohort_index = run_day.toordinal() % cohort_count
    return accounts[cohort_index::cohort_count]


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


def panel_scores(feeds: dict[str, list[dict]], reference, max_age_hours: float = 168,
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
        verified = [m for m in settled if m.get("comment_verified")]
        hist = baselines.get(account)
        if settled:
            cur_likes = median(m["like_count"] or 0 for m in settled)
        elif hist:
            cur_likes = hist["median_likes"]
        else:
            cur_likes = median(m["like_count"] or 0 for m in usable)
        if len(verified) >= 3:
            cur_comments = median(m["comment_count"] or 0 for m in verified)
            cur_ratio = median((m["comment_count"] or 0) / (m["like_count"] + 1) for m in verified)
            if hist:
                base_comments = 0.5 * hist["median_comments"] + 0.5 * cur_comments
                base_ratio = 0.5 * hist["median_ratio"] + 0.5 * cur_ratio
            else:
                base_comments, base_ratio = cur_comments, cur_ratio
        elif hist:
            base_comments, base_ratio = hist["median_comments"], hist["median_ratio"]
        else:
            base_comments, base_ratio = None, None
        base_likes = (0.5 * hist["median_likes"] + 0.5 * cur_likes) if hist and settled else cur_likes

        for item in usable:
            age = hours_since(item["taken_at"], reference)
            if age is None or age > max_age_hours:
                continue
            weight = max(0.25, min(1.0, age / 24))
            comments, likes = item["comment_count"] or 0, item["like_count"] or 0
            over_likes = (likes + 20) / (base_likes * weight + 20)
            if base_comments is not None and item.get("comment_verified"):
                over_comments = (comments + 5) / (base_comments * weight + 5)
                ratio_lift = (comments / (likes + 1)) / base_ratio if base_ratio > 0 else 1.0
                heat_basis = "comments_and_likes"
            else:
                over_comments, ratio_lift, heat_basis = None, None, "likes_only"
            cap_val = item.get("caption")
            cap_str = cap_val.get("text", "") if isinstance(cap_val, dict) else str(cap_val or "")
            item["panel"] = {"account": account, "group": item.get("panel_group"),
                             "ritual": bool(RITUAL.search(cap_str)),
                             "over_comments": round(over_comments, 2) if over_comments is not None else None,
                             "over_likes": round(over_likes, 2),
                             "ratio_lift": round(ratio_lift, 2) if ratio_lift is not None else None,
                             "heat_basis": heat_basis,
                             "heat": round(math.log(over_comments) + 0.5 * math.log(over_likes)
                                           + 0.5 * math.log(max(ratio_lift, 0.1)), 3) if heat_basis == "comments_and_likes"
                             else round(0.5 * math.log(over_likes), 3)}
            hot.append(item)
    return sorted(hot, key=lambda m: m["panel"]["heat"], reverse=True)


_INDIA_RELEVANT_RE = re.compile(
    r"\b("
    # parties
    r"bjp|congress|inc|aap|tmc|dmk|sp|bsp|brs|shiv sena|shivsena|aimim|ncp|jdu|rjd|ysrcp|tdp|"
    # politicians / leaders
    r"modi|rahul|gandhi|amit shah|kejriwal|mamata|yogi|nitish|chandrashekhar|akhilesh|pawar|"
    r"mk stalin|karunanidhi|jayalalitha|gehlot|siddaramaiah|kharge|smriti|piyush goyal|"
    # governance / political terms
    r"parliament|loksabha|rajya sabha|vidhansabha|election|vote|mandate|constitution|"
    r"pm |chief minister|home minister|finance minister|government|cabinet|minister|"
    r"nda|upa|india alliance|mahagathbandhan|opposition|ruling party|"
    r"cbi|ed |enforcement directorate|ias |ips |rss|vhp|bajrang|sangh|"
    # economy
    r"economy|gdp|inflation|rupee|budget|gst|tax|sensex|nifty|rbi|sebi|"
    r"unemployment|poverty|fuel price|petrol|diesel|onion price|food price|"
    r"make in india|startup india|upi|digital india|"
    # india-specific social / political issues
    r"farmer|agriculture|protest|agitation|bandh|strike|reservation|caste|"
    r"mandir|masjid|waqf|article 370|caa|nrc|ram mandir|ayodhya|"
    r"kerala|karnataka|andhra|telangana|gujarat|maharashtra|rajasthan|"
    r"punjab|haryana|uttarakhand|himachal|jammu|kashmir|ladakh|"
    r"corruption|scam|arrest|raid|chargesheet|court|supreme court|high court"
    r")\b",
    re.I,
)

def _is_india_relevant(entry: dict) -> bool:
    """Proactive radar and X-trends terms are kept as-is; filter Google Trends by political/economy keywords."""
    if entry.get("source") in ("proactive_radar", "x_trends"):
        return True
    return bool(_INDIA_RELEVANT_RE.search(entry.get("term", "")))


RESERVED_POLITICS_HASHTAGS = 5


def _extract_organic_hashtags(bundle: dict, limit: int) -> list[str]:
    """Extract the most-used India-political hashtags from panel captions collected this run.
    Returns up to `limit` lowercase tag strings (no #). Falls back to POLITICS_HASHTAGS to fill gaps."""
    tag_counts: dict[str, int] = defaultdict(int)
    for items in bundle["feeds"].values():
        for item in items:
            caption = item.get("caption") or {}
            text = caption.get("text", "") if isinstance(caption, dict) else str(caption)
            for tag in re.findall(r"#(\w+)", text.lower()):
                if len(tag) >= 4 and _INDIA_RELEVANT_RE.search(tag):
                    tag_counts[tag] += 1
    reserved = [tag.lstrip("#").lower() for tag in POLITICS_HASHTAGS[:RESERVED_POLITICS_HASHTAGS]]
    organic = [tag for tag, _ in sorted(tag_counts.items(), key=lambda x: -x[1])]
    fallback = [tag.lstrip("#").lower() for tag in POLITICS_HASHTAGS]
    return list(dict.fromkeys(reserved + organic + fallback))[:limit]


def _build_commenter_overlap(bundle: dict) -> dict:
    """Map commenter identifier → list of post codes they appeared in across all fetched comment threads."""
    user_posts: dict[str, list[str]] = defaultdict(list)
    for media in _all_media(bundle):
        for comment in (media.get("comments") or []):
            # normalized comments store 'author' (username) and 'author_id' (numeric)
            uid = comment.get("author") or comment.get("author_id") or ""
            if uid:
                user_posts[uid].append(media["code"])
    return {u: list(dict.fromkeys(posts)) for u, posts in user_posts.items() if len(set(posts)) >= 2}


async def collect_instagram(plan: dict, transport, reference, panel: dict, terms: list[dict],
                           panel_comment_posts: int = 100, term_comment_posts: int = 25, seed_hashtags: int = 25,
                           baselines: dict[str, dict] | None = None, early_comment_posts: int = 40,
                           comment_posts: int = 3, panel_total_budget: int = 800,
                           deep_comment_pages: int = 4, deep_reply_threads: int = 6) -> dict:
    """Every Instagram request of a run, in priority order (budget may end the run early). No LLM calls."""
    bundle = {"feeds": {}, "missing_accounts": [], "terms": {}, "probes": {}, "seeds": {}, "comments": {},
              "stopped": None}

    _ig_slots = asyncio.Semaphore(10)

    async def call(method, *args, _timeout: int = 45, **kwargs):
        if bundle["stopped"]:
            return None
        try:
            async with _ig_slots:
                result = method(*args, **kwargs)
                # Hiker collection is async; saved-capture replay is synchronous.
                if asyncio.iscoroutine(result):
                    return await asyncio.wait_for(result, timeout=_timeout)
                return result
        except asyncio.TimeoutError:
            print(f"[collect] timeout ({_timeout}s) on {getattr(method, '__name__', method)}", flush=True)
            return None
        except Exception as error:
            text = str(error).lower()
            if "budget" in text or "blocked" in text or "challenge" in text or "login" in text:
                bundle["stopped"] = f"{type(error).__name__}: {str(error)[:120]}"
                print(f"[collect] stopping Instagram requests: {bundle['stopped']}", flush=True)
            else:
                print(f"[collect] {getattr(method, '__name__', method)} failed: {type(error).__name__}: "
                      f"{str(error)[:120]}", flush=True)
            return None

    async def fetch_comments(item):
        if item["code"] in bundle["comments"]:
            return
        sampled = await call(transport.comments, item["id"], _timeout=150)
        if sampled:
            bundle["comments"][item["code"]] = sampled

    selected_accounts = panel_accounts_for_run(panel, plan, reference)
    _sem = asyncio.Semaphore(25)

    async def _fetch_account(account):
        async with _sem:
            items = await call(transport.user_feed, account["username"], panel.get("doc_id"))
        if items is None:
            bundle["missing_accounts"].append(account["username"])
        else:
            for item in items:
                item["panel_group"] = account["group"]
            bundle["feeds"][account["username"]] = items
            print(f"[collect] panel: {len(bundle['feeds'])} accounts read, "
                  f"{len(bundle['missing_accounts'])} missing; "
                  f"selected {len(selected_accounts)}/{len(panel['accounts'])}", flush=True)

    await asyncio.gather(*[_fetch_account(acct) for acct in selected_accounts], return_exceptions=True)

    feed_items = sorted((m for items in bundle["feeds"].values() for m in items if m.get("taken_at")),
                        key=lambda m: m["taken_at"], reverse=True)
    for items in bundle["feeds"].values():
        for m in items:
            m["comment_verified"] = False
    spent = 0
    for start in range(0, len(feed_items), 10):
        if spent >= panel_total_budget:
            break
        batch = feed_items[start:start + 10]
        used_before = transport.used
        totals = await asyncio.gather(*[call(transport.comment_total, m["id"], m["taken_at"]) for m in batch],
                                      return_exceptions=True)
        spent += transport.used - used_before
        for m, total in zip(batch, totals):
            if isinstance(total, int):
                m["feed_comment_count"] = m.get("comment_count")
                m["comment_count"] = total
                m["comment_verified"] = True
    verified = sum(m.get("comment_verified", False) for items in bundle["feeds"].values() for m in items)
    total_posts = sum(len(items) for items in bundle["feeds"].values())
    print(f"[collect] verified comment totals: {verified}/{total_posts} panel posts ({spent} requests)", flush=True)

    scored = [m for m in panel_scores(bundle["feeds"], reference, baselines=baselines) if not m["panel"]["ritual"]]
    early = sorted((m for m in scored if (hours_since(m["taken_at"], reference) or 1e9) <= 12
                    and (m["comment_count"] or 0) >= 15
                    and ((m["panel"]["over_comments"] or 0) >= 2 or (m["panel"]["ratio_lift"] or 0) >= 1.5)),
                   key=lambda m: m["panel"]["heat"] + math.log(12 / max(hours_since(m["taken_at"], reference), 0.5)),
                   reverse=True)
    hot = [m for m in scored if (m["panel"]["over_comments"] or 0) >= 1.5 or (m["panel"]["ratio_lift"] or 0) >= 2]
    panel_comment_targets = list({m["code"]: m for m in early[:early_comment_posts] + hot}.values())[:panel_comment_posts]

    from identify_what_matters.proactive_radar import identify_proactive_candidates, generate_proactive_probe_queries
    proactive_cands = identify_proactive_candidates(bundle, reference, baselines=baselines)
    proactive_queries = generate_proactive_probe_queries(proactive_cands, max_queries=8)
    if proactive_queries:
        print(f"[collect] proactive radar discovered {len(proactive_queries)} emerging entity seeds: {proactive_queries}", flush=True)
    bundle["proactive_candidates"] = proactive_cands
    bundle["proactive_queries"] = proactive_queries

    all_search_terms = terms + [{"term": q, "source": "proactive_radar"} for q in proactive_queries]
    search_needed = [e for e in all_search_terms
                     if e["term"] not in bundle["terms"] and _is_india_relevant(e)]

    probe_keys = set(plan.get("probe_keys", []))
    probe_topics = [t for t in plan.get("topics", []) if t["topic_key"] in probe_keys]
    hashtag_targets = _extract_organic_hashtags(bundle, seed_hashtags)
    print(f"[collect] organic hashtags from panel captions: {hashtag_targets}", flush=True)

    _done = {"panel_comments": 0, "searches": 0, "hashtags": 0, "probes": 0}
    filtered_out = len([e for e in all_search_terms if e["term"] not in bundle["terms"]]) - len(search_needed)
    print(f"[collect] parallel phase: {len(panel_comment_targets)} panel comments, "
          f"{len(search_needed)} searches ({filtered_out} non-political filtered), "
          f"{len(hashtag_targets)} hashtags, {len(probe_topics)} probes "
          f"(budget used: {transport.used})", flush=True)

    async def _do_panel_comment(item):
        await fetch_comments(item)
        _done["panel_comments"] += 1
        print(f"[collect] panel comments: {_done['panel_comments']}/{len(panel_comment_targets)} done", flush=True)

    async def _do_search(entry):
        results = await call(transport.search, entry["term"])
        if results is not None:
            bundle["terms"][entry["term"]] = results
        _done["searches"] += 1
        status = "failed" if results is None else f"{len(results)} posts"
        print(f"[collect] searches: {_done['searches']}/{len(search_needed)} '{entry['term']}'"
              f" -> {status}", flush=True)

    async def _do_hashtag(tag):
        result = await call(transport.hashtag, tag)
        if result:
            bundle["seeds"][tag] = result["media"]
        _done["hashtags"] += 1
        status = "failed" if result is None else f"{len(result['media'])} posts"
        print(f"[collect] hashtags: {_done['hashtags']}/{len(hashtag_targets)} #{tag}"
              f" -> {status}", flush=True)

    async def _do_probe(topic):
        if not bundle["stopped"]:
            bundle["probes"][topic["topic_key"]] = await probe(
                topic, transport, reference, comment_posts=2, comment_pages=1, reply_threads=0)
        _done["probes"] += 1
        title_short = (topic.get("title") or topic["topic_key"])[:60]
        print(f"[collect] probes: {_done['probes']}/{len(probe_topics)} '{topic['topic_key']}' — {title_short}", flush=True)

    await asyncio.gather(
        *[_do_panel_comment(item) for item in panel_comment_targets],
        *[_do_search(e) for e in search_needed],
        *[_do_hashtag(tag) for tag in hashtag_targets],
        *[_do_probe(t) for t in probe_topics],
        return_exceptions=True,
    )

    fresh_term_posts = []
    for results in bundle["terms"].values():
        fresh = [m for m in results if (hours_since(m["taken_at"], reference) or 1e9) <= 168]
        fresh_term_posts += sorted(fresh, key=lambda m: m.get("comment_count") or 0, reverse=True)[:2]
    term_comment_targets = sorted(fresh_term_posts, key=lambda m: m.get("comment_count") or 0, reverse=True)[:term_comment_posts]
    _done_tc = {"n": 0}
    print(f"[collect] term comments: {len(term_comment_targets)} top-commented posts (budget used: {transport.used})", flush=True)

    async def _do_term_comment(item):
        await fetch_comments(item)
        _done_tc["n"] += 1
        print(f"[collect] term comments: {_done_tc['n']}/{len(term_comment_targets)} done", flush=True)

    await asyncio.gather(*[_do_term_comment(item) for item in term_comment_targets], return_exceptions=True)

    deep_targets = []
    for topic in probe_topics:
        media = (bundle["probes"].get(topic["topic_key"]) or {}).get("media", [])
        commentable = [m for m in media if (m.get("comment_count") or 0) > 0]
        ranked = sorted(commentable, key=lambda m: m["comment_count"], reverse=True)[:comment_posts]
        deep_targets += [m for m in ranked if not m.get("comments")]

    async def _do_deep_comment(media):
        sampled = await call(transport.comments, media["id"], _timeout=150,
                             max_pages=deep_comment_pages, max_reply_threads=deep_reply_threads)
        if sampled:
            bundle["comments"][media["code"]] = sampled
            media["comments"] = sampled["comments"]

    print(f"[collect] deep comments: {len(deep_targets)} posts across {len(probe_topics)} topics", flush=True)
    for start in range(0, len(deep_targets), 10):
        await asyncio.gather(*[_do_deep_comment(m) for m in deep_targets[start:start + 10]], return_exceptions=True)
    for media in _all_media(bundle):
        sampled = bundle["comments"].get(media["code"])
        if sampled and not media.get("comments"):
            media["comments"] = sampled["comments"]
            if media.get("comment_count") is None:
                media["comment_count"] = sampled["total"]

    bundle["commenter_overlap"] = _build_commenter_overlap(bundle)
    cross_topic = sum(1 for posts in bundle["commenter_overlap"].values() if len(posts) >= 3)
    print(f"[collect] commenter overlap: {len(bundle['commenter_overlap'])} users on 2+ posts, "
          f"{cross_topic} on 3+ posts", flush=True)

    overlap_users = sorted((u for u in bundle["commenter_overlap"] if not u.isdigit()),
                           key=lambda u: -len(set(bundle["commenter_overlap"][u])))[:50]
    bundle["profiles"] = {}

    async def _do_profile(username):
        profile = await call(transport.profile, username)
        if profile:
            bundle["profiles"][username] = profile

    for start in range(0, len(overlap_users), 10):
        await asyncio.gather(*[_do_profile(u) for u in overlap_users[start:start + 10]], return_exceptions=True)
    print(f"[collect] commenter profiles: {len(bundle['profiles'])}/{len(overlap_users)} fetched", flush=True)

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
        if age is None or age > 168 or (len(scrub(item["caption"])) <= 15 and not item.get("comments")):
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
            and ((panel.get("over_comments") or 0) >= 3 or (panel.get("ratio_lift") or 0) >= 2))


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

"""Proactive identification of emerging Instagram discussions and debates.

Instead of waiting for Google News RSS or Google Trends to report what already happened
(which lags Instagram debate by 6-36 hours), this module drives discovery *bottom-up*
from native Instagram signals:
1. Sentinel Panel Post Velocity & Ratio Deviation (dC/dt, ratio lift vs. account baseline, Poisson comment bursts)
2. Bottom-up Vernacular Entity & Keyword Mining from fast-moving comment trees
3. 1-hop Autonomous Snowball Probing on surging entities (without pre-baked keywords)
4. Predictive Lifecycle Categorization (spark vs breakout vs saturated)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from statistics import median

from identify_what_matters.utils import hours_since as _hours_since

# Vernacular & conversational stopwords across English, Hindi, and Hinglish comments
# to isolate real named entities, places, controversies, and events.
STOPWORDS = {
    # English conversational filler
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
    "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "bro", "bhai", "sir", "madam", "pls", "please", "omg", "lol", "lmao", "post", "video", "reel",
    "admin", "page", "like", "follow", "share", "comment", "true", "false", "real", "fake", "right", "wrong",
    "w", "massive", "w", "huge", "rip", "legend", "goat", "caption", "thoughts", "agree", "disagree",

    # Hindi / Hinglish filler
    "kya", "kyu", "kyun", "kaise", "kab", "kaha", "kahan", "hai", "hain", "tha", "thi", "the", "ho", "hona",
    "bhi", "toh", "to", "hi", "he", "aur", "ya", "par", "pe", "mein", "me", "se", "ko", "ka", "ki", "ke",
    "ye", "yeh", "wo", "woh", "voh", "kuch", "sab", "koi", "kisi", "apna", "apni", "apne", "mera", "meri",
    "mere", "tera", "teri", "tere", "unka", "unki", "unke", "iska", "iski", "iske", "karo", "karna", "kiya",
    "dekh", "dekho", "bolo", "bol", "batao", "bhai", "bhaiya", "yaar", "yr", "log", "logo", "desh", "ab", "aaj",
    "kal", "hoga", "hogi", "honge", "raha", "rahi", "rahe", "baat", "sahi", "galat", "achha", "accha", "bura",
}


def hours_since(value, reference: datetime) -> float | None:
    return _hours_since(value, reference, min_hours=0.05)


# High-priority political and civic entities, conflict terms, and undercurrent keywords
POLITICAL_CIVIC_PATTERNS = re.compile(
    r"\b(modi|rahul|yogi|kejriwal|congress|bjp|bsp|sp|aap|tmc|dmk|aiadmk|shivsena|ncp|cpim|"
    r"election|chunav|vote|voter|sir|cec|evm|booth|gandhi|nehru|fadnavis|shinde|pawar|"
    r"thackeray|stalin|mamata|lallantop|sarkar|neta|desh|samvidhan|constitution|court|sc|"
    r"cbi|ed|fir|police|arrest|hinsa|violence|danga|clash|protest|andolan|dharna|morcha|"
    r"waqf|caste|jaati|dalit|obc|reservation|aarakshan|sanatan|hindu|muslim|mandir|masjid|"
    r"paperleak|exam|ugc|neet|upsc|railway|train|kavach|scam|ghotala|corruption|bhrashtachar|"
    r"collector|dm|mla|mp|cm|pm|mantri|minister|budget|tax|ladli|agniveer|delhi|mumbai|"
    r"up|bihar|bengal|punjab|kashmir|haryana|maharashtra|rajasthan|gujarat|karnataka|tamilnadu)\b",
    re.I
)


BURST_WINDOW_HOURS = 2.0
BURST_HISTORY_HOURS = 24.0
BURST_ALPHA = 0.01
BURST_MIN_WINDOW_COMMENTS = 5
BURST_MIN_COMMENTS = 10
BURST_RATE_FLOOR = 0.5  # expected comments per window when history is empty


def _poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def comment_burst(comment_ages_hours: list[float]) -> dict:
    """Poisson burst test on comment arrival times.

    Counts comments in the last BURST_WINDOW_HOURS and compares them with the arrival rate
    observed over the preceding history window. A burst is flagged when the recent count is
    improbable under that rate (upper-tail Poisson p-value below BURST_ALPHA). Ages come from
    the sampled comments, so counts are lower bounds and the test is conservative.
    """
    ages = [a for a in comment_ages_hours if a is not None and a >= 0]
    if len(ages) < BURST_MIN_COMMENTS:
        return {"is_burst": False, "recent": 0, "expected": None, "p_value": None, "sampled": len(ages)}
    recent = sum(1 for a in ages if a <= BURST_WINDOW_HOURS)
    history = sum(1 for a in ages if BURST_WINDOW_HOURS < a <= BURST_HISTORY_HOURS)
    history_rate = history / (BURST_HISTORY_HOURS - BURST_WINDOW_HOURS)
    expected = max(BURST_RATE_FLOOR, history_rate * BURST_WINDOW_HOURS)
    p_value = _poisson_sf(recent, expected)
    return {
        "is_burst": recent >= BURST_MIN_WINDOW_COMMENTS and p_value < BURST_ALPHA,
        "recent": recent,
        "expected": round(expected, 2),
        "p_value": p_value,
        "sampled": len(ages),
    }


def calculate_velocity_signals(media_item: dict, reference: datetime,
                                account_baseline: dict | None = None) -> dict:
    """Calculate early comment velocity (dC/dt), like velocity, ratio lift, political boost, and comment bursts.

    The fixed threshold rules are kept as the fallback. A Poisson burst test over sampled comment
    arrival times adds a second, rate-relative trigger when enough comments are available.
    """
    age_hours = hours_since(media_item.get("taken_at"), reference) or 1.0
    comments = float(media_item.get("comment_count") or 0)
    likes = float(media_item.get("like_count") or 0)

    # 1. First-derivative velocities
    comment_velocity = comments / age_hours  # comments per hour
    like_velocity = likes / age_hours        # likes per hour
    current_ratio = comments / (likes + 1.0)

    # 2. Comparison against account baseline
    baseline_comments = (account_baseline or {}).get("median_comments", 50.0)
    baseline_likes = (account_baseline or {}).get("median_likes", 1000.0)
    baseline_ratio = (account_baseline or {}).get("median_ratio", 0.03)

    # Scaled expectation based on post age (Minot et al. 24h accumulation)
    time_weight = max(0.15, min(1.0, age_hours / 24.0))
    expected_comments = baseline_comments * time_weight
    expected_ratio = baseline_ratio

    comment_acceleration = (comments + 5.0) / (expected_comments + 5.0)
    ratio_deviation = current_ratio / expected_ratio if expected_ratio > 0 else 1.0

    # 3. Political & Civic Undercurrent Affinity Detection
    caption_text = media_item.get("caption") or ""
    if isinstance(caption_text, dict):
        caption_text = caption_text.get("text", "")
    has_political_affinity = bool(POLITICAL_CIVIC_PATTERNS.search(caption_text))

    # 4. Proactive Heat Score
    # Highly rewards posts young in age with high comment velocity, surging ratio, and political relevance
    recency_boost = max(1.0, 12.0 / age_hours) if age_hours <= 12 else 1.0
    political_boost = 1.6 if has_political_affinity else 1.0

    proactive_score = (
        (math.log1p(comment_velocity) * 1.5
         + math.log1p(comment_acceleration) * 1.2
         + math.log1p(max(0.0, ratio_deviation - 1.0)) * 1.8)
        * recency_boost * political_boost
    )

    burst = comment_burst([hours_since(c.get("created_at"), reference)
                           for c in (media_item.get("comments") or []) if isinstance(c, dict)])

    is_flame = (
        (age_hours <= 8 and comment_velocity >= 20 and ratio_deviation >= 1.8)
        or (comment_acceleration >= 3.0 and comments >= 80)
        or (has_political_affinity and comment_velocity >= 15 and ratio_deviation >= 1.5)
        or burst["is_burst"]
    )

    return {
        "age_hours": round(age_hours, 2),
        "comment_velocity": round(comment_velocity, 2),
        "like_velocity": round(like_velocity, 2),
        "comment_acceleration": round(comment_acceleration, 2),
        "ratio_deviation": round(ratio_deviation, 2),
        "proactive_score": round(proactive_score, 3),
        "is_flame": is_flame,
        "is_burst": burst["is_burst"],
        "burst": burst,
        "is_political": has_political_affinity,
    }


def extract_emerging_entities(comments: list[dict], min_occurrences: int = 2) -> list[dict]:
    """Mine high-frequency capitalized phrases, hashtags, and vernacular terms from comment trees.

    In Indian social media, the controversy's focal entity/person/event is mentioned
    repeatedly in comments even when the caption is a meme punchline.
    """
    token_counts = Counter()
    phrase_counts = Counter()
    hashtag_counts = Counter()

    for c in comments:
        text = c.get("text", "")
        if not text:
            continue

        # Extract hashtags
        tags = re.findall(r"#([A-Za-z0-9_]{3,30})", text)
        for tag in tags:
            tag_clean = tag.lower()
            if tag_clean not in STOPWORDS:
                hashtag_counts[tag_clean] += 1 + int(c.get("like_count") or 0 > 10)

        # Clean text
        clean = re.sub(r"https?://\S+|@\w+|[^\w\s]", " ", text)
        words = clean.split()

        # Unigrams (case-insensitive for vernacular entities)
        for w in words:
            wl = w.lower()
            if len(wl) >= 4 and wl not in STOPWORDS and not wl.isdigit():
                token_counts[wl] += 1

        # Bigrams / Named Entity Candidates
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            w1l, w2l = w1.lower(), w2.lower()
            if (
                len(w1l) >= 3 and len(w2l) >= 3
                and w1l not in STOPWORDS and w2l not in STOPWORDS
                and not (w1l.isdigit() or w2l.isdigit())
            ):
                # If capitalized, double weight
                weight = 2 if (w1[0].isupper() and w2[0].isupper()) else 1
                phrase_counts[f"{w1l} {w2l}"] += weight

    candidates = []
    # Collect top phrases
    for phrase, count in phrase_counts.most_common(15):
        if count >= min_occurrences:
            candidates.append({"entity": phrase, "count": count, "type": "bigram"})

    # Collect top tokens
    for token, count in token_counts.most_common(15):
        if count >= min_occurrences + 1 and not any(token in p["entity"] for p in candidates):
            candidates.append({"entity": token, "count": count, "type": "unigram"})

    # Collect top hashtags
    for tag, count in hashtag_counts.most_common(10):
        if count >= min_occurrences:
            candidates.append({"entity": f"#{tag}", "count": count, "type": "hashtag"})

    return sorted(candidates, key=lambda x: x["count"], reverse=True)

def extract_caption_entities(caption: str) -> list[dict]:
    """Extract key named entities and hashtags from flame post captions."""
    entities = []
    # Hashtags
    tags = re.findall(r"#([A-Za-z0-9_]{3,30})", caption)
    for t in tags:
        tl = t.lower()
        if tl not in STOPWORDS and tl not in {"reels", "informative", "education", "shorts", "explore", "trending"}:
            entities.append({"entity": f"#{t}", "count": 2, "type": "hashtag"})

    # Political and capitalized entities
    matches = POLITICAL_CIVIC_PATTERNS.findall(caption)
    for m in matches:
        ml = m.lower()
        if ml not in {e["entity"].lower().lstrip("#") for e in entities}:
            entities.append({"entity": m, "count": 3, "type": "political_entity"})

    # Capitalized sequences (e.g. "Election Commission", "Gyanesh Kumar")
    clean = re.sub(r"https?://\S+|@\w+|[^\w\s]", " ", caption)
    words = clean.split()
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 and w2 and w1[0].isupper() and w2[0].isupper():
            comb = f"{w1} {w2}"
            if comb.lower() not in STOPWORDS and not any(comb.lower() in e["entity"].lower() for e in entities):
                entities.append({"entity": comb, "count": 2, "type": "bigram"})

    return entities

def identify_proactive_candidates(bundle: dict, reference: datetime,
                                  baselines: dict[str, dict] | None = None) -> list[dict]:
    """Scan all panel feeds, compute velocity/flame scores, and extract surging entities to form probe seeds."""
    baselines = baselines or {}
    flame_posts = []

    for account, items in bundle.get("feeds", {}).items():
        base = baselines.get(account)
        for post in items:
            if not isinstance(post, dict) or not post.get("taken_at") or post.get("pinned") or post.get("paid"):
                continue
            vel = calculate_velocity_signals(post, reference, base)
            post["proactive_radar"] = vel
            if vel["is_flame"] or vel["proactive_score"] >= 8.0:
                flame_posts.append(post)

    flame_posts.sort(key=lambda p: p["proactive_radar"]["proactive_score"], reverse=True)

    # For each flame post, extract comment entities if comments exist
    candidates = []
    comments_store = bundle.get("comments", {})
    for post in flame_posts:
        post_comments = post.get("comments")
        if not post_comments:
            # Check by pk (int or str) or code
            entry = (
                comments_store.get(str(post.get("pk")))
                or comments_store.get(post.get("pk"))
                or comments_store.get(post.get("code"))
                or {}
            )
            post_comments = entry.get("comments", []) if isinstance(entry, dict) else entry

        entities = extract_emerging_entities(post_comments)
        cap = post.get("caption") or ""
        cap_text = cap.get("text", "") if isinstance(cap, dict) else str(cap)

        # If comments didn't yield enough entities, also mine key entities from flame post caption
        if len(entities) < 2 and cap_text:
            caption_entities = extract_caption_entities(cap_text)
            entities = entities + [e for e in caption_entities if e["entity"] not in {x["entity"] for x in entities}]

        candidates.append({
            "post_code": post.get("code"),
            "post_pk": post.get("pk"),
            "account": post.get("panel_group") or "panel",
            "caption": cap_text[:120],
            "proactive_score": post["proactive_radar"]["proactive_score"],
            "comment_velocity": post["proactive_radar"]["comment_velocity"],
            "ratio_deviation": post["proactive_radar"]["ratio_deviation"],
            "age_hours": post["proactive_radar"]["age_hours"],
            "surging_entities": entities[:4],
        })

    return candidates


def generate_proactive_probe_queries(proactive_candidates: list[dict], max_queries: int = 15) -> list[str]:
    """Convert mined comment entities and flame post keywords into autonomous next-hop search queries.
    
    Prioritizes political undercurrent entities and specific conflict keywords.
    """
    queries = []
    seen = set()

    # Prioritize political candidates first
    sorted_candidates = sorted(
        proactive_candidates,
        key=lambda c: (1 if POLITICAL_CIVIC_PATTERNS.search(c.get("caption", "")) else 0, c.get("proactive_score", 0)),
        reverse=True
    )

    for cand in sorted_candidates:
        for ent in cand.get("surging_entities", []):
            q = ent["entity"].lstrip("#").strip()
            if len(q) >= 3 and q.lower() not in seen and q.lower() not in STOPWORDS:
                seen.add(q.lower())
                queries.append(q)
                if len(queries) >= max_queries:
                    return queries
    return queries


def classify_lifecycle_trajectory(topic: dict, reference: datetime) -> str:
    """Classify topic trajectory based on the lag between Instagram emergence and external news presence.

    - spark: High Instagram comment velocity / debate, < 6 hours old, zero mainstream press.
             (Most valuable proactive signal)
    - breakout: Expanding across 2+ independent accounts / creators, debate intensifying.
    - peak_debate: Maximum comment-to-like ratio lift and side polarization.
    - saturated: News publishers > 5, press has covered it, discussion shifting to commentary.
    """
    ig = topic.get("instagram_metrics", {})
    publishers = topic.get("publishers", 0)
    panel_accounts = ig.get("panel_accounts", 0)
    debate = (topic.get("brief") or {}).get("debate") or {}
    is_debate = debate.get("is_debate", False)

    first_seen_hours = hours_since(topic.get("first_seen"), reference) or 0.0

    if publishers == 0 and panel_accounts >= 1:
        if first_seen_hours <= 8:
            return "spark"
        return "breakout"

    if is_debate and ig.get("comment_like_ratio", 0) >= 0.04:
        return "peak_debate"

    if publishers >= 4:
        return "saturated"

    return "breakout" if panel_accounts >= 2 else "spark"


def breakout_assessment(topic: dict, reference: datetime) -> dict:
    """Where a topic is in its life, from the ages, spread and over-performance of its own posts.

    Stages: pre_breakout (young, over-performing, few accounts, no press) -> breaking_out (several accounts
    joined in the last 6 h) -> peak (broad or covered by press) -> saturated (wide press) -> simmering (active,
    ongoing debate over days) -> settled (debate concluded, no recent activity in >48h or low momentum).
    """
    media = topic.get("instagram", {}).get("media", [])
    if topic.get("brief"):
        on_topic = set(topic["brief"].get("on_topic_codes", []))
        media = [m for m in media if m.get("code") in on_topic]
    aged = [(hours_since(m.get("taken_at"), reference), m) for m in media]
    aged = [(age, m) for age, m in aged if age is not None and age <= 168]
    if not aged:
        return {"stage": "quiet", "score": 0.0, "reasons": [], "signals": {}}
    young = [m for age, m in aged if age <= 6]
    early = [m for age, m in aged if age <= 12]
    day = [m for age, m in aged if age <= 24]
    multi_day = [m for age, m in aged if age <= 72]
    newest = min(age for age, _ in aged)
    authors_6h = len({m.get("author_id") for m in young})
    authors_24h = len({m.get("author_id") for m in day})
    authors_7d = len({m.get("author_id") for _, m in aged})
    # Likes-only posts carry None for over_comments/ratio_lift; treat them as 0.
    max_over = max(((m["panel"].get("over_comments") or 0.0) for m in early if m.get("panel")), default=0.0)
    max_lift = max(((m["panel"].get("ratio_lift") or 0.0) for m in early if m.get("panel")), default=0.0)
    comment_rate = sum((m.get("comment_count") or 0) / max(age, 0.5) for age, m in aged if age <= 12)
    comments_total = sum(m.get("comment_count") or 0 for _, m in aged)
    sampled = [c for _, m in aged for c in m.get("comments", [])]
    replies = sum(c.get("reply_count") or 0 for c in sampled) / len(sampled) if sampled else 0.0
    publishers = topic.get("publishers", 0) or 0
    previous = (topic.get("previous") or {}).get("instagram_heat")
    current = (topic.get("scores") or {}).get("instagram_heat")
    growth = current / previous if previous and current else None

    # Multi-day activity momentum:
    multiday_momentum = 0.4 * math.log1p(comments_total / 100) + 0.3 * min(authors_7d, 5)

    score = (1.2 * math.log1p(max_over) + 0.8 * math.log1p(max(max_lift - 1, 0)) + 0.5 * math.log1p(comment_rate / 10)
             + 0.6 * min(authors_6h, 5) + (1.0 if newest <= 3 else 0.5 if newest <= 6 else 0.0)
             + 0.5 * math.log1p(replies) + (0.8 * math.log1p(growth - 1) if growth and growth > 1 else 0.0)
             + multiday_momentum)
    score /= 1 + 0.5 * publishers
    if comments_total > 5000:
        score *= 0.7

    # Active multi-day debate condition:
    # If posts exist across multiple days with active participation (>=3 accounts or comments >= 150),
    # the topic remains an active 'simmering' multi-day debate rather than prematurely 'settled'.
    has_active_multiday_debate = (len(aged) >= 3 and (authors_7d >= 3 or comments_total >= 150 or replies >= 0.5))

    if newest > 48 or (newest > 24 and not has_active_multiday_debate):
        stage = "settled"
    elif publishers >= 5:
        stage = "saturated"
    elif publishers >= 3 or authors_24h >= 8:
        stage = "peak"
    elif authors_6h >= 3 or (authors_24h >= 3 and len(young) >= 0.5 * len(day)):
        stage = "breaking_out"
    elif newest <= 12 and (max_over >= 2 or max_lift >= 1.5 or comment_rate >= 30 or (growth or 0) >= 1.3):
        stage = "pre_breakout"
    else:
        stage = "simmering"

    # Neural trajectory model calibration if topic text is present
    topic_text = " ".join(filter(None, [topic.get("title"), (topic.get("brief") or {}).get("summary")]))
    neural_reason = None
    if topic_text.strip():
        try:
            from identify_what_matters.strategic_classifier import classify_strategic
            neural_s = classify_strategic(topic_text)
            n_stage = neural_s.get("trajectory_stage")
            n_conf = neural_s.get("trajectory_stage_conf", 0.0)
            if n_stage and n_conf >= 0.60:
                stage_map = {
                    "pre_breakout": "pre_breakout",
                    "breaking_out": "breaking_out",
                    "peak_saturation": "saturated" if publishers >= 4 else "peak",
                    "simmering_entropy": "simmering",
                    "settling_quiet": "settled"
                }
                mapped_stage = stage_map.get(n_stage)
                if mapped_stage and stage in ("simmering", "quiet", "pre_breakout"):
                    stage = mapped_stage
                    neural_reason = f"Strategic neural classifier detected {n_stage.replace('_', ' ')} dynamics ({n_conf:.0%} conf)"
        except Exception:
            pass

    reasons = [f"newest post {newest:.1f} h old",
               f"{authors_6h} account(s) posted in the last 6 h, {authors_24h} in 24 h"]
    if max_over >= 1.5:
        reasons.append(f"an early post has {max_over}x its account's usual comments")
    if max_lift >= 1.3:
        reasons.append(f"comment:like {max_lift}x the account's norm (people arguing, not just liking)")
    if comment_rate >= 5:
        reasons.append(f"{comment_rate:.0f} comments/h on posts under 12 h")
    if replies >= 0.5:
        reasons.append(f"{replies:.1f} replies per sampled comment")
    if growth:
        reasons.append(f"Instagram heat {'up' if growth >= 1 else 'down'} {abs(growth - 1):.0%} since the last run")
    reasons.append("no news coverage yet" if publishers == 0 else f"{publishers} news publisher(s)")
    if neural_reason:
        reasons.append(neural_reason)
    return {"stage": stage, "score": round(score, 3), "reasons": reasons,
            "signals": {"newest_hours": round(newest, 2), "authors_6h": authors_6h, "authors_24h": authors_24h,
                        "max_over_comments": max_over, "max_ratio_lift": max_lift,
                        "comments_per_hour_12h": round(comment_rate, 1), "replies_per_comment": round(replies, 2),
                        "publishers": publishers, "growth_vs_last_run": round(growth, 2) if growth else None}}

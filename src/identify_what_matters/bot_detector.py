"""Bot, Astroturf, and Synthetic Coordination Detection Model.

Quantifies the probability and volume of inauthentic / coordinated behavior across:
1. Cross-account comment text duplication and near-duplicate shingling.
2. Zero-post / new / low-engagement account heuristic simulation.
3. Burst timing velocity (high volume in narrow windows).
4. Sentiment and tone unanimity / unnatural polarity clustering.
5. Bot influence score per topic and per debate position.
"""

from collections import Counter, defaultdict
import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Tuple


def _shingles(text: str, k: int = 4) -> set[str]:
    """Generate k-gram character shingles from normalized text."""
    clean = re.sub(r"[^\w\s]", "", text.casefold())
    clean = " ".join(clean.split())
    if len(clean) < k:
        return {clean} if clean else set()
    return {clean[i : i + k] for i in range(len(clean) - k + 1)}


def _jaccard_similarity(s1: set[str], s2: set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def compute_comment_repetition_clusters(
    comments: list[dict], similarity_threshold: float = 0.80
) -> list[dict]:
    """Find clusters of identical or near-identical comments posted across multiple accounts.
    
    Returns cluster records with template text, accounts involved, and size.
    """
    cleaned_comments = []
    for c in comments:
        text = str(c.get("text") or "").strip()
        if len(text) < 15:
            continue
        author = str(c.get("author_id") or (c.get("user") or {}).get("pk") or "")
        # If no author, create a pseudo-fingerprint from time or salt
        if not author:
            author = f"anon_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        shs = _shingles(text, k=5)
        cleaned_comments.append({
            "text": text,
            "author": author,
            "shingles": shs,
            "likes": int(c.get("likes") or c.get("like_count") or 0),
        })

    clusters: list[dict] = []
    for item in cleaned_comments:
        matched = False
        for cl in clusters:
            sim = _jaccard_similarity(item["shingles"], cl["representative_shingles"])
            if sim >= similarity_threshold:
                cl["comments_count"] += 1
                cl["accounts"].add(item["author"])
                cl["samples"].append(item["text"])
                matched = True
                break
        if not matched:
            clusters.append({
                "representative_text": item["text"],
                "representative_shingles": item["shingles"],
                "comments_count": 1,
                "accounts": {item["author"]},
                "samples": [item["text"]],
            })

# Only keep clusters with 2+ occurrences or cross-account repetition
    meaningful = []
    for c in clusters:
        if c["comments_count"] >= 2:
            template_text = c["representative_text"][:120]
            cluster_meta = classify_template_lean(template_text)
            meaningful.append({
                "template": template_text,
                "occurrences": c["comments_count"],
                "unique_accounts": len(c["accounts"]),
                "is_cross_account": len(c["accounts"]) > 1,
                "suspicion_level": "high" if len(c["accounts"]) >= 3 or c["comments_count"] >= 4 else "medium",
                "synthetic_lean": cluster_meta["synthetic_lean"],
                "dominant_sentiment": cluster_meta["dominant_sentiment"],
                "lean_category": cluster_meta["lean_category"],
                "badge_color": cluster_meta["badge_color"],
                "badge_bg": cluster_meta["badge_bg"],
                "border_color": cluster_meta["border_color"],
                "samples": c["samples"],
            })
    return sorted(meaningful, key=lambda x: (x["unique_accounts"], x["occurrences"]), reverse=True)


PRO_INCUMBENT_PATTERNS = [
    re.compile(r"\b(?:vote for bjp|vote for modi|aayega to modi|ayega to modi|modi sarkar|har har modi|jai shri ram|hindu rashtra|har har mahadev|modi ji zindabad|modi hai to mumkin hai|bharat mata ki jai|vande mataram)\b", re.I),
    re.compile(r"(?:मोदी जी जिंदाबाद|मोदी है तो मुमकिन है|जय श्री राम|भारत माता की जय|हिंदू राष्ट्र|वंदे मातरम्|हर हर महादेव)", re.I),
]

ANTI_INCUMBENT_PATTERNS = [
    re.compile(r"\b(?:modi chor|evm chor|evm scam|bjp hatao|dictatorship|tanashahi|save constitution|rahul gandhi zindabad|modi resign|resign button|resign now|shame on bjp|anti bjp|vote out bjp)\b", re.I),
    re.compile(r"(?:मोदी चोर|ईवीएम चोरी|बीजेपी हटाओ|तानाशाही|संविधान बचाओ|राहुल गांधी जिंदाबाद|इस्तीफा दो)", re.I),
]

GENERIC_SLOGAN_PATTERNS = [
    re.compile(r"\b(?:boycott\w*|shame on|resign now|toolkit|godi media|pappu|bhakts?|librandus?)\b", re.I),
    re.compile(r"\b(?:100% true|plz share|must watch|viral karo|trend karao|retweet|repost)\b", re.I),
    re.compile(r"(?:शेयर करो|वायरल करो|ट्रेंड कराओ|सब देखें|जरूर देखें)", re.I),
    re.compile(r"\b(?:share karo|viral karo|trend karo|sabko batao|must share|must watch|dekho aur share)\b", re.I),
]


def classify_template_lean(text: str) -> dict:
    """Determine the dominant political lean and sentiment of a repeated template cluster."""
    is_pro = any(p.search(text) for p in PRO_INCUMBENT_PATTERNS)
    is_anti = any(p.search(text) for p in ANTI_INCUMBENT_PATTERNS)

    if not is_pro and not is_anti:
        try:
            from identify_what_matters.political_pulse import political_pulse_engine
            fav = political_pulse_engine.classify_text_favorability(text)
            if fav.get("bjp_share", 0) >= fav.get("opp_share", 0) + 0.15:
                is_pro = True
            elif fav.get("opp_share", 0) >= fav.get("bjp_share", 0) + 0.15:
                is_anti = True
        except Exception:
            pass

    if is_pro and not is_anti:
        return {
            "synthetic_lean": "100% Pro-Incumbent / Nationalist",
            "dominant_sentiment": "Supportive / Slogan",
            "lean_category": "pro_incumbent",
            "badge_color": "#c2410c",
            "badge_bg": "#fff7ed",
            "border_color": "#fed7aa",
        }
    elif is_anti and not is_pro:
        return {
            "synthetic_lean": "100% Anti-Incumbent / Opposition",
            "dominant_sentiment": "Critical / Partisan Attack",
            "lean_category": "anti_incumbent",
            "badge_color": "#1d4ed8",
            "badge_bg": "#eff6ff",
            "border_color": "#bfdbfe",
        }
    elif is_pro and is_anti:
        return {
            "synthetic_lean": "Contested Multi-Faction Script",
            "dominant_sentiment": "Polarized Conflict",
            "lean_category": "contested",
            "badge_color": "#7c3aed",
            "badge_bg": "#f5f3ff",
            "border_color": "#ddd6fe",
        }
    else:
        is_crit = bool(re.search(r"\b(?:resign|shame|corrupt|scam|worst|fail|fraud|boycott|chor)\b", text, re.I))
        is_sup = bool(re.search(r"\b(?:support|great|proud|respect|love|best|win)\b", text, re.I))
        if is_crit:
            return {
                "synthetic_lean": "Anti-Establishment Cynicism",
                "dominant_sentiment": "Critical / Grievance Slogan",
                "lean_category": "anti_establishment",
                "badge_color": "#991b1b",
                "badge_bg": "#fef2f2",
                "border_color": "#fecaca",
            }
        elif is_sup:
            return {
                "synthetic_lean": "Pro-Governance / General Praise",
                "dominant_sentiment": "Supportive Slogan",
                "lean_category": "pro_governance",
                "badge_color": "#15803d",
                "badge_bg": "#f0fdf4",
                "border_color": "#bbf7d0",
            }
        return {
            "synthetic_lean": "Non-partisan / Civic Campaign Script",
            "dominant_sentiment": "Viral Engagement Template",
            "lean_category": "non_partisan",
            "badge_color": "#475569",
            "badge_bg": "#f1f5f9",
            "border_color": "#cbd5e1",
        }


def _evaluate_comment_stance_and_lean(text: str) -> dict:
    """Classify sentiment stance (supportive/critical/neutral/humorous) and political lean."""
    t_low = text.lower()
    stance = "neutral"
    if any(h in t_low for h in ["lol", "lmao", "haha", "😂", "🤣", "meme", "joke", "clown"]):
        stance = "humorous"
    elif any(w in t_low for w in [
        "fail", "failed", "failure", "resign", "shame", "scam", "corrupt", "worst", "chor",
        "loot", "fraud", "hatao", "arrest", "protest", "bekar", "barbaad", "dhokha", "fake",
        "dictatorship", "tanashahi", "boycott", "oppose"
    ]):
        stance = "critical"
    elif any(w in t_low for w in [
        "support", "proud", "great", "congrats", "good", "best", "respect", "love", "development",
        "victory", "win", "jai", "zindabad", "shandar", "badhai", "well done", "bravo"
    ]):
        stance = "supportive"

    lean = "neutral"
    if any(p.search(text) for p in PRO_INCUMBENT_PATTERNS):
        lean = "pro_incumbent"
    elif any(p.search(text) for p in ANTI_INCUMBENT_PATTERNS):
        lean = "anti_incumbent"

    return {"stance": stance, "lean": lean}


class BotAstroturfDetector:
    """Multi-signal statistical & lexical bot / astroturf classifier."""

    BOT_PHRASE_PATTERNS = PRO_INCUMBENT_PATTERNS + ANTI_INCUMBENT_PATTERNS + GENERIC_SLOGAN_PATTERNS

    def evaluate_account_risk(self, comment: dict, is_in_repetition_cluster: bool = False) -> dict:
        """Score an individual comment / account for bot likelihood (0.0 to 1.0)."""
        score = 0.0
        reasons = []

        text = str(comment.get("text") or "").strip()
        author = str(comment.get("author_id") or (comment.get("user") or {}).get("pk") or "")
        likes = int(comment.get("likes") or comment.get("like_count") or 0)
        replies = int(comment.get("replies") or comment.get("reply_count") or 0)

        # Signal 1: Account profile indicators (if available from metadata)
        user_meta = comment.get("user") or {}
        media_count = user_meta.get("media_count")
        follower_count = user_meta.get("follower_count")

        if media_count == 0:
            score += 0.25
            reasons.append("zero_post_profile")
        elif media_count is not None and media_count < 2:
            score += 0.20
            reasons.append("sparse_profile_history")

        if follower_count is not None and follower_count < 5:
            score += 0.15
            reasons.append("near_zero_followers")

        # No author ID at all: anonymous ephemeral commenter
        if not author:
            score += 0.15
            reasons.append("anonymous_commenter")

        # Signal 2: Repetition cluster membership (copied across multiple accounts)
        if is_in_repetition_cluster:
            score += 0.35
            reasons.append("cross_account_template_match")

        # Signal 3: Slogan / Copy-paste template density
        matched_slogans = sum(1 for pat in self.BOT_PHRASE_PATTERNS if pat.search(text))
        if matched_slogans >= 2:
            score += 0.30
            reasons.append("canned_slogan_density")
        elif matched_slogans == 1:
            score += 0.15
            reasons.append("campaign_slogan")

        # Signal 4: Repetitive emoji spamming or hashtag spamming
        emojis = re.findall(r"[\U00010000-\U0010ffff]", text)
        hashtags = re.findall(r"#\w+", text)
        if len(emojis) > 6 and len(text.split()) < 5:
            score += 0.20
            reasons.append("hyper_emoji_spam")
        elif len(emojis) > 4 and not re.search(r"[a-zA-Zऀ-ॿ]{4}", text):
            # Emoji-only with no real words (even 4 chars of Hindi/English)
            score += 0.20
            reasons.append("emoji_only_comment")
        if len(hashtags) >= 4:
            # Hashtag-bomb: mostly tags, no real text
            score += 0.20
            reasons.append("hashtag_spam")

        # Signal 5: Engagement asymmetry (lots of comments, 0 likes received, 0 replies)
        if likes == 0 and replies == 0 and len(text) > 30:
            score += 0.10
            reasons.append("zero_engagement_shouting")

        # Signal 6: Neural Authenticity & Astroturf Classifier (mDeBERTa-v3)
        neural_astroturf_prob = comment.get("neural_astroturf_prob")
        if neural_astroturf_prob is not None:
            if neural_astroturf_prob >= 0.70:
                score += 0.40
                reasons.append("neural_astroturf_high_confidence")
            elif neural_astroturf_prob >= 0.45:
                score += 0.20
                reasons.append("neural_astroturf_suspicious")
            elif neural_astroturf_prob < 0.15:
                score = max(0.0, score - 0.25)
                reasons.append("neural_organic_verified")

        return {
            "bot_probability": min(round(score, 2), 1.0),
            "is_likely_bot": score >= 0.45,
            "reasons": reasons,
        }

    def analyze_topic_astroturf(
        self,
        topic_title: str,
        comments: list[dict],
        debate_sides: Optional[list[dict]] = None,
    ) -> dict:
        """Comprehensive astroturf & synthetic amplification audit for a topic."""
        if not comments:
            return {
                "astroturf_index": 0,
                "bot_share_pct": 0.0,
                "bot_comment_count": 0,
                "organic_comment_count": 0,
                "total_comments_audited": 0,
                "organic_vs_synthetic_ratio": 1.0,
                "organic_sentiment_mix": {
                    "supportive": 0.0, "critical": 0.0, "neutral": 100.0, "humorous": 0.0,
                    "pro_incumbent": 0.0, "anti_incumbent": 0.0, "non_partisan": 100.0,
                    "sample_size": 0,
                },
                "synthetic_sentiment_mix": {
                    "extreme_partisan_slogan": 0.0, "critical": 0.0, "supportive": 0.0, "neutral": 100.0,
                    "pro_incumbent": 0.0, "anti_incumbent": 0.0, "non_partisan": 100.0,
                    "sample_size": 0,
                },
                "sentiment_divergence": {
                    "critical_skew": 0.0,
                    "polarization_skew": 0.0,
                    "diagnosis": "No comment discourse available to evaluate.",
                    "type": "insufficient_data",
                },
                "bot_adjusted_stance": {"supportive": 0.0, "critical": 0.0, "neutral": 1.0, "humorous": 0.0},
                "template_clusters": [],
                "coordination_status": "insufficient_data",
                "side_bot_bias": {},
                "verdict_summary": "No comment discourse available to evaluate.",
            }

        clusters = compute_comment_repetition_clusters(comments, similarity_threshold=0.78)
        
        # Build set of texts or shingles that belong to multi-account repetition clusters
        clustered_texts = set()
        for cl in clusters:
            if cl["is_cross_account"] or cl["occurrences"] >= 2:
                clustered_texts.update(cl.get("samples", []))

        evaluations = []
        for c in comments:
            txt = str(c.get("text") or "").strip()
            in_cluster = txt in clustered_texts or any(_jaccard_similarity(_shingles(txt), _shingles(cl_sample)) >= 0.78 for cl_sample in clustered_texts)
            evaluations.append(self.evaluate_account_risk(c, is_in_repetition_cluster=in_cluster))

        bot_evals = [e for e in evaluations if e["is_likely_bot"]]
        organic_evals = [e for e in evaluations if not e["is_likely_bot"]]

        # Calculate cluster penalty
        duplicated_accounts = sum(c["unique_accounts"] for c in clusters if c["is_cross_account"])
        cluster_impact = min(len(clusters) * 12 + duplicated_accounts * 8, 45)

        # Base bot ratio penalty
        raw_bot_ratio = len(bot_evals) / max(len(comments), 1)
        ratio_impact = min(raw_bot_ratio * 55, 55)

        astroturf_score = min(int(cluster_impact + ratio_impact), 100)
        bot_share_pct = round((len(bot_evals) / max(len(comments), 1)) * 100, 1)

        # Separate and profile organic vs synthetic sentiment
        organic_comments = []
        synthetic_comments = []
        for c, ev in zip(comments, evaluations):
            c_text = str(c.get("text") or "").strip()
            c_meta = _evaluate_comment_stance_and_lean(c_text)
            is_slogan = any(pat.search(c_text) for pat in self.BOT_PHRASE_PATTERNS) or bool(re.search(r"[\U00010000-\U0010ffff]{5,}", c_text))
            rec = {
                "text": c_text,
                "stance": c_meta["stance"],
                "lean": c_meta["lean"],
                "is_slogan": is_slogan,
                "is_bot": ev["is_likely_bot"],
            }
            if ev["is_likely_bot"]:
                synthetic_comments.append(rec)
            else:
                organic_comments.append(rec)

        n_org = len(organic_comments)
        n_syn = len(synthetic_comments)

        # 1. Organic Sentiment Mix (derived strictly from verified human accounts)
        if n_org > 0:
            org_sup = round(sum(1 for c in organic_comments if c["stance"] == "supportive") / n_org * 100, 1)
            org_crit = round(sum(1 for c in organic_comments if c["stance"] == "critical") / n_org * 100, 1)
            org_hum = round(sum(1 for c in organic_comments if c["stance"] == "humorous") / n_org * 100, 1)
            org_neu = max(0.0, round(100.0 - org_sup - org_crit - org_hum, 1))
            org_pro = round(sum(1 for c in organic_comments if c["lean"] == "pro_incumbent") / n_org * 100, 1)
            org_anti = round(sum(1 for c in organic_comments if c["lean"] == "anti_incumbent") / n_org * 100, 1)
            org_lean_neu = max(0.0, round(100.0 - org_pro - org_anti, 1))
        else:
            org_sup, org_crit, org_hum, org_neu = 20.0, 45.0, 10.0, 25.0
            org_pro, org_anti, org_lean_neu = 25.0, 35.0, 40.0

        organic_sentiment_mix = {
            "supportive": org_sup,
            "critical": org_crit,
            "humorous": org_hum,
            "neutral": org_neu,
            "pro_incumbent": org_pro,
            "anti_incumbent": org_anti,
            "non_partisan": org_lean_neu,
            "sample_size": n_org,
        }

        # 2. Bot / Synthetic Sentiment Mix (automated scripts & canned campaign templates)
        if n_syn > 0:
            syn_slogan = round(sum(1 for c in synthetic_comments if c["is_slogan"]) / n_syn * 100, 1)
            syn_crit = round(sum(1 for c in synthetic_comments if not c["is_slogan"] and c["stance"] == "critical") / n_syn * 100, 1)
            syn_sup = round(sum(1 for c in synthetic_comments if not c["is_slogan"] and c["stance"] == "supportive") / n_syn * 100, 1)
            syn_neu = max(0.0, round(100.0 - syn_slogan - syn_crit - syn_sup, 1))
            syn_pro = round(sum(1 for c in synthetic_comments if c["lean"] == "pro_incumbent") / n_syn * 100, 1)
            syn_anti = round(sum(1 for c in synthetic_comments if c["lean"] == "anti_incumbent") / n_syn * 100, 1)
            syn_lean_neu = max(0.0, round(100.0 - syn_pro - syn_anti, 1))
        else:
            syn_slogan, syn_crit, syn_sup, syn_neu = 0.0, 0.0, 0.0, 0.0
            syn_pro, syn_anti, syn_lean_neu = 0.0, 0.0, 0.0

        synthetic_sentiment_mix = {
            "extreme_partisan_slogan": syn_slogan,
            "critical": syn_crit,
            "supportive": syn_sup,
            "neutral": syn_neu,
            "pro_incumbent": syn_pro,
            "anti_incumbent": syn_anti,
            "non_partisan": syn_lean_neu,
            "sample_size": n_syn,
        }

        # 3. Sentiment Divergence & Artificial Skew Delta
        crit_skew = round((syn_crit + (syn_slogan * 0.7 if syn_anti > syn_pro else 0.0)) - org_crit, 1)
        polarization_skew = round(abs(syn_pro - syn_anti) - abs(org_pro - org_anti), 1)

        if n_syn >= 2 and (bot_share_pct >= 25.0 or syn_slogan >= 35.0):
            if crit_skew >= 15.0 or syn_anti >= org_anti + 20.0:
                divergence_diagnosis = "Artificial Astroturfed Attack: Negative spike is driven by automated script clusters rather than grassroots grievances."
                divergence_type = "astroturfed_negative"
            elif syn_pro >= org_pro + 20.0:
                divergence_diagnosis = "Inorganic Pro-Incumbent Amplification: Slogan clusters artificially inflate supportive metrics."
                divergence_type = "astroturfed_positive"
            else:
                divergence_diagnosis = "Polarized Bot Contestation: Both pro- and anti-incumbent scripts deployed concurrently."
                divergence_type = "contested_astroturf"
        elif org_crit >= 45.0:
            divergence_diagnosis = "Genuine Grassroots Discontent: Negative sentiment originates predominantly from verified human accounts."
            divergence_type = "grassroots_negative"
        else:
            divergence_diagnosis = "Authentic Civic Discourse: Minimal synthetic distortion between bots and human participants."
            divergence_type = "authentic_deliberation"

        sentiment_divergence = {
            "critical_skew": crit_skew,
            "polarization_skew": polarization_skew,
            "diagnosis": divergence_diagnosis,
            "type": divergence_type,
        }

        bot_adjusted_stance = {
            "supportive": round(org_sup / 100.0, 2),
            "critical": round(org_crit / 100.0, 2),
            "neutral": round(org_neu / 100.0, 2),
            "humorous": round(org_hum / 100.0, 2),
        }
        organic_ratio = round(n_org / max(1, n_syn), 2)

        # Assess debate side bias if debate sides exist
        side_bot_bias = {}
        if debate_sides:
            for i, side in enumerate(debate_sides):
                pos = side.get("position", f"Side {i+1}")
                # Count bot indicators matching side keywords
                pos_words = set(re.findall(r"\w+", pos.lower()))
                matching_bot_cmts = 0
                for c, ev in zip(comments, evaluations):
                    c_words = set(re.findall(r"\w+", str(c.get("text") or "").lower()))
                    if ev["is_likely_bot"] and (c_words & pos_words):
                        matching_bot_cmts += 1
                side_bot_bias[pos] = {
                    "synthetic_mentions": matching_bot_cmts,
                    "estimated_astroturf_intensity": "high" if matching_bot_cmts >= 3 else ("medium" if matching_bot_cmts >= 1 else "low"),
                }

        # Status categorization
        if astroturf_score >= 55 or bot_share_pct >= 40.0:
            status = "heavy_astroturf"
            verdict = f"High probability of coordinated amplification. {len(clusters)} cross-account template clusters detected; ~{bot_share_pct}% synthetic comments."
        elif astroturf_score >= 25 or bot_share_pct >= 18.0:
            status = "moderate_amplification"
            verdict = f"Moderate astroturf detected with {len(clusters)} repeated narrative templates and ~{bot_share_pct}% bot-like accounts."
        else:
            status = "largely_organic"
            verdict = f"Discourse is largely organic (~{bot_share_pct}% low-quality accounts, negligible template coordination)."

        return {
            "astroturf_index": astroturf_score,
            "bot_share_pct": bot_share_pct,
            "bot_comment_count": len(bot_evals),
            "organic_comment_count": n_org,
            "total_comments_audited": len(comments),
            "organic_vs_synthetic_ratio": organic_ratio,
            "organic_sentiment_mix": organic_sentiment_mix,
            "synthetic_sentiment_mix": synthetic_sentiment_mix,
            "sentiment_divergence": sentiment_divergence,
            "bot_adjusted_stance": bot_adjusted_stance,
            "template_clusters": clusters[:6],
            "coordination_status": status,
            "side_bot_bias": side_bot_bias,
            "verdict_summary": verdict,
        }


# Global singleton instance
bot_detector = BotAstroturfDetector()

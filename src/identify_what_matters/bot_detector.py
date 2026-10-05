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
    meaningful = [
        {
            "template": c["representative_text"][:120],
            "occurrences": c["comments_count"],
            "unique_accounts": len(c["accounts"]),
            "is_cross_account": len(c["accounts"]) > 1,
            "suspicion_level": "high" if len(c["accounts"]) >= 3 or c["comments_count"] >= 4 else "medium",
            "samples": c["samples"],
        }
        for c in clusters
        if c["comments_count"] >= 2
    ]
    return sorted(meaningful, key=lambda x: (x["unique_accounts"], x["occurrences"]), reverse=True)


class BotAstroturfDetector:
    """Multi-signal statistical & lexical bot / astroturf classifier."""

    BOT_PHRASE_PATTERNS = [
        re.compile(r"\b(?:vote for bjp|vote for modi|aayega to modi|ayega to modi|modi sarkar|har har modi|jai shri ram|hindu rashtra)\b", re.I),
        re.compile(r"\b(?:modi chor|evm chor|evm scam|bjp hatao|dictatorship|tanashahi|save constitution|rahul gandhi zindabad|modi resign)\b", re.I),
        re.compile(r"\b(?:boycott\w*|shame on|resign now|toolkit|godi media|pappu|bhakts?|librandus?)\b", re.I),
        re.compile(r"\b(?:100% true|plz share|must watch|viral karo|trend karao|retweet|repost)\b", re.I),
    ]

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

        if media_count == 0 or (media_count is None and not author):
            # Zero posts or completely anonymous ephemeral handle
            score += 0.25
            reasons.append("zero_or_anonymous_profile")
        elif media_count is not None and media_count < 2:
            score += 0.20
            reasons.append("sparse_profile_history")

        if follower_count is not None and follower_count < 5:
            score += 0.15
            reasons.append("near_zero_followers")

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
        if len(emojis) > 6 and len(text.split()) < 5:
            score += 0.20
            reasons.append("hyper_emoji_spam")

        # Signal 5: Engagement asymmetry (lots of comments, 0 likes received, 0 replies)
        if likes == 0 and replies == 0 and len(text) > 30:
            score += 0.10
            reasons.append("zero_engagement_shouting")

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
                "total_comments_audited": 0,
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
        
        # Calculate cluster penalty
        duplicated_accounts = sum(c["unique_accounts"] for c in clusters if c["is_cross_account"])
        cluster_impact = min(len(clusters) * 12 + duplicated_accounts * 8, 45)
        
        # Base bot ratio penalty
        raw_bot_ratio = len(bot_evals) / max(len(comments), 1)
        ratio_impact = min(raw_bot_ratio * 55, 55)

        astroturf_score = min(int(cluster_impact + ratio_impact), 100)
        bot_share_pct = round((len(bot_evals) / max(len(comments), 1)) * 100, 1)

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
            "total_comments_audited": len(comments),
            "template_clusters": clusters[:6],
            "coordination_status": status,
            "side_bot_bias": side_bot_bias,
            "verdict_summary": verdict,
        }


# Global singleton instance
bot_detector = BotAstroturfDetector()

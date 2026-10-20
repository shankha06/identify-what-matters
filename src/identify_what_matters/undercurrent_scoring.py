"""Multi-signal fusion for the political "Undercurrent Score".

Key design: surface the gap between what a POST frames and how the AUDIENCE actually reacted.

Signal hierarchy (best → fallback):
  post framing   → LLM brief's political_framing field (explicit direction from Gemini)
                 → keyword classifier on post caption (weak, often neutral)
  comment reaction → political_pulse percentages (BJP% / Opp% / Neutral% from comment pool)
                   → keyword majority-vote on comments (old path, weak)

Framing-Reaction Gap is computed as a continuous 0–1 score (not binary), so topics where
comments are mildly contrary to framing still get partial credit rather than zero.

The undercurrent_narrative field gives a plain-English sentence surfacing the divergence —
this is what the dashboard should show instead of the raw score alone.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from identify_what_matters import political_classifier as pc

NON_NEUTRAL = {"pro_incumbent", "anti_incumbent", "pro_opposition", "anti_opposition"}
_LEAN_MAP = {
    "pro_incumbent":   1.0,
    "anti_opposition": 0.5,
    "neutral":         0.0,
    "pro_opposition":  -0.5,
    "anti_incumbent":  -1.0,
}


def _caption_text(topic: dict) -> str:
    brief = topic.get("brief") or {}
    for candidate in (brief.get("summary"), topic.get("title")):
        if candidate and str(candidate).strip():
            return str(candidate)
    return ""


def _post_framing(topic: dict) -> tuple[str, float, str]:
    """Returns (direction, confidence, source). Priority: LLM brief > keyword classifier."""
    brief = topic.get("brief") or {}
    pf = brief.get("political_framing") or {}
    direction = pf.get("direction", "")
    confidence = float(pf.get("confidence") or 0.0)
    if direction in _LEAN_MAP and confidence > 0.05:
        basis = pf.get("basis", "")
        return direction, confidence, f"llm ({basis})" if basis else "llm"

    # Keyword fallback on post caption
    post_text = _caption_text(topic)
    if post_text:
        cls = pc.classify(post_text)
        kw_dir = cls["direction"]["value"]
        kw_conf = cls["direction"]["confidence"]
        return kw_dir, kw_conf, cls["direction"]["source"]

    return "neutral", 0.0, "none"


def _comment_reaction(
    political_pulse: dict | None,
    comment_pool: list[dict],
    bot_share: float,
) -> tuple[str, float, int, float, float, str]:
    """Returns (direction, confidence, n_engagements, bjp_pct, opp_pct, source)."""
    if political_pulse and political_pulse.get("engagement_count", 0) >= 3:
        bjp_pct = float(political_pulse.get("bjp_favor_pct") or 0)
        opp_pct = float(political_pulse.get("opp_favor_pct") or 0)
        n = int(political_pulse.get("engagement_count") or 0)

        if bjp_pct >= opp_pct + 10:
            direction = "pro_incumbent"
            dominant = bjp_pct
        elif opp_pct >= bjp_pct + 10:
            direction = "anti_incumbent"
            dominant = opp_pct
        else:
            direction = "neutral"
            dominant = 100 - bjp_pct - opp_pct

        confidence = round((dominant / 100) * max(0.0, 1.0 - bot_share), 3)
        return direction, confidence, n, bjp_pct, opp_pct, "political_pulse"

    # Fallback: keyword majority-vote on raw comments
    texts = [c.get("text", "") for c in (comment_pool or []) if c.get("text", "").strip()]
    if not texts:
        return "neutral", 0.0, 0, 0.0, 0.0, "none"
    classed = [pc.classify(t) for t in texts]
    votes = Counter(c["direction"]["value"] for c in classed)
    top_dir, top_n = votes.most_common(1)[0]
    agreement = top_n / len(classed)
    confidence = round(agreement * max(0.0, 1.0 - bot_share), 3)
    return top_dir, confidence, len(classed), 0.0, 0.0, "keyword_classifier"


def _framing_gap(post_lean: float, comment_lean: float, pulse_bjp: float, pulse_opp: float,
                 reaction_source: str) -> float:
    """Continuous 0–1 gap score. Uses pulse percentages for comment lean when available."""
    if reaction_source == "political_pulse" and (pulse_bjp + pulse_opp) > 0:
        # Richer signal: (-1..+1) from actual percentages
        comment_lean_precise = (pulse_bjp - pulse_opp) / 100.0
        gap = abs(post_lean - comment_lean_precise)
    else:
        gap = abs(post_lean - comment_lean)
    return round(min(gap / 2.0, 1.0), 3)  # normalise 0–2 range → 0–1


def _narrative(
    post_dir: str, post_conf: float, post_source: str,
    comment_dir: str, comment_conf: float,
    bjp_pct: float, opp_pct: float, n: int,
    gap: float, bot_share_pct: float,
    brief: dict,
) -> str:
    bot_note = f", bot-discounted ({bot_share_pct:.0f}% suspected coordination)" if bot_share_pct > 15 else ""
    pulse_note = f"{bjp_pct:.0f}% BJP-lean / {opp_pct:.0f}% opposition-lean / {100-bjp_pct-opp_pct:.0f}% neutral across {n} engagements" if n > 0 else f"{n} engagements"

    if n == 0:
        return "No comment engagements measured — cannot determine audience reaction."

    if gap < 0.12:
        if comment_dir == "neutral" and bjp_pct < 20 and opp_pct < 20:
            return (f"Audience reaction is diffuse ({pulse_note}{bot_note}). "
                    "No strong political lean detected in comments — topic may be genuinely non-partisan or discussion is early-stage.")
        dir_label = "pro-incumbent" if comment_dir == "pro_incumbent" else "anti-incumbent" if comment_dir == "anti_incumbent" else comment_dir.replace("_", " ")
        return (f"Framing and audience reaction align ({dir_label}). "
                f"Comment pool: {pulse_note}{bot_note}.")

    if gap < 0.35:
        return (f"Mild divergence between post framing ({post_dir.replace('_', ' ')}) and comment reaction. "
                f"Comment pool: {pulse_note}{bot_note}. "
                "Audience is partially pushing back against the post's narrative frame.")

    # Significant gap — generate specific narrative
    if post_dir in ("pro_incumbent",) and opp_pct > 35:
        return (f"Post frames this as an incumbent achievement or defence, "
                f"but {opp_pct:.0f}% of comment engagement is opposition/critical{bot_note}. "
                f"Strong public scepticism runs against the post's narrative ({pulse_note}).")

    if post_dir in ("anti_incumbent",) and bjp_pct > 35:
        return (f"Post is critically framed against the ruling party, "
                f"but {bjp_pct:.0f}% of comment engagement is pro-incumbent{bot_note}. "
                f"BJP supporters are defending in the comments despite the critical framing ({pulse_note}).")

    if post_dir in ("anti_opposition",) and opp_pct > 35:
        return (f"Post attacks the opposition, but {opp_pct:.0f}% of comment engagement is opposition-leaning{bot_note}. "
                f"Comments are defending the opposition against the post's narrative ({pulse_note}).")

    if post_dir in ("pro_opposition",) and bjp_pct > 35:
        return (f"Post is favourable to the opposition, but {bjp_pct:.0f}% of comment engagement is pro-BJP{bot_note}. "
                f"Ruling party supporters are pushing back in comments ({pulse_note}).")

    # Generic gap
    return (f"Notable divergence (gap score {gap:.2f}): post framing is {post_dir.replace('_', ' ')} "
            f"but comment pool shows {pulse_note}{bot_note}.")


def score_topic(
    topic: dict,
    bot_audit: dict | None = None,
    comment_pool: list[dict] | None = None,
    political_pulse: dict | None = None,
) -> dict:
    """Score one topic's political undercurrent.

    political_pulse: output of evaluate_topic_pulse() — BJP%/Opp%/Neutral% per comment pool.
    Callers in make_full_dashboard.py should pass it directly (it's already computed).
    """
    brief = topic.get("brief") or {}
    if comment_pool is None:
        comment_pool = brief.get("comment_pool") or []
    bot_audit = bot_audit or {}
    bot_share = float(bot_audit.get("bot_share_pct") or 0) / 100
    bot_share_pct = float(bot_audit.get("bot_share_pct") or 0)

    scores = topic.get("scores") or {}
    trend_score = float(scores.get("trend_score") or 0)
    debate_score = float(scores.get("debate_score") or 0)
    engagement_per_hour = float((topic.get("instagram_metrics") or {}).get("engagement_per_hour") or 0)

    # --- POST FRAMING ---
    post_dir, post_conf, post_source = _post_framing(topic)

    # --- COMMENT REACTION ---
    comment_dir, comment_conf, n_engagements, bjp_pct, opp_pct, reaction_source = \
        _comment_reaction(political_pulse, comment_pool, bot_share)

    # --- ISSUE / NARRATIVE from post classifier ---
    post_text = _caption_text(topic)
    post_cls = pc.classify(post_text) if post_text else None
    
    # Topic level political gate: check topic category and post_cls
    topic_cat = topic.get("category", "")
    is_pol_cat = topic_cat in ("politics_governance", "civic_issues")
    
    issue_cluster = (post_cls or {}).get("issue_cluster", {}).get("value", "none") if post_cls else "none"
    if issue_cluster == "none":
        from identify_what_matters.political_classifier import tag_issue_cluster
        issue_cluster = tag_issue_cluster(post_text or topic.get("title", ""))
    
    # Non-political categories (e.g. cinema, entertainment, weather, disaster) must not inherit political issue clusters
    if not is_pol_cat:
        issue_cluster = "none"

    narrative_type = (post_cls or {}).get("narrative_type", {}).get("value", "other") if post_cls else "other"

    # Override issue_cluster from political_pulse community lens if richer
    if (political_pulse or {}).get("primary_community_lens") and issue_cluster == "none" and is_pol_cat:
        c_lens = (political_pulse.get("primary_community_lens") or "none").lower().replace(" ", "_")
        if c_lens not in ("secular_/_civic_/_general", "general", "n/a"):
            issue_cluster = c_lens

    # --- FRAMING-REACTION GAP (continuous 0–1) ---
    post_lean = _LEAN_MAP.get(post_dir, 0.0)
    comment_lean = _LEAN_MAP.get(comment_dir, 0.0)
    gap = _framing_gap(post_lean, comment_lean, bjp_pct, opp_pct, reaction_source)

    disagreement = gap >= 0.35 and abs(post_lean) >= 0.3

    # --- IS POLITICAL ---
    is_political = (
        bool(post_cls and post_cls.get("is_political", {}).get("value"))
        or post_dir in NON_NEUTRAL
        or comment_dir in NON_NEUTRAL
        or bjp_pct + opp_pct > 15
    )

    # --- UNDERCURRENT SCORE ---
    # Enhanced with Neural Strategic Criticality Score:
    # Blend momentum, debate intensity, framing gap, and neural criticality probability
    topic_summary_text = " ".join(filter(None, [topic.get("title"), (brief or {}).get("summary")]))
    neural_crit_boost = 0.5
    neural_strategic_meta = None
    try:
        from identify_what_matters.strategic_classifier import classify_strategic
        s_intel = classify_strategic(topic_summary_text)
        if s_intel:
            neural_crit_boost = s_intel.get("criticality_score", 0.5)
            neural_strategic_meta = s_intel
    except Exception:
        pass

    if is_political:
        undercurrent_score = round((
            0.20 * min(trend_score / 10, 1.0)
            + 0.15 * min(debate_score / 10, 1.0)
            + 0.15 * comment_conf
            + 0.20 * gap
            + 0.10 * min(engagement_per_hour / 50, 1.0)
            + 0.15 * neural_crit_boost
            + 0.05 * (1.0 - bot_share)
        ) * 10, 2)
    else:
        undercurrent_score = 0.0

    # --- NARRATIVE ---
    undercurrent_narrative = _narrative(
        post_dir, post_conf, post_source,
        comment_dir, comment_conf,
        bjp_pct, opp_pct, n_engagements,
        gap, bot_share_pct,
        brief,
    )

    # Agreement label for backward compat display
    if n_engagements == 0:
        agreement_label = "no comment engagements measured"
    elif reaction_source == "political_pulse":
        agreement_label = (f"{bjp_pct:.0f}% BJP-lean, {opp_pct:.0f}% opp-lean, "
                           f"{100-bjp_pct-opp_pct:.0f}% neutral across {n_engagements} engagements")
        if bot_share_pct > 15:
            agreement_label += f", discounted for {bot_share_pct:.0f}% suspected coordination"
    else:
        agreement_label = f"{n_engagements} comments sampled (keyword classifier)"

    return {
        "is_political": is_political,
        "issue_cluster": issue_cluster,
        "narrative_type": narrative_type,
        "post_direction": post_dir,
        "post_direction_confidence": round(post_conf, 3),
        "post_direction_source": post_source,
        "comment_direction": comment_dir,
        "comment_direction_source": reaction_source,
        "comment_sample_size": n_engagements,
        "comment_direction_confidence": comment_conf,
        "bjp_pct": round(bjp_pct, 1),
        "opp_pct": round(opp_pct, 1),
        "neutral_pct": round(100 - bjp_pct - opp_pct, 1),
        "framing_reaction_gap": gap,
        "framing_vs_reaction_disagreement": disagreement,
        "signal_agreement_label": agreement_label,
        "undercurrent_score": undercurrent_score,
        "undercurrent_narrative": undercurrent_narrative,
        "neural_strategic": neural_strategic_meta,
        "models_in_use": pc.models_available(),
    }


def summarize_models_status() -> dict[str, Any]:
    available = pc.models_available()
    return {"tasks_on_trained_model": sum(available.values()), "tasks_total": len(available), "detail": available}

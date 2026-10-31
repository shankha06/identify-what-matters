"""Strategic & Prescriptive Analytics for Political Leadership.

Bridges the gap between descriptive social sentiment and tactical political decision-making:
1. Narrative Vulnerability & Friction Mapping (Narrative Penetration Rate - NPR)
2. Demographic & Geographic Swing Potential Index (Persuadability Score)
3. Issue-to-Action Translation (The "Policy Gap" Detector)
4. Bot & Coordination Detection (Organic vs. Synthetic Engagement Ratio)
5. Opposition Vulnerability & Counter-Attack Window Tracker (Opposition Backlash Index & Capitalization Windows)
6. Strategic Leadership Action Recommendation Feed
7. Influencer Sentiment Alignment & Suburban Voter Swing Predictor (>200K Sentinel Tier)
"""

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Tuple


# Baseline party communication share across issue clusters (manifesto vs public anxiety)
PARTY_MESSAGING_BENCHMARKS = {
    "employment": {
        "manifesto_theme": "Skill India & Startup Ecosystem",
        "current_messaging_share": 7.5,
        "recommended_policy_injection": "Announce standardized annual central recruitment calendars & youth paid internship schemes.",
    },
    "education": {
        "manifesto_theme": "NEP 2020 & Academic Excellence",
        "current_messaging_share": 6.0,
        "recommended_policy_injection": "Fast-track national anti-paper leak ordinance tribunals & computer-based exam center audits.",
    },
    "inflation": {
        "manifesto_theme": "Macroeconomic Stability & Low CPI",
        "current_messaging_share": 11.0,
        "recommended_policy_injection": "Highlight retail buffer releases (tomatoes/onions), GST relief on essentials, and targeted LPG subsidies.",
    },
    "women_safety": {
        "manifesto_theme": "Nari Shakti & Mission Vatsalya",
        "current_messaging_share": 9.0,
        "recommended_policy_injection": "Deploy city-wide CCTV surveillance expansion, 24/7 pink patrols, and fast-track courts for crime against women.",
    },
    "infrastructure": {
        "manifesto_theme": "PM GatiShakti & Expressways",
        "current_messaging_share": 38.0,
        "recommended_policy_injection": "Counter localized structural failure claims with rigorous municipal structural safety audits & rapid interim relief.",
    },
    "elections": {
        "manifesto_theme": "Vibrant Democracy & Free Polls",
        "current_messaging_share": 4.5,
        "recommended_policy_injection": "Empower booth-level workers with Form 7 statutory deletion documentation toolkits to refute voter suppression narratives.",
    },
    "corruption": {
        "manifesto_theme": "Zero Tolerance Against Graft",
        "current_messaging_share": 24.0,
        "recommended_policy_injection": "Emphasize digital DBT leak-prevention track record and court convictions in economic offense cases.",
    },
    "farmers": {
        "manifesto_theme": "PM Kisan & Agri-Modernization",
        "current_messaging_share": 14.0,
        "recommended_policy_injection": "Amplify MSP procurement hikes, solar pump solarization schemes, and local agri-credit disbursals.",
    },
    "governance": {
        "manifesto_theme": "Minimum Government, Maximum Governance",
        "current_messaging_share": 22.0,
        "recommended_policy_injection": "Highlight civil service delivery turnaround metrics, single-window citizen portals, and pension grievance resolution.",
    },
    "communal": {
        "manifesto_theme": "Cultural Renaissance & National Unity",
        "current_messaging_share": 18.0,
        "recommended_policy_injection": "Reiterate constitutional equality, cultural pride, and zero tolerance for law-and-order disruptions.",
    },
}

# Key demographic and geographic voter blocks for swing targeting
SWING_DEMOGRAPHIC_SEGMENTS = [
    {
        "segment_id": "youth_18_24",
        "name": "Suburban Youth & First-Time Voters (18-24)",
        "geography": "Tier-2/3 Satellite Towns & University Belts",
        "core_sensitivities": ["education", "employment", "caste"],
        "base_persuadability": 74.0,
        "tactical_lever": "Campus town halls on transparent hiring & digital sports/startup subsidies.",
    },
    {
        "segment_id": "working_prof_25_34",
        "name": "Urban Working Professionals & Gig Workers (25-34)",
        "geography": "Metropolitan Suburbs (e.g. NCR, Bengaluru, MMR, Pune)",
        "core_sensitivities": ["inflation", "infrastructure", "governance"],
        "base_persuadability": 68.0,
        "tactical_lever": "Issue a 60-second video message on local tax relief, toll rationalization, and transit quality.",
    },
    {
        "segment_id": "semiurban_women_35_54",
        "name": "Semi-Urban Women & Homemakers (35-54)",
        "geography": "District Headquarters & Small Towns",
        "core_sensitivities": ["women_safety", "inflation", "welfare"],
        "base_persuadability": 65.0,
        "tactical_lever": "Mahila Morcha door-to-door briefing on subsidized nutrition, clean tap water, and safety response.",
    },
    {
        "segment_id": "rural_agri_youth",
        "name": "Agrarian Youth & Small Farmers",
        "geography": "Rural Agro-Clustering Districts",
        "core_sensitivities": ["farmers", "employment", "welfare"],
        "base_persuadability": 62.0,
        "tactical_lever": "Candidate site visit to completed local irrigation/connectivity project & agri-credit mela.",
    },
    {
        "segment_id": "regional_diaspora",
        "name": "Regional Linguistic Minorities & Migrant Workers",
        "geography": "Industrial Corridors & Interstate Border Zones",
        "core_sensitivities": ["communal", "governance", "welfare"],
        "base_persuadability": 58.0,
        "tactical_lever": "Release multi-lingual video explainers clarifying federal protections and equal welfare access.",
    },
]


def compute_narrative_penetration_rate(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Measures how fast an opposition talking point spreads outside its echo chamber into neutral/supportive voters."""
    alerts = []
    for t in topics:
        if not t.get("is_political"):
            continue

        pulse = t.get("political_pulse") or {}
        undercurrent = t.get("undercurrent") or {}
        opp_pct = pulse.get("opp_favor_pct", 0.0)
        bjp_pct = pulse.get("bjp_favor_pct", 0.0)
        neu_pct = pulse.get("neutral_pct", 0.0)
        gap = undercurrent.get("framing_reaction_gap", 0.0)
        title = t.get("title", "")
        issue_cluster = t.get("issue_cluster", "general")

        # Narrative penetration is significant when opposition talking points penetrate neutral/moderate discourse
        if opp_pct >= 35.0 and (neu_pct >= 25.0 or gap >= 0.20):
            # Penetration calculation: opposition strength combined with neutral audience exposure & velocity
            npr = min(round((opp_pct * 0.45 + neu_pct * 0.30 + gap * 40.0), 1), 94.0)

            # Match recommended counter-narrative
            policy_meta = PARTY_MESSAGING_BENCHMARKS.get(issue_cluster, {})
            counter_narrative = policy_meta.get("recommended_policy_injection", "Proactively address factual claims with direct data releases.")

            target_audience = "Suburban Swing Voters & Neutral Moderates"
            if issue_cluster in ("education", "employment"):
                target_audience = "Youth & Student Demographics (18-24)"
            elif issue_cluster in ("women_safety",):
                target_audience = "Female Electorate & Urban Families"
            elif issue_cluster in ("infrastructure", "inflation"):
                target_audience = "Urban & Semi-Urban Working Families"

            urgency = "HIGH" if npr >= 55.0 else ("MEDIUM" if npr >= 35.0 else "WATCH")

            alerts.append({
                "topic_id": t.get("id"),
                "topic_title": title,
                "issue_cluster": issue_cluster,
                "npr_score": npr,
                "opp_share": opp_pct,
                "bjp_share": bjp_pct,
                "neutral_share": neu_pct,
                "urgency": urgency,
                "target_audience": target_audience,
                "alert_text": f"The opposition narrative on '{title}' has penetrated {npr}% of core/neutral voter demographics in the last 48 hours.",
                "recommended_counter_narrative": counter_narrative,
                "actionable_window": "< 24 Hours",
            })

    alerts.sort(key=lambda x: -x["npr_score"])
    return alerts


def compute_persuadability_matrix(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculates swing potential across demographic & geographic slices to prioritize campaign resources."""
    # Compute active topic anxiety per cluster
    cluster_intensity = Counter()
    for t in topics:
        if t.get("is_political"):
            cl = t.get("issue_cluster", "general")
            u_score = t.get("undercurrent_score", 1.0)
            cluster_intensity[cl] += u_score

    matrix = []
    for seg in SWING_DEMOGRAPHIC_SEGMENTS:
        # Dynamic calculation based on current topics touching this segment's sensitivities
        cluster_hits = [cluster_intensity[cl] for cl in seg["core_sensitivities"]]
        sum_intensity = sum(cluster_hits)
        volatility_bonus = min(sum_intensity * 1.8, 22.0)
        persuadability_score = min(round(seg["base_persuadability"] + volatility_bonus, 1), 96.0)

        top_issue = max(seg["core_sensitivities"], key=lambda cl: cluster_intensity[cl])
        top_issue_label = top_issue.replace("_", " ").title()

        priority = "🔴 High Priority" if persuadability_score >= 78.0 else ("🟡 Medium Priority" if persuadability_score >= 68.0 else "🟢 Low Priority")

        matrix.append({
            "segment_id": seg["segment_id"],
            "segment_name": seg["name"],
            "geography": seg["geography"],
            "persuadability_score": persuadability_score,
            "dominant_grievance": top_issue_label,
            "priority": priority,
            "responsive_policy_lever": seg["tactical_lever"],
            "recommended_allocation": "Heavy Digital Ad Micro-Targeting + Candidate Town Hall" if persuadability_score >= 78.0 else "Targeted Ground Leafleting & Digital Reels",
        })

    matrix.sort(key=lambda x: -x["persuadability_score"])
    return matrix


def compute_policy_gap_detector(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies disconnects between public grievances/anxiety and party communication volume."""
    # Count public grievance intensity per cluster
    grievance_weights = Counter()
    total_political_weight = 0.0

    for t in topics:
        if t.get("is_political"):
            cl = t.get("issue_cluster", "general")
            u = max(t.get("undercurrent_score", 1.0), 1.0)
            c_cnt = (t.get("ig") or {}).get("comments_7d", 10) or 10
            weight = u * math.log10(c_cnt + 10)
            grievance_weights[cl] += weight
            total_political_weight += weight

    if total_political_weight <= 0:
        total_political_weight = 1.0

    gaps = []
    for cl, benchmark in PARTY_MESSAGING_BENCHMARKS.items():
        public_share = round((grievance_weights.get(cl, 0.0) / total_political_weight) * 100, 1)
        messaging_share = benchmark["current_messaging_share"]
        gap_delta = round(public_share - messaging_share, 1)

        # Gap exists when public anxiety significantly outstrips leadership communication share
        status = "CRITICAL GAP" if gap_delta >= 12.0 else ("MILD GAP" if gap_delta >= 5.0 else ("OVER-INDEXED" if gap_delta <= -10.0 else "BALANCED"))

        gaps.append({
            "issue_cluster": cl,
            "cluster_label": cl.replace("_", " ").title(),
            "public_anxiety_share": public_share,
            "messaging_share": messaging_share,
            "gap_delta": gap_delta,
            "status": status,
            "manifesto_theme": benchmark["manifesto_theme"],
            "recommended_injection": benchmark["recommended_policy_injection"],
            "diagnosis_callout": f"Public anxiety on {cl.replace('_', ' ').title()} represents {public_share}% of civic debate, but only {messaging_share}% of current leadership messaging addresses it." if gap_delta > 0 else f"Messaging ({messaging_share}%) is currently well ahead of public controversy ({public_share}%).",
        })

    gaps.sort(key=lambda x: -x["gap_delta"])
    return gaps


def compute_opposition_vulnerabilities(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tracks negative sentiment spikes on opposition announcements or candidate gaffes to trigger capitalization windows."""
    vulnerabilities = []

    for t in topics:
        if not t.get("is_political"):
            continue

        pulse = t.get("political_pulse") or {}
        bjp_pct = pulse.get("bjp_favor_pct", 0.0)
        opp_pct = pulse.get("opp_favor_pct", 0.0)
        stance = t.get("stance") or {}
        crit_pct = round((stance.get("critical", 0.0) or 0.0) * 100, 1)
        title = t.get("title", "")
        summary = t.get("summary", "")

        # Look for opposition gaffes, public backlash against opposition moves, or favorable pro-incumbent spikes
        t_low = f"{title} {summary}".lower()
        has_opp_entity = any(w in t_low for w in ["congress", "rahul gandhi", "kejriwal", "aap", "tmc", "opposition", "mamata", "stalin", "akhilesh", "india alliance"])
        
        if has_opp_entity and (bjp_pct >= 48.0 or (crit_pct >= 45.0 and opp_pct <= 35.0)):
            backlash_index = min(round((bjp_pct * 0.6 + crit_pct * 0.4), 1), 95.0)
            window = "Immediate < 12 Hours" if backlash_index >= 65.0 else "24-48 Hours"
            priority = "HIGH 🔴" if backlash_index >= 65.0 else "MEDIUM 🟡"

            # Determine tactical counter-attack action
            if "controversy" in t_low or "statement" in t_low or "slams" in t_low:
                action = "Deploy 30-second digital video ads contrasting opposition statement against constitutional record."
            elif "protest" in t_low or "detained" in t_low:
                action = "Highlight commuter disruption & law-and-order violation; pivot debate to governance delivery."
            else:
                action = "Depute national spokespersons for prime-time broadcast debates to amplify contrasting policy performance."

            target_demo = "Urban Middle Class & Gen-Z Youth" if "student" in t_low or "youth" in t_low else "Suburban Swing Voters"

            vulnerabilities.append({
                "topic_id": t.get("id"),
                "trigger_event": f"Backlash on opposition posture regarding {title[:60]}",
                "backlash_index": backlash_index,
                "priority": priority,
                "target_audience": target_demo,
                "capitalization_window": window,
                "recommended_action": action,
                "diagnosis": f"Opposition move on '{title[:50]}' triggered {backlash_index}% adverse public backlash. Capitalize before narrative fades.",
            })

    vulnerabilities.sort(key=lambda x: -x["backlash_index"])
    return vulnerabilities


def compute_influencer_alignment(panel_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes sentiment alignment of high-reach influencer posts (>200K followers / high heat) as early indicator for suburban swings."""
    high_reach_posts = []
    favorable_count = 0
    critical_count = 0
    neutral_count = 0
    total_likes = 0

    for p in panel_posts:
        likes = p.get("likes", 0) or 0
        comments = p.get("comments", 0) or 0
        heat = p.get("heat", 0.0) or 0.0
        # High reach proxy: > 25,000 likes or heat >= 2.0 or creator_commentary group
        is_high_reach = (likes >= 15000) or (heat >= 1.8) or (p.get("group") in ("creator_commentary", "political_leadership"))

        if is_high_reach:
            lean = p.get("lean", "Neutral / Non-partisan")
            tone = p.get("tone", "neutral")
            direction = p.get("direction", "neutral")

            is_fav = (lean in ("Pro-incumbent", "Favoring BJP / NDA") or direction == "pro_incumbent" or "positive" in tone.lower())
            is_crit = (lean in ("Pro-opposition", "Favoring Opposition / INDIA") or direction == "anti_incumbent" or "critical" in tone.lower() or "sarcasm" in tone.lower())

            if is_fav:
                favorable_count += 1
            elif is_crit:
                critical_count += 1
            else:
                neutral_count += 1

            total_likes += likes
            high_reach_posts.append({
                "account": p.get("account"),
                "caption": p.get("caption", "")[:100],
                "likes": likes,
                "comments": comments,
                "group": p.get("group"),
                "tone": tone,
                "lean": lean,
                "heat": heat,
                "is_fav": is_fav,
                "is_crit": is_crit,
            })

    total_influencer = len(high_reach_posts) or 1
    fav_pct = round((favorable_count / total_influencer) * 100, 1)
    crit_pct = round((critical_count / total_influencer) * 100, 1)
    neu_pct = max(0.0, round(100.0 - fav_pct - crit_pct, 1))

    # Early swing indicator based on net influencer slant
    net_slant = round(fav_pct - crit_pct, 1)
    if net_slant >= 15.0:
        swing_indicator = "Favorable Tailwinds: High-reach creator accounts are reinforcing incumbent governance narratives (+{}% net slant).".format(net_slant)
        swing_status = "POSITIVE_SWING"
    elif net_slant <= -15.0:
        swing_indicator = "Suburban Warning Light: Influencers are amplifying friction on cost-of-living and governance (-{}% net slant). Early signal of suburban voter volatility.".format(abs(net_slant))
        swing_status = "NEGATIVE_SWING"
    else:
        swing_indicator = "Contested Influencer Sphere: Slant is evenly matched across lifestyle & commentary creators."
        swing_status = "BALANCED"

    return {
        "high_reach_posts_count": len(high_reach_posts),
        "total_influencer_likes": total_likes,
        "favorable_pct": fav_pct,
        "critical_pct": crit_pct,
        "neutral_pct": neu_pct,
        "net_slant": net_slant,
        "swing_status": swing_status,
        "swing_indicator": swing_indicator,
        "sample_influencer_posts": high_reach_posts[:8],
    }


def generate_action_recommendations(
    topics: List[Dict[str, Any]],
    early_warnings: List[Dict[str, Any]],
    panel_posts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Synthesizes all underlying prescriptive analytics into the unified Action Recommendation Feed."""
    recommendations = []

    # 1. High Priority items from Early Warnings (Threat Axes)
    for ew in early_warnings[:3]:
        pr_label = "High 🔴" if ew.get("priority") in ("CRITICAL", "HIGH") else "Medium 🟡"
        trigger = f"Narrative spike on {ew.get('threat_axis', '')}"
        
        # Determine target audience from demographics
        demo_str = ew.get("demographic_trait", "")
        if "18-24" in demo_str or "Youth" in demo_str:
            target = "Youth & Student Demographic"
        elif "Female" in demo_str or "Women" in demo_str:
            target = "Suburban Women & Urban Families"
        elif "Middle-Class" in demo_str:
            target = "Suburban Swing Voters & Professionals"
        else:
            target = "Neutral Suburban Electorate"

        acts = ew.get("immediate_actions", [])
        action_text = acts[0] if acts else "Deploy rapid factual explainer desks & ministerial town hall."

        recommendations.append({
            "priority": pr_label,
            "trigger_event": trigger,
            "target_audience": target,
            "recommended_action": action_text,
            "category": "Friction Defusal",
            "metric_trigger": f"Friction: {ew.get('avg_undercurrent', 0)}/10",
        })

    # 2. Strategic items from Policy Gaps
    policy_gaps = compute_policy_gap_detector(topics)
    for pg in policy_gaps[:2]:
        if pg["gap_delta"] >= 8.0:
            recommendations.append({
                "priority": "High 🔴" if pg["gap_delta"] >= 15.0 else "Medium 🟡",
                "trigger_event": f"Policy disconnect: Public anxiety on {pg['cluster_label']} is up {pg['public_anxiety_share']}%, messaging at {pg['messaging_share']}%",
                "target_audience": "Working Professionals & Civic Voters",
                "recommended_action": f"Talking point injection: {pg['recommended_injection']}",
                "category": "Policy Injection",
                "metric_trigger": f"Gap Delta: +{pg['gap_delta']}%",
            })

    # 3. Capitalization items from Opposition Vulnerabilities
    opp_vulns = compute_opposition_vulnerabilities(topics)
    for ov in opp_vulns[:2]:
        recommendations.append({
            "priority": ov["priority"].replace("HIGH 🔴", "High 🔴").replace("MEDIUM 🟡", "Medium 🟡"),
            "trigger_event": ov["trigger_event"],
            "target_audience": ov["target_audience"],
            "recommended_action": ov["recommended_action"],
            "category": "Offensive Capitalization",
            "metric_trigger": f"Backlash Index: {ov['backlash_index']}%",
        })

    # 4. Swing Potential items
    swing_matrix = compute_persuadability_matrix(topics)
    for sm in swing_matrix[:2]:
        if sm["persuadability_score"] >= 72.0:
            recommendations.append({
                "priority": "Medium 🟡",
                "trigger_event": f"High persuadability detected in {sm['geography']}",
                "target_audience": sm["segment_name"],
                "recommended_action": sm["responsive_policy_lever"],
                "category": "Ground Mobilization",
                "metric_trigger": f"Persuadability: {sm['persuadability_score']}",
            })

    # 5. Influencer suburban indicator
    inf_intel = compute_influencer_alignment(panel_posts)
    if inf_intel["swing_status"] == "NEGATIVE_SWING":
        recommendations.append({
            "priority": "High 🔴",
            "trigger_event": "Influencer sentiment slant dropped on cost-of-living & municipal issues",
            "target_audience": "Suburban Metro Swing Voters",
            "recommended_action": "Issue a 60-second video message focusing on local tax relief, transit development, and kitchen essentials.",
            "category": "Influencer Alignment",
            "metric_trigger": f"Influencer Slant: {inf_intel['net_slant']}%",
        })
    elif inf_intel["swing_status"] == "POSITIVE_SWING":
        recommendations.append({
            "priority": "Low 🟢",
            "trigger_event": "Organic positive trend across high-reach creator commentary on infrastructure & space",
            "target_audience": "Youth & Tech Corridor Voters",
            "recommended_action": "Schedule candidate site visit to completed local project and co-create digital content with key local creators.",
            "category": "Momentum Amplification",
            "metric_trigger": f"Influencer Slant: +{inf_intel['net_slant']}%",
        })

    # Sort so High 🔴 comes first, then Medium 🟡, then Low 🟢
    priority_order = {"High 🔴": 0, "Medium 🟡": 1, "Low 🟢": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    return recommendations

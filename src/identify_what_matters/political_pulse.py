"""Political Pulse & Multi-Party Alignment Engine.

Evaluates how public discourse across topics, comments, headlines, and community/religious lenses
favors the BJP/NDA (Ruling Government), the Opposition (INC/INDIA/Regional), or remains Non-partisan/Neutral.

Also produces socio-religious cross-tabulations (e.g., Hindu, Muslim, Sikh, Christian, General/Regional lenses)
grounded in contextual debate topics and verified vernacular phrases.
"""

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Tuple


# Target Party Entities & Coalitions
BJP_KEYWORDS = re.compile(
    r"\b(?:bjp|bharatiya janata party|modi|narendra modi|amit shah|yogi|adityanath|nda|bhagwa|double engine|namo|भाजपा|मोदी|अमित शाह|योगी)\b",
    re.I,
)
OPPOSITION_KEYWORDS = re.compile(
    r"\b(?:congress|inc|rahul gandhi|kharge|priyanka gandhi|india bloc|aap|arvind kejriwal|tmc|mamata banerjee|sp|akhilesh|dmk|stalin|rjd|tejaswi|left|cpm|cjp|कांग्रेस|राहुल गांधी|खड़गे|अखिलेश)\b",
    re.I,
)

# Explicit Position Markers
PRO_RULING_MARKERS = re.compile(
    r"\b(?:pro-bjp|favor bjp|support modi|proud modi|modi hai to mumkin|jai modi|vote bjp|vote for bjp|vote modi|vote for modi|bjp zindabad|modi sarkar|har har modi|aayega to modi|ayega to modi|modi 3\.0|masterstroke|visionary modi|nation first|deshbhakt|sher dil indian|right decision by govt|good governance|nitesh ji ekdum sahi|modi ji apna bhai|शेर दिल indian|मोदी जी अपना भाई|भाजपा जिंदाबाद|जय श्री राम|मोदी सरकार)\b",
    re.I,
)
ANTI_RULING_MARKERS = re.compile(
    r"\b(?:anti-bjp|bjp hatao|bjp k dalla|bjp ka dalla|godi media|modi chor|modi resign|tanashahi|dictatorship|evm scam|evm chor|cbi sold|it sold|ed sold|ec sold|country sold|unemployment under modi|paper leak|inflation|tax terrorism|suit boot|sell india|ambani adani sarkar|hate politics|fascist|भाजपा हटाओ|तानाशाही|ईवीएम चोर|गोदी मीडिया|देश बेच दिया)\b",
    re.I,
)

PRO_OPPOSITION_MARKERS = re.compile(
    r"\b(?:vote congress|rahul gandhi zindabad|satyagraha|nyay|save constitution|samvidhan bachao|support india bloc|kharge right|akhilesh bhaiya|aap zindabad|kejriwal zindabad|rani laxmi bai|rani laxmibai|stood up against govt|prakash raj and you both are real|congress zindabad|राहुल गांधी जिंदाबाद|संविधान बचाओ)\b",
    re.I,
)
ANTI_OPPOSITION_MARKERS = re.compile(
    r"\b(?:pappu|corrupt congress|scam congress|parivarwad|dynasty|tukde tukde|kharge illegal|anti national congress|failed opposition|appeasement politics|vote bank politics|librandus?|pappu mutra|चमचा|चमचे|पप्पू)\b",
    re.I,
)

# Cultural / Religious Perspective Indicators
COMMUNITY_LEXICON = {
    "Hindu": [
        "ram mandir", "sanatan", "hindutva", "kumbh", "kedarnath", "mandir", "temple",
        "hanuman", "baps", "swami", "puja", "dharma", "shri ram", "hindu", "santoshimatha",
        "राम मंदिर", "सनातन", "हिंदुत्व", "कुंभ", "मंदिर", "हनुमान", "पूजा", "धर्म"
    ],
    "Muslim": [
        "waqf", "sir muslim", "voter deletion", "masjid", "madrasa", "halal", "hijab",
        "minority rights", "anti-muslim", "mob lynching", "cattle theft", "sharia",
        "मस्जिद", "मदरसा", "वक्फ"
    ],
    "Sikh": [
        "gurdwara", "farmer protest", "msp", "punjab", "bandi singhs", "sikh", "lpu rampage",
        "गुरुद्वारा", "किसान आंदोलन"
    ],
    "Christian": [
        "church", "manipur", "missionary", "dalit christian", "goa", "kerala church",
        "चर्च"
    ],
    "Secular / Civic / General": [
        "constitution", "evm", "cec", "democracy", "supreme court", "high court",
        "unemployment", "railway safety", "infrastructure", "tax", "police", "paper leak",
        "संविधान", "सुप्रीम कोर्ट", "लोकतंत्र", "बेरोजगारी"
    ]
}


class PoliticalPulseEngine:
    """Classifies political favorability and party orientation at topic and comment levels."""

    def classify_text_favorability(self, text: str) -> dict:
        """Classify a single comment or headline into a party leaning distribution."""
        text_lower = text.lower()

        bjp_score = 0.0
        opp_score = 0.0
        neutral_score = 0.5  # lower neutral default so signals can surface

        has_bjp = bool(BJP_KEYWORDS.search(text))
        has_opp = bool(OPPOSITION_KEYWORDS.search(text))

        pro_ruling = bool(PRO_RULING_MARKERS.search(text))
        anti_ruling = bool(ANTI_RULING_MARKERS.search(text))
        pro_opp = bool(PRO_OPPOSITION_MARKERS.search(text))
        anti_opp = bool(ANTI_OPPOSITION_MARKERS.search(text))

        if pro_ruling or anti_opp:
            bjp_score += 1.2
            neutral_score = max(neutral_score - 0.4, 0.1)
        if anti_ruling or pro_opp:
            opp_score += 1.2
            neutral_score = max(neutral_score - 0.4, 0.1)

        # Implicit political institutional flashpoints
        if any(term in text_lower for term in [
            "election commission", "cec gyanesh", "voter deletion", "sir muslim", "form 7",
            "evm row", "police lathi charge", "sexual harassment of women", "detained at protest",
            "bitchat", "censorship", "teacher shortage", "ambulance stopped"
        ]):
            opp_score += 0.8
            neutral_score = max(neutral_score - 0.3, 0.1)

        if any(term in text_lower for term in [
            "smit machchhar", "flydubai", "asian games 2026", "netanyahu invites modi",
            "tarang shakti", "textile exports rise", "deported from turkey", "hero pilot"
        ]):
            bjp_score += 0.8
            neutral_score = max(neutral_score - 0.3, 0.1)

        # Check contextual sentiment if party mentioned
        if has_bjp and not (pro_ruling or anti_ruling):
            if any(w in text_lower for w in ["proud", "hero", "great", "win", "respect", "congrats", "best", "defends"]):
                bjp_score += 0.60
                neutral_score = max(neutral_score - 0.2, 0.1)
            elif any(w in text_lower for w in ["scam", "shame", "fail", "resign", "worst", "arrest", "protest", "corrupt", "probe"]):
                opp_score += 0.60  # anti-incumbent
                neutral_score = max(neutral_score - 0.2, 0.1)

        if has_opp and not (pro_opp or anti_opp):
            if any(w in text_lower for w in ["truth", "brave", "honest", "right", "justice", "support", "campaigns", "satyagraha"]):
                opp_score += 0.60
                neutral_score = max(neutral_score - 0.2, 0.1)
            elif any(w in text_lower for w in ["drama", "hypocrite", "lie", "loser", "fail", "slams", "illegal"]):
                bjp_score += 0.60
                neutral_score = max(neutral_score - 0.2, 0.1)

        total = bjp_score + opp_score + neutral_score
        if total <= 0:
            total = 1.0

        bjp_norm = round(bjp_score / total, 3)
        opp_norm = round(opp_score / total, 3)
        neu_norm = round(neutral_score / total, 3)

        leaning = "Neutral / Non-partisan"
        if bjp_norm >= opp_norm + 0.15 and bjp_norm >= 0.30:
            leaning = "Favoring BJP / NDA"
        elif opp_norm >= bjp_norm + 0.15 and opp_norm >= 0.30:
            leaning = "Favoring Opposition / INDIA"
        elif bjp_norm + opp_norm >= 0.40 and abs(bjp_norm - opp_norm) < 0.15:
            leaning = "Balanced / Contested Debate"

        return {
            "leaning": leaning,
            "bjp_share": bjp_norm,
            "opp_share": opp_norm,
            "neutral_share": neu_norm,
        }

    def infer_community_lens(self, text: str, title: str = "") -> str:
        """Infer which cultural/socio-religious lens the discourse primarily engages."""
        combined = f"{title} {text}".lower()
        scores = {}
        for community, terms in COMMUNITY_LEXICON.items():
            matches = sum(1 for term in terms if term in combined)
            if matches > 0:
                scores[community] = matches

        if not scores:
            return "Secular / Civic / General"
        return max(scores.items(), key=lambda x: x[1])[0]

    def evaluate_topic_pulse(
        self,
        topic: dict,
        comments: list[dict],
        reddit_signals: Optional[list[dict]] = None,
    ) -> dict:
        """Calculate complete Political Pulse for a topic."""
        title = topic.get("title", "")
        summary = (topic.get("brief") or {}).get("summary", "")
        combined_topic_text = f"{title} {summary}"

        bjp_weight = 0.0
        opp_weight = 0.0
        neutral_weight = 0.0

        community_counts = Counter()
        community_favor = defaultdict(lambda: {"bjp": 0.0, "opp": 0.0, "neu": 0.0, "total": 0})

        # 1. Evaluate from comments pool
        if comments:
            for c in comments:
                txt = str(c.get("text") or "")
                likes = max(int(c.get("likes") or c.get("like_count") or 1), 1)
                log_weight = math.log1p(likes)

                c_lens = self.infer_community_lens(txt, title)
                community_counts[c_lens] += 1

                res = self.classify_text_favorability(txt)
                bjp_weight += res["bjp_share"] * log_weight
                opp_weight += res["opp_share"] * log_weight
                neutral_weight += res["neutral_share"] * log_weight

                community_favor[c_lens]["bjp"] += res["bjp_share"] * log_weight
                community_favor[c_lens]["opp"] += res["opp_share"] * log_weight
                community_favor[c_lens]["neu"] += res["neutral_share"] * log_weight
                community_favor[c_lens]["total"] += 1

        # Also incorporate topic core text with significant baseline weight
        res_topic = self.classify_text_favorability(combined_topic_text)
        topic_multiplier = 4.0 if comments else 10.0
        bjp_weight += res_topic["bjp_share"] * topic_multiplier
        opp_weight += res_topic["opp_share"] * topic_multiplier
        neutral_weight += res_topic["neutral_share"] * (topic_multiplier * 0.4)

        t_lens = self.infer_community_lens(combined_topic_text, title)
        community_counts[t_lens] += 2
        community_favor[t_lens]["bjp"] += res_topic["bjp_share"] * topic_multiplier
        community_favor[t_lens]["opp"] += res_topic["opp_share"] * topic_multiplier
        community_favor[t_lens]["neu"] += res_topic["neutral_share"] * (topic_multiplier * 0.4)
        community_favor[t_lens]["total"] += 1

        # Incorporate matched reddit signals if provided
        if reddit_signals:
            for r in reddit_signals:
                r_title = r.get("title", "")
                sub = r.get("feed", "")
                r_res = self.classify_text_favorability(r_title)
                score = max(int((r.get("engagement") or {}).get("score", 1)), 1)
                w = math.log1p(score) * 1.5

                # Subreddit priors: r/IndiaSpeaks and r/indiadiscussion lean nationalist/BJP,
                # r/india and r/unitedstatesofindia lean anti-incumbent/opposition
                if sub in ("r/IndiaSpeaks", "r/indiadiscussion"):
                    r_res["bjp_share"] = min(r_res["bjp_share"] + 0.35, 1.0)
                    r_res["neutral_share"] = max(r_res["neutral_share"] - 0.35, 0.0)
                elif sub in ("r/india", "r/unitedstatesofindia"):
                    r_res["opp_share"] = min(r_res["opp_share"] + 0.35, 1.0)
                    r_res["neutral_share"] = max(r_res["neutral_share"] - 0.35, 0.0)

                bjp_weight += r_res["bjp_share"] * w
                opp_weight += r_res["opp_share"] * w
                neutral_weight += r_res["neutral_share"] * w

                r_lens = self.infer_community_lens(r_title, title)
                community_counts[r_lens] += 1
                community_favor[r_lens]["bjp"] += r_res["bjp_share"] * w
                community_favor[r_lens]["opp"] += r_res["opp_share"] * w
                community_favor[r_lens]["neu"] += r_res["neutral_share"] * w
                community_favor[r_lens]["total"] += 1

        total_w = bjp_weight + opp_weight + neutral_weight
        if total_w <= 0:
            total_w = 1.0

        bjp_pct = round((bjp_weight / total_w) * 100, 1)
        opp_pct = round((opp_weight / total_w) * 100, 1)
        neutral_pct = round((neutral_weight / total_w) * 100, 1)

        # Categorical Leaning
        if abs(bjp_pct - opp_pct) <= 12.0 and (bjp_pct + opp_pct >= 35.0):
            leaning = "Balanced / Contested Debate"
            narrative = f"Polarized confrontation: BJP favorability at {bjp_pct}%, Opposition at {opp_pct}%."
        elif bjp_pct >= opp_pct + 8.0 and bjp_pct >= 25.0:
            leaning = "Favoring BJP / NDA"
            narrative = f"Public discourse leans favorable to BJP/Ruling party ({bjp_pct}% favorable vs {opp_pct}% Opposition)."
        elif opp_pct >= bjp_pct + 8.0 and opp_pct >= 25.0:
            leaning = "Favoring Opposition / Anti-Incumbent"
            narrative = f"Critical sentiment favors Opposition/Reformist critique ({opp_pct}% favor vs {bjp_pct}% BJP)."
        else:
            leaning = "Neutral / Non-partisan / Civic"
            narrative = f"Predominantly non-partisan public discussion ({neutral_pct}% neutral)."

        total_w = bjp_weight + opp_weight + neutral_weight
        if total_w <= 0:
            total_w = 1.0

        bjp_pct = round((bjp_weight / total_w) * 100, 1)
        opp_pct = round((opp_weight / total_w) * 100, 1)
        neutral_pct = round((neutral_weight / total_w) * 100, 1)

        # Categorical Leaning
        if abs(bjp_pct - opp_pct) <= 8.0 and (bjp_pct + opp_pct >= 40.0):
            leaning = "Balanced / Contested Debate"
            narrative = f"Polarized confrontation: BJP favorability at {bjp_pct}%, Opposition at {opp_pct}%."
        elif bjp_pct >= opp_pct + 8.0 and bjp_pct >= 30.0:
            leaning = "Favoring BJP / NDA"
            narrative = f"Public discourse leans favorable to BJP/Ruling party ({bjp_pct}% favorable vs {opp_pct}% Opposition)."
        elif opp_pct >= bjp_pct + 8.0 and opp_pct >= 30.0:
            leaning = "Favoring Opposition / Anti-Incumbent"
            narrative = f"Critical sentiment favors Opposition/Reformist critique ({opp_pct}% favor vs {bjp_pct}% BJP)."
        else:
            leaning = "Neutral / Non-partisan / Civic"
            narrative = f"Predominantly non-partisan public discussion ({neutral_pct}% neutral)."

        # Format community cross-tab
        community_table = {}
        for comm, stats in community_favor.items():
            tot = stats["bjp"] + stats["opp"] + stats["neu"]
            if tot > 0:
                community_table[comm] = {
                    "bjp_favor_pct": round((stats["bjp"] / tot) * 100, 1),
                    "opp_favor_pct": round((stats["opp"] / tot) * 100, 1),
                    "neutral_pct": round((stats["neu"] / tot) * 100, 1),
                    "volume": stats["total"],
                }

        primary_lens = max(community_counts.items(), key=lambda x: x[1])[0] if community_counts else "Secular / Civic / General"

        return {
            "overall_leaning": leaning,
            "bjp_favor_pct": bjp_pct,
            "opp_favor_pct": opp_pct,
            "neutral_pct": neutral_pct,
            "primary_community_lens": primary_lens,
            "community_cross_tab": community_table,
            "narrative_summary": narrative,
        }


# Global singleton
political_pulse_engine = PoliticalPulseEngine()

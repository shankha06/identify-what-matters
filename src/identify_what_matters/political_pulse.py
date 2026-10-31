"""Political Pulse & Multi-Party Alignment Engine.

Evaluates how public discourse across topics, comments, headlines, and community/religious lenses
favors the BJP/NDA (Ruling Government), the Opposition (INC/INDIA/Regional), or remains Non-partisan/Neutral.

Also produces socio-religious cross-tabulations (e.g., Hindu, Muslim, Sikh, Christian, General/Regional lenses)
grounded in contextual debate topics and verified vernacular phrases.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

_BERT_PARTY_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models" / "bert-party-support-17"
if not _BERT_PARTY_DIR.exists():
    _BERT_PARTY_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models" / "bert-party-support"
_NEURAL_PARTY_TOKENIZER = None
_NEURAL_PARTY_MODEL = None

def _get_neural_party_model():
    global _NEURAL_PARTY_TOKENIZER, _NEURAL_PARTY_MODEL
    if _NEURAL_PARTY_MODEL is None and _BERT_PARTY_DIR.exists():
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            _NEURAL_PARTY_TOKENIZER = AutoTokenizer.from_pretrained(_BERT_PARTY_DIR)
            _NEURAL_PARTY_MODEL = AutoModelForSequenceClassification.from_pretrained(_BERT_PARTY_DIR)
            _NEURAL_PARTY_MODEL.eval()
            try:
                _NEURAL_PARTY_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                _NEURAL_PARTY_MODEL.to("cpu")
        except Exception:
            _NEURAL_PARTY_MODEL = False
    return _NEURAL_PARTY_TOKENIZER, (_NEURAL_PARTY_MODEL if _NEURAL_PARTY_MODEL is not False else None)

_NEURAL_PARTY_CACHE: dict[str, Optional[Tuple[str, float]]] = {}
_NEURAL_PARTY_DIST_CACHE: dict[str, dict[str, float]] = {}


def prewarm_party_neural(texts: List[str], batch_size: int = 32) -> None:
    """Run the party model over unseen texts in padded batches and cache (label, confidence) results."""
    import torch
    tok, mod = _get_neural_party_model()
    if tok is None or mod is None:
        return
    todo = list(dict.fromkeys(t for t in texts if t and t.strip() and t not in _NEURAL_PARTY_CACHE))
    if not todo:
        return
    if hasattr(mod.config, "id2label") and mod.config.id2label:
        idx2party = [mod.config.id2label[i] for i in range(len(mod.config.id2label))]
    else:
        idx2party = ["BJP", "Congress", "AAP", "TMC", "SP", "DMK", "None"]
    device = next(mod.parameters()).device
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(mod(**enc).logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            prob_rows = probs.cpu().numpy()
        for text, i, c, p_row in zip(chunk, idx.tolist(), conf.tolist(), prob_rows):
            _NEURAL_PARTY_CACHE[text] = (idx2party[i] if i < len(idx2party) else "None", float(c))
            _NEURAL_PARTY_DIST_CACHE[text] = {p: float(p_row[pi]) for pi, p in enumerate(idx2party)}
    if device.type == "cuda":
        mod.to("cpu")
        import torch
        torch.cuda.empty_cache()


def predict_party_neural(text: str) -> Optional[Tuple[str, float]]:
    """Party label and confidence for one text (served from the batch cache)."""
    if not text.strip():
        return None
    return _NEURAL_PARTY_CACHE.get(text)


def predict_party_neural_distribution(text: str) -> Optional[Dict[str, float]]:
    """Return full neural probability distribution across individual parties (served from batch cache)."""
    if not text.strip():
        return None
    return _NEURAL_PARTY_DIST_CACHE.get(text)


# Target Party Entities & Coalitions
BJP_KEYWORDS = re.compile(
    r"\b(?:bjp|bharatiya janata party|modi|narendra modi|amit shah|yogi|adityanath|nda|bhagwa|double engine|namo|"
    r"jpnadda|smriti irani|anurag thakur|tejasvi surya|devendra fadnavis|fadnavis|himanta biswa|himanta|shivraj|nityanand rai|rss|sangh|"
    r"भाजपा|मोदी|अमित शाह|योगी|नमो|भगवा|फडणवीस|फडतवीस)\b",
    re.I,
)
OPPOSITION_KEYWORDS = re.compile(
    r"\b(?:congress|inc|rahul gandhi|kharge|priyanka gandhi|india bloc|aap|arvind kejriwal|tmc|mamata banerjee|"
    r"sp|akhilesh|akhilesh yadav|dmk|stalin|mk stalin|rjd|tejaswi|left|cpm|cpim|cjp|jdu|nitish kumar|"
    r"samajwadi|shiv sena ubt|uddhav thackeray|sharad pawar|supriya sule|"
    r"कांग्रेस|राहुल गांधी|खड़गे|अखिलेश|केजरीवाल|ममता|आदिवासी)\b",
    re.I,
)

# Explicit Position Markers — strong direct signals
PRO_RULING_MARKERS = re.compile(
    r"\b(?:pro.?bjp|favor bjp|support modi|proud modi|modi hai to mumkin|jai modi|vote bjp|vote for bjp|"
    r"vote modi|vote for modi|bjp zindabad|bjp jindabad|only bjp|next also bjp|bjp win|congress mukt bharat|"
    r"modi sarkar|har har modi|aayega to modi|ayega to modi|25yearsofseva|seva jyoti|"
    r"modi 3\.0|masterstroke|visionary modi|nation first|deshbhakt|right decision by govt|good governance|"
    r"modi ji ne|modi government|nda government|india shining|modi magic|ham bjp ke sath|"
    r"शेर दिल|मोदी जी अपना भाई|भाजपा जिंदाबाद|जय श्री राम|मोदी सरकार|भारत माता की जय|जय हिंद)\b",
    re.I,
)
ANTI_RULING_MARKERS = re.compile(
    r"\b(?:anti.?bjp|bjp hatao|bjp k dalla|bjp ka dalla|bjp dalali|godi media|modi chor|modi resign|tanashahi|dictatorship|"
    r"evm scam|evm chor|evm hacking|cbi sold|it sold|ed sold|ec sold|country sold|istifa do|gyanu|gyanesh|"
    r"unemployment under modi|paper leak|tax terrorism|suit boot|sell india|ambani adani sarkar|"
    r"fascist|hate this government|govt failed|government failed|bjp failed|ed raid|cbi misuse|"
    r"crony capitalism|electoral bond scam|press freedom under attack|"
    r"निष्क्रिय.*सरकार|नालायक.*सरकार|या शासनाचा धिक्कार|मुस्कटदाबी|"
    r"भाजपा हटाओ|तानाशाही|ईवीएम चोर|गोदी मीडिया|देश बेच दिया|सरकार विफल|इस्तीफा दो)\b",
    re.I,
)

PRO_OPPOSITION_MARKERS = re.compile(
    r"\b(?:vote congress|vote for congress|support congress|support rahul gandhi|rahul gandhi zindabad|rahul gandhi jindabad|satyagraha|nyay guarantee|save constitution|samvidhan bachao|"
    r"support india bloc|akhilesh bhaiya|aap zindabad|aap jindabad|kejriwal zindabad|stood up against govt|"
    r"congress zindabad|congress jindabad|india alliance|india bloc|opposition united|rahul ji|rahul sir|"
    r"inqualab zindabad|inquilab zindabad|more power to you students|well done students|"
    r"राहुल गांधी जिंदाबाद|संविधान बचाओ|विपक्ष एकजुट|कांग्रेस जिंदाबाद|इंकलाब जिंदाबाद)\b",
    re.I,
)
ANTI_OPPOSITION_MARKERS = re.compile(
    r"\b(?:pappu|pappu comedy show|corrupt congress|scam congress|pariwarwad|parivarwad|dynasty politics|tukde tukde|"
    r"kharge illegal|failed opposition|appeasement politics|vote bank politics|librandus?|"
    r"congress ka haath|gandhi pariwar|nehru family|shahi pariwar|anti national congress|"
    r"चमचा|चमचे|पप्पू|परिवारवाद|तुष्टिकरण)\b",
    re.I,
)

# Negation and failure patterns for bidirectional valence calibration
NEGATION_PRECEDING_RE = re.compile(
    r"\b(?:not|no|never|don\'t|dont|doesn\'t|doesnt|won\'t|wont|cannot|cant|can\'t|stop|against|hate|fail|failed|failure|nahi|nahin|na|mat|bnd|band)\b",
    re.I,
)
FOLLOWING_FAILURE_RE = re.compile(
    r"\b(?:failed|failure|fail|scam|chor|hatao|resign|fraud|corrupt|loot|flop|disaster)\b",
    re.I,
)


def check_phrase_valence(text: str, pattern: re.Pattern) -> Tuple[int, int]:
    """Check matches for a pattern and determine whether they are positive or negated/pejorative.

    Returns (pro_hits, negated_hits).
    A phrase is negated if preceded by negation words within 5 words, or followed by failure words within 3 words.
    """
    pro_hits = 0
    negated_hits = 0
    for match in pattern.finditer(text):
        start, end = match.start(), match.end()
        prec = text[:start].strip().split()
        if any(NEGATION_PRECEDING_RE.search(w) for w in prec[-5:]):
            negated_hits += 1
            continue
        foll = text[end:].strip().split()
        if any(FOLLOWING_FAILURE_RE.search(w) for w in foll[:3]):
            negated_hits += 1
            continue
        pro_hits += 1
    return pro_hits, negated_hits


# ---- Implicit Signal Banks ----
# These are strong contextual signals that don't mention parties explicitly but indicate political lean

# Pro-incumbent / Pro-government signals: governance achievements, national pride, development
_PRO_RULING_IMPLICIT = [
    # Development & infrastructure achievements
    "viksit bharat", "atmanirbhar bharat", "digital india", "new india", "amrit kaal", "amrit mahotsav",
    "india growing", "india shining", "india development", "make in india", "startup india",
    "jan dhan", "pmjdy", "pm kisan", "ayushman bharat", "ujjwala yojana", "har ghar jal",
    "swachh bharat", "pradhan mantri awas", "pm awas", "housing for all",
    "vande bharat", "bullet train", "expressway", "highway", "new airport", "metro expansion",
    "g20 india", "g20 presidency", "india's soft power", "global south leader",
    "upi global", "upi payment", "digital payment revolution",
    # ISRO / national achievement
    "isro achievement", "chandrayaan success", "gaganyaan", "india in space", "proud of isro",
    "our scientists", "india proud", "bharat ka gaurv",
    # Cultural / religious achievements under current govt
    "ram mandir inauguration", "ram mandir bhoomi", "pran pratishtha", "kashi vishwanath",
    "kedarnath development", "char dham yatra", "yoga day", "international yoga",
    # Military / security
    "surgical strike", "balakot", "galwan", "brave army", "jai hind proud",
    "india strong", "national security", "border security",
    # Economic positive framing
    "fastest growing economy", "india gdp", "india 5 trillion", "economic growth",
    "foreign investment", "fdi record", "stock market all time high",
    # BJP/NDA wins / election victories
    "bjp wins", "nda wins", "modi wave", "india mandate",
]

# Anti-incumbent / Anti-government signals: policy failures, governance critique
_ANTI_RULING_IMPLICIT = [
    # Paper leaks / exam controversies
    "paper leak", "neet paper", "upsc paper", "exam leak", "question paper leak",
    "ssc paper", "railway paper", "exam cancelled", "neet ug scam",
    # Employment / unemployment
    "unemployment", "berozgari", "no jobs", "job crisis", "youth unemployment",
    "recruitment scam", "agniveer controversy", "agniveer scheme", "short term army",
    # Inflation / cost of living
    "petrol price hike", "diesel price hike", "lpg hike", "mehengai", "inflation high",
    "price rise", "onion price", "tomato price", "grocery expensive", "cost of living",
    # Governance failures
    "government failed", "system failed", "policy failure", "government ignored",
    "minister resign", "accountability needed", "no accountability",
    # Democratic concerns
    "press freedom", "media censorship", "journalist arrested", "free speech threat",
    "democracy under threat", "constitution under threat", "federal structure",
    "opposition harassed", "opposition arrested", "ed raid on opposition",
    "it raid on journalists", "cbi misuse", "weaponizing agencies",
    # Electoral integrity
    "evm tampered", "voter id deleted", "voter list manipulation", "booth capturing",
    "election commission biased", "cec controversy",
    # Minorities / social issues
    "mob lynching", "bulldozer justice", "anti-muslim", "religious discrimination",
    "minority rights", "communal violence", "hate speech",
    # Corruption
    "adani controversy", "adani modi", "crony capitalism", "electoral bond",
    "corporate lobbying", "suit boot sarkar",
    # Welfare gaps
    "manrega cut", "welfare scheme cancelled", "ration card denied", "ayushman rejected",
]

# Pro-opposition / Pro-accountability signals
_PRO_OPP_IMPLICIT = [
    "save constitution", "save democracy", "india alliance", "united opposition",
    "india bloc", "alternative government", "alternative vision",
    "rahul gandhi bharat jodo", "bharat jodo yatra", "padyatra",
    "nyay scheme", "minimum income guarantee", "iga",
    "farmer rights", "msp guarantee", "kisan andolan support",
    "student rights", "student protests support", "neet reform needed",
    "dalit rights", "obc census", "caste census support", "social justice",
    "reservation protection", "constitution protection",
]

# Anti-opposition signals (used to boost BJP score)
_ANTI_OPP_IMPLICIT = [
    "congress corruption", "upa scam", "manmohan era", "2g scam", "coal scam", "cwg scam",
    "nehru mistakes", "60 saal loota", "70 saal loota", "parivar politics",
    "dynasty raj", "dynastic party", "gandhi family control",
    "kejriwal liquor scam", "delhi liquor policy", "aap scam",
    "mamata violence", "tmc violence", "bengal violence", "opposition drama",
    "opposition no alternative", "opposition no vision", "opposition failed",
    "pappu politics", "rahul gandhi no experience",
]

# Issue-cluster to political direction priors for India 2024-26
# Key insight: issues are not neutral - they carry political valence in Indian context
_ISSUE_CLUSTER_PRIORS = {
    # Anti-incumbent issues (opposition typically benefits from criticism)
    "employment": ("opp", 0.4),       # unemployment is PM criticism
    "inflation": ("opp", 0.45),        # inflation hits ruling party
    "education": ("opp", 0.35),        # paper leaks → anti-govt
    "corruption": ("opp", 0.4),        # corruption → ruling party criticism
    "elections": ("neutral", 0.0),     # elections → contested
    # Pro-incumbent issues (ruling party benefits from achievements)
    "infrastructure": ("bjp", 0.35),   # roads/airports/railways = BJP achievement narrative
    "defence": ("bjp", 0.4),          # military/ISRO = nationalism = BJP
    "welfare": ("bjp", 0.3),           # PM schemes credit to ruling party
    "governance": ("opp", 0.3),        # governance criticism → anti-incumbent
    "farmers": ("opp", 0.3),           # farm laws backlash → anti-BJP
    "communal": ("bjp", 0.25),         # Hindu framing → BJP lens
    "caste": ("opp", 0.3),             # caste census → opposition demand
    "women_safety": ("opp", 0.25),    # crime critique → anti-govt
    "economy": ("neutral", 0.1),       # mixed (growth vs inequality)
    "none": ("neutral", 0.0),
}


# Cultural / Religious Perspective Indicators
COMMUNITY_LEXICON = {
    "Hindu": [
        "ram mandir", "sanatan", "hindutva", "kumbh", "kedarnath", "mandir", "temple",
        "hanuman", "baps", "swami", "puja", "dharma", "shri ram", "hindu", "vhp",
        "bajrang dal", "vishwa hindu", "ghar wapsi",
        "राम मंदिर", "सनातन", "हिंदुत्व", "कुंभ", "मंदिर", "हनुमान", "पूजा", "धर्म"
    ],
    "Muslim": [
        "waqf", "waqf board", "masjid", "madrasa", "halal", "hijab",
        "sharia", "uapa", "pfi", "aimim", "owaisi", "muslim personal law",
        "मस्जिद", "मदरसा", "वक्फ"
    ],
    "Sikh": [
        "gurdwara", "bandi singhs", "sikh", "khalistan",
        "harmandir sahib", "sgpc", "akali dal", "gurpurab", "langar",
        "गुरुद्वारा"
    ],
    "Christian": [
        "church", "missionary", "dalit christian", "kerala church",
        "christmas", "easter", "csi", "cross", "bible",
        "चर्च"
    ],
    "Secular / Civic / General": [
        "voter deletion", "electoral roll", "farmer protest", "msp", "punjab",
        "minority rights", "anti-muslim", "mob lynching", "cattle theft",
        "constitution", "evm", "cec", "democracy", "supreme court", "high court",
        "unemployment", "railway safety", "infrastructure", "tax", "police", "paper leak",
        "hospital", "school", "college", "education", "health", "welfare", "pension",
        "संविधान", "सुप्रीम कोर्ट", "लोकतंत्र", "बेरोजगारी", "किसान आंदोलन"
    ]
}


class PoliticalPulseEngine:
    """Classifies political favorability and party orientation at topic and comment levels."""

    def classify_text_favorability(self, text: str) -> dict:
        """Classify a single comment or headline into a party leaning distribution.

        Incorporates negation handling, following failure words, implicit signals,
        and fine-grained party probability estimation.
        """
        text_lower = text.lower()

        bjp_score = 0.0
        opp_score = 0.0
        # Start at 0.20 — if genuine political signals exist, they will calibrate and dominate
        neutral_score = 0.20

        has_bjp = bool(BJP_KEYWORDS.search(text))
        has_opp = bool(OPPOSITION_KEYWORDS.search(text))

        pro_r_hits, neg_r_hits = check_phrase_valence(text, PRO_RULING_MARKERS)
        anti_r_hits = 1 if ANTI_RULING_MARKERS.search(text) else 0
        pro_o_hits, neg_o_hits = check_phrase_valence(text, PRO_OPPOSITION_MARKERS)
        anti_o_hits = 1 if ANTI_OPPOSITION_MARKERS.search(text) else 0

        # Strong explicit signals with negation inversion
        if pro_r_hits > 0 or anti_o_hits > 0:
            bjp_score += 1.2
            neutral_score = max(neutral_score - 0.1, 0.05)
        if neg_r_hits > 0 or anti_r_hits > 0 or pro_o_hits > 0:
            opp_score += 1.2
            neutral_score = max(neutral_score - 0.1, 0.05)
        if neg_o_hits > 0:
            bjp_score += 0.8
            neutral_score = max(neutral_score - 0.1, 0.05)

        # Implicit pro-ruling signals
        pro_impl_hits = sum(1 for term in _PRO_RULING_IMPLICIT if term in text_lower)
        if pro_impl_hits > 0 and neg_r_hits == 0 and anti_r_hits == 0:
            bjp_score += 0.6 * min(pro_impl_hits, 2)
            neutral_score = max(neutral_score - 0.05 * pro_impl_hits, 0.05)

        # Implicit anti-ruling signals
        anti_impl_hits = sum(1 for term in _ANTI_RULING_IMPLICIT if term in text_lower)
        if anti_impl_hits > 0 and pro_r_hits == 0:
            opp_score += 0.6 * min(anti_impl_hits, 2)
            neutral_score = max(neutral_score - 0.05 * anti_impl_hits, 0.05)

        # Implicit pro-opposition signals
        pro_opp_impl = sum(1 for term in _PRO_OPP_IMPLICIT if term in text_lower)
        if pro_opp_impl > 0 and neg_o_hits == 0:
            opp_score += 0.5 * min(pro_opp_impl, 2)
            neutral_score = max(neutral_score - 0.05, 0.05)

        # Implicit anti-opposition signals (boost BJP)
        anti_opp_impl = sum(1 for term in _ANTI_OPP_IMPLICIT if term in text_lower)
        if anti_opp_impl > 0:
            bjp_score += 0.5 * min(anti_opp_impl, 2)
            neutral_score = max(neutral_score - 0.05, 0.05)

        # Contextual sentiment when party is mentioned but no explicit position marker
        if has_bjp and (pro_r_hits == 0 and neg_r_hits == 0 and anti_r_hits == 0) and pro_impl_hits == 0 and anti_impl_hits == 0:
            if any(w in text_lower for w in [
                "proud", "hero", "great", "win", "respect", "congrats", "best", "defends",
                "strong", "good work", "well done", "appreciate", "thank", "achievement",
                "development", "growth", "progress", "support modi", "good"
            ]):
                bjp_score += 0.55
                neutral_score = max(neutral_score - 0.05, 0.05)
            elif any(w in text_lower for w in [
                "scam", "shame", "fail", "resign", "worst", "arrest", "arrested", "protest", "corrupt",
                "probe", "fraud", "loot", "criminal", "exposed", "accountability", "unacceptable",
                "detained", "suppressed", "censored", "crackdown", "lathi"
            ]):
                opp_score += 0.55
                neutral_score = max(neutral_score - 0.05, 0.05)

        if has_opp and (pro_o_hits == 0 and neg_o_hits == 0 and anti_o_hits == 0) and pro_opp_impl == 0 and anti_opp_impl == 0:
            if any(w in text_lower for w in [
                "truth", "brave", "honest", "right", "justice", "support", "campaigns",
                "satyagraha", "hope", "future", "better", "fight for", "standing up",
                "protest", "detained", "arrested", "march", "demonstration", "rally", "rally for",
                "demand", "accountability", "speaking up", "voice for"
            ]):
                opp_score += 0.60
                neutral_score = max(neutral_score - 0.05, 0.05)
            elif any(w in text_lower for w in [
                "drama", "hypocrite", "lie", "loser", "fail", "slams", "illegal",
                "dynast", "corrupt", "pappu", "rejected", "irrelevant", "opportunist"
            ]):
                bjp_score += 0.55
                neutral_score = max(neutral_score - 0.05, 0.05)

        # Specific institutional controversy signals (election commission, CEC, EVMs)
        if any(t in text_lower for t in [
            "gyanesh kumar", "gyanesh", "cec ", "election commissioner", "cec controversy",
            "election commission", "voter id deleted", "voter list", "form 7",
        ]):
            opp_score += 0.7
            neutral_score = max(neutral_score - 0.1, 0.05)

        # If neither BJP nor opposition mentioned but strong governance/political language exists
        if not has_bjp and not has_opp and bjp_score == 0 and opp_score == 0:
            if any(w in text_lower for w in [
                "government should", "government must", "government failed", "this government",
                "incumbent", "ruling party", "accountability", "resign", "protest against govt"
            ]):
                opp_score += 0.3  # Civic criticism → slight anti-incumbent lean
                neutral_score = max(neutral_score - 0.05, 0.05)

        total = bjp_score + opp_score + neutral_score
        if total <= 0:
            total = 1.0

        bjp_norm = round(bjp_score / total, 3)
        opp_norm = round(opp_score / total, 3)
        neu_norm = round(neutral_score / total, 3)

        leaning = "Neutral / Non-partisan"
        if bjp_norm >= opp_norm + 0.18 and bjp_norm >= 0.25:
            leaning = "Favoring BJP / NDA"
        elif opp_norm >= bjp_norm + 0.18 and opp_norm >= 0.25:
            leaning = "Favoring Opposition / INDIA"
        elif bjp_norm + opp_norm >= 0.35 and abs(bjp_norm - opp_norm) < 0.18:
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

    def _apply_issue_prior(self, issue_cluster: str, bjp_w: float, opp_w: float, neu_w: float,
                           strength: float = 1.0) -> tuple[float, float, float]:
        """Nudge weighted scores using the political valence of an issue cluster."""
        direction, magnitude = _ISSUE_CLUSTER_PRIORS.get(issue_cluster, ("neutral", 0.0))
        if magnitude <= 0 or direction == "neutral":
            return bjp_w, opp_w, neu_w
        base = bjp_w + opp_w + neu_w
        nudge = base * magnitude * strength
        if direction == "bjp":
            bjp_w += nudge
            neu_w = max(neu_w - nudge * 0.5, 0.0)
        elif direction == "opp":
            opp_w += nudge
            neu_w = max(neu_w - nudge * 0.5, 0.0)
        return bjp_w, opp_w, neu_w

    def evaluate_topic_pulse(
        self,
        topic: dict,
        comments: list[dict],
        reddit_signals: Optional[list[dict]] = None,
        ml_classifier_fn=None,
    ) -> dict:
        """Aggregate per-engagement political lean across all comments.

        Each comment/engagement is scored individually.  Result = % of
        like-weighted engagements that favored BJP vs Opposition vs Neutral,
        plus granular multi-party support breakdown (BJP, Congress, AAP, TMC, SP, DMK).
        Topic text is never scored — only what commenters said counts.

        ml_classifier_fn: optional callable(text) -> str  (party label such as
        "BJP", "Congress", "AAP", …, or "None").  Used as a fallback when the
        keyword classifier returns neutral for a comment with len > 8.
        """
        title = topic.get("title", "")

        bjp_weight = 0.0
        opp_weight = 0.0
        neutral_weight = 0.0
        engagement_count = 0

        party_weights = defaultdict(float)

        community_counts = Counter()
        community_favor = defaultdict(lambda: {"bjp": 0.0, "opp": 0.0, "neu": 0.0, "total": 0})

        _BJP_ALLIED = {"BJP"}
        _OPP_ALLIED = {"Congress", "AAP", "TMC", "SP", "DMK", "RJD", "NCP", "JMM", "CPI", "CPM"}

        def _ml_lean(txt: str) -> Optional[Tuple[str, float]]:
            if len(txt) <= 8:
                return None
            # Check neural transformer first
            neural_res = predict_party_neural(txt)
            if neural_res and neural_res[0] != "None" and neural_res[1] >= 0.50:
                p, c = neural_res
                if p in _BJP_ALLIED:
                    return "bjp", c
                if p in _OPP_ALLIED:
                    return "opp", c
            # Fall back to supplied ml_classifier_fn if available
            if ml_classifier_fn is not None:
                try:
                    party = ml_classifier_fn(txt)
                    if party in _BJP_ALLIED:
                        return "bjp", 0.65
                    if party in _OPP_ALLIED:
                        return "opp", 0.65
                except Exception:
                    pass
            return None

        # 1. Score each Instagram / collected comment as one engagement
        for c in (comments or []):
            txt = str(c.get("text") or "")
            if not txt.strip():
                continue
            likes = max(int(c.get("likes") or c.get("like_count") or 1), 1)
            log_weight = math.log1p(likes)
            engagement_count += 1

            c_lens = self.infer_community_lens(txt, title)
            community_counts[c_lens] += 1

            res = self.classify_text_favorability(txt)

            # ML fallback: if keyword says neutral, ask the neural / ML party classifier
            if res["leaning"] == "Neutral / Non-partisan":
                ml_eval = _ml_lean(txt)
                if ml_eval is not None:
                    side, conf = ml_eval
                    boost = min(0.65, max(0.50, conf * 0.7))
                    neu = max(0.15, 1.0 - boost - 0.15)
                    other = 1.0 - boost - neu
                    if side == "bjp":
                        res = {"bjp_share": boost, "opp_share": other, "neutral_share": neu, "leaning": "bjp"}
                    elif side == "opp":
                        res = {"bjp_share": other, "opp_share": boost, "neutral_share": neu, "leaning": "opp"}

            bjp_weight += res["bjp_share"] * log_weight
            opp_weight += res["opp_share"] * log_weight
            neutral_weight += res["neutral_share"] * log_weight

            # Track fine-grained party distribution
            p_dist = predict_party_neural_distribution(txt)
            if p_dist:
                for party_name, party_p in p_dist.items():
                    party_weights[party_name] += party_p * log_weight

            community_favor[c_lens]["bjp"] += res["bjp_share"] * log_weight
            community_favor[c_lens]["opp"] += res["opp_share"] * log_weight
            community_favor[c_lens]["neu"] += res["neutral_share"] * log_weight
            community_favor[c_lens]["total"] += 1

        # 2. Reddit signals with subreddit priors (each Reddit post title = one engagement)
        if reddit_signals:
            for r in reddit_signals:
                r_title = r.get("title", "")
                if not r_title.strip():
                    continue
                sub = r.get("feed", "")
                r_res = self.classify_text_favorability(r_title)
                score = max(int((r.get("engagement") or {}).get("score", 1)), 1)
                w = math.log1p(score) * 1.5
                engagement_count += 1

                # Subreddit priors: r/IndiaSpeaks and r/indiadiscussion lean nationalist/BJP,
                # r/india and r/unitedstatesofindia lean anti-incumbent/opposition
                if sub in ("r/IndiaSpeaks", "r/indiadiscussion"):
                    r_res["bjp_share"] = min(r_res["bjp_share"] + 0.30, 1.0)
                    r_res["neutral_share"] = max(r_res["neutral_share"] - 0.30, 0.0)
                elif sub in ("r/india", "r/unitedstatesofindia"):
                    r_res["opp_share"] = min(r_res["opp_share"] + 0.30, 1.0)
                    r_res["neutral_share"] = max(r_res["neutral_share"] - 0.30, 0.0)

                bjp_weight += r_res["bjp_share"] * w
                opp_weight += r_res["opp_share"] * w
                neutral_weight += r_res["neutral_share"] * w

                # Track fine-grained party distribution for reddit titles
                r_dist = predict_party_neural_distribution(r_title)
                if r_dist:
                    for party_name, party_p in r_dist.items():
                        party_weights[party_name] += party_p * w

                r_lens = self.infer_community_lens(r_title, title)
                community_counts[r_lens] += 1
                community_favor[r_lens]["bjp"] += r_res["bjp_share"] * w
                community_favor[r_lens]["opp"] += r_res["opp_share"] * w
                community_favor[r_lens]["neu"] += r_res["neutral_share"] * w
                community_favor[r_lens]["total"] += 1

        # When no comment engagements exist, derive pulse from topic political framing and story text
        if engagement_count == 0:
            brief = topic.get("brief") or {}
            pf = brief.get("political_framing") or {}
            p_dir = pf.get("direction", "")
            p_conf = float(pf.get("confidence") or 0.0)

            # Check story narrative text
            story_parts = [topic.get("title", "")]
            if brief.get("summary"):
                story_parts.append(brief.get("summary"))
            if brief.get("why_trending"):
                story_parts.append(brief.get("why_trending"))
            story_text = " ".join(story_parts)
            story_fav = self.classify_text_favorability(story_text)

            bjp_share = 0.0
            opp_share = 0.0
            neu_share = 1.0

            if p_dir == "pro_incumbent" and p_conf > 0.2:
                bjp_share = max(0.60, round(p_conf * 0.85, 2))
                opp_share = round(max(0.05, (1.0 - bjp_share) * 0.25), 2)
                neu_share = round(max(0.0, 1.0 - bjp_share - opp_share), 2)
            elif p_dir == "anti_opposition" and p_conf > 0.2:
                bjp_share = max(0.55, round(p_conf * 0.75, 2))
                opp_share = round(max(0.05, (1.0 - bjp_share) * 0.25), 2)
                neu_share = round(max(0.0, 1.0 - bjp_share - opp_share), 2)
            elif p_dir == "anti_incumbent" and p_conf > 0.2:
                opp_share = max(0.60, round(p_conf * 0.85, 2))
                bjp_share = round(max(0.05, (1.0 - opp_share) * 0.25), 2)
                neu_share = round(max(0.0, 1.0 - bjp_share - opp_share), 2)
            elif p_dir == "pro_opposition" and p_conf > 0.2:
                opp_share = max(0.60, round(p_conf * 0.85, 2))
                bjp_share = round(max(0.05, (1.0 - opp_share) * 0.25), 2)
                neu_share = round(max(0.0, 1.0 - bjp_share - opp_share), 2)
            elif story_fav["bjp_share"] > 0.3 or story_fav["opp_share"] > 0.3:
                bjp_share = story_fav["bjp_share"]
                opp_share = story_fav["opp_share"]
                neu_share = story_fav["neutral_share"]

            tot_s = bjp_share + opp_share + neu_share or 1.0
            b_pct = round((bjp_share / tot_s) * 100, 1)
            o_pct = round((opp_share / tot_s) * 100, 1)
            n_pct = round((neu_share / tot_s) * 100, 1)

            if b_pct >= o_pct + 12 and b_pct >= 25:
                leaning = "Favoring BJP / NDA (Narrative Frame)"
            elif o_pct >= b_pct + 12 and o_pct >= 25:
                leaning = "Favoring Opposition / Anti-Incumbent (Narrative Frame)"
            elif b_pct + o_pct >= 30:
                leaning = "Balanced / Contested (Narrative Frame)"
            else:
                leaning = "Neutral / Non-partisan / Civic"

            primary_lens = self.infer_community_lens(story_text, topic.get("title", ""))

            return {
                "overall_leaning": leaning,
                "bjp_favor_pct": b_pct,
                "opp_favor_pct": o_pct,
                "neutral_pct": n_pct,
                "party_breakdown": {"BJP": b_pct, "Opposition": o_pct, "Neutral": n_pct},
                "engagement_count": 0,
                "primary_community_lens": primary_lens,
                "community_cross_tab": {},
                "narrative_summary": f"Derived from political framing & narrative context ({p_dir or 'neutral'} frame).",
                "is_fallback": True,
            }

        total_w = bjp_weight + opp_weight + neutral_weight
        if total_w <= 0:
            total_w = 1.0

        bjp_pct = round((bjp_weight / total_w) * 100, 1)
        opp_pct = round((opp_weight / total_w) * 100, 1)
        neutral_pct = round((neutral_weight / total_w) * 100, 1)

        # Multi-party breakdown computation
        party_breakdown = {}
        tot_pw = sum(party_weights.values())
        if tot_pw > 0:
            for p_k, p_v in party_weights.items():
                party_breakdown[p_k] = round((p_v / tot_pw) * 100, 1)
        else:
            party_breakdown = {"BJP": bjp_pct, "Opposition": opp_pct, "Neutral": neutral_pct}

        # Categorical Leaning — lower thresholds to reduce false "Non-partisan"
        n = engagement_count
        if abs(bjp_pct - opp_pct) <= 10.0 and (bjp_pct + opp_pct >= 35.0):
            leaning = "Balanced / Contested Debate"
            narrative = f"Divided engagements ({n} comments): {bjp_pct}% BJP-leaning, {opp_pct}% Opposition — polarized debate."
        elif bjp_pct >= opp_pct + 8.0 and bjp_pct >= 22.0:
            leaning = "Favoring BJP / NDA"
            narrative = f"{bjp_pct}% of {n} engagements supported BJP/NDA vs {opp_pct}% for the Opposition."
        elif opp_pct >= bjp_pct + 8.0 and opp_pct >= 22.0:
            leaning = "Favoring Opposition / Anti-Incumbent"
            narrative = f"{opp_pct}% of {n} engagements supported Opposition/anti-incumbent vs {bjp_pct}% for BJP."
        else:
            leaning = "Neutral / Non-partisan / Civic"
            narrative = f"Mostly neutral engagements ({n} comments): {bjp_pct}% BJP-leaning, {opp_pct}% opp-leaning, {neutral_pct}% neutral."

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
            "party_breakdown": party_breakdown,
            "engagement_count": engagement_count,
            "primary_community_lens": primary_lens,
            "community_cross_tab": community_table,
            "narrative_summary": narrative,
        }


# Global singleton
political_pulse_engine = PoliticalPulseEngine()

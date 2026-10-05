"""Aggregate demographic inference from public profile text, handles, captions, and comments.

Provides:
  * Deterministic Indian given name & honorific lexicons for gender (male, female, org, unknown).
  * Syntactic, linguistic, and regex rules for age bracket (13-17, 18-24, 25-34, 35-54, 55+, unknown).
  * State and metro location mapping.
  * Cross-tabulation of stance and tone across demographic groups with full confidence tracking.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any
import joblib

# ---------------------------------------------------------------- LOAD TRAINED CLASSIFIERS ----
MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models"
_AGE_MODEL = None
_GENDER_MODEL = None
_MENTAL_STATE_MODEL = None

def _get_models():
    global _AGE_MODEL, _GENDER_MODEL, _MENTAL_STATE_MODEL
    if _AGE_MODEL is None and (MODELS_DIR / "age_classifier.joblib").exists():
        try:
            _AGE_MODEL = joblib.load(MODELS_DIR / "age_classifier.joblib")
            _GENDER_MODEL = joblib.load(MODELS_DIR / "gender_classifier.joblib")
            _MENTAL_STATE_MODEL = joblib.load(MODELS_DIR / "mental_state_classifier.joblib")
        except Exception:
            pass
    return _AGE_MODEL, _GENDER_MODEL, _MENTAL_STATE_MODEL

# ------------------------------------------------------------------ GENDER LEXICONS ----
MALE_NAMES = {
    "aakash", "aarav", "aayush", "abhay", "abhijeet", "abhinav", "abhishek", "adarsh", "aditya",
    "ajay", "ajit", "akash", "akhil", "akshay", "aman", "amar", "amit", "amrit", "anand", "aniket",
    "anil", "ankit", "ankush", "anmol", "anuj", "anupam", "aravind", "arjun", "arun", "arvind",
    "ashish", "ashok", "ashwin", "atul", "avinash", "ayush", "balram", "bharat", "bhaskar", "brijesh",
    "chandan", "chetan", "chirag", "darshan", "deepak", "dev", "devendra", "dharmendra", "dhruv",
    "dinesh", "divyansh", "durgesh", "ganesh", "gaurav", "gautam", "girish", "gopal", "govind",
    "gulshan", "hardik", "hari", "harish", "harsh", "harshit", "hemant", "himanshu", "hitesh",
    "imran", "inder", "ishaan", "jagdish", "jai", "jaideep", "jaspal", "jatin", "jay", "jayant",
    "jitendra", "kailash", "kamal", "karan", "kartik", "keshav", "kishan", "kishore", "krishna",
    "kunal", "lalit", "lokesh", "madhav", "mahesh", "manas", "manav", "mandeep", "mani", "manish",
    "manoj", "mayank", "mithun", "mohit", "mukesh", "mukul", "murli", "naresh", "naveen", "navneet",
    "neeraj", "nikhil", "nilesh", "nitin", "nripendra", "omkar", "pankaj", "paras", "pardeep",
    "parth", "pawan", "piyush", "pradeep", "prakash", "pramod", "pranav", "pranay", "prashant",
    "prateek", "praveen", "prem", "prince", "prithvi", "raghav", "rahul", "raj", "rajan", "rajat",
    "rajeev", "rajesh", "rajiv", "rajkumar", "rakesh", "ram", "raman", "ramesh", "ranjan", "ranveer",
    "ratan", "ravi", "ravindra", "rishabh", "rishi", "ritesh", "rohan", "rohit", "ronit", "sachin",
    "sagar", "sahil", "samar", "sameer", "sandeep", "sanjay", "sanjeev", "santosh", "sarvesh",
    "satish", "satyendra", "saurabh", "shailesh", "shakti", "shantanu", "shashank", "shashi",
    "shekhar", "shirish", "shivam", "shobhit", "shyam", "siddharth", "sohan", "somesh", "sourabh",
    "subhash", "sudarshan", "sudhir", "sujit", "sumit", "sunder", "sunil", "suraj", "suresh",
    "surya", "sushil", "swapnil", "tanmay", "tarun", "tejas", "tushar", "uday", "ujjwal", "umesh",
    "utkarsh", "vaibhav", "varun", "ved", "vicky", "vidur", "vijay", "vikas", "vikram", "vinay",
    "vinod", "vipin", "viplav", "virendra", "vishal", "vishnu", "vishwas", "vivek", "yash", "yogesh"
}

FEMALE_NAMES = {
    "aarti", "aashi", "aditi", "aishwarya", "akanksha", "alka", "amita", "amrita", "anamika", "ananya",
    "anchal", "anita", "anjali", "anjani", "ankita", "annu", "anshu", "antardrishti", "anupama",
    "anuradha", "anushree", "aparna", "archana", "aradhana", "arti", "aruna", "asha", "ashwini",
    "asita", "asmita", "avantika", "babita", "barkha", "bhavna", "bhumi", "chahat", "chandni",
    "chaya", "chetna", "chitra", "damini", "deeksha", "deepa", "deepali", "deepika", "deepti",
    "devika", "dhwani", "disha", "divya", "dolly", "drishti", "ekta", "garima", "gargi", "gauri",
    "gayatri", "geeta", "geetanjali", "gunjan", "hansa", "harini", "heena", "hema", "himani",
    "indu", "ishani", "ishita", "jagriti", "janki", "jaya", "jayashree", "jeetu", "jhanvi", "juhi",
    "jyotsna", "kajal", "kalpana", "kamla", "kamini", "kanak", "kanchan", "kanika", "karishma",
    "kavita", "khushi", "khushboo", "komal", "kriti", "kritika", "kusum", "lakshmi", "lata", "lavanya",
    "leela", "madhu", "madhuri", "mahima", "malini", "mamta", "manisha", "manjari", "manju",
    "mansi", "meena", "meenakshi", "meera", "megha", "mitali", "mohini", "mona", "monika", "mridula",
    "mukta", "namrata", "nandini", "nandita", "navya", "neelam", "neelima", "neetu", "neha",
    "nidhi", "nikita", "nirmala", "nisha", "nishita", "nita", "nivedita", "nutan", "pallavi",
    "paridhi", "parul", "payal", "pooja", "poonam", "prabha", "prachi", "pragya", "pratibha",
    "pratima", "preeti", "prerna", "priya", "priyanka", "pushpa", "rachna", "radha", "radhika",
    "ragini", "rajni", "rakhi", "rashmi", "rekha", "renu", "richa", "riddhi", "rina", "ritu",
    "riya", "roshni", "ruchi", "rupa", "rupali", "sadhna", "sakshi", "saloni", "samiksha", "sangeeta",
    "sanika", "sanjana", "sapna", "sarita", "saroj", "savita", "seema", "shaili", "shalini",
    "shanti", "sharda", "sheetal", "shikha", "shilpa", "shivani", "shobha", "shraddha", "shreya",
    "shubha", "shweta", "simran", "smita", "sneha", "snehlata", "sonal", "sonali", "sonia", "sudha",
    "sukanya", "suman", "sumitra", "sunaina", "sunita", "supriya", "surabhi", "surbhi", "sushma",
    "swati", "tanu", "tanvi", "tanya", "tara", "trisha", "tulsi", "uma", "urvashi", "usha",
    "vaishali", "vandana", "vani", "varsha", "vasudha", "veena", "vibha", "vidya", "vinita",
    "vrinda", "yamini", "yashoda"
}

AMBIGUOUS_NAMES = {
    "kiran", "sonu", "jyoti", "sandhya", "harpreet", "gurpreet", "amar", "deep", "preet", "jaspreet",
    "manpreet", "simar", "navjot", "kamalpreet", "sukhi"
}

MALE_HONORIFICS = {"mr", "shri", "sri", "pt", "pandit", "thakur", "sardar", "bhai", "bhaiya"}
FEMALE_HONORIFICS = {"mrs", "ms", "smt", "smti", "kumari", "miss", "didi", "bhabhi", "bhabhiji"}
ORG_HINTS = {
    "official", "news", "media", "party", "org", "trust", "foundation", "ngo", "tv", "press",
    "samiti", "sangathan", "morcha", "yuva", "committee", "association", "updates", "times",
    "channel", "network", "group", "page", "fanclub", "fan", "army", "support", "daily", "bulletin",
    "herald", "post", "gazette", "chronicle", "live", "portal", "express", "studio", "radio"
}

# ------------------------------------------------------------------- AGE REGEX & CUES ----
BIRTH_YEAR_RE = re.compile(r"\b(?:b\.?|born|est\.?)\s*(19[5-9]\d|20[0-2]\d)\b", re.I)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
BATCH_RE = re.compile(r"\b(?:batch|class)\s*(?:of)?\s*[\'’]?(\d{2,4})\b", re.I)

AGE_CUES = [
    ({"school", "class 10", "10th", "teenager", "board exam", "cbse"}, "13-17", 0.50),
    ({"student", "clg", "college", "btech", "b.tech", "neet", "jee", "aspirant",
      "class 12", "12th", "undergrad", "campus", "delhi university", "du", "iit", "nit"}, "18-24", 0.60),
    ({"engineer", "working", "job", "professional", "mba", "startup", "founder",
      "developer", "analyst", "consultant", "software", "corporate", "wfh"}, "25-34", 0.50),
    ({"father of", "mother of", "parent", "papa", "mom of", "husband", "wife",
      "family man", "proud parent"}, "35-54", 0.55),
    ({"retired", "ex-serviceman", "pensioner", "grandfather", "dada", "nana",
      "senior citizen", "veteran"}, "55+", 0.65),
]

# Linguistic markers in comment texts - Gen Z / Youth slang
GEN_Z_SLANG = {
    "bro", "bruh", "ngl", "fr", "frfr", "dead", "skull", "clown", "ratio", "w", "l",
    "cap", "nocap", "cringe", "rizz", "sigma", "beta", "delulu", "vibe", "lit", "lowkey",
    "omg", "idk", "wtf", "tbh", "smh", "fml", "savage", "flex", "slay", "stan", "fire",
    "scene", "bussin", "sus", "yeet", "cooked", "gyatt", "skibidi", "bhai", "bhaiya"
}

# Career & Working Professional / Taxpayer markers (25-34 age bracket proxy)
PROFESSIONAL_SLANG = {
    "salary", "tax", "taxes", "income", "emi", "loan", "corporate", "office", "manager",
    "ctc", "wfh", "hike", "switch", "startup", "developer", "engineer", "nifty", "sensex",
    "stock", "invest", "sip", "mutual", "fund", "pf", "epf", "gst", "inflation", "economy",
    "appraisal", "notice", "client", "workload", "recession", "it", "tcs", "infosys"
}

# Elder / Senior / Civic Respect markers (35-54 & 55+ age bracket proxy)
MATURE_SLANG = {
    "sir", "ji", "sahab", "pranam", "namaste", "blessings", "bless", "sanskar", "maryada",
    "dharma", "shri", "nation", "patriot", "desh", "constitution", "parliament", "sadhu",
    "swami", "children", "bache", "family", "generation", "values", "retirement", "pension"
}

# Female conversational cues & address markers
FEMALE_CONVERSATIONAL = {
    "didi", "behen", "bhabhi", "sister", "girl", "queen", "woman", "ladki", "she", "her",
    "women", "nari", "maa", "ammi", "di", "mam", "maam", "auntie", "saheli", "mummy"
}

# Male conversational cues & address markers
MALE_CONVERSATIONAL = {
    "bhai", "bhaiya", "bro", "bruh", "man", "king", "boy", "ladka", "he", "him",
    "bhaijaan", "veer", "sir", "dost", "yaar", "uncle", "chacha", "bapu"
}

AGE_BRACKETS = ["13-17", "18-24", "25-34", "35-54", "55+", "unknown"]

# ---------------------------------------------------------------- LOCATION MAPPINGS ----
INDIAN_CITIES = {
    "mumbai": ("Maharashtra", "urban"), "delhi": ("Delhi", "urban"), "bangalore": ("Karnataka", "urban"),
    "bengaluru": ("Karnataka", "urban"), "hyderabad": ("Telangana", "urban"), "ahmedabad": ("Gujarat", "urban"),
    "chennai": ("Tamil Nadu", "urban"), "kolkata": ("West Bengal", "urban"), "surat": ("Gujarat", "urban"),
    "pune": ("Maharashtra", "urban"), "jaipur": ("Rajasthan", "urban"), "lucknow": ("Uttar Pradesh", "urban"),
    "kanpur": ("Uttar Pradesh", "urban"), "nagpur": ("Maharashtra", "urban"), "patna": ("Bihar", "urban"),
    "indore": ("Madhya Pradesh", "urban"), "bhopal": ("Madhya Pradesh", "urban"), "ludhiana": ("Punjab", "urban"),
    "vadodara": ("Gujarat", "urban"), "agra": ("Uttar Pradesh", "semi-urban"), "varanasi": ("Uttar Pradesh", "semi-urban"),
    "amritsar": ("Punjab", "urban"), "ranchi": ("Jharkhand", "urban"), "guwahati": ("Assam", "urban"),
    "chandigarh": ("Chandigarh", "urban"), "kochi": ("Kerala", "urban"), "thiruvananthapuram": ("Kerala", "urban"),
    "dehradun": ("Uttarakhand", "urban"), "shimla": ("Himachal Pradesh", "semi-urban"), "srinagar": ("Jammu and Kashmir", "urban"),
    "jammu": ("Jammu and Kashmir", "urban"), "raipur": ("Chhattisgarh", "urban"), "bhubaneswar": ("Odisha", "urban")
}

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "chandigarh",
    "jammu and kashmir", "ladakh", "puducherry"
}


@dataclass
class ProfileDemographics:
    gender: str = "unknown"
    gender_confidence: float = 0.0
    age_bracket: str = "unknown"
    age_confidence: float = 0.0
    location_state: str | None = None
    location_city: str | None = None
    location_confidence: float = 0.0
    is_organization: bool = False
    evidence: list[str] = field(default_factory=list)


def infer_gender(display_name: str, bio: str = "") -> tuple[str, float, str | None]:
    """Infers gender from author name, username, or text cues."""
    full_text = f"{display_name} {bio}".lower()
    words = re.findall(r"[a-z]+", full_text)
    wordset = set(words)

    if wordset & ORG_HINTS:
        return "other", 0.50, "organisation-or-page"

    if wordset & FEMALE_HONORIFICS:
        return "female", 0.75, "honorific"
    if wordset & MALE_HONORIFICS:
        return "male", 0.75, "honorific"

    tokens = re.findall(r"[a-z]+", display_name.lower())
    for token in tokens:
        if token in AMBIGUOUS_NAMES:
            return "unknown", 0.20, "ambiguous-given-name"
        if token in FEMALE_NAMES:
            return "female", 0.70, "given-name"
        if token in MALE_NAMES:
            return "male", 0.70, "given-name"

    if any(p in full_text for p in ("she/her", "her pronouns")):
        return "female", 0.90, "stated-pronouns"
    if any(p in full_text for p in ("he/him", "his pronouns")):
        return "male", 0.90, "stated-pronouns"

    return "unknown", 0.0, None


def infer_age(text: str) -> tuple[str, float, str | None]:
    """Infers coarse age group from bio cues, graduation years, or text style."""
    low = text.lower()
    now_year = 2026

    m = BIRTH_YEAR_RE.search(low) or BATCH_RE.search(low)
    if m:
        raw = m.group(1)
        year = int(raw) if len(raw) == 4 else (2000 + int(raw) if int(raw) < 50 else 1900 + int(raw))
        age = (now_year - year) if BIRTH_YEAR_RE.search(low) else (now_year - year) + 22
        if 13 <= age <= 17:
            return "13-17", 0.85, "stated-year"
        if 18 <= age <= 24:
            return "18-24", 0.85, "stated-year"
        if 25 <= age <= 34:
            return "25-34", 0.85, "stated-year"
        if 35 <= age <= 54:
            return "35-54", 0.85, "stated-year"
        if age >= 55:
            return "55+", 0.85, "stated-year"

    for cues, bracket, conf in AGE_CUES:
        if any(c in low for c in cues):
            return bracket, conf, "bio-cue"

    tokens = set(re.findall(r"\b[a-z]+\b", low))
    if tokens & GEN_Z_SLANG:
        return "18-24", 0.50, "youth-slang"
    if tokens & PROFESSIONAL_SLANG:
        return "25-34", 0.50, "career-professional"
    if tokens & MATURE_SLANG:
        return "35-54", 0.45, "civic-respect"

    # ML Classifier fallback
    age_m, _, _ = _get_models()
    if age_m is not None and len(text.strip()) > 15:
        try:
            probs = age_m.predict_proba([text])[0]
            max_idx = probs.argmax()
            conf = float(probs[max_idx])
            bracket = str(age_m.classes_[max_idx])
            if conf >= 0.35:
                return bracket, round(conf, 2), "ml-classifier"
        except Exception:
            pass

    return "unknown", 0.0, None


def infer_location(text: str) -> tuple[str | None, str | None, float]:
    """Infers Indian state and city if mentioned in text or through Indic regional scripts."""
    # Script-level mental & regional presence
    if re.search(r"[\u0d00-\u0d7f]", text):  # Malayalam
        return "Kerala", None, 0.85
    if re.search(r"[\u0b80-\u0bff]", text):  # Tamil
        return "Tamil Nadu", None, 0.85
    if re.search(r"[\u0c00-\u0c7f]", text):  # Telugu
        return "Andhra Pradesh / Telangana", None, 0.85
    if re.search(r"[\u0c80-\u0cff]", text):  # Kannada
        return "Karnataka", None, 0.85
    if re.search(r"[\u0980-\u09ff]", text):  # Bengali
        return "West Bengal", None, 0.85
    if re.search(r"[\u0a80-\u0aff]", text):  # Gujarati
        return "Gujarat", None, 0.85
    if re.search(r"[\u0a00-\u0a7f]", text):  # Gurmukhi / Punjabi
        return "Punjab", None, 0.85

    low = text.lower()
    for city, (state, _) in INDIAN_CITIES.items():
        if re.search(r"\b" + re.escape(city) + r"\b", low):
            return state, city.title(), 0.75
    for state in INDIAN_STATES:
        if re.search(r"\b" + re.escape(state) + r"\b", low):
            return state.title(), None, 0.70
    return None, None, 0.0


def infer_profile(name_or_handle: str, bio: str = "", location: str = "") -> ProfileDemographics:
    """Full demographic inference on an author or user profile."""
    clean_name = re.sub(r"[_.\d]+", " ", name_or_handle).strip()
    g, g_conf, g_ev = infer_gender(clean_name, bio)
    a, a_conf, a_ev = infer_age(f"{bio} {location}")
    st, city, l_conf = infer_location(f"{location} {bio}")

    ev = []
    if g_ev: ev.append(f"gender:{g_ev}")
    if a_ev: ev.append(f"age:{a_ev}")
    if st: ev.append(f"loc:{st}")

    is_org = (g == "other" and g_ev == "organisation-or-page")
    return ProfileDemographics(
        gender=g,
        gender_confidence=g_conf,
        age_bracket=a,
        age_confidence=a_conf,
        location_state=st,
        location_city=city,
        location_confidence=l_conf,
        is_organization=is_org,
        evidence=ev
    )


def infer_commenter_demographics(comment: dict) -> ProfileDemographics:
    """Infers demographic profile for an individual comment from author_id, text, and metadata."""
    raw_user = comment.get("user") or {}
    username = raw_user.get("username") or comment.get("username") or ""
    text = comment.get("text", "")

    if username:
        return infer_profile(username, bio=text)

    # First-person self-identification and linguistic cues in comment text
    g, g_conf, g_ev = "unknown", 0.0, None
    low = text.lower()
    words = set(re.findall(r"[a-z]+", low))
    if re.search(r"\b(as a woman|as a girl|female here|ladki|bhabhi)\b", low):
        g, g_conf, g_ev = "female", 0.65, "self-ident"
    elif re.search(r"\b(as a man|as a guy|ladka|bhai|bhaiya)\b", low):
        g, g_conf, g_ev = "male", 0.65, "self-ident"
    elif words & FEMALE_CONVERSATIONAL:
        g, g_conf, g_ev = "female", 0.45, "conversational-cue"
    elif words & MALE_CONVERSATIONAL:
        g, g_conf, g_ev = "male", 0.45, "conversational-cue"
    else:
        # ML Classifier fallback for subtle gender cues
        _, gen_m, _ = _get_models()
        if gen_m is not None and len(text.strip()) > 15:
            try:
                probs = gen_m.predict_proba([text])[0]
                max_idx = probs.argmax()
                conf = float(probs[max_idx])
                cls = str(gen_m.classes_[max_idx])
                if cls in ("female", "male") and conf >= 0.45:
                    g, g_conf, g_ev = cls, round(conf, 2), "ml-classifier"
            except Exception:
                pass

    a, a_conf, a_ev = infer_age(text)
    st, city, l_conf = infer_location(text)

    ev = []
    if g_ev: ev.append(f"gender:{g_ev}")
    if a_ev: ev.append(f"age:{a_ev}")
    if st: ev.append(f"loc:{st}")

    return ProfileDemographics(
        gender=g,
        gender_confidence=g_conf,
        age_bracket=a,
        age_confidence=a_conf,
        location_state=st,
        location_city=city,
        location_confidence=l_conf,
        evidence=ev
    )


def compute_demographic_breakdowns(topic: dict) -> dict:
    """Computes demographic cross-tabs (gender, age, location) across topic post creators and commenters.
    Integrates with debate sides and stance to highlight demographic influence."""
    brief = topic.get("brief") or {}
    measured = brief.get("measured_opinion") or {}
    media = topic.get("instagram", {}).get("media", [])
    on_topic = set(brief.get("on_topic_codes", []))

    # 1. Author Demographics (Post creators)
    authors_seen = set()
    author_gender_counts = Counter()
    author_age_counts = Counter()
    author_loc_counts = Counter()
    author_known_gender = 0
    author_known_age = 0
    author_known_loc = 0

    for m in media:
        auth = m.get("author")
        if auth and auth not in authors_seen:
            authors_seen.add(auth)
            caption = m.get("caption", "")
            d = infer_profile(auth, bio=caption[:150])
            author_gender_counts[d.gender] += 1
            author_age_counts[d.age_bracket] += 1
            if d.location_state:
                author_loc_counts[d.location_state] += 1
            if d.gender != "unknown":
                author_known_gender += 1
            if d.age_bracket != "unknown":
                author_known_age += 1
            if d.location_state:
                author_known_loc += 1

    total_authors = len(authors_seen) or 1

    # 2. Commenter demographics & Stance Cross-Tabs
    # Read from media comments if available, or fall back to brief.comment_pool
    comments_pool = [c for m in media if not on_topic or m.get("code") in on_topic
                     for c in m.get("comments", []) if c.get("text")]
    if not comments_pool and brief.get("comment_pool"):
        comments_pool = [c for c in brief.get("comment_pool", []) if c.get("text")]

    sides = measured.get("sides") or []
    comment_gender_counts = Counter()
    comment_age_counts = Counter()
    comment_loc_counts = Counter()

    side_demographics = [{"side": i, "position": s.get("position", ""),
                          "gender": Counter(), "age": Counter()}
                         for i, s in enumerate(sides)]

    comm_known_gender = 0
    comm_known_age = 0
    comm_known_loc = 0

    for i, c in enumerate(comments_pool):
        demo = infer_commenter_demographics(c)
        comment_gender_counts[demo.gender] += 1
        comment_age_counts[demo.age_bracket] += 1
        if demo.location_state:
            comment_loc_counts[demo.location_state] += 1

        if demo.gender != "unknown":
            comm_known_gender += 1
        if demo.age_bracket != "unknown":
            comm_known_age += 1
        if demo.location_state:
            comm_known_loc += 1

        side_idx = c.get("side")
        if isinstance(side_idx, int) and 0 <= side_idx < len(sides):
            side_demographics[side_idx]["gender"][demo.gender] += 1
            side_demographics[side_idx]["age"][demo.age_bracket] += 1

    total_comments = len(comments_pool) or 1

    # Format side breakdowns
    formatted_sides = []
    for sd in side_demographics:
        g_tot = sum(sd["gender"].values()) or 1
        a_tot = sum(sd["age"].values()) or 1
        formatted_sides.append({
            "position": sd["position"],
            "gender_split": {k: round(v / g_tot, 2) for k, v in sd["gender"].items() if k != "unknown"},
            "top_age": max(sd["age"].items(), key=lambda x: x[1])[0] if sd["age"] else "unknown",
        })

    # Blended audience profile for this topic
    blended_gender = Counter()
    blended_age = Counter()
    blended_loc = Counter()

    for k, v in author_gender_counts.items():
        blended_gender[k] += v
    for k, v in comment_gender_counts.items():
        blended_gender[k] += v

    for k, v in author_age_counts.items():
        blended_age[k] += v
    for k, v in comment_age_counts.items():
        blended_age[k] += v

    for k, v in author_loc_counts.items():
        blended_loc[k] += v
    for k, v in comment_loc_counts.items():
        blended_loc[k] += v

    total_audience = (len(authors_seen) + len(comments_pool)) or 1
    total_known_gender = author_known_gender + comm_known_gender
    total_known_age = author_known_age + comm_known_age
    total_known_loc = author_known_loc + comm_known_loc

    coverage = {
        "accounts_in_scope": total_audience,
        "authors_in_scope": len(authors_seen),
        "comments_in_scope": len(comments_pool),
        "gender_resolved_pct": round((total_known_gender / total_audience) * 100, 1),
        "age_resolved_pct": round((total_known_age / total_audience) * 100, 1),
        "location_resolved_pct": round((total_known_loc / total_audience) * 100, 1),
    }

    # Significant demographic signals
    skews = []
    gen_known = sum(v for k, v in blended_gender.items() if k in ("male", "female"))
    if gen_known >= 5:
        m_pct = blended_gender.get("male", 0) / gen_known
        f_pct = blended_gender.get("female", 0) / gen_known
        if f_pct >= 0.55:
            skews.append(f"Female-majority participation ({round(f_pct*100)}% of resolved gender)")
        elif m_pct >= 0.75:
            skews.append(f"Heavily male-dominated ({round(m_pct*100)}% of resolved gender)")

    age_known = sum(v for k, v in blended_age.items() if k != "unknown")
    if age_known >= 5:
        youth_pct = (blended_age.get("18-24", 0) + blended_age.get("13-17", 0)) / age_known
        if youth_pct >= 0.60:
            skews.append(f"Youth/Gen-Z concentrated ({round(youth_pct*100)}% of resolved age in 13-24)")
        elif blended_age.get("25-34", 0) / age_known >= 0.50:
            skews.append(f"Working professional skew ({round((blended_age['25-34']/age_known)*100)}% in 25-34)")

    top_regs = [k for k, _ in blended_loc.most_common(4)]
    if top_regs:
        skews.append(f"Key regional focus: {', '.join(top_regs)}")

    # 3. Psychological Archetypes / Mental State Distribution
    mental_counts = Counter()
    _, _, state_m = _get_models()
    if state_m is not None:
        texts_to_score = [c.get("text", "") for c in comments_pool if len(c.get("text", "").strip()) > 10]
        if texts_to_score:
            try:
                preds = state_m.predict(texts_to_score[:100])
                for p in preds:
                    mental_counts[str(p)] += 1
            except Exception:
                pass

    total_mental = sum(mental_counts.values()) or 1
    mental_distribution = {k: round(v / total_mental, 2) for k, v in mental_counts.items()}

    return {
        "coverage": coverage,
        "gender_distribution": {k: round(v / total_audience, 2) for k, v in blended_gender.items()},
        "age_distribution": {k: round(v / total_audience, 2) for k, v in blended_age.items()},
        "top_regions": top_regs,
        "side_breakdowns": formatted_sides,
        "mental_state_distribution": mental_distribution,
        "signals": skews,
    }

"""Aggregate demographic inference from public profile text, handles, captions, and comments.

Provides:
  * Deterministic Indian given name & honorific lexicons for gender (male, female, org, unknown).
  * Syntactic, linguistic, and regex rules for age bracket (13-17, 18-24, 25-34, 35-54, 55+, unknown).
  * State and metro location mapping.
  * ML classifiers for age, gender, mental state, impatience, irrationality, party support.
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
from functools import lru_cache

import joblib
import numpy as np
import torch

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- LOAD TRAINED CLASSIFIERS ----
MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models"
BERT_PARTY_DIR = MODELS_DIR / "bert-party-support-17" if (MODELS_DIR / "bert-party-support-17").exists() else MODELS_DIR / "bert-party-support"
NEURAL_DEMO_DIR = MODELS_DIR / "neural_contrastive_demographics"

_AGE_MODEL = None
_GENDER_MODEL = None
_MENTAL_STATE_MODEL = None
_IMPATIENCE_MODEL = None
_IRRATIONALITY_MODEL = None
_PARTY_MODEL = None
_BERT_PARTY_TOKENIZER = None
_BERT_PARTY_MODEL = None
_NEURAL_DEMO_TOKENIZER = None
_NEURAL_DEMO_MODEL = None
_NEURAL_DEMO_CONFIG = None

def _get_neural_demographics_model():
    """Lazy loader for multi-task contrastive neural demographic model."""
    global _NEURAL_DEMO_TOKENIZER, _NEURAL_DEMO_MODEL, _NEURAL_DEMO_CONFIG
    if _NEURAL_DEMO_MODEL is None and NEURAL_DEMO_DIR.exists() and (NEURAL_DEMO_DIR / "demographic_heads.pt").exists():
        try:
            from transformers import AutoTokenizer
            import sys
            repo_root = str(Path(__file__).resolve().parents[2])
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from train_contrastive_demographics_vm import MultiTaskContrastiveDemographics, MENTAL_CLASSES, AGE_CLASSES

            _NEURAL_DEMO_TOKENIZER = AutoTokenizer.from_pretrained(NEURAL_DEMO_DIR)
            model = MultiTaskContrastiveDemographics(
                str(NEURAL_DEMO_DIR),
                num_mental=len(MENTAL_CLASSES),
                num_age=len(AGE_CLASSES)
            )
            ckpt = torch.load(NEURAL_DEMO_DIR / "demographic_heads.pt", map_location="cpu")
            model.mental_head.load_state_dict(ckpt["mental_head"])
            model.age_head.load_state_dict(ckpt["age_head"])
            model.eval()
            try:
                model.to(_DEVICE)
            except Exception:
                model.to("cpu")
            _NEURAL_DEMO_MODEL = model
            _NEURAL_DEMO_CONFIG = ckpt.get("config", {})
        except Exception:
            _NEURAL_DEMO_MODEL = False
    return _NEURAL_DEMO_TOKENIZER, (_NEURAL_DEMO_MODEL if _NEURAL_DEMO_MODEL is not False else None), _NEURAL_DEMO_CONFIG

_NEURAL_DEMO_CACHE: dict[str, tuple] = {}


def prewarm_neural_demographics(texts: list[str], batch_size: int = 32) -> None:
    """Run the contrastive demographic model over unseen texts in padded batches and cache (mental, age) softmax vectors."""
    tok, mod, _ = _get_neural_demographics_model()
    if tok is None or mod is None:
        return
    todo = list(dict.fromkeys(t for t in texts if t and t.strip() and t not in _NEURAL_DEMO_CACHE))
    if not todo:
        return
    device = next(mod.parameters()).device
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            m_logits, a_logits, _, _ = mod(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            m_probs = torch.softmax(m_logits, dim=-1).cpu().numpy()
            a_probs = torch.softmax(a_logits, dim=-1).cpu().numpy()
        for text, m_p, a_p in zip(chunk, m_probs, a_probs):
            _NEURAL_DEMO_CACHE[text] = (m_p, a_p)
    if device.type == "cuda":
        mod.to("cpu")
        torch.cuda.empty_cache()


def _neural_demo_probs(text: str):
    """(mental, age) softmax vectors for one text, served from the batch cache."""
    if not text.strip():
        return None
    if text not in _NEURAL_DEMO_CACHE:
        prewarm_neural_demographics([text])
    return _NEURAL_DEMO_CACHE.get(text)


def predict_demographics_neural(text: str) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
    """Predict mental state and age bracket using fine-tuned contrastive DeBERTa model."""
    try:
        from train_contrastive_demographics_vm import ID2MENTAL, ID2AGE
        probs = _neural_demo_probs(text)
        if probs is None:
            return None, None
        m_probs, a_probs = probs
        m_idx = int(np.argmax(m_probs))
        a_idx = int(np.argmax(a_probs))
        mental_res = (ID2MENTAL[m_idx], round(float(m_probs[m_idx]), 3))
        age_res = (ID2AGE[a_idx], round(float(a_probs[a_idx]), 3))
        return mental_res, age_res
    except Exception:
        return None, None
    except Exception:
        return None, None

def _get_bert_party_model():
    """Lazy loader for fine-tuned mDeBERTa-v3 neural party support classifier."""
    global _BERT_PARTY_TOKENIZER, _BERT_PARTY_MODEL
    if _BERT_PARTY_MODEL is None and BERT_PARTY_DIR.exists():
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            _BERT_PARTY_TOKENIZER = AutoTokenizer.from_pretrained(BERT_PARTY_DIR)
            _BERT_PARTY_MODEL = AutoModelForSequenceClassification.from_pretrained(BERT_PARTY_DIR)
            _BERT_PARTY_MODEL.eval()
            try:
                _BERT_PARTY_MODEL.to(_DEVICE)
            except Exception:
                _BERT_PARTY_MODEL.to("cpu")
        except Exception:
            _BERT_PARTY_MODEL = False
    return _BERT_PARTY_TOKENIZER, (_BERT_PARTY_MODEL if _BERT_PARTY_MODEL is not False else None)

_NEURAL_PARTY_CACHE: dict[str, tuple[str, float] | None] = {}


def _party_idx2label(mod) -> list[str]:
    if hasattr(mod.config, "id2label") and mod.config.id2label:
        return [mod.config.id2label[i] for i in range(len(mod.config.id2label))]
    return ["BJP", "Congress", "AAP", "TMC", "SP", "DMK", "None"]


def prewarm_party_neural(texts: list[str], batch_size: int = 32) -> None:
    """Run the party model over unseen texts in padded batches and cache (label, confidence) results."""
    tok, mod = _get_bert_party_model()
    if tok is None or mod is None:
        return
    todo = list(dict.fromkeys(t for t in texts if t and t.strip() and t not in _NEURAL_PARTY_CACHE))
    if not todo:
        return
    idx2party = _party_idx2label(mod)
    device = next(mod.parameters()).device
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(mod(**enc).logits, dim=-1)
            conf, idx = probs.max(dim=-1)
        for text, i, c in zip(chunk, idx.tolist(), conf.tolist()):
            _NEURAL_PARTY_CACHE[text] = (idx2party[i] if i < len(idx2party) else "None", float(c))
    if device.type == "cuda":
        mod.to("cpu")
        torch.cuda.empty_cache()


def predict_party_neural(text: str) -> tuple[str, float] | None:
    """Predict political party leaning using fine-tuned neural transformer (served from the batch cache)."""
    if not text.strip():
        return None
    if text not in _NEURAL_PARTY_CACHE:
        try:
            prewarm_party_neural([text])
        except Exception:
            return None
    return _NEURAL_PARTY_CACHE.get(text)

def _get_models():
    global _AGE_MODEL, _GENDER_MODEL, _MENTAL_STATE_MODEL, _IMPATIENCE_MODEL, _IRRATIONALITY_MODEL, _PARTY_MODEL
    if _AGE_MODEL is None and (MODELS_DIR / "age_classifier.joblib").exists():
        try:
            _AGE_MODEL = joblib.load(MODELS_DIR / "age_classifier.joblib")
            _GENDER_MODEL = joblib.load(MODELS_DIR / "gender_classifier.joblib")
            _MENTAL_STATE_MODEL = joblib.load(MODELS_DIR / "mental_state_classifier.joblib")
            _IMPATIENCE_MODEL = joblib.load(MODELS_DIR / "impatience_classifier.joblib")
            _IRRATIONALITY_MODEL = joblib.load(MODELS_DIR / "irrationality_classifier.joblib")
            _PARTY_MODEL = joblib.load(MODELS_DIR / "party_support_classifier.joblib")
        except Exception:
            pass
    return _AGE_MODEL, _GENDER_MODEL, _MENTAL_STATE_MODEL, _IMPATIENCE_MODEL, _IRRATIONALITY_MODEL, _PARTY_MODEL

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
    mental_state: str = "unknown"
    mental_state_confidence: float = 0.0
    impatience: str = "neutral"
    impatience_confidence: float = 0.0
    irrationality: str = "neutral"
    irrationality_confidence: float = 0.0
    party_support: str = "None"
    party_confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


def _ml_predict(text, model, classes):
    if model is None or len(text.strip()) < 15:
        return None, 0.0
    try:
        probs = model.predict_proba([text])[0]
        max_idx = probs.argmax()
        conf = float(probs[max_idx])
        cls = str(model.classes_[max_idx])
        if conf >= 0.35:
            return cls, round(conf, 2)
    except Exception:
        pass
    return None, 0.0


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

    words = set(re.findall(r"\b[a-z0-9\+\.\-]+\b", low))

    for cues, bracket, conf in AGE_CUES:
        for c in cues:
            if " " in c:
                if c in low:
                    return bracket, conf, "bio-cue"
            else:
                if c in words:
                    return bracket, conf, "bio-cue"

    # ML Classifier fallback
    age_m, _, _, _, _, _ = _get_models()
    if age_m is not None and len(text.strip()) > 15:
        cls, conf = _ml_predict(text, age_m, AGE_BRACKETS)
        if cls:
            return cls, conf, "ml-classifier"

    return "unknown", 0.0, None


def infer_location(text: str) -> tuple[str | None, str | None, float]:
    """Infers Indian state and city if mentioned in text or through Indic regional scripts."""
    if re.search(r"[\u0d00-\u0d7f]", text):
        return "Kerala", None, 0.85
    if re.search(r"[\u0b80-\u0bff]", text):
        return "Tamil Nadu", None, 0.85
    if re.search(r"[\u0c00-\u0c7f]", text):
        return "Andhra Pradesh / Telangana", None, 0.85
    if re.search(r"[\u0c80-\u0cff]", text):
        return "Karnataka", None, 0.85
    if re.search(r"[\u0980-\u09ff]", text):
        return "West Bengal", None, 0.85
    if re.search(r"[\u0a80-\u0aff]", text):
        return "Gujarat", None, 0.85
    if re.search(r"[\u0a00-\u0a7f]", text):
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
    """Infers demographic and behavioral profile for an individual comment."""
    raw_user = comment.get("user") or {}
    username = raw_user.get("username") or comment.get("username") or ""
    text = comment.get("text") or comment.get("body") or ""

    if username:
        d = infer_profile(username, bio=text[:200])
    else:
        d = ProfileDemographics()

    low = text.lower()

    # Gender conversational cues if username didn't resolve
    if d.gender == "unknown":
        if re.search(r"\b(as a woman|as a girl|female here|ladki|bhabhi)\b", low):
            d.gender, d.gender_confidence, _ = "female", 0.65, "self-ident"
        elif re.search(r"\b(as a man|as a guy|ladka|bhai|bhaiya)\b", low):
            d.gender, d.gender_confidence, _ = "male", 0.65, "self-ident"
        elif any(w in low for w in FEMALE_CONVERSATIONAL):
            d.gender, d.gender_confidence, _ = "female", 0.45, "conversational-cue"
        elif any(w in low for w in MALE_CONVERSATIONAL):
            d.gender, d.gender_confidence, _ = "male", 0.45, "conversational-cue"

    # Infer age from heuristics first
    if d.age_bracket == "unknown":
        a, a_conf, a_ev = infer_age(text)
        if a != "unknown":
            d.age_bracket, d.age_confidence = a, a_conf

    # Get ML models
    age_m, gen_m, state_m, imp_m, irr_m, party_m = _get_models()

    # Neural multi-task contrastive demographic model with calibrated ensembling
    neural_mental, neural_age = predict_demographics_neural(text)
    if neural_mental:
        # If linear mental state classifier exists, evaluate calibrated ensemble (0.65 Neural + 0.35 Linear)
        if state_m is not None and hasattr(state_m, "predict_proba"):
            try:
                neural_probs = _neural_demo_probs(text)[0]
                from train_contrastive_demographics_vm import MENTAL_CLASSES, ID2MENTAL
                lin_probs = state_m.predict_proba([text])[0]
                lin_classes = list(state_m.classes_)
                lin_reordered = np.array([lin_probs[lin_classes.index(c)] if c in lin_classes else 0.0 for c in MENTAL_CLASSES])
                ens_m = 0.65 * neural_probs + 0.35 * lin_reordered
                best_idx = int(np.argmax(ens_m))
                d.mental_state = ID2MENTAL[best_idx]
                d.mental_state_confidence = round(float(ens_m[best_idx]), 3)
            except Exception:
                d.mental_state, d.mental_state_confidence = neural_mental
        else:
            d.mental_state, d.mental_state_confidence = neural_mental

    # Age bracket prediction: Combine lifecycle priors + ML classifier + neural representation
    if d.age_bracket == "unknown":
        if age_m is not None and hasattr(age_m, "predict_proba"):
            try:
                import numpy as np
                from train_contrastive_demographics_vm import AGE_CLASSES, ID2AGE
                lin_probs = age_m.predict_proba([text])[0]
                lin_classes = list(age_m.classes_)
                lin_reordered = np.array([lin_probs[lin_classes.index(c)] if c in lin_classes else 0.0 for c in AGE_CLASSES])

                # Domain lifecycle prior weights
                prior = np.zeros(len(AGE_CLASSES))
                if any(k in low for k in ['retire', 'pension', 'budhape', 'budhapa', '500 years', 'since 200', 'since 19', '30 saal', '35 saal', 'planning commission', 'in my youth', 'elders', 'fixed deposit interest', 'grandchild', 'pota', 'poti']):
                    prior[AGE_CLASSES.index('55+')] += 0.35
                if any(k in low for k in ['cbse', 'board exam', 'term 1', 'term 2', '10th', '12th', 'science exam', 'maths exam', 'school', 'mummy', 'homework', 'syllabus', 'exam hai']):
                    prior[AGE_CLASSES.index('13-17')] += 0.40
                if any(k in low for k in ['college', 'semester', 'placement', 'fresher', 'campus', 'hostel', 'bro,', 'bro ', 'cutoff', 'agniveer', 'nta', 'neet', 'jee', 'b.tech', 'undergrad']):
                    prior[AGE_CLASSES.index('18-24')] += 0.25
                if any(k in low for k in ['school fees', 'daughter', 'son', 'home loan', 'c-section', 'income tax', 'cpi', 'rbi', 'gsdp', 'fiscal deficit', 'cabinet', 'parliament']):
                    prior[AGE_CLASSES.index('35-54')] += 0.20

                if neural_age:
                    neural_a_probs = _neural_demo_probs(text)[1]
                    ens_a = 0.65 * lin_reordered + 0.25 * neural_a_probs + 0.35 * prior
                else:
                    ens_a = lin_reordered + 0.35 * prior

                best_a_idx = int(np.argmax(ens_a))
                d.age_bracket = ID2AGE[best_a_idx]
                d.age_confidence = round(float(min(ens_a[best_a_idx], 0.95)), 3)
            except Exception:
                if neural_age:
                    d.age_bracket, d.age_confidence = neural_age
        elif neural_age:
            d.age_bracket, d.age_confidence = neural_age

    # ML Classifier fallbacks for all behavioral traits
    combined_text = f"{username} {text}"

    if d.gender == "unknown" and gen_m is not None:
        cls, conf = _ml_predict(combined_text, gen_m, ["male", "female", "neutral"])
        if cls:
            d.gender, d.gender_confidence = cls, conf

    if d.age_bracket == "unknown" and age_m is not None:
        cls, conf = _ml_predict(combined_text, age_m, AGE_BRACKETS)
        if cls:
            d.age_bracket, d.age_confidence = cls, conf

    if d.mental_state == "unknown" and state_m is not None:
        cls, conf = _ml_predict(text, state_m, sorted(state_m.classes_))
        if cls:
            d.mental_state, d.mental_state_confidence = cls, conf

    if d.impatience == "neutral" and imp_m is not None:
        cls, conf = _ml_predict(text, imp_m, sorted(imp_m.classes_))
        if cls:
            d.impatience, d.impatience_confidence = cls, conf

    if d.irrationality == "neutral" and irr_m is not None:
        cls, conf = _ml_predict(text, irr_m, sorted(irr_m.classes_))
        if cls:
            d.irrationality, d.irrationality_confidence = cls, conf

    if d.party_support == "None":
        neural_res = predict_party_neural(text)
        if neural_res and neural_res[0] != "None":
            d.party_support, d.party_confidence = neural_res
        elif party_m is not None:
            cls, conf = _ml_predict(text, party_m, sorted(party_m.classes_))
            if cls:
                d.party_support, d.party_confidence = cls, conf

    # Heuristic impatience/irrationality
    if d.impatience == "neutral" and d.mental_state == "Civic Anger & Institutional Skepticism":
        d.impatience = "impatient"
        d.impatience_confidence = 0.45
    if d.irrationality == "neutral" and "conspiracy" in low:
        d.irrationality = "irrational"
        d.irrationality_confidence = 0.50

    return d


def compute_demographic_breakdowns(topic: dict) -> dict:
    """Computes demographic cross-tabs across topic post creators and commenters."""
    brief = topic.get("brief") or {}
    measured = brief.get("measured_opinion") or {}
    media = topic.get("instagram", {}).get("media", [])
    on_topic = set(brief.get("on_topic_codes", []))

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

    comments_pool = [c for m in media if not on_topic or m.get("code") in on_topic
                     for c in m.get("comments", []) if c.get("text")]
    if not comments_pool and brief.get("comment_pool"):
        comments_pool = [c for c in brief.get("comment_pool", []) if c.get("text")]

    sides = measured.get("sides") or []
    comment_gender_counts = Counter()
    comment_age_counts = Counter()
    comment_loc_counts = Counter()
    mental_state_counts = Counter()
    impatience_counts = Counter()
    irrationality_counts = Counter()
    party_counts = Counter()

    side_demographics = [{"side": i, "position": s.get("position", ""),
                          "gender": Counter(), "age": Counter(),
                          "impatience": Counter(), "irrationality": Counter(), "party": Counter()}
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
        mental_state_counts[demo.mental_state] += 1
        impatience_counts[demo.impatience] += 1
        irrationality_counts[demo.irrationality] += 1
        party_counts[demo.party_support] += 1

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
            side_demographics[side_idx]["impatience"][demo.impatience] += 1
            side_demographics[side_idx]["irrationality"][demo.irrationality] += 1
            side_demographics[side_idx]["party"][demo.party_support] += 1

    total_comments = len(comments_pool) or 1

    formatted_sides = []
    for sd in side_demographics:
        g_tot = sum(sd["gender"].values()) or 1
        a_tot = sum(sd["age"].values()) or 1
        formatted_sides.append({
            "position": sd["position"],
            "gender_split": {k: round(v / g_tot, 2) for k, v in sd["gender"].items() if k != "unknown"},
            "top_age": max(sd["age"].items(), key=lambda x: x[1])[0] if sd["age"] else "unknown",
            "impatience_share": {k: round(v / sum(sd["impatience"].values()), 2) for k, v in sd["impatience"].items()},
            "irrationality_share": {k: round(v / sum(sd["irrationality"].values()), 2) for k, v in sd["irrationality"].items()},
            "party_support": {k: round(v / sum(sd["party"].values()), 2) for k, v in sd["party"].items()},
        })

    blended_gender = Counter()
    blended_age = Counter()
    blended_loc = Counter()
    blended_imp = Counter()
    blended_irr = Counter()
    blended_party = Counter()

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

    for k, v in impatience_counts.items():
        blended_imp[k] += v
    for k, v in irrationality_counts.items():
        blended_irr[k] += v
    for k, v in party_counts.items():
        blended_party[k] += v

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

    imp_total = sum(blended_imp.values()) or 1
    imp_dist = {k: round(v / imp_total, 2) for k, v in blended_imp.items()}
    if imp_dist.get("impatient", 0) >= 0.55:
        skews.append("High impatience in audience discourse")

    irr_total = sum(blended_irr.values()) or 1
    irr_dist = {k: round(v / irr_total, 2) for k, v in blended_irr.items()}
    if irr_dist.get("irrational", 0) >= 0.30:
        skews.append("Elevated irrationality markers detected")

    party_total = sum(blended_party.values()) or 1
    party_dist = {k: round(v / party_total, 2) for k, v in blended_party.items()}
    top_parties = [k for k, v in party_dist.items() if v >= 0.10 and k != "None"]
    if top_parties:
        skews.append(f"Political leanings: {', '.join(top_parties)}")

    return {
        "coverage": coverage,
        "gender_distribution": {k: round(v / total_audience, 2) for k, v in blended_gender.items()},
        "age_distribution": {k: round(v / total_audience, 2) for k, v in blended_age.items()},
        "top_regions": top_regs,
        "side_breakdowns": formatted_sides,
        "mental_state_distribution": {k: round(v / total_comments, 2) for k, v in mental_state_counts.items()},
        "impatience_distribution": imp_dist,
        "irrationality_distribution": irr_dist,
        "party_support_distribution": party_dist,
        "signals": skews,
    }

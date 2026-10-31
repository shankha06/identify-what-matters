"""Model-aware political/civic text classification.

Single entry point - `classify(text)` - for every stage of the data-requirements doc's pipeline
(political/non-political -> issue cluster -> sentiment -> target/direction -> narrative type) on ONE
piece of text (a post caption or a comment). For each task it:

  1. Uses the trained classifier in artifacts/political_models/<task>_classifier.joblib if present
     (produced by `political_corpus_trainer.py` as a CPU baseline, or later by a VM fine-tune - both
     write to the same path, so dropping in a better model needs no code change here).
  2. Falls back to political_pulse.py's keyword/regex heuristics otherwise.

Every result reports its own `source` ("model" or "keyword") and a `confidence`, so callers can do real
signal fusion (see undercurrent_scoring.py) instead of treating a keyword guess and a calibrated model
probability as the same kind of number. This mirrors demographics.py's lexicon-then-ML pattern applied
to the political pipeline, and replaces the keyword-only `tag_issue_cluster()` that used to live directly
in make_full_dashboard.py.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from identify_what_matters import onnx_backend
from identify_what_matters.political_pulse import political_pulse_engine

MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "political_models"
MIN_TEXT_LEN = 8

TASKS = ("is_political", "issue_cluster", "sentiment", "target", "direction", "narrative_type")

# The data-requirements doc's issue-cluster taxonomy. Keyword lists follow its own "Example signals"
# column; this is the single source of truth other modules should import (instead of redefining it).
ISSUE_CLUSTER_KEYWORDS = {
    "employment": ("job", "jobs", "unemployment", "recruitment", "exam", "vacancy", "hiring", "naukri"),
    "inflation": ("inflation", "price", "prices", "lpg", "petrol", "diesel", "food price", "mehangai"),
    "economy": ("gdp", "economy", "growth", "business", "msme", "sensex", "nifty", "market"),
    "farmers": ("msp", "farmer", "farmers", "agriculture", "procurement", "kisan", "crop"),
    "education": ("neet", "upsc", "ssc", "paper leak", "exam", "board exam", "admission", "school", "college", "university", "campus", "student", "students"),
    "defence": ("military", "agniveer", "china", "pakistan", "army", "border", "defence", "loc", "air force", "iaf"),
    "corruption": ("scam", "cbi", "corruption", "bribery", "ghotala", "bhrashtachar", "money laundering"),
    "communal": ("hindu", "muslim", "temple", "temples", "mosque", "mandir", "masjid", "communal", "riot", "rss", "vande mataram", "sanatan", "kashi"),
    "women_safety": ("women safety", "rape", "harassment", "women employment", "mahila", "chappal protest"),
    "caste": ("caste", "reservation", "caste census", "obc", "dalit", "aarakshan", "tribal", "adivasi"),
    "infrastructure": ("road", "roads", "railway", "airport", "metro", "highway", "infrastructure", "building collapse", "collapsed", "collapse", "bridge"),
    "welfare": ("scheme", "subsidy", "dbt", "yojana", "welfare", "pension", "ration", "seva"),
    "governance": ("bureaucracy", "delivery", "governance", "administration", "collector", "sdm", "public service", "policy"),
    "elections": ("vote", "voter", "candidate", "alliance", "seat", "election", "chunav", "evm", "booth", "electoral roll"),
}

# Regex word-boundary patterns for high-risk ambiguous political acronyms
SPECIAL_CLUSTER_PATTERNS = {
    "corruption": re.compile(r"\b(ed|cbi|lokayukta)\b", re.I),
    "defence": re.compile(r"\b(iaf|drdo|cds)\b", re.I),
}

IS_POLITICAL_RE = re.compile(
    r"\b(?:modi|bjp|congress|election|electoral|minister|government|parliament|assembly|policy|protest|"
    r"kejriwal|yogi|dmk|tvk|rahul gandhi|mamata|stalin|aap|tmc|cpim|rjd|jdu|bsp|ncp|shiv ?sena)\b", re.I)

# Entity recognizer patterns for high-precision entity target routing
ENTITY_TARGET_PATTERNS = {
    "BJP": re.compile(r"\b(bjp|modi|narendra modi|amit shah|yogi|nda|bhartiya janata)\b", re.I),
    "Congress": re.compile(r"\b(congress|rahul gandhi|kharge|priyanka gandhi|inc)\b", re.I),
    "AAP": re.compile(r"\b(aap|kejriwal|aam aadmi|bhagwant mann)\b", re.I),
    "TMC": re.compile(r"\b(tmc|mamata|trinamool)\b", re.I),
    "DMK": re.compile(r"\b(dmk|stalin|udayanidhi)\b", re.I),
    "SP": re.compile(r"\b(samajwadi|akhilesh)\b", re.I),
    "RJD": re.compile(r"\b(rjd|tejashwi|lalu)\b", re.I),
    "CPIM": re.compile(r"\b(cpim|cpi|left front|pinarayi)\b", re.I),
    "Shiv Sena": re.compile(r"\b(shiv sena|shinde|uddhav|thackeray)\b", re.I),
    "NCP": re.compile(r"\b(ncp|sharad pawar|ajit pawar)\b", re.I),
    "Election Commission": re.compile(r"\b(election commission|eci|cec|gyanesh kumar)\b", re.I),
    "Judiciary": re.compile(r"\b(supreme court|high court|cji|chief justice|judiciary|bench)\b", re.I),
    "Police": re.compile(r"\b(police|cops|dsp|sho|constable|dgp)\b", re.I),
    "Government": re.compile(r"\b(government|centre|administration|ministry|cabinet)\b", re.I),
    "Opposition": re.compile(r"\b(opposition|india alliance|allies|indi alliance)\b", re.I),
}

# Domain-specific narrative markers for calibrated narrative type boost
NARRATIVE_PATTERNS = {
    "satire_meme": re.compile(r"\b(meme|lol|pappu|jumla|fek|feku|circus|joke|clown|funny|hilarious)\b|[🤡😂🤣💀☠️]", re.I),
    "fact_check": re.compile(r"\b(fact check|fake news|debunk|misleading|claim check|verified|alt news|boomlive)\b", re.I),
    "mobilization_call": re.compile(r"\b(protest|march|boycott|call to action|join us|rally|strike|chalo|dharna|andolan)\b", re.I),
    "corruption_allegation": re.compile(r"\b(scam|ghotala|bhrashtachar|bribery|kickback|ed raid|cbi raid|money laundering|hawala)\b", re.I),
    "communal_framing": re.compile(r"\b(hindu|muslim|mandir|masjid|love jihad|waqf|appeasement|dharma|communal)\b", re.I),
    "government_achievement": re.compile(r"\b(inaugurate|record high|milestone|world class|vande bharat|achievement|gdp growth|historic feat)\b", re.I),
    "governance_criticism": re.compile(r"\b(failure|mismanagement|crisis|collapse|poor administration|corruption|incompetence|price rise)\b", re.I),
}

_LEANING_TO_DIRECTION = {
    "Favoring BJP / NDA": "pro_incumbent",
    "Favoring Opposition / INDIA": "anti_incumbent",
    "Balanced / Contested Debate": "neutral",
    "Neutral / Non-partisan": "neutral",
}
_LEANING_TO_TARGET = {
    "Favoring BJP / NDA": "Government",
    "Favoring Opposition / INDIA": "Opposition",
    "Balanced / Contested Debate": "Government",
    "Neutral / Non-partisan": "None",
}


def tag_issue_cluster(text: str) -> str:
    """Keyword-only; kept as the fallback path for `classify()`'s "issue_cluster" task."""
    low = text.lower()
    best, best_hits = "none", 0
    for cluster, keywords in ISSUE_CLUSTER_KEYWORDS.items():
        hits = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", low))
        if hits > best_hits:
            best, best_hits = cluster, hits

    for cluster, pat in SPECIAL_CLUSTER_PATTERNS.items():
        if pat.search(text):
            if best_hits == 0:
                best, best_hits = cluster, 1

    return best


@lru_cache(maxsize=None)
def _load_transformer(task: str):
    """Loads a fine-tuned Hugging Face transformer model if available."""
    task_dir = MODELS_DIR / f"mdeberta-{task}"
    if not (task_dir / "config.json").exists():
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from identify_what_matters.utils import get_torch_device
        device = get_torch_device()
        tok = AutoTokenizer.from_pretrained(str(task_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(task_dir)).to(device).eval()
        run_file = task_dir / "training_run.json"
        classes = json.loads(run_file.read_text()).get("classes") if run_file.exists() else None
        return tok, model, device, classes
    except Exception:
        return None


def _task_classes(task_dir: Path) -> list | None:
    run_file = task_dir / "training_run.json"
    return json.loads(run_file.read_text()).get("classes") if run_file.exists() else None


def _transformer_available(task: str) -> bool:
    """True when an ONNX or PyTorch transformer exists for `task`; does not load PyTorch when ONNX is present."""
    task_dir = MODELS_DIR / f"mdeberta-{task}"
    if onnx_backend.onnx_path(task_dir) is not None:
        return True
    return _load_transformer(task) is not None


_TRANSFORMER_CACHE: dict[tuple[str, str], tuple] = {}


def prewarm_political_transformers(texts: list[str], tasks: tuple = ("is_political", "direction"), batch_size: int = 32) -> None:
    """Batch-evaluates texts through the fine-tuned transformers (using ONNX if available, else PyTorch) and populates _TRANSFORMER_CACHE."""
    for task in tasks:
        task_dir = MODELS_DIR / f"mdeberta-{task}"
        todo = list(dict.fromkeys(t for t in texts if t and (task, t) not in _TRANSFORMER_CACHE))
        if not todo:
            continue
        if onnx_backend.onnx_path(task_dir) is not None:
            try:
                probs = onnx_backend.seq_cls_probs(task_dir, todo, max_length=160, batch_size=batch_size)
                classes = _task_classes(task_dir)
                for text_item, prob_row in zip(todo, probs):
                    _TRANSFORMER_CACHE[(task, text_item)] = (prob_row, classes)
                continue
            except Exception:
                pass

        bundle = _load_transformer(task)
        if bundle is None:
            continue
        import torch
        tok, tf_model, device, classes = bundle
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            inputs = {k: v.to(device) for k, v in tok(chunk, truncation=True, max_length=160, padding=True, return_tensors="pt").items()}
            with torch.no_grad():
                logits = tf_model(**inputs).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for text_item, prob_row in zip(chunk, probs):
                _TRANSFORMER_CACHE[(task, text_item)] = (prob_row, classes)


def _transformer_probs(task: str, text: str):
    """(class probabilities, class names) from the ONNX model if exported, else PyTorch; None if unavailable."""
    if (task, text) in _TRANSFORMER_CACHE:
        return _TRANSFORMER_CACHE[(task, text)]
    task_dir = MODELS_DIR / f"mdeberta-{task}"
    if onnx_backend.onnx_path(task_dir) is not None:
        probs = onnx_backend.seq_cls_probs(task_dir, [text], max_length=160)[0]
        return probs, _task_classes(task_dir)
    bundle = _load_transformer(task)
    if bundle is None:
        return None
    import torch
    tok, tf_model, device, classes = bundle
    inputs = {k: v.to(device) for k, v in tok([text], truncation=True, max_length=160, padding=True, return_tensors="pt").items()}
    with torch.no_grad():
        logits = tf_model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs, classes


@lru_cache(maxsize=None)
def _load_model(task: str):
    path = MODELS_DIR / f"{task}_classifier.joblib"
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def models_available() -> dict[str, bool]:
    """What's actually loaded right now, for dashboards/diagnostics to report honestly."""
    res = {}
    for task in TASKS:
        res[task] = _transformer_available(task) or (_load_model(task) is not None)
    return res


def _model_predict(task: str, text: str) -> dict[str, Any] | None:
    if len(text.strip()) < MIN_TEXT_LEN:
        return None

    # Task-Specific Hybrid Ensemble Architecture:
    # 1. 'target': Entity-calibrated ensemble combining explicit named entity patterns with calibrated linear features
    if task == "target":
        for ent, pat in ENTITY_TARGET_PATTERNS.items():
            if pat.search(text):
                return {"value": ent, "confidence": 0.85, "source": "entity_calibrated_ensemble"}
        model = _load_model("target")
        if model is not None:
            try:
                probs = model.predict_proba([text])[0]
                idx = probs.argmax()
                return {"value": str(model.classes_[idx]), "confidence": round(float(probs[idx]), 3), "source": "model"}
            except Exception:
                pass

    # 2. 'issue_cluster': Calibrated hybrid boosting keyword domain priors onto linear model probabilities
    if task == "issue_cluster":
        model = _load_model("issue_cluster")
        if model is not None:
            try:
                classes = list(model.classes_)
                probs = model.predict_proba([text])[0].copy()
                kw_cluster = tag_issue_cluster(text)
                if kw_cluster in classes and kw_cluster != "none":
                    probs[classes.index(kw_cluster)] += 0.35
                idx = probs.argmax()
                return {"value": str(classes[idx]), "confidence": round(float(probs[idx] / sum(probs)), 3), "source": "hybrid_ensemble"}
            except Exception:
                pass

    # 3. 'narrative_type': Calibrated hybrid boosting discourse markers onto linear model probabilities
    if task == "narrative_type":
        model = _load_model("narrative_type")
        if model is not None:
            try:
                classes = list(model.classes_)
                probs = model.predict_proba([text])[0].copy()
                for narr, pat in NARRATIVE_PATTERNS.items():
                    if pat.search(text) and narr in classes:
                        probs[classes.index(narr)] += 0.20
                idx = probs.argmax()
                return {"value": str(classes[idx]), "confidence": round(float(probs[idx] / sum(probs)), 3), "source": "hybrid_ensemble"}
            except Exception:
                pass

    # 4. For 'is_political' and 'direction': Use fine-tuned multilingual transformer first
    tf_probs = _transformer_probs(task, text)
    if tf_probs is not None:
        try:
            probs, classes = tf_probs
            
            # Direction calibration: enforce strong lexicon priors with negation awareness
            if task == "direction":
                from identify_what_matters.political_pulse import (
                    PRO_RULING_MARKERS, ANTI_RULING_MARKERS, PRO_OPPOSITION_MARKERS,
                    _PRO_RULING_IMPLICIT, _ANTI_RULING_IMPLICIT, check_phrase_valence
                )
                low = text.lower()
                pro_r_hits, neg_r_hits = check_phrase_valence(text, PRO_RULING_MARKERS)
                pro_o_hits, neg_o_hits = check_phrase_valence(text, PRO_OPPOSITION_MARKERS)
                anti_r_hits = 1 if ANTI_RULING_MARKERS.search(text) else 0

                pro_gov = (pro_r_hits > 0 or any(w in low for w in _PRO_RULING_IMPLICIT)) and neg_r_hits == 0 and anti_r_hits == 0
                anti_gov = anti_r_hits > 0 or neg_r_hits > 0 or any(w in low for w in _ANTI_RULING_IMPLICIT)
                pro_opp = pro_o_hits > 0 and neg_o_hits == 0

                if pro_gov and not anti_gov:
                    if "pro_incumbent" in classes:
                        probs[classes.index("pro_incumbent")] += 0.50
                    if "anti_incumbent" in classes:
                        probs[classes.index("anti_incumbent")] = max(0.0, probs[classes.index("anti_incumbent")] - 0.30)
                elif anti_gov and not pro_gov:
                    if "anti_incumbent" in classes:
                        probs[classes.index("anti_incumbent")] += 0.60
                    if "pro_incumbent" in classes:
                        probs[classes.index("pro_incumbent")] = max(0.0, probs[classes.index("pro_incumbent")] - 0.50)
                elif pro_opp and not pro_gov:
                    if "pro_opposition" in classes:
                        probs[classes.index("pro_opposition")] += 0.50
                    if "pro_incumbent" in classes:
                        probs[classes.index("pro_incumbent")] = max(0.0, probs[classes.index("pro_incumbent")] - 0.30)

            idx = int(probs.argmax())
            val = classes[idx] if classes and idx < len(classes) else str(idx)
            if task == "is_political":
                val = val == "True" or val is True
            return {"value": val, "confidence": round(float(probs[idx] / sum(probs)), 3), "source": "transformer"}
        except Exception:
            pass

    # 5. General fall back to joblib baseline
    model = _load_model(task)
    if model is None:
        return None
    try:
        probs = model.predict_proba([text])[0]
        idx = probs.argmax()
        value = str(model.classes_[idx])
        if task == "is_political":
            value = value == "True"  # trained on str(bool) labels; keep the task's real type, not "False"-is-truthy
        return {"value": value, "confidence": round(float(probs[idx]), 3), "source": "model"}
    except Exception:
        return None


def _keyword_fallback(task: str, text: str, pulse: dict) -> dict[str, Any]:
    if task == "is_political":
        return {"value": bool(IS_POLITICAL_RE.search(text)), "confidence": 0.5, "source": "keyword"}
    if task == "issue_cluster":
        cluster = tag_issue_cluster(text)
        return {"value": cluster, "confidence": 0.4 if cluster != "none" else 0.0, "source": "keyword"}
    if task == "sentiment":
        if pulse["bjp_share"] + pulse["opp_share"] >= 0.5:
            return {"value": "mixed" if abs(pulse["bjp_share"] - pulse["opp_share"]) < 0.1 else "negative",
                   "confidence": 0.3, "source": "keyword"}
        return {"value": "neutral", "confidence": 0.3, "source": "keyword"}
    if task == "target":
        return {"value": _LEANING_TO_TARGET[pulse["leaning"]], "confidence": 0.3, "source": "keyword"}
    if task == "direction":
        return {"value": _LEANING_TO_DIRECTION[pulse["leaning"]], "confidence": 0.3, "source": "keyword"}
    if task == "narrative_type":
        # political_pulse.py has no keyword taxonomy for narrative framing at all - nothing to fall back to.
        return {"value": "other", "confidence": 0.0, "source": "none"}
    raise ValueError(f"unknown task: {task}")


def classify(text: str) -> dict[str, dict[str, Any]]:
    """Classifies one post/comment across every pipeline task. Never raises on bad/short text -
    returns low-confidence fallbacks instead, since this runs over thousands of real comments."""
    text = text or ""
    try:
        pulse = political_pulse_engine.classify_text_favorability(text)
    except Exception:
        pulse = {"leaning": "Neutral / Non-partisan", "bjp_share": 0.0, "opp_share": 0.0, "neutral_share": 1.0}
    result = {}
    for task in TASKS:
        result[task] = _model_predict(task, text) or _keyword_fallback(task, text, pulse)
    return result

"""Dynamic, ML-driven account discovery and tiering - extends config/panel.json beyond a static,
hand-curated list.

config/panel.json today is a fixed {username, group} list a human curates once. This module makes the
account universe grow itself, the same way `hiker_client.py`'s `_proposals()` already discovers new
*hashtags* from captions - generalized to *accounts*:

  1. `record_run_observations` - after every collect run, every post's author across the WHOLE
     collection (panel feeds, trend searches, hashtag seeds, topic probes - not just the panel) is
     logged to an append-only ledger (output/discovery/account_ledger.jsonl): known panel accounts
     with their human-assigned `group` (supervision), and every OTHER author as an unlabelled
     candidate sighting with its own post text, engagement, and a political_classifier read.

  2. `train_tier_classifier` - a TF-IDF + Logistic Regression classifier (same recipe as
     demographics_trainer.py / political_corpus_trainer.py) learns each panel group's own writing
     signature directly from the ledger's real captured captions - not from a hand-written description
     of what a "political_satire" account sounds like.

  3. `score_candidates` - aggregates a candidate's sightings across every run recorded in the ledger
     (one run is weak evidence; sustained political engagement across several is what matters),
     predicts its tier/group with the trained classifier, and ranks candidates into a reviewable
     proposals file - never auto-added to config/panel.json. Same human-in-the-loop policy panel.json
     already states for its own disabled proposed accounts.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from identify_what_matters import political_classifier as pc

LEDGER_PATH = Path("output/discovery/account_ledger.jsonl")
PROPOSALS_PATH = Path("output/discovery/account_candidates.json")
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "political_models" / "account_tier_classifier.joblib"
MIN_SIGHTINGS_FOR_PROPOSAL = 2
MIN_RUNS_FOR_PROPOSAL = 2
MIN_LEDGER_ROWS_TO_TRAIN = 80


def _caption_text(item: dict) -> str:
    caption = item.get("caption")
    if isinstance(caption, dict):
        return caption.get("text", "") or ""
    return str(caption or item.get("caption_text") or "")


def record_run_observations(bundle: dict, panel: dict, run_id: str,
                            output_dir: str | Path = "output/discovery") -> dict:
    """Appends one row per distinct (author, run) to the ledger: labelled for known panel accounts,
    unlabelled candidate sightings for everyone else. Safe to call every run; never raises on
    malformed items (defensive .get() throughout) since this must never break the main collect path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    known = {a["username"]: a["group"] for a in panel.get("accounts", [])}
    now = datetime.now(timezone.utc).isoformat()

    by_author: dict[str, list[dict]] = defaultdict(list)
    for username, items in (bundle.get("feeds") or {}).items():
        for item in items or []:
            by_author[username].append(item)
    for items in list((bundle.get("terms") or {}).values()) + list((bundle.get("seeds") or {}).values()):
        for item in items or []:
            author = item.get("author")
            if author and author not in by_author:
                by_author[author].append(item)
    for result in (bundle.get("probes") or {}).values():
        for item in (result or {}).get("media", []) or []:
            author = item.get("author")
            if author and author not in by_author:
                by_author[author].append(item)

    rows = []
    for author, items in by_author.items():
        captions = [_caption_text(item) for item in items if _caption_text(item).strip()]
        if not captions:
            continue
        engagement = [(item.get("like_count") or 0) + 3 * (item.get("comment_count") or 0) for item in items]
        political_hits = sum(1 for c in captions if pc.classify(c)["is_political"]["value"])
        rows.append({
            "run_id": run_id, "observed_at": now, "username": author,
            "known_group": known.get(author), "verified": bool(items[0].get("author_verified")),
            "post_count": len(items), "captions": captions[:5], "avg_engagement": round(sum(engagement) / len(engagement), 1),
            "political_share": round(political_hits / len(captions), 2),
        })
    if rows:
        with LEDGER_PATH.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"ledger_rows_appended": len(rows), "known_accounts_seen": sum(1 for r in rows if r["known_group"]),
            "candidate_accounts_seen": sum(1 for r in rows if not r["known_group"])}


def _read_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    rows = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # skip truncated lines from interrupted writes
    return rows


def train_tier_classifier() -> dict:
    """Trains the account-tier classifier from the ledger's labelled (known-account) rows. Returns a
    summary dict; raises only if there genuinely isn't enough labelled data yet (honest failure, not
    a silent no-op) so callers/CLI users know to collect more runs first."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rows = [r for r in _read_ledger() if r.get("known_group")]
    if len(rows) < MIN_LEDGER_ROWS_TO_TRAIN:
        raise RuntimeError(
            f"Only {len(rows)} labelled ledger rows (need {MIN_LEDGER_ROWS_TO_TRAIN}+); "
            "run collect a few more times with record_run_observations wired in first."
        )
    texts = [" ".join(r["captions"]) for r in rows]
    labels = [r["known_group"] for r in rows]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20_000)),
        ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)),
    ])
    pipe.fit(texts, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return {"trained_rows": len(texts), "classes": sorted(set(labels))}


def _load_tier_classifier():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def classify_account_tier(captions: list[str]) -> dict:
    """Predicts a candidate account's group from its own caption text. Falls back to a generic
    'unclassified' verdict (not a guess dressed up as a label) when no classifier is trained yet."""
    model = _load_tier_classifier()
    text = " ".join(captions)
    if model is None or not text.strip():
        return {"predicted_group": "unclassified", "confidence": 0.0, "source": "none"}
    probs = model.predict_proba([text])[0]
    idx = probs.argmax()
    return {"predicted_group": str(model.classes_[idx]), "confidence": round(float(probs[idx]), 3), "source": "model"}


def score_candidates(min_sightings: int = MIN_SIGHTINGS_FOR_PROPOSAL,
                     min_runs: int = MIN_RUNS_FOR_PROPOSAL) -> list[dict]:
    """Aggregates candidate (non-panel) sightings across every ledger row, scores and ranks them."""
    rows = [r for r in _read_ledger() if not r.get("known_group")]
    by_account: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_account[row["username"]].append(row)

    candidates = []
    for username, sightings in by_account.items():
        runs = {s["run_id"] for s in sightings}
        if len(sightings) < min_sightings or len(runs) < min_runs:
            continue
        all_captions = [c for s in sightings for c in s["captions"]]
        avg_engagement = sum(s["avg_engagement"] for s in sightings) / len(sightings)
        political_share = sum(s["political_share"] for s in sightings) / len(sightings)
        tier = classify_account_tier(all_captions)
        promotion_score = round(
            (0.4 * min(political_share, 1.0) + 0.3 * min(avg_engagement / 5000, 1.0)
             + 0.2 * min(len(runs) / 5, 1.0) + 0.1 * tier["confidence"]) * 10, 2
        )
        candidates.append({
            "username": username, "sightings": len(sightings), "distinct_runs": len(runs),
            "avg_engagement": round(avg_engagement, 1), "political_share": round(political_share, 2),
            "predicted_group": tier["predicted_group"], "tier_confidence": tier["confidence"],
            "tier_source": tier["source"], "promotion_score": promotion_score,
            "sample_caption": all_captions[0][:160] if all_captions else "",
            "status": "proposal_requires_human_review",
        })
    return sorted(candidates, key=lambda c: -c["promotion_score"])


def write_proposals(output_dir: str | Path = "output/discovery") -> dict:
    output_dir = Path(output_dir)
    candidates = score_candidates()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classifier_trained": MODEL_PATH.exists(),
        "candidates": candidates[:50],
        "policy": "Proposals only - verify handle, public status, India relevance and permissions "
                 "before adding to config/panel.json, exactly as panel.json's own panel_policy requires "
                 "for its disabled proposed accounts.",
    }
    (output_dir).mkdir(parents=True, exist_ok=True)
    (output_dir / "account_candidates.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

"""Longitudinal store and emergence detection for Instagram discussion topics.

Separates three questions the earlier engine conflated:
  * what is being discussed now (support from distinct authors, posts and comments),
  * what is new or accelerating (exposure-normalised rate vs a trailing baseline),
  * what is hot (engagement velocity from repeated snapshots, else an age-normalised estimate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from identify_what_matters.utils import parse_time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
    status TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS posts (post_id TEXT PRIMARY KEY, taken_at TEXT, author_id TEXT,
    caption TEXT, first_observed_at TEXT, domains TEXT);
CREATE TABLE IF NOT EXISTS post_sources (post_id TEXT, run_id TEXT, source_type TEXT, query TEXT,
    domain TEXT, observed_at TEXT, PRIMARY KEY (post_id, run_id, source_type, query));
CREATE TABLE IF NOT EXISTS snapshots (post_id TEXT, observed_at TEXT, like_count INTEGER,
    comment_count INTEGER, view_count INTEGER, PRIMARY KEY (post_id, observed_at));
CREATE TABLE IF NOT EXISTS comments (comment_id TEXT PRIMARY KEY, post_id TEXT, parent_id TEXT,
    author_id TEXT, created_at TEXT, text TEXT, like_count INTEGER, observed_at TEXT);
CREATE TABLE IF NOT EXISTS exposure (run_id TEXT, source_type TEXT, query TEXT, domain TEXT,
    observed_at TEXT, returned_count INTEGER, PRIMARY KEY (run_id, source_type, query));
"""

STOPWORDS = set("""
a an the and or but if of to in on at for from by with about as is are was were be been being am
this that these those it its i me my we our you your he she they them his her their what which who
whom how why when where there here not no yes so too very just can could should would will shall
do does did done have has had having get got go going gone make made more most much many some any
all also only than then now new one two up down out over again into after before like really
yaar bhai hai hain ho hota hoti hote ka ki ke ko se me mein par pe aur ya na nahi kya kyu kyun
toh to bhi hi ab jab tab kab sab ye yeh woh wo vo is us un in kar kare karo karna kiya diya liya
tha thi the raha rahi rahe gaya gayi gaye wala wali wale ek do apna apni apne mera meri mere tera
teri tere hum tum aap log kuch bahut bohot sirf abhi
follow following like likes share comment comments save tag dm link bio subscribe click shop order
buy available now today check page account reel reels post posts video watch drop insta instagram
""".split())
TOKEN = re.compile(r"[#]?[\w\u0900-\u0d7f]+", re.UNICODE)
SCRIPTS = {"hi_or_mr": "\u0900-\u097f", "bn": "\u0980-\u09ff", "pa": "\u0a00-\u0a7f",
           "gu": "\u0a80-\u0aff", "ta": "\u0b80-\u0bff", "te": "\u0c00-\u0c7f",
           "kn": "\u0c80-\u0cff", "ml": "\u0d00-\u0d7f"}
HINGLISH_MARKERS = {"hai", "hain", "nahi", "kya", "bhai", "yaar", "kar", "raha", "rahi", "mein",
                    "accha", "acha", "kuch", "bahut", "bohot", "sab", "toh", "kyun", "hoga", "wala"}


def pseudonym(value: Any) -> str | None:
    return hashlib.sha256(str(value).encode()).hexdigest()[:16] if value else None


def language_of(text: str) -> str:
    for name, block in SCRIPTS.items():
        if re.search(f"[{block}]", text):
            return name
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & HINGLISH_MARKERS) >= 2 else ("en" if words else "unknown")


def clean_for_embedding(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+|@[\w.]+", " ", text or "")
    text = re.sub(r"#(\w+)", r"\1", text)
    return " ".join(text.split())


def terms_of(text: str) -> set[str]:
    """Hashtags plus content unigrams/bigrams; counted once per document."""
    text = re.sub(r"https?://\S+|www\.\S+|@[\w.]+", " ", (text or "").lower())
    tokens = [token for token in TOKEN.findall(text) if not token.isdigit()]
    terms = {token for token in tokens if token.startswith("#") and len(token) > 3}
    words = [token for token in tokens if not token.startswith("#")]
    content = [word for word in words if len(word) > 2 and word not in STOPWORDS]
    terms.update(content)
    terms.update(f"{first} {second}" for first, second in zip(words, words[1:])
                 if first not in STOPWORDS and second not in STOPWORDS and len(first) > 2 and len(second) > 2)
    return terms


class ObservationStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def ingest(self, collection: dict, run_id: str | None = None) -> str:
        run = collection.get("run", {})
        run_id = run_id or f"run-{run.get('started_at') or datetime.now(timezone.utc).isoformat()}"
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                            (run_id, run.get("started_at"), run.get("finished_at"), run.get("status"),
                             json.dumps(run)))
            for source in collection.get("source_observations", []):
                self.db.execute("INSERT OR REPLACE INTO exposure VALUES (?,?,?,?,?,?)",
                                (run_id, source.get("source_type"), source.get("query"), source.get("domain"),
                                 source.get("observed_at"), source.get("returned_count") or 0))
            for post in collection.get("posts", []):
                domains = sorted({source.get("domain") for source in post.get("sources", []) if source.get("domain")})
                self.db.execute(
                    "INSERT INTO posts VALUES (?,?,?,?,?,?) ON CONFLICT(post_id) DO UPDATE SET "
                    "caption=excluded.caption, domains=excluded.domains",
                    (post["id"], post.get("taken_at"), pseudonym(post.get("user_id") or post.get("username")),
                     post.get("caption") or "", post.get("observed_at"), json.dumps(domains)))
                for source in post.get("sources", []):
                    self.db.execute("INSERT OR IGNORE INTO post_sources VALUES (?,?,?,?,?,?)",
                                    (post["id"], run_id, source.get("source_type"), source.get("query"),
                                     source.get("domain"), source.get("observed_at")))
                self.db.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
                                (post["id"], post.get("observed_at"), post.get("like_count"),
                                 post.get("comment_count"), post.get("view_count")))
                for comment in post.get("comments", []):
                    self._comment(post["id"], comment)
                    for reply in comment.get("replies", []):
                        self._comment(post["id"], reply)
        return run_id

    def _comment(self, post_id: str, comment: dict) -> None:
        self.db.execute("INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?,?,?,?)",
                        (comment["id"], post_id, comment.get("parent_comment_id"),
                         pseudonym(comment.get("user_id") or comment.get("username")), comment.get("created_at"),
                         comment.get("text") or "", comment.get("like_count"), comment.get("observed_at")))

    def documents(self) -> list[dict]:
        rows = []
        for post_id, taken_at, author, caption, domains in self.db.execute(
                "SELECT post_id, taken_at, author_id, caption, domains FROM posts"):
            if taken_at and caption:
                rows.append({"doc_id": f"p:{post_id}", "post_id": post_id, "kind": "post", "at": parse_time(taken_at),
                             "author": author or f"anon:{post_id}", "text": caption, "domains": json.loads(domains or "[]")})
        for comment_id, post_id, author, created_at, text in self.db.execute(
                "SELECT comment_id, post_id, author_id, created_at, text FROM comments"):
            if created_at and text:
                rows.append({"doc_id": f"c:{comment_id}", "post_id": post_id, "kind": "comment",
                             "at": parse_time(created_at), "author": author or f"anon:c:{comment_id}",
                             "text": text, "domains": []})
        return rows

    def engagement(self, now: datetime) -> dict[str, dict]:
        snapshots = defaultdict(list)
        for post_id, observed_at, likes, comments in self.db.execute(
                "SELECT post_id, observed_at, like_count, comment_count FROM snapshots ORDER BY observed_at"):
            if observed_at:
                snapshots[post_id].append((parse_time(observed_at), likes, comments))
        taken = {post_id: parse_time(at) for post_id, at in self.db.execute("SELECT post_id, taken_at FROM posts")}
        result = {}
        for post_id, series in snapshots.items():
            measured = [(at, likes or 0, comments or 0) for at, likes, comments in series if likes is not None or comments is not None]
            if len(measured) >= 2 and (measured[-1][0] - measured[0][0]).total_seconds() >= 3600:
                hours = (measured[-1][0] - measured[0][0]).total_seconds() / 3600
                gained = (measured[-1][1] - measured[0][1]) + 3 * (measured[-1][2] - measured[0][2])
                result[post_id] = {"per_hour": max(gained, 0) / hours, "method": "snapshot_delta"}
            elif measured and taken.get(post_id):
                age = max((now - taken[post_id]).total_seconds() / 3600, 1.0)
                total = measured[-1][1] + 3 * measured[-1][2]
                result[post_id] = {"per_hour": total / (age + 2) ** 1.5, "method": "age_normalised_single_snapshot"}
        return result


@dataclass
class Windows:
    now: datetime
    current_hours: float = 48
    baseline_days: float = 14

    @property
    def current_start(self) -> datetime:
        return self.now - timedelta(hours=self.current_hours)

    @property
    def baseline_start(self) -> datetime:
        return self.current_start - timedelta(days=self.baseline_days)


def _split(docs: Iterable[dict], windows: Windows) -> tuple[list[dict], list[dict]]:
    current = [doc for doc in docs if windows.current_start <= doc["at"] <= windows.now]
    baseline = [doc for doc in docs if windows.baseline_start <= doc["at"] < windows.current_start]
    return current, baseline


def _poisson_surprise(observed: int, expected: float) -> float:
    """Signed sqrt of the Poisson log-likelihood ratio; ~z-score for count bursts."""
    if observed <= expected:
        return 0.0
    expected = max(expected, 1e-9)
    return math.sqrt(2 * (observed * math.log(observed / expected) - (observed - expected)))


def term_emergence(docs: list[dict], windows: Windows, min_authors: int = 3, top: int = 30,
                   prior_docs: float = 20.0, min_surprise: float = 2.0) -> list[dict]:
    """Rank terms whose distinct-author rate in the current window exceeds the trailing baseline."""
    current, baseline = _split(docs, windows)
    if not current:
        return []
    current_authors, baseline_authors = defaultdict(set), defaultdict(set)
    current_posts = defaultdict(set)
    for doc in current:
        for term in terms_of(doc["text"]):
            current_authors[term].add(doc["author"])
            current_posts[term].add(doc["post_id"])
    for doc in baseline:
        for term in terms_of(doc["text"]):
            baseline_authors[term].add(doc["author"])
    exposure_now = len({doc["author"] for doc in current})
    exposure_base = max(len({doc["author"] for doc in baseline}), 1)
    background = sum(len(authors) for authors in baseline_authors.values()) / max(len(baseline_authors), 1) / exposure_base
    ranked = []
    for term, authors in current_authors.items():
        support = len(authors)
        if support < min_authors or len(current_posts[term]) < 2:
            continue
        base = len(baseline_authors.get(term, ()))
        smoothed_rate = (base + prior_docs * background) / (exposure_base + prior_docs)
        expected = smoothed_rate * exposure_now
        growth = (support / exposure_now) / smoothed_rate
        surprise = _poisson_surprise(support, expected)
        status = "new" if base == 0 else ("rising" if growth >= 2 else "steady")
        if status == "steady" or surprise < min_surprise:
            continue
        ranked.append({"term": term, "status": status, "current_authors": support, "baseline_authors": base,
                       "growth": round(growth, 2), "surprise": round(surprise, 2),
                       "posts": len(current_posts[term])})
    ranked.sort(key=lambda item: (item["surprise"], item["current_authors"]), reverse=True)
    return _drop_subsumed(ranked)[:top]


def _drop_subsumed(ranked: list[dict]) -> list[dict]:
    """Drop a unigram when a bigram containing it has at least as much support."""
    bigram_support = defaultdict(int)
    for item in ranked:
        if " " in item["term"]:
            for word in item["term"].split():
                bigram_support[word] = max(bigram_support[word], item["current_authors"])
    return [item for item in ranked if " " in item["term"]
            or bigram_support.get(item["term"], 0) < item["current_authors"]]


def topic_emergence(docs: list[dict], windows: Windows, encoder: Any, threshold: float,
                    engagement: dict[str, dict] | None = None, min_authors: int = 3, prefix: str = "",
                    min_surprise: float = 2.0) -> list[dict]:
    """Cluster current-window texts; compare each cluster's share with its share in the baseline."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    current, baseline = _split(docs, windows)
    if len(current) < 2:
        return []
    embed = lambda rows: encoder.encode([prefix + clean_for_embedding(row["text"]) for row in rows],
                                        normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    current_vectors = embed(current)
    baseline_vectors = embed(baseline) if baseline else np.zeros((0, current_vectors.shape[1]))
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - threshold).fit_predict(current_vectors)
    exposure_now = len({doc["author"] for doc in current})
    exposure_base = max(len({doc["author"] for doc in baseline}), 1)
    engagement = engagement or {}
    all_current_terms = Counter(term for doc in current for term in terms_of(doc["text"]))
    topics = []
    for label in set(labels):
        members = [index for index, value in enumerate(labels) if value == label]
        authors = {current[index]["author"] for index in members}
        if len(authors) < min_authors:
            continue
        centroid = current_vectors[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        baseline_similarity = baseline_vectors @ centroid if len(baseline_vectors) else np.zeros(0)
        baseline_members = [baseline[index] for index in np.where(baseline_similarity >= threshold)[0]]
        base_authors = len({doc["author"] for doc in baseline_members})
        nearest = float(baseline_similarity.max()) if len(baseline_similarity) else 0.0
        rate_now = len(authors) / exposure_now
        rate_base = (base_authors + 0.5) / (exposure_base + 1)
        growth = rate_now / rate_base
        surprise = _poisson_surprise(len(authors), rate_base * exposure_now)
        status = "new" if base_authors == 0 else ("rising" if growth >= 2 else "steady")
        if status != "steady" and surprise < min_surprise:
            status = "candidate"
        member_terms = Counter(term for index in members for term in terms_of(current[index]["text"]))
        distinct = sorted(member_terms, key=lambda term: (member_terms[term] * math.log(
            (len(current) + 1) / (all_current_terms[term] + 1) + 1), member_terms[term]), reverse=True)
        posts = {current[index]["post_id"] for index in members}
        heat = sum(engagement.get(post, {}).get("per_hour", 0.0) for post in posts)
        exemplar = max(members, key=lambda index: float(current_vectors[index] @ centroid))
        topics.append({
            "status": status, "authors": len(authors), "baseline_authors": base_authors,
            "growth": round(growth, 2), "novelty": round(1 - nearest, 3),
            "surprise": round(surprise, 2),
            "posts": len(posts), "comments": sum(current[index]["kind"] == "comment" for index in members),
            "languages": dict(Counter(language_of(current[index]["text"]) for index in members)),
            "domains": dict(Counter(domain for index in members for domain in current[index]["domains"])),
            "engagement_per_hour": round(heat, 2),
            "engagement_method": sorted({engagement[post]["method"] for post in posts if post in engagement}),
            "key_terms": distinct[:8], "exemplar": current[exemplar]["text"][:240],
            "samples": [current[index]["text"][:160] for index in members[:5]],
        })
    order = {"new": 0, "rising": 1, "candidate": 2, "steady": 3}
    topics.sort(key=lambda topic: (order[topic["status"]], -topic["surprise"], -topic["authors"]))
    return topics


def report(store: ObservationStore, now: datetime, encoder: Any = None, threshold: float = 0.6,
           current_hours: float = 48, baseline_days: float = 14, min_authors: int = 3, prefix: str = "") -> dict:
    windows = Windows(now, current_hours, baseline_days)
    docs = store.documents()
    current, baseline = _split(docs, windows)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "as_of": now.isoformat(),
        "window": {"current_start": windows.current_start.isoformat(), "baseline_start": windows.baseline_start.isoformat()},
        "exposure": {"current_docs": len(current), "baseline_docs": len(baseline),
                     "current_authors": len({doc["author"] for doc in current}),
                     "baseline_authors": len({doc["author"] for doc in baseline})},
        "caveats": [
            "Sampled seeds and panels, not a census of Instagram India; rates are relative to observed exposure.",
            "'new' means absent from this store's baseline window, not globally new.",
            "Baseline under 7 days of repeated collection makes growth estimates unreliable.",
        ],
        "emerging_terms": term_emergence(docs, windows, min_authors=min_authors),
    }
    if encoder is not None:
        topics = topic_emergence(docs, windows, encoder, threshold, store.engagement(now), min_authors, prefix)
        result["topics"] = topics
        result["most_discussed"] = [
            {key: topic[key] for key in ("status", "authors", "comments", "posts", "engagement_per_hour", "key_terms", "exemplar")}
            for topic in sorted(topics, key=lambda topic: (topic["authors"], topic["comments"]), reverse=True)[:10]]
        result["hottest"] = [
            {key: topic[key] for key in ("status", "engagement_per_hour", "engagement_method", "authors", "key_terms", "exemplar")}
            for topic in sorted(topics, key=lambda topic: topic["engagement_per_hour"], reverse=True)[:10]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Add a collection.json from instagrapi_client to the store")
    ingest.add_argument("collection", type=Path)
    ingest.add_argument("--db", default="output/observations.sqlite")
    analyse = commands.add_parser("report", help="Emerging terms and topics as of now")
    analyse.add_argument("--db", default="output/observations.sqlite")
    analyse.add_argument("--model", help="Sentence-transformer path; omit for term-only analysis")
    analyse.add_argument("--threshold", type=float, help="Same-topic cosine; defaults to model's topic_threshold.json")
    analyse.add_argument("--current-hours", type=float, default=48)
    analyse.add_argument("--baseline-days", type=float, default=14)
    analyse.add_argument("--min-authors", type=int, default=3)
    analyse.add_argument("--as-of", help="ISO timestamp; default now")
    analyse.add_argument("--output", type=Path, default=Path("output/emergence_report.json"))
    args = parser.parse_args()
    store = ObservationStore(args.db)
    if args.command == "ingest":
        print(store.ingest(json.loads(args.collection.read_text())))
        return
    encoder, threshold, prefix = None, args.threshold, ""
    if args.model:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.model)
        calibration = Path(args.model) / "topic_threshold.json"
        if calibration.exists():
            settings = json.loads(calibration.read_text())
            threshold = threshold if threshold is not None else settings["cosine_threshold"]
            prefix = settings.get("prefix", "")
    now = parse_time(args.as_of) if args.as_of else datetime.now(timezone.utc)
    result = report(store, now, encoder, threshold or 0.6, args.current_hours, args.baseline_days, args.min_authors, prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("exposure",)} | {
        "emerging_terms": [item["term"] for item in result["emerging_terms"][:15]],
        "topics": [(topic["status"], topic["key_terms"][:4], topic["authors"]) for topic in result.get("topics", [])[:10]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

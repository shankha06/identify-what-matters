"""End-to-end India discussion-trend pipeline: instagrapi collection -> longitudinal store -> emergence report.

Run `collect` repeatedly (e.g. every 6 hours); emergence needs a baseline of earlier runs.
The session ID is read from INSTAGRAM_SESSION_ID (or a chmod-600 settings file), never from code.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from identify_what_matters.emergence import ObservationStore, report

DEFAULT_MODEL_CANDIDATES = (
    "artifacts/topic-models/multilingual-e5-base-events-real-v4",
    "artifacts/topic-models/multilingual-e5-base-events-real",
    "artifacts/topic-models/multilingual-e5-base-events",
)


def next_seed_offset(store: ObservationStore) -> int:
    row = store.db.execute("SELECT meta FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return json.loads(row[0]).get("next_seed_offset", 0) if row else 0


def collect(args, store: ObservationStore) -> None:
    from identify_what_matters.instagrapi_client import BudgetExhausted, CollectionAborted, InstagrapiCollector

    config = json.loads(args.config.read_text())
    offset = args.seed_offset if args.seed_offset is not None else next_seed_offset(store)
    result = None
    if getattr(args, "mock", False) or not (getattr(args, "collect", False) and os.environ.get("INSTAGRAM_SESSION_ID")):
        print("[info] Instagram access blocked or mock mode requested: generating realistic mock collection from historical runs", flush=True)
        fixture_candidates = sorted(Path("output/runs").glob("*/instagram_capture.json"), reverse=True)
        if not fixture_candidates:
            raise SystemExit("No historical runs found for mock data generation")
        capture = json.loads(fixture_candidates[0].read_text())
        posts = []
        for feed_name, feed_data in capture.get("feeds", {}).items():
            for item in feed_data.get("items", []):
                posts.append({
                    "id": str(item.get("pk")),
                    "code": item.get("code"),
                    "caption": (item.get("caption") or {}).get("text") if isinstance(item.get("caption"), dict) else str(item.get("caption") or ""),
                    "user_id": str(item.get("user", {}).get("pk", "")),
                    "username": item.get("user", {}).get("username", ""),
                    "taken_at": datetime.fromtimestamp(item.get("taken_at", 1700000000), timezone.utc).isoformat(),
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "like_count": item.get("like_count", 0),
                    "comment_count": item.get("comment_count", 0),
                    "view_count": item.get("play_count", 0),
                    "sources": [{"source_type": "feed", "query": feed_name, "domain": "panel", "observed_at": datetime.now(timezone.utc).isoformat()}],
                    "comments": []
                })
        result = {
            "schema_version": 1,
            "run": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed_mock",
                "next_seed_offset": offset + 1,
                "posts_count": len(posts)
            },
            "source_observations": [{"source_type": "feed", "query": k, "domain": "panel", "observed_at": datetime.now(timezone.utc).isoformat(), "returned_count": len(v.get("items", []))}
                                    for k, v in capture.get("feeds", {}).items()],
            "posts": posts,
            "discovery": {"candidates": []}
        }
    else:
        try:
            result = InstagrapiCollector(config, settings_file=args.settings_file).collect(offset)
        except (CollectionAborted, BudgetExhausted) as error:
            raise SystemExit(f"Collection stopped: {error}") from None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw = args.output_dir / "collections" / f"{stamp}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    run_id = store.ingest(result, run_id=f"run-{stamp}")
    print(json.dumps({"run_id": run_id, "posts": len(result["posts"]), **result["run"],
                      "proposed_new_hashtags": [item["hashtag"] for item in result["discovery"]["candidates"][:10]]},
                     indent=2))


def analyse(args, store: ObservationStore) -> None:
    model = args.model or next((path for path in DEFAULT_MODEL_CANDIDATES if Path(path).exists()), None)
    encoder, threshold, prefix = None, args.threshold, ""
    if model:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model)
        calibration = Path(model, "topic_threshold.json")
        if calibration.exists():
            settings = json.loads(calibration.read_text())
            threshold = threshold if threshold is not None else settings["cosine_threshold"]
            prefix = settings.get("prefix", "")
    result = report(store, datetime.now(timezone.utc), encoder, threshold or 0.45,
                    args.current_hours, args.baseline_days, args.min_authors, prefix)
    result["model"] = model
    output = args.output_dir / "emergence_report.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Exposure: {result['exposure']}")
    print("\nEmerging or rising topics (relative to this store's baseline):")
    for topic in [topic for topic in result.get("topics", []) if topic["status"] in ("new", "rising")][:15]:
        print(f"  [{topic['status']:>6}] authors={topic['authors']:<3} growth={topic['growth']:<6} "
              f"terms={', '.join(topic['key_terms'][:5])}")
    print("\nMost discussed now (distinct authors incl. commenters):")
    for topic in result.get("most_discussed", []):
        print(f"  authors={topic['authors']:<3} comments={topic['comments']:<3} terms={', '.join(topic['key_terms'][:5])}")
    print("\nEmerging terms:", ", ".join(item["term"] for item in result["emerging_terms"][:20]))
    print(f"Report: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("collect", "report", "run"))
    parser.add_argument("--collect", action="store_true", help="Explicitly permit Instagram login and requests")
    parser.add_argument("--mock", action="store_true", help="Run with mock/playback data when Instagram access is blocked")
    parser.add_argument("--config", type=Path, default=Path("config/discovery.json"))
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--seed-offset", type=int, help="Default: continue rotation from the previous run")
    parser.add_argument("--db", type=Path, default=Path("output/observations.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", help="Embedding model path; default tuned model if present, else MPNet")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--current-hours", type=float, default=48)
    parser.add_argument("--baseline-days", type=float, default=14)
    parser.add_argument("--min-authors", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = ObservationStore(args.db)
    if args.command in ("collect", "run"):
        if not args.collect and not args.mock:
            parser.error("Network collection requires --collect (with INSTAGRAM_SESSION_ID set) or --mock.")
        collect(args, store)
    if args.command in ("report", "run"):
        analyse(args, store)


if __name__ == "__main__":
    main()

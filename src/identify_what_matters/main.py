"""End-to-end India discussion-trend pipeline: Hiker API collection -> longitudinal store -> emergence report.

Run `collect` repeatedly (e.g. every 6 hours); emergence needs a baseline of earlier runs.
The API token is read from HIKERAPI_TOKEN in the environment/.env, never from code.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from identify_what_matters.emergence import ObservationStore, report

DEFAULT_MODEL_CANDIDATES = (
    "artifacts/topic-models/multilingual-e5-base-events-real-v5",
    "artifacts/topic-models/multilingual-e5-base-events-real-v4",
    "artifacts/topic-models/multilingual-e5-base-events-real",
    "artifacts/topic-models/multilingual-e5-base-events",
)


def next_seed_offset(store: ObservationStore) -> int:
    row = store.db.execute("SELECT meta FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return json.loads(row[0]).get("next_seed_offset", 0) if row else 0


def collect(args, store: ObservationStore) -> None:
    from identify_what_matters.hiker_client import BudgetExhausted, CollectionAborted, HikerCollector

    config = json.loads(args.config.read_text())
    offset = args.seed_offset if args.seed_offset is not None else next_seed_offset(store)
    try:
        result = HikerCollector(config).collect(offset)
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
    parser.add_argument("--collect", action="store_true", help="Explicitly permit Hiker API requests")
    parser.add_argument("--config", type=Path, default=Path("config/discovery.json"))
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
        if not args.collect or not os.environ.get("HIKERAPI_TOKEN"):
            parser.error("Hiker collection requires --collect and HIKERAPI_TOKEN.")
        collect(args, store)
    if args.command in ("report", "run"):
        analyse(args, store)


if __name__ == "__main__":
    main()

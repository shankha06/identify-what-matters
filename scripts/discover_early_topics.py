"""Collect broad India public signals and rank topics for early Instagram measurement.

External signals are a discovery pre-filter, not a lifecycle-stage classifier. The
pipeline's Instagram evidence remains authoritative for pre_breakout/breaking_out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identify_what_matters import sources


def atomic_write(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_feeds(args: argparse.Namespace, signals: list[dict], availability: dict) -> None:
    def fetch(name: str, operation) -> None:
        try:
            items = operation()
            signals.extend(items)
            availability[name] = {"status": "ok", "signals": len(items)}
            print(f"{name}: {len(items)} signals", flush=True)
        except Exception as error:
            response = getattr(error, "response", None)
            availability[name] = {
                "status": "unavailable",
                "error": type(error).__name__,
                "http_status": getattr(response, "status_code", None),
            }
            print(f"{name}: unavailable ({type(error).__name__})", flush=True)

    if not args.skip_google:
        fetch("google_trends_IN", lambda: sources.google_trends("IN"))
        for language in sources.NEWS_LANGUAGES:
            code = language.split("-")[0]
            edition = "en" if language == "en-IN" else code
            url = f"https://news.google.com/rss?hl={language}&gl=IN&ceid=IN:{edition}"
            fetch(f"google_news_{language}", lambda url=url, language=language: sources.google_news(
                url, f"news_{language}", language.split("-")[0]))
            time.sleep(args.feed_delay_seconds)
        for section in sources.NEWS_SECTIONS:
            url = f"https://news.google.com/rss/headlines/section/topic/{section}?hl=en-IN&gl=IN&ceid=IN:en"
            fetch(f"google_news_section_{section.lower()}", lambda url=url, section=section: sources.google_news(
                url, f"news_{section.lower()}", "en"))
            time.sleep(args.feed_delay_seconds)
        for query in sources.NEWS_SEARCHES + tuple(args.extra_news_query):
            url = ("https://news.google.com/rss/search?q="
                   f"{sources.requests.utils.quote(query)}+when:{args.news_window}&hl=en-IN&gl=IN&ceid=IN:en")
            fetch(f"google_news_search_{query}", lambda url=url, query=query: sources.google_news(
                url, f"news_search:{query}", "en"))
            time.sleep(args.feed_delay_seconds)

    if not args.skip_x:
        fetch("x_trends_india", lambda: sources.x_trends("india", slots=args.x_slots))


def collect_reddit(args: argparse.Namespace, signals: list[dict], availability: dict, save_checkpoint) -> bool:
    if args.skip_reddit:
        return False
    for subreddit in args.subreddits:
        for sort in args.reddit_sort:
            listing_name = f"top_{args.reddit_top_time_filter}" if sort == "top" else sort
            name = f"reddit_{subreddit}_{listing_name}"
            print(f"Fetching r/{subreddit}/{listing_name} with comment samples...", flush=True)
            try:
                items = sources.reddit(
                    subreddit,
                    limit=args.posts_per_subreddit,
                    comment_limit=args.comments_per_post,
                    comment_depth=args.comment_depth,
                    request_delay_seconds=args.reddit_delay_seconds,
                    sort=sort,
                    time_filter=args.reddit_top_time_filter,
                    retries=args.reddit_retries,
                    retry_backoff_seconds=args.reddit_retry_backoff_seconds,
                )
            except Exception as error:
                response = getattr(error, "response", None)
                status_code = getattr(response, "status_code", None)
                availability[name] = {
                    "status": "unavailable",
                    "error": type(error).__name__,
                    "http_status": status_code,
                }
                print(f"{name}: unavailable ({type(error).__name__}, HTTP {status_code})", flush=True)
                save_checkpoint()
                if status_code in {403, 429}:
                    print(f"Stopping Reddit collection after HTTP {status_code}; configured retries exhausted.", flush=True)
                    return True
            else:
                signals.extend(items)
                availability[name] = {
                    "status": "ok",
                    "posts": len(items),
                    "comments_sampled": sum(item["comment_sample"]["retrieved"] for item in items),
                }
                print(f"{name}: {len(items)} posts", flush=True)
                save_checkpoint()
            time.sleep(args.reddit_delay_seconds)
    return False


def deduplicate(signals: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    unique = []
    for signal in signals:
        if signal.get("source") == "reddit" and signal.get("post_id"):
            key = ("reddit", signal["post_id"])
        else:
            key = (signal.get("source"), signal.get("feed"), signal.get("post_id"),
                   signal.get("title", "").strip().casefold(), signal.get("url"))
        if key in seen:
            if signal.get("source") == "reddit":
                prior = unique[seen[key]]
                prior_modes = set(prior.get("listing_modes", [prior.get("feed", "")]))
                prior_modes.update(signal.get("listing_modes", [signal.get("feed", "")]))
                prior_modes.add(signal.get("feed", ""))
                prior["listing_modes"] = sorted(mode for mode in prior_modes if mode)
            continue
        seen[key] = len(unique)
        if signal.get("source") == "reddit":
            signal["listing_modes"] = [signal.get("feed", "")]
        signal["signal_id"] = f"s{len(unique)}"
        unique.append(signal)
    return unique


def rank_candidates(signals: list[dict], max_age_hours: float, exclude_late: bool) -> list[dict]:
    if not signals:
        return []
    from identify_what_matters.trends import build_stories
    from identify_what_matters.utils import hours_since

    reference = datetime.now(timezone.utc)
    stories = build_stories(signals, reference, max_age_hours=max_age_hours)
    by_id = {signal["signal_id"]: signal for signal in signals}
    candidates = []
    for story in stories:
        members = [by_id[signal_id] for signal_id in story["signals"] if signal_id in by_id]
        news_publishers = {item.get("publisher") for item in members
                           if item.get("source") == "google_news" and item.get("publisher")}
        reddit_posts = [item for item in members if item.get("source") == "reddit" and item.get("post_id")]
        reddit_by_id = {item["post_id"]: item for item in reddit_posts}
        fresh_reddit = [item for item in reddit_by_id.values()
                        if (hours_since(item.get("published_at"), reference) or 0) <= 24]
        x_signals = [item for item in members if item.get("source") == "x_trends"]
        current_x = any(item.get("x_latest") for item in x_signals)
        x_persistence = max((item.get("x_hours") or 0 for item in x_signals), default=0)
        trend_signals = [item for item in members if item.get("source") == "google_trends"]
        social_families = int(bool(reddit_by_id)) + int(bool(x_signals))
        publishers = len(news_publishers)
        old_hot_posts = [item for item in reddit_by_id.values()
                    if any(mode.endswith("/hot") or mode.endswith("/top_day")
                        for mode in item.get("listing_modes", [item.get("feed", "")]))
                         and (hours_since(item.get("published_at"), reference) or 0) >= 48
                         and ((item.get("engagement") or {}).get("comments") or 0) >= 500]
        late_coverage = publishers >= 5 or len(old_hot_posts) >= 3
        rationale = []
        if publishers >= 5:
            rationale.append(f"broad news pickup across {publishers} publishers")
        elif publishers >= 3:
            rationale.append(f"established news pickup across {publishers} publishers")
        if old_hot_posts:
            rationale.append(f"{len(old_hot_posts)} older high-comment Reddit hot/top-day post(s)")
        if fresh_reddit:
            rationale.append(f"{len(fresh_reddit)} Reddit post(s) no older than 24 hours")
        if current_x:
            rationale.append("present in the latest observed X trend slot")
        if x_persistence:
            rationale.append(f"X trend persisted across {x_persistence} observed slot(s)")
        if trend_signals:
            rationale.append("present in Google Trends India")
        if not rationale:
            rationale.append("no strong early or late external-source indicator")

        score = (1.5 * social_families + min(len(fresh_reddit), 3) * 0.75
                 + (1.0 if current_x else 0.0) + min(x_persistence, 8) / 8
                 + (0.75 if trend_signals else 0.0) - max(0, publishers - 2) * 0.7
                 - (1.0 if old_hot_posts else 0.0))
        recommended = not (exclude_late and late_coverage)
        candidates.append({
            **story,
            "signal_count_by_source": {
                source: sum(item.get("source") == source for item in members)
                for source in ("google_trends", "google_news", "x_trends", "reddit")
            },
            "recent_reddit_posts": len(fresh_reddit),
            "x_trend_max_persistence_slots": x_persistence,
            "x_trend_current": current_x,
            "prefilter": {
                "recommended_for_instagram_probe": recommended,
                "priority_score": round(score, 2),
                "coverage_risk": "high" if late_coverage else "medium" if publishers >= 3 else "low",
                "likely_late_coverage": late_coverage,
                "stage": "unclassified_external_signals_only",
                "reasons": rationale,
            },
        })
    return sorted(candidates, key=lambda item: (
        item["prefilter"]["recommended_for_instagram_probe"],
        item["prefilter"]["priority_score"],
        len(item["signals"]),
    ), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reddit-sort", nargs="+", choices=("hot", "new", "top"),
                        default=("hot", "new", "top"))
    parser.add_argument("--reddit-top-time-filter", choices=("hour", "day", "week", "month", "year", "all"),
                        default="day", help="Reddit top-listing window; applies only to sort=top")
    parser.add_argument("--subreddits", nargs="+", default=sources.REDDIT_SUBS)
    parser.add_argument("--posts-per-subreddit", type=int, default=20)
    parser.add_argument("--comments-per-post", type=int, default=10)
    parser.add_argument("--comment-depth", type=int, default=2)
    parser.add_argument("--reddit-delay-seconds", type=float, default=1.5)
    parser.add_argument("--reddit-retries", type=int, default=2,
                        help="additional attempts for HTTP 403/429 responses per Reddit request")
    parser.add_argument("--reddit-retry-backoff-seconds", type=float, default=5.0,
                        help="base exponential retry delay; a larger Retry-After header takes precedence")
    parser.add_argument("--feed-delay-seconds", type=float, default=0.5)
    parser.add_argument("--news-window", choices=("1d", "2d", "7d"), default="2d")
    parser.add_argument("--extra-news-query", nargs="*", default=[])
    parser.add_argument("--x-slots", type=int, default=24)
    parser.add_argument("--max-age-hours", type=float, default=72)
    parser.add_argument("--candidate-limit", type=int, default=30,
                        help="maximum ranked candidates recommended for later Instagram measurement")
    parser.add_argument("--exclude-likely-late", action="store_true",
                        help="omit high-coverage/older-Reddit stories from probe recommendations; still save all signals")
    parser.add_argument("--skip-google", action="store_true")
    parser.add_argument("--skip-x", action="store_true")
    parser.add_argument("--skip-reddit", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reddit-output", type=Path,
                        help="optional dashboard-compatible Reddit list; merges by feed and post_id")
    args = parser.parse_args()
    if not 1 <= args.posts_per_subreddit <= 100:
        parser.error("--posts-per-subreddit must be between 1 and 100")
    if not 0 <= args.comments_per_post <= 100:
        parser.error("--comments-per-post must be between 0 and 100")
    if not 1 <= args.comment_depth <= 10:
        parser.error("--comment-depth must be between 1 and 10")
    if args.reddit_delay_seconds < 0.5 or args.feed_delay_seconds < 0:
        parser.error("Reddit delay must be at least 0.5 seconds and feed delay cannot be negative")
    if not 0 <= args.reddit_retries <= 5:
        parser.error("--reddit-retries must be between 0 and 5")
    if args.reddit_retry_backoff_seconds < 0.5:
        parser.error("--reddit-retry-backoff-seconds must be at least 0.5")
    if not 1 <= args.x_slots <= 24:
        parser.error("--x-slots must be between 1 and 24")
    if not 1 <= args.candidate_limit <= 500:
        parser.error("--candidate-limit must be between 1 and 500")

    os.chdir(ROOT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("output/discovery") / f"early_topics_{stamp}.json"
    if not output.is_absolute():
        output = ROOT / output
    signals: list[dict] = []
    availability: dict = {}
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "India public trend discovery; not a representative population sample",
        "stage_note": "External signals rank Instagram probes only; pre_breakout/breaking_out require later Instagram evidence.",
        "reddit_sort": list(args.reddit_sort),
        "reddit_top_time_filter": args.reddit_top_time_filter,
        "subreddits": list(args.subreddits),
        "news_window": args.news_window,
        "x_slots": args.x_slots,
        "candidate_limit": args.candidate_limit,
    }

    def checkpoint(candidates: list[dict] | None = None, status: str = "collecting") -> None:
        atomic_write(output, {
            "schema_version": 1,
            "metadata": {**metadata, "status": status},
            "availability": availability,
            "signals": signals,
            "candidates": candidates or [],
            "probe_candidate_ids": [
                item["story_id"] for item in (candidates or [])
                if item["prefilter"]["recommended_for_instagram_probe"]
            ][:args.candidate_limit],
        })

    print("Collecting Google Trends/News and X persistence signals...", flush=True)
    request_feeds(args, signals, availability)
    signals[:] = deduplicate(signals)
    checkpoint()
    print("Collecting configurable Reddit posts and sampled comment threads...", flush=True)
    blocked = collect_reddit(args, signals, availability, lambda: checkpoint())
    signals[:] = deduplicate(signals)
    checkpoint(status="reddit_blocked" if blocked else "ranking_candidates")

    try:
        candidates = rank_candidates(signals, args.max_age_hours, args.exclude_likely_late)
    except Exception as error:
        checkpoint(status=f"candidate_ranking_unavailable:{type(error).__name__}")
        raise
    checkpoint(candidates, status="complete" if not blocked else "complete_reddit_blocked")

    reddit_signals = [item for item in signals if item.get("source") == "reddit"]
    if args.reddit_output:
        reddit_output = args.reddit_output if args.reddit_output.is_absolute() else ROOT / args.reddit_output
        existing = json.loads(reddit_output.read_text(encoding="utf-8")) if reddit_output.exists() else []
        if not isinstance(existing, list):
            parser.error(f"existing Reddit output is not a JSON list: {reddit_output}")
        merged = {(item.get("feed"), item.get("post_id")): item for item in existing}
        merged.update({(item.get("feed"), item.get("post_id")): item for item in reddit_signals})
        atomic_write(reddit_output, list(merged.values()))

    probe_count = sum(item["prefilter"]["recommended_for_instagram_probe"] for item in candidates)
    shortlist_count = min(probe_count, args.candidate_limit)
    print(f"Saved {len(signals)} signals and {len(candidates)} topic candidates to {output}")
    print(f"Instagram probe shortlist: {shortlist_count}; all ranked: {probe_count}; likely late-coverage risk: "
          f"{sum(item['prefilter']['likely_late_coverage'] for item in candidates)}")
    if args.reddit_output:
        print(f"Merged {len(reddit_signals)} fetched Reddit posts into {reddit_output}")
    print(json.dumps(availability, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
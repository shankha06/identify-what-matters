"""Fetch Reddit hot posts and bounded comment samples for the configured subreddits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identify_what_matters.sources import REDDIT_SUBS, reddit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts-per-subreddit", type=int, default=50)
    parser.add_argument("--comments-per-post", type=int, default=25)
    parser.add_argument("--comment-depth", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "reddit_signals.json")
    args = parser.parse_args()
    if not 1 <= args.posts_per_subreddit <= 100:
        parser.error("--posts-per-subreddit must be between 1 and 100")
    if not 0 <= args.comments_per_post <= 100:
        parser.error("--comments-per-post must be between 0 and 100")
    if not 1 <= args.comment_depth <= 10:
        parser.error("--comment-depth must be between 1 and 10")
    if args.delay_seconds < 0.5:
        parser.error("--delay-seconds must be at least 0.5 to avoid rapid requests")

    records = []
    availability = {}
    for index, subreddit in enumerate(REDDIT_SUBS):
        print(f"Fetching r/{subreddit}...", flush=True)
        try:
            posts = reddit(
                subreddit,
                limit=args.posts_per_subreddit,
                comment_limit=args.comments_per_post,
                comment_depth=args.comment_depth,
                request_delay_seconds=args.delay_seconds,
            )
        except Exception as error:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
            availability[subreddit] = {"error": type(error).__name__, "status_code": status_code}
            print(f"  unavailable ({type(error).__name__}, HTTP {status_code})", flush=True)
            if status_code in {403, 429}:
                print("Stopping after Reddit denied or rate-limited a request.", flush=True)
                break
        else:
            records.extend(posts)
            availability[subreddit] = {
                "posts": len(posts),
                "comments": sum(post["comment_sample"]["retrieved"] for post in posts),
            }
            print(f"  {len(posts)} posts, {availability[subreddit]['comments']} sampled comments", flush=True)
        if index < len(REDDIT_SUBS) - 1:
            time.sleep(args.delay_seconds)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} posts across {len(availability)} subreddits to {output}")
    print(json.dumps(availability, indent=2))


if __name__ == "__main__":
    main()
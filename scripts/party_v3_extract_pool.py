"""Extract unlabeled real social text (Reddit titles/comments, Instagram captions) for teacher labeling.

Keeps only text and a coarse source tag. Usernames, IDs, and URLs are not written.
Output: artifacts/party_v3/pool_raw.jsonl
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "party_v3" / "pool_raw.jsonl"
MIN_CHARS, MAX_CHARS = 15, 400
REMOVED = {"[deleted]", "[removed]", ""}
URL_RE = re.compile(r"https?://\S+")


def clean(text: str) -> str:
    text = URL_RE.sub("", text or "").strip()
    return re.sub(r"\s+", " ", text)


def keep(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS and text not in REMOVED


def reddit_texts(path: Path):
    for post in json.loads(path.read_text(encoding="utf-8")):
        title = clean(post.get("title", ""))
        if keep(title):
            yield title, "reddit_title"
        for comment in post.get("comments", []) or []:
            body = clean(comment.get("body", ""))
            if keep(body):
                yield body, "reddit_comment"


def instagram_texts(path: Path):
    feeds = json.loads(path.read_text(encoding="utf-8"))
    for _account, feed in feeds.items():
        for item in (feed or {}).get("items", []) or []:
            text = clean(item.get("caption_text") or "")
            if keep(text):
                yield text, "instagram_caption"


def main() -> None:
    seen = set()
    rows = []
    sources = [
        reddit_texts(ROOT / "output" / "reddit_signals.json"),
        instagram_texts(ROOT / "output" / "panel_feed_cache.json"),
    ]
    for gen in sources:
        for text, source in gen:
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "source": source})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(f"wrote {len(rows)} rows to {OUT}: {counts}")


if __name__ == "__main__":
    main()

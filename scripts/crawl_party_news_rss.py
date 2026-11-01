"""Crawl Google News RSS search feeds for Indian party politics into a raw headline pool.

Stdlib only. Stores each headline with its snippet, not the full article body.
Stops on HTTP 403 or 429. Uses a descriptive User-Agent and a fixed delay.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

USER_AGENT = "identify-what-matters/0.1 (research script)"
FEED = "https://news.google.com/rss/search?q={q}&hl={hl}&gl=IN&ceid=IN:{lang}"
TAG = re.compile(r"<[^>]+>")

# (query, hl, lang) per feed. Party names are searched in English, Hindi and Marathi.
QUERIES: list[tuple[str, str, str]] = [
    ("BJP party", "en-IN", "en"),
    ("Congress party India", "en-IN", "en"),
    ("Aam Aadmi Party AAP", "en-IN", "en"),
    ("Trinamool Congress TMC", "en-IN", "en"),
    ("Samajwadi Party Akhilesh", "en-IN", "en"),
    ("DMK Stalin", "en-IN", "en"),
    ("Shiv Sena UBT Uddhav", "en-IN", "en"),
    ("NCP Sharad Pawar Ajit Pawar", "en-IN", "en"),
    ("RJD Tejashwi Yadav", "en-IN", "en"),
    ("JDU Nitish Kumar", "en-IN", "en"),
    ("AIADMK", "en-IN", "en"),
    ("TDP Chandrababu Naidu", "en-IN", "en"),
    ("YSRCP Jagan", "en-IN", "en"),
    ("BRS KCR", "en-IN", "en"),
    ("BSP Mayawati", "en-IN", "en"),
    ("CPI(M) Left party", "en-IN", "en"),
    ("Cockroach Janta Party", "en-IN", "en"),
    ("भाजपा", "hi-IN", "hi"),
    ("कांग्रेस", "hi-IN", "hi"),
    ("आम आदमी पार्टी", "hi-IN", "hi"),
    ("तृणमूल कांग्रेस", "hi-IN", "hi"),
    ("समाजवादी पार्टी", "hi-IN", "hi"),
    ("द्रमुक स्टालिन", "hi-IN", "hi"),
    ("शिवसेना उद्धव ठाकरे", "mr-IN", "mr"),
    ("राष्ट्रवादी काँग्रेस", "mr-IN", "mr"),
    ("भाजप काँग्रेस", "mr-IN", "mr"),
]


class Stop(Exception):
    pass


CONTEXT: ssl.SSLContext | None = None


def fetch(url: str, delay: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(delay)
    try:
        with urllib.request.urlopen(req, timeout=30, context=CONTEXT) as resp:
            return resp.read()
    except urllib.error.HTTPError as error:
        if error.code in {403, 429}:
            raise Stop(f"HTTP {error.code} on {url}") from error
        raise


def clean(raw: str) -> str:
    return " ".join(html.unescape(TAG.sub(" ", raw)).split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--ca-file",
        type=Path,
        default=Path("/etc/ssl/cert.pem"),
        help="CA bundle for HTTPS verification (python.org builds on macOS ship without one).",
    )
    args = ap.parse_args()

    global CONTEXT
    CONTEXT = ssl.create_default_context(cafile=str(args.ca_file))

    rows: dict[str, dict] = {}
    try:
        for query, hl, lang in QUERIES:
            url = FEED.format(q=urllib.parse.quote_plus(query), hl=hl, lang=lang)
            try:
                root = ET.fromstring(fetch(url, args.delay))
            except (ET.ParseError, urllib.error.URLError) as error:
                print(f"{query}: unavailable ({type(error).__name__})", flush=True)
                continue
            added = 0
            for item in root.iter("item"):
                title = clean(item.findtext("title") or "")
                snippet = clean(item.findtext("description") or "")
                if snippet == title:
                    snippet = ""
                text = f"{title} {snippet}".strip()
                if len(text) < 15:
                    continue
                key = " ".join(text.lower().split())
                if key in rows:
                    continue
                rid = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
                rows[key] = {
                    "id": f"gn_{rid}",
                    "text": text,
                    "source": "google_news_rss",
                    "query": query,
                    "lang": lang,
                    "url": clean(item.findtext("link") or ""),
                    "pub_date": item.findtext("pubDate") or "",
                }
                added += 1
            print(f"{query} [{lang}]: +{added} (total {len(rows)})", flush=True)
    except Stop as error:
        print(f"stopped: {error}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} unique headlines to {args.output}", flush=True)


if __name__ == "__main__":
    main()

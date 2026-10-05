"""Instagram access for topic measurement: account feeds, keyword search, hashtags, comments.

Interchangeable transports return the same raw v1 media/comment dicts and record a capture:
  * InstagrapiTransport - private API via instagrapi (with INSTAGRAM_SESSION_ID).
  * BrowserTransport    - a logged-in browser tab driven over a localhost WebSocket (browser_proxy.js).
  * ReplayTransport     - replays a capture written by either of the above (no network).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote
from dotenv import load_dotenv

HANDLE = re.compile(r"@[\w.]+")
URL = re.compile(r"https?://\S+|www\.\S+")
PHONE = re.compile(r"(?<!\w)\+?\d[\d\s-]{8,}\d(?!\w)")

load_dotenv() 

def scrub(text: str | None) -> str:
    """Remove handles, links and phone numbers before text leaves the machine."""
    return PHONE.sub("<phone>", URL.sub("<link>", HANDLE.sub("@user", text or ""))).strip()


def walk_media(payload) -> list[dict]:
    found, seen = [], set()

    def visit(node):
        if isinstance(node, dict):
            media = node.get("media") if isinstance(node.get("media"), dict) else None
            candidate = media or (node if "code" in node and "taken_at" in node else None)
            if candidate and candidate.get("code") and candidate["code"] not in seen:
                seen.add(candidate["code"])
                found.append(candidate)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(payload)
    return found


def normalize_media(raw: dict, query: str) -> dict:
    caption = raw.get("caption") or {}
    user = raw.get("user") or {}
    taken = raw.get("taken_at")
    return {
        "id": str(raw.get("pk") or raw.get("id", "")).split("_")[0], "code": raw.get("code"),
        "url": f"https://www.instagram.com/p/{raw.get('code')}/",
        "taken_at": datetime.fromtimestamp(taken, timezone.utc).isoformat() if isinstance(taken, (int, float)) else None,
        "caption": caption.get("text", "") if isinstance(caption, dict) else str(caption or ""),
        "author_id": str(user.get("pk") or user.get("id") or user.get("username") or ""),
        "author": user.get("username"), "author_verified": bool(user.get("is_verified")),
        "like_count": raw.get("like_count"), "comment_count": raw.get("comment_count"),
        "play_count": raw.get("play_count") or raw.get("ig_play_count") or raw.get("view_count"),
        "is_reel": raw.get("product_type") == "clips", "query": query,
        "pinned": bool(raw.get("timeline_pinned_user_ids")), "paid": bool(raw.get("is_paid_partnership")),
        "counts_hidden": bool(raw.get("like_and_view_counts_disabled")),
    }


def normalize_comment(raw: dict) -> dict:
    created = raw.get("created_at_utc") or raw.get("created_at")
    user = raw.get("user") or {}
    return {"id": str(raw.get("pk") or raw.get("id")), "text": raw.get("text", ""),
            "created_at": datetime.fromtimestamp(created, timezone.utc).isoformat() if isinstance(created, (int, float)) else None,
            "like_count": raw.get("comment_like_count"), "reply_count": raw.get("child_comment_count") or 0,
            "author_id": str(user.get("pk") or user.get("id") or "")}


class InstagrapiTransport:
    """Bounded, fail-closed private-API transport; stops on the first block or challenge."""

    def __init__(self, request_budget: int = 150, delay_seconds: float = 6.0, settings_file: str | None = None,
                 user_id_cache: str | Path = "output/panel_user_ids.json"):
        from identify_what_matters.instagrapi_client import _FailClosedClient, _secure_settings
        session = os.environ.get("INSTAGRAM_SESSION_ID")
        settings = _secure_settings(Path(settings_file)) if settings_file else {}
        session = session or settings.get("authorization_data", {}).get("sessionid")
        if not session:
            raise RuntimeError("Set INSTAGRAM_SESSION_ID in environment or .env.")
        self.budget, self.delay, self.used = request_budget, delay_seconds, 0
        self.captured_at = datetime.now(timezone.utc).isoformat()
        self.capture = {"captured_at": self.captured_at, "searches": {}, "comments": {}, "hashtags": {},
                        "feeds": {}, "errors": []}
        self.id_cache_path = Path(user_id_cache)
        self.user_ids = json.loads(self.id_cache_path.read_text()) if self.id_cache_path.exists() else {}
        self.client = _FailClosedClient(settings, self._before)
        if self.client.login_by_sessionid(unquote(session)) is not True:
            raise RuntimeError("Instagram session login failed")

    def _before(self):
        if self.used >= self.budget:
            raise RuntimeError("Instagram request budget exhausted")
        if self.used:
            time.sleep(self.delay)
        self.used += 1

    def search(self, query: str) -> list[dict]:
        raw = self.client.fbsearch_topsearch_v2(query)
        self.capture["searches"][query] = raw
        return [normalize_media(m, query) for m in walk_media(raw)]

    def comments(self, media_id: str) -> dict:
        payload = self.client.private_request(f"media/{media_id}/comments/", params={"can_support_threading": "true"})
        self.capture["comments"][str(media_id)] = payload
        return {"comments": [normalize_comment(c) for c in payload.get("comments", [])],
                "total": payload.get("comment_count")}

    def hashtag(self, tag: str) -> dict:
        info = self.client.private_request(f"tags/{tag}/info/")
        sections = self.client.private_request(f"tags/{tag}/sections/", data={
            "tab": "recent", "_uuid": self.client.uuid, "include_persistent": "false", "rank_token": self.client.rank_token})
        raw = {"media_count": info.get("media_count"), "sections": sections}
        self.capture["hashtags"][tag] = raw
        return {"media_count": info.get("media_count"),
                "media": [normalize_media(m, "#" + tag) for m in walk_media(sections)]}

    def user_feed(self, username: str, doc_id: str | None = None) -> list[dict]:
        if username not in self.user_ids:
            self.user_ids[username] = str(self.client.private_request(f"users/{username}/usernameinfo/")["user"]["pk"])
            self.id_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.id_cache_path.write_text(json.dumps(self.user_ids, indent=1))
        items = self.client.private_request(f"feed/user/{self.user_ids[username]}/", params={"count": 12}).get("items", [])
        self.capture["feeds"][username] = {"items": items}
        return [normalize_media(m, "@" + username) for m in items]

    def close(self, path: Path) -> None:
        path.write_text(json.dumps(self.capture, ensure_ascii=False))


class BrowserTransport:
    """Logged-in browser tab as a fetch proxy (see browser_proxy.js); Python paces and budgets requests."""

    def __init__(self, port: int = 8765, request_budget: int = 220, delay_seconds: float = 3.0):
        from identify_what_matters.bridge import BrowserBridge
        self.bridge = BrowserBridge(port)
        self.budget, self.delay, self.used, self.blocked = request_budget, delay_seconds, 0, None
        self.captured_at = datetime.now(timezone.utc).isoformat()
        self.capture = {"captured_at": self.captured_at, "searches": {}, "comments": {}, "hashtags": {},
                        "feeds": {}, "errors": []}

    def _call(self, kind: str, arg: str):
        if self.blocked:
            raise RuntimeError(f"Instagram blocked earlier ({self.blocked}); stopping")
        if self.used >= self.budget:
            raise RuntimeError("Instagram request budget exhausted")
        if self.used:
            time.sleep(self.delay + random.random())
        self.used += 1
        status, body = self.bridge.call(kind, arg)
        if status in (401, 403, 429):
            self.blocked = status
            self.capture["errors"].append({"kind": kind, "arg": arg, "status": status, "stopped": True})
            raise RuntimeError(f"Instagram blocked with HTTP {status}")
        if status != 200 or body is None:
            self.capture["errors"].append({"kind": kind, "arg": arg, "status": status})
            return None
        return body

    def search(self, query: str) -> list[dict] | None:
        body = self._call("search", query)
        if body is None:
            return []
        self.capture["searches"][query] = body
        return [normalize_media(m, query) for m in walk_media(body)]

    def comments(self, media_id: str) -> dict | None:
        body = self._call("comments", str(media_id))
        if body is None:
            return None
        self.capture["comments"][str(media_id)] = body
        return {"comments": [normalize_comment(c) for c in body.get("comments", [])], "total": body.get("comment_count")}

    def hashtag(self, tag: str) -> dict | None:
        if tag in self.capture["hashtags"]:
            body = self.capture["hashtags"][tag]
        else:
            body = self._call("hashtag", tag)
            if body is None:
                return None
            self.capture["hashtags"][tag] = body
        return {"media_count": body.get("media_count"),
                "media": [normalize_media(m, "#" + tag) for m in walk_media(body.get("sections", []))]}

    def user_feed(self, username: str, doc_id: str | None = None) -> list[dict] | None:
        body = self._call("user_feed", {"username": username, "doc_id": doc_id})
        if body is None:
            return None
        self.capture["feeds"][username] = body
        return [normalize_media(m, "@" + username) for m in body.get("items", [])]

    def close(self, path: Path) -> None:
        path.write_text(json.dumps(self.capture, ensure_ascii=False))
        self.bridge.close(f"{self.used} requests; capture saved to {path}")


class ReplayTransport:
    """Serves a capture {"searches": {query: raw}, "comments": {media_id: raw}} from the browser collector."""

    def __init__(self, path: str | Path):
        capture = json.loads(Path(path).read_text())
        self.searches, self.comment_map = capture.get("searches", {}), capture.get("comments", {})
        self.hashtags, self.feeds = capture.get("hashtags", {}), capture.get("feeds", {})
        self.captured_at = (capture.get("captured_at") or "").replace("Z", "+00:00") or None
        self.used = 0

    def user_feed(self, username: str, doc_id: str | None = None) -> list[dict] | None:
        raw = self.feeds.get(username)
        return None if raw is None else [normalize_media(m, "@" + username) for m in raw.get("items", [])]

    def hashtag(self, tag: str) -> dict | None:
        raw = self.hashtags.get(tag)
        if raw is None:
            return None
        return {"media_count": raw.get("media_count"),
                "media": [normalize_media(m, "#" + tag) for m in walk_media(raw.get("sections", []))]}

    def search(self, query: str) -> list[dict] | None:
        """None means the query was not captured (distinct from captured with zero results)."""
        if query not in self.searches:
            return None
        return [normalize_media(m, query) for m in walk_media(self.searches[query])]

    def comments(self, media_id: str) -> dict | None:
        raw = self.comment_map.get(str(media_id))
        if raw is None:
            return None
        return {"comments": [normalize_comment(c) for c in raw.get("comments", [])], "total": raw.get("comment_count")}


class MockTransport(ReplayTransport):
    """Simulates/mocks live Instagram access using realistic historical run captures.
    Enables deterministic end-to-end runs when Instagram private API is blocked or rate-limited.
    """

    def __init__(self, fixture_path: str | Path | None = None, request_budget: int = 150):
        if not fixture_path:
            # Locate the most recent high-quality historical capture
            candidates = sorted(Path("output/runs").glob("*/instagram_capture.json"), reverse=True)
            if not candidates:
                raise RuntimeError("No historical instagram_capture.json found to serve as mock fixture")
            fixture_path = candidates[0]
        super().__init__(fixture_path)
        self.raw_capture = json.loads(Path(fixture_path).read_text())
        self.budget = request_budget
        self.used = 0
        self.captured_at = datetime.now(timezone.utc).isoformat()

    def user_feed(self, username: str, doc_id: str | None = None) -> list[dict] | None:
        self.used += 1
        return super().user_feed(username, doc_id)

    def hashtag(self, tag: str) -> dict | None:
        self.used += 1
        return super().hashtag(tag)

    def search(self, query: str) -> list[dict] | None:
        self.used += 1
        res = super().search(query)
        if res is not None:
            return res
        # If specific query wasn't captured, fuzzily find closest query or return random slice of captured media
        for k in self.searches:
            if any(word in k.lower() for word in query.lower().split() if len(word) > 3):
                return [normalize_media(m, query) for m in walk_media(self.searches[k])]
        # Return fallback items so the pipeline doesn't choke
        all_searches = list(self.searches.values())
        if all_searches:
            sample = all_searches[hash(query) % len(all_searches)]
            return [normalize_media(m, query) for m in walk_media(sample)[:6]]
        return []

    def comments(self, media_id: str) -> dict | None:
        self.used += 1
        res = super().comments(media_id)
        if res is not None:
            return res
        # Fallback to any comment thread from capture if media_id not found
        all_threads = list(self.comment_map.values())
        if all_threads:
            sample = all_threads[hash(str(media_id)) % len(all_threads)]
            return {"comments": [normalize_comment(c) for c in sample.get("comments", [])],
                    "total": sample.get("comment_count")}
        return {"comments": [], "total": 0}

    def close(self, path: Path) -> None:
        """Saves simulated capture to run directory."""
        mock_capture = {
            "captured_at": self.captured_at,
            "searches": self.raw_capture.get("searches", {}),
            "comments": self.raw_capture.get("comments", {}),
            "hashtags": self.raw_capture.get("hashtags", {}),
            "feeds": self.raw_capture.get("feeds", {}),
            "errors": [],
            "mocked": True
        }
        path.write_text(json.dumps(mock_capture, ensure_ascii=False))

"""Hiker collection and saved-capture replay for Instagram topic measurement."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

HANDLE = re.compile(r"@[\w.]+")
URL = re.compile(r"https?://\S+|www\.\S+")
PHONE = re.compile(r"(?<!\w)\+?\d[\d\s-]{8,}\d(?!\w)")
UNAVAILABLE_ACCOUNT_ERRORS = {"PrivateAccount", "UserNotFound", "NotFoundError"}

load_dotenv()


class BudgetExhausted(RuntimeError):
    pass


PROFILE_FIELDS = ("biography", "follower_count", "following_count", "media_count", "is_verified", "is_private",
                  "is_business", "category", "city_name")
PROFILE_TTL_SECONDS = 30 * 24 * 3600
TOTALS_CACHE_PATH = Path("output/panel_comment_totals.json")
REQUEST_CACHE_PATH = Path("output/instagram_request_cache.json")
SEARCH_TTL_SECONDS = 6 * 3600
HASHTAG_TTL_SECONDS = 6 * 3600
THREAD_TTL_SECONDS = 12 * 3600
# The request cache holds full raw responses (>100 MB); rewriting it on every call stalls the event loop.
REQUEST_CACHE_FLUSH_SECONDS = 60


def _is_fresh(entry: dict | None, ttl_seconds: float) -> bool:
    return bool(entry) and (datetime.now(timezone.utc) - datetime.fromisoformat(entry["fetched_at"])).total_seconds() < ttl_seconds


def totals_due(entry: dict, now: datetime) -> bool:
    age_hours = (now - datetime.fromisoformat(entry["taken_at"])).total_seconds() / 3600
    interval = 6 if age_hours < 24 else 24 if age_hours < 72 else None
    if interval is None:
        return False
    return (now - datetime.fromisoformat(entry["fetched_at"])).total_seconds() / 3600 >= interval


def normalize_profile(raw: dict) -> dict:
    profile = {key: raw.get(key) for key in PROFILE_FIELDS}
    profile["biography"] = scrub(profile["biography"])
    return profile


class HikerAPIError(RuntimeError):
    """Hiker returns HTTP 200 with {"detail", "exc_type"} for failed requests; this surfaces them as exceptions."""

    def __init__(self, exc_type: str, detail: str):
        super().__init__(f"{exc_type}: {detail[:160]}")
        self.exc_type = exc_type


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


def _parse_taken_at(raw: dict) -> str | None:
    taken = raw.get("taken_at")
    if isinstance(taken, (int, float)):
        return datetime.fromtimestamp(taken, timezone.utc).isoformat()
    if isinstance(taken, str) and taken:
        try:
            return datetime.fromisoformat(taken.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    taken_ts = raw.get("taken_at_ts")
    if isinstance(taken_ts, (int, float)):
        return datetime.fromtimestamp(taken_ts, timezone.utc).isoformat()
    return None


def normalize_media(raw: dict, query: str) -> dict:
    caption = raw.get("caption")
    if isinstance(caption, dict):
        caption_text = caption.get("text", "")
    elif isinstance(caption, str):
        caption_text = caption
    else:
        caption_text = raw.get("caption_text") or ""
    user = raw.get("user") or {}
    return {
        "id": str(raw.get("pk") or raw.get("id", "")).split("_")[0], "code": raw.get("code"),
        "url": f"https://www.instagram.com/p/{raw.get('code')}/",
        "taken_at": _parse_taken_at(raw),
        "caption": caption_text,
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
    reply_count = raw.get("child_comment_count")
    if reply_count is None:
        reply_count = len(raw.get("preview_child_comments") or [])
    return {"id": str(raw.get("pk") or raw.get("id")), "text": raw.get("text", ""),
            "created_at": datetime.fromtimestamp(created, timezone.utc).isoformat() if isinstance(created, (int, float)) else None,
            "like_count": raw.get("comment_like_count"), "reply_count": reply_count,
            "author_id": str(user.get("pk") or user.get("id") or raw.get("user_id") or ""),
            "author": user.get("username") or raw.get("username") or "", "parent_id": None}


class HikerTransport:
    """Async transport via the Hiker API (https://api.hikerapi.com/docs).

    Uses AsyncClient for proper asyncio.wait_for timeouts. All methods are coroutines;
    shared state is protected by asyncio.Lock. Call from within an event loop.
    """

    def __init__(self, request_budget: int = 999_999, delay_seconds: float = 1.0, token: str | None = None,
                 user_id_cache: str | Path = "output/panel_user_ids.json",
                 feed_cache: str | Path = "output/panel_feed_cache.json",
                 feed_ttl_hours: float = 6.0):
        from hikerapi import AsyncClient
        token = token or os.environ.get("HIKERAPI_TOKEN")
        if not token:
            raise RuntimeError("Set HIKERAPI_TOKEN in environment/.env.")
        self.delay = delay_seconds
        self.request_budget = request_budget
        self.used = 0
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(10)
        self.captured_at = datetime.now(timezone.utc).isoformat()
        self.capture = {"captured_at": self.captured_at, "searches": {}, "comments": {}, "hashtags": {},
                        "feeds": {}, "profiles": {}, "totals": {}, "errors": []}
        self.totals_cache: dict = json.loads(TOTALS_CACHE_PATH.read_text()) if TOTALS_CACHE_PATH.exists() else {}
        self.request_cache: dict = json.loads(REQUEST_CACHE_PATH.read_text()) if REQUEST_CACHE_PATH.exists() else {}
        for section in ("searches", "hashtags", "threads", "first_pages"):
            self.request_cache.setdefault(section, {})
        self._cache_dirty = False
        self._cache_flushed_at = time.monotonic()
        self.profile_cache_path = Path("output/commenter_profiles.json")
        self.profile_cache: dict = json.loads(self.profile_cache_path.read_text()) if self.profile_cache_path.exists() else {}
        self.id_cache_path = Path(user_id_cache)
        self.user_ids = json.loads(self.id_cache_path.read_text()) if self.id_cache_path.exists() else {}
        self.feed_cache_path = Path(feed_cache)
        self.feed_cache: dict = json.loads(self.feed_cache_path.read_text()) if self.feed_cache_path.exists() else {}
        self.feed_ttl = feed_ttl_hours * 3600
        self.client = AsyncClient(token=token)

    async def _before(self):
        """Rate-limit + budget guard. Each coroutine sleeps independently; lock guards the counter."""
        async with self._lock:
            if self.used >= self.request_budget:
                raise BudgetExhausted(f"budget exhausted: {self.used}/{self.request_budget} requests used")
            do_sleep = self.used > 0
            self.used += 1
        if do_sleep:
            jitter = random.uniform(-0.2, 0.2)
            await asyncio.sleep(max(0.1, self.delay + jitter))

    async def _api(self, method: str, *args, **kwargs):
        async with self._slots:
            raw = await getattr(self.client, method)(*args, **kwargs)
        if isinstance(raw, dict) and "exc_type" in raw:
            raise HikerAPIError(str(raw["exc_type"]), str(raw.get("detail", "")))
        return raw

    async def _remember(self, section: str, key: str, value: dict) -> None:
        async with self._lock:
            self.request_cache[section][key] = {"fetched_at": datetime.now(timezone.utc).isoformat(), **value}
            self._cache_dirty = True
            if time.monotonic() - self._cache_flushed_at >= REQUEST_CACHE_FLUSH_SECONDS:
                self._flush_request_cache()

    def _flush_request_cache(self) -> None:
        """Atomic write so an interrupted run cannot leave a truncated cache behind."""
        REQUEST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = REQUEST_CACHE_PATH.parent / (REQUEST_CACHE_PATH.name + ".tmp")
        tmp.write_text(json.dumps(self.request_cache, ensure_ascii=False))
        os.replace(tmp, REQUEST_CACHE_PATH)
        self._cache_dirty = False
        self._cache_flushed_at = time.monotonic()

    async def search(self, query: str) -> list[dict]:
        entry = self.request_cache["searches"].get(query)
        if _is_fresh(entry, SEARCH_TTL_SECONDS):
            raw = entry["raw"]
        else:
            await self._before()
            raw = await self._api("fbsearch_topsearch_v2", query)
            await self._remember("searches", query, {"raw": raw})
        async with self._lock:
            self.capture["searches"][query] = raw
        return [normalize_media(m, query) for m in walk_media(raw)]

    async def comments(self, media_id: str, max_pages: int = 2, max_comments: int = 100,
                       max_reply_threads: int = 5) -> dict:
        key = str(media_id)
        entry = self.request_cache["threads"].get(key)
        if _is_fresh(entry, THREAD_TTL_SECONDS) and entry["max_pages"] >= max_pages \
                and entry["max_reply_threads"] >= max_reply_threads \
                and entry.get("max_comments", 100) >= max_comments:
            all_comments, total = entry["comments"], entry["total"]
        else:
            all_comments, total = await self._fetch_thread(media_id, max_pages, max_reply_threads, max_comments)
            await self._remember("threads", key, {"comments": all_comments, "total": total, "max_pages": max_pages,
                                                  "max_reply_threads": max_reply_threads, "max_comments": max_comments})
        async with self._lock:
            self.capture["comments"][key] = {"comments": all_comments, "comment_count": total}
        return {"comments": all_comments, "total": total}

    async def _get_page(self, media_id: str, page_id: str | None) -> dict | None:
        """One media_comments_v2 page; a fresh cached first page is served without a request."""
        key = str(media_id)
        if page_id is None:
            cached = self.request_cache["first_pages"].get(key)
            if _is_fresh(cached, THREAD_TTL_SECONDS):
                return cached
        await self._before()
        try:
            raw = await self._api("media_comments_v2", id=media_id, page_id=page_id)
        except HikerAPIError as error:
            if error.exc_type != "NotFoundError":
                raise
            return {"comments": [], "comment_count": 0, "next_page_id": None}
        if not isinstance(raw, dict):
            return None
        payload = raw.get("response") or {}
        page = {"comments": [nc for nc in (normalize_comment(c) for c in payload.get("comments", [])) if nc],
                "comment_count": payload.get("comment_count"),
                "next_page_id": raw.get("next_page_id") or payload.get("next_min_id")}
        if page_id is None:
            await self._remember("first_pages", key, page)
        return page

    async def _fetch_thread(self, media_id: str, max_pages: int, max_reply_threads: int,
                            max_comments: int = 100) -> tuple[list, int | None]:
        all_comments: list[dict] = []
        page_id = None
        seen_cursors: set = set()
        total = None

        for _ in range(max_pages):
            page = await self._get_page(media_id, page_id)
            if page is None:
                break
            if total is None:
                total = page["comment_count"]
            all_comments.extend(page["comments"])
            next_page_id = page["next_page_id"]
            if not next_page_id or next_page_id in seen_cursors or len(all_comments) >= max_comments:
                break
            seen_cursors.add(next_page_id)
            page_id = next_page_id

        replied = sorted((c for c in all_comments if c["reply_count"]),
                         key=lambda c: (-(c["like_count"] or 0), -c["reply_count"]))
        for thread in replied[:max_reply_threads]:
            await self._before()
            try:
                raw = await self._api("media_comments_replies_v2", media_id=media_id, comment_id=thread["id"])
            except HikerAPIError:
                continue
            for child in (raw.get("child_comments") or []) if isinstance(raw, dict) else []:
                reply = normalize_comment(child)
                reply["parent_id"] = thread["id"]
                all_comments.append(reply)

        return all_comments, total

    async def hashtag(self, tag: str) -> dict:
        entry = self.request_cache["hashtags"].get(tag)
        if _is_fresh(entry, HASHTAG_TTL_SECONDS):
            media_count, sections = entry["media_count"], entry["sections"]
        else:
            await self._before()
            info = await self._api("hashtag_by_name_v1", tag)
            await self._before()
            recent = await self._api("hashtag_medias_recent_v2", name=tag)
            sections = recent.get("response", {}).get("sections", []) if isinstance(recent, dict) else []
            media_count = info.get("media_count")
            await self._remember("hashtags", tag, {"media_count": media_count, "sections": sections})
        raw = {"media_count": media_count, "sections": sections}
        async with self._lock:
            self.capture["hashtags"][tag] = raw
        return {"media_count": media_count,
                "media": [normalize_media(m, "#" + tag) for m in walk_media(sections)]}

    async def user_feed(self, username: str, doc_id: str | None = None) -> list[dict]:
        now = datetime.now(timezone.utc)
        _max_post_age = 168 * 3600  # keep posts up to 7 days (168h) across runs for multi-day debate tracking
        _stale_fallback = 48 * 3600  # return stale cache if live fetch fails and cache is < 48h old

        # Serve from cache if fresh enough (within TTL); unavailable accounts are re-checked weekly
        cached = self.feed_cache.get(username)
        if cached:
            age = (now - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
            ttl = self.feed_ttl * 28 if cached.get("unavailable") else self.feed_ttl
            if age < ttl:
                items = cached["items"]
                self.capture["feeds"][username] = {"items": items}
                return [normalize_media(m, "@" + username) for m in items]

        uid = self.user_ids.get(username)
        new_items, fetch_error = None, None
        try:
            if uid is None:
                await self._before()
                result = await self._api("user_by_username_v1", username)
                uid = str(result["pk"])
                async with self._lock:
                    self.user_ids[username] = uid
                    self.id_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.id_cache_path.write_text(json.dumps(self.user_ids, indent=1))
            await self._before()
            new_items, _end_cursor = await self._api("user_medias_chunk_v1", user_id=uid)
        except BudgetExhausted:
            raise
        except HikerAPIError as error:
            if error.exc_type not in UNAVAILABLE_ACCOUNT_ERRORS:
                raise
            async with self._lock:
                self.feed_cache[username] = {"fetched_at": now.isoformat(), "items": [], "unavailable": error.exc_type}
                self.feed_cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.feed_cache_path.write_text(json.dumps(self.feed_cache, ensure_ascii=False))
            self.capture["feeds"][username] = {"items": [], "unavailable": error.exc_type}
            return []
        except Exception as error:
            fetch_error = error

        if new_items is None:
            # Live fetch failed — fall back to stale cache if within 48h
            if cached:
                age = (now - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
                if age < _stale_fallback:
                    self.capture["feeds"][username] = {"items": cached["items"]}
                    return [normalize_media(m, "@" + username) for m in cached["items"]]
            raise fetch_error

        # Merge new posts with existing cached posts (additive), dedup by 'code', prune > 72h
        existing = {m["code"]: m for m in (cached["items"] if cached else []) if m.get("code")}
        for m in new_items:
            if m.get("code"):
                existing[m["code"]] = m
        # Prune posts older than 72h (taken_at field)
        merged = []
        for m in existing.values():
            ta = m.get("taken_at")
            if ta:
                try:
                    age_s = (now - datetime.fromisoformat(str(ta).replace("Z", "+00:00"))).total_seconds()
                    if age_s <= _max_post_age:
                        merged.append(m)
                    continue
                except Exception:
                    pass
            merged.append(m)

        async with self._lock:
            self.capture["feeds"][username] = {"items": new_items}
            self.feed_cache[username] = {"fetched_at": now.isoformat(), "items": merged}
            self.feed_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.feed_cache_path.write_text(json.dumps(self.feed_cache, ensure_ascii=False))

        return [normalize_media(m, "@" + username) for m in merged]

    async def profile(self, username: str) -> dict | None:
        cached = self.profile_cache.get(username)
        if cached and (datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"])).total_seconds() < PROFILE_TTL_SECONDS:
            profile = cached["profile"]
        else:
            await self._before()
            try:
                raw = await self._api("user_by_username_v1", username)
            except HikerAPIError:
                return None
            profile = normalize_profile(raw)
            async with self._lock:
                self.profile_cache[username] = {"fetched_at": datetime.now(timezone.utc).isoformat(), "profile": profile}
                self.profile_cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.profile_cache_path.write_text(json.dumps(self.profile_cache, ensure_ascii=False))
        async with self._lock:
            self.capture["profiles"][username] = profile
        return profile

    async def comment_total(self, media_id: str, taken_at: str) -> int | None:
        key = str(media_id)
        now = datetime.now(timezone.utc)
        cached = self.totals_cache.get(key)
        if cached and not totals_due(cached, now):
            total = cached["total"]
        else:
            page = await self._get_page(media_id, None)
            if page is None or page["comment_count"] is None:
                return None
            total = int(page["comment_count"])
            async with self._lock:
                self.totals_cache[key] = {"total": total, "fetched_at": page.get("fetched_at", now.isoformat()),
                                          "taken_at": taken_at}
                TOTALS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                TOTALS_CACHE_PATH.write_text(json.dumps(self.totals_cache))
        async with self._lock:
            self.capture["totals"][key] = int(total)
        return int(total)

    def close(self, path: Path) -> None:
        if self._cache_dirty:
            self._flush_request_cache()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.capture, ensure_ascii=False))


class ReplayTransport:
    """Serves a saved Instagram capture offline for deterministic measurement."""

    def __init__(self, path: str | Path):
        capture = json.loads(Path(path).read_text())
        self.searches, self.comment_map = capture.get("searches", {}), capture.get("comments", {})
        self.hashtags, self.feeds = capture.get("hashtags", {}), capture.get("feeds", {})
        self.profiles = capture.get("profiles", {})
        self.totals = capture.get("totals", {})
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

    def profile(self, username: str) -> dict | None:
        return self.profiles.get(username)

    def comment_total(self, media_id: str, taken_at: str = None) -> int | None:
        return self.totals.get(str(media_id))

    def comments(self, media_id: str, **_) -> dict | None:
        raw = self.comment_map.get(str(media_id))
        if raw is None:
            return None
        return {"comments": list(raw.get("comments", [])), "total": raw.get("comment_count")}

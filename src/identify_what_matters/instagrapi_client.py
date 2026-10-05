"""Opt-in, bounded public-content collection; importing this module makes no requests."""

import argparse
import json
import logging
import math
import os
import re
import stat
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from instagrapi import Client
from instagrapi.exceptions import CommentsDisabled, NotFoundError, PrivateAccount

# Errors about one account/post/tag (private, deleted, comments off); the session itself is still healthy.
ITEM_ERRORS = (PrivateAccount, NotFoundError, CommentsDisabled)
MAX_ITEM_ERROR_STREAK = 3


class CollectionAborted(RuntimeError):
    """Collection failed; no partial result should be treated as a successful run."""


class BudgetExhausted(RuntimeError):
    pass


class ItemUnavailable(RuntimeError):
    """One account, post or hashtag cannot be read; skip it and continue the run."""


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _timestamp(value: Any) -> str | None:
    try:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat() if value.tzinfo else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc).isoformat() if parsed.tzinfo else None
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _secure_settings(file_path: Path) -> dict:
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise CollectionAborted("Settings must be an owner-only regular file (chmod 600).")
            settings = json.load(stream)
        if not isinstance(settings, dict):
            raise ValueError
        return settings
    except CollectionAborted:
        raise
    except Exception:
        raise CollectionAborted("Cannot read secure local settings JSON.") from None


class _FailClosedClient(Client):
    """Prevent library retry/fallback paths from hiding transport failures."""

    def __init__(self, settings: dict, before_request: Any):
        self.failure = None
        self.item_error_streak = 0
        self.before_request = before_request
        self.raw_medias = {}
        self.raw_users = {}
        quiet_logger = logging.Logger("bounded_instagrapi", level=logging.CRITICAL + 1)
        quiet_logger.addHandler(logging.NullHandler())
        settings = deepcopy(settings)
        settings.update({
            "session_retry_total": 0,
            "session_retry_statuses": [],
            "public_request_retries_count": 1,
            "public_request_retries_timeout": 0,
            "private_transport": "requests",
            "public_transport": "requests",
        })
        super().__init__(settings=settings, logger=quiet_logger, delay_range=None,
                         session_retry_total=0, session_retry_statuses=[],
                         public_request_retries_count=1, private_transport="requests",
                         public_transport="requests")
        self.handle_exception = self._raise_failure
        self.private_request_logger = quiet_logger
        self.private.trust_env = False
        self.public.trust_env = False
        self.set_proxy(None)

    @staticmethod
    def _raise_failure(client: Any, error: Exception) -> None:
        raise error

    def _remember_raw(self, value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            identifier = value.get("pk") or value.get("id")
            if identifier is not None:
                identifier = str(identifier).split("_")[0]
                if key == "user":
                    self.raw_users[identifier] = {
                        "is_private": value.get("is_private"),
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                    }
                if "media_type" in value or "code" in value:
                    self.raw_medias[identifier] = {
                        name: value.get(name) for name in
                        ("taken_at", "like_count", "comment_count", "view_count", "video_view_count",
                         "play_count", "video_play_count", "like_and_view_counts_disabled")
                    }
            for child_key, child in value.items():
                self._remember_raw(child, child_key)
        elif isinstance(value, list):
            for child in value:
                self._remember_raw(child, key)

    def _send_private_request(self, endpoint: str, **kwargs: Any) -> Any:
        if self.failure:
            raise self.failure
        try:
            self.before_request()
            result = super()._send_private_request(endpoint, **kwargs)
        except BudgetExhausted as error:
            self.failure = error
            raise
        except ITEM_ERRORS as error:
            # Instagram also answers soft blocks with "Not authorized to view user", so a streak is treated as a block.
            self.item_error_streak += 1
            if self.item_error_streak > MAX_ITEM_ERROR_STREAK:
                self.failure = CollectionAborted(
                    f"Instagram collection blocked: {self.item_error_streak} consecutive {type(error).__name__} errors.")
                raise self.failure from None
            raise ItemUnavailable(f"{type(error).__name__} on {endpoint}") from None
        except Exception as error:
            self.failure = CollectionAborted(
                f"Instagram collection blocked by {type(error).__name__}; stopped without retry or challenge resolution."
            )
            raise self.failure from None
        self.item_error_streak = 0
        self._remember_raw(self.last_json)
        return result

    def user_info_v1(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().user_info_v1(*args, **kwargs)
        except (CollectionAborted, BudgetExhausted):
            raise
        except Exception as error:
            self.failure = CollectionAborted(
                f"Profile validation failed with {type(error).__name__}; stopped without alternate authentication probes."
            )
            raise self.failure from None


class InstagrapiCollector:
    """One-shot collector. Inject a fake client and sleeper for offline verification."""

    def __init__(self, config: dict, client: Any = None, settings_file: str | Path | None = None,
                 sleeper: Any = time.sleep, clock: Any = None):
        self.config = deepcopy(config)
        self.client = client
        self.settings_file = Path(settings_file) if settings_file else None
        self.sleeper = sleeper
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.calls = 0
        self.private_requests = 0
        self.started = False
        self.public_users = {}
        self.source_observations = []
        self.limits = self.config["limits"]
        bounds = {
            "request_budget": (1, 100), "max_seeds": (1, 24), "posts_per_seed": (1, 12),
            "recent_posts_per_seed": (1, 12), "top_posts_per_seed": (0, 4),
            "comment_posts": (0, 12), "comment_pages_per_post": (1, 3),
            "comments_per_post": (1, 40), "reply_parents_per_post": (0, 3),
            "reply_pages_per_parent": (1, 2), "replies_per_parent": (1, 10),
            "recent_days": (1, 30),
        }
        for name, (minimum, maximum) in bounds.items():
            value = self.limits[name]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        delay = self.limits["delay_seconds"]
        if not isinstance(delay, (int, float)) or not math.isfinite(delay) or not 5 <= delay <= 120:
            raise ValueError("delay_seconds must be between 5 and 120")
        fraction = self.config["discovery"]["exploration_fraction"]
        if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 0.5:
            raise ValueError("exploration_fraction must be between 0 and 0.5")

    def _now(self) -> str:
        value = _timestamp(self.clock())
        if value is None:
            raise ValueError("clock must return a timezone-aware timestamp")
        return value

    def _before_request(self) -> None:
        if self.private_requests >= self.limits["request_budget"]:
            raise BudgetExhausted("Private request budget exhausted")
        if self.private_requests:
            self.sleeper(self.limits["delay_seconds"])
        self.private_requests += 1

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.calls >= self.limits["request_budget"]:
            raise BudgetExhausted("Client method-call budget exhausted")
        if self.calls and not isinstance(self.client, _FailClosedClient):
            self.sleeper(self.limits["delay_seconds"])
        self.calls += 1
        try:
            result = getattr(self.client, method)(*args, **kwargs)
            if getattr(self.client, "failure", None):
                raise self.client.failure
            return result
        except (BudgetExhausted, CollectionAborted):
            raise
        except Exception as error:
            raise CollectionAborted(
                f"Collection stopped at {method}: {type(error).__name__}; no retry attempted."
            ) from None

    @staticmethod
    def _evidence(seed: dict) -> dict:
        evidence = seed.get("india_evidence") or {}
        if (seed["type"] == "user" and evidence.get("status") == "human_audited"
                and evidence.get("reference") and evidence.get("audited_by")
                and _timestamp(evidence.get("audited_at"))):
            return deepcopy(evidence)
        return {"status": "unknown", "reason": "Hashtag, language, or unaudited panel is not geography evidence."}

    def _is_public(self, user: Any) -> bool:
        identifier = str(_get(user, "pk") or "")
        if not identifier:
            return False
        if identifier not in self.public_users:
            profile = self._call("user_info_v1", identifier)
            raw_users = getattr(self.client, "raw_users", None)
            flag = raw_users.get(identifier, {}).get("is_private") if raw_users is not None else _get(profile, "is_private")
            self.public_users[identifier] = flag is False
        return self.public_users[identifier]

    def _post(self, media: Any, source: dict) -> dict:
        identifier = str(_get(media, "pk") or _get(media, "id"))
        raw_medias = getattr(self.client, "raw_medias", None)
        raw = raw_medias.get(identifier.split("_")[0], {}) if raw_medias is not None else media
        caption = _get(media, "caption_text", _get(media, "caption", ""))
        if isinstance(caption, dict):
            caption = caption.get("text", "")
        hidden = _get(raw, "like_and_view_counts_disabled") is True
        return {
            "id": identifier, "code": _get(media, "code"), "shortcode": _get(media, "code"),
            "caption": caption or "", "taken_at": _timestamp(_get(raw, "taken_at")),
            "username": _get(_get(media, "user"), "username"),
            "user_id": str(_get(_get(media, "user"), "pk") or "") or None,
            "like_count": None if hidden else _count(_get(raw, "like_count")),
            "comment_count": _count(_get(raw, "comment_count")),
            "view_count": None if hidden else _count(_get(raw, "view_count") if _get(raw, "view_count") is not None else _get(raw, "video_view_count")),
            "play_count": None if hidden else _count(_get(raw, "play_count") if _get(raw, "play_count") is not None else _get(raw, "video_play_count")),
            "is_video": _get(media, "media_type") == 2,
            "observed_at": source["observed_at"], "sources": [source],
            "india_evidence": [source["india_evidence"]], "geography": "unknown",
            "public_status": "verified_public_at_collection", "comments": [],
            "comments_sampling": {"status": "not_sampled"},
        }

    def _comment(self, comment: Any, parent: str | None = None) -> dict | None:
        identifier = _get(comment, "pk", _get(comment, "id"))
        if identifier is None:
            return None
        return {
            "id": str(identifier), "text": _get(comment, "text", "") or "",
            "created_at": _timestamp(_get(comment, "created_at_utc", _get(comment, "created_at"))),
            "like_count": _count(_get(comment, "like_count", _get(comment, "comment_like_count"))),
            "parent_comment_id": parent or _get(comment, "replied_to_comment_id"),
            "user_id": str(_get(_get(comment, "user"), "pk") or "") or None,
            "observed_at": self._now(),
        }

    def _comments(self, post: dict, reason: str) -> None:
        sampling = post["comments_sampling"] = {
            "status": "sampled", "selection": reason, "server_order": "unspecified_not_guaranteed_chronological",
            "method": "media_comments_v1_chunk", "pages": [], "reply_pages": [],
        }
        comments = {}
        min_cursor, max_cursor = "", ""
        visited = set()
        for page in range(self.limits["comment_pages_per_post"]):
            cursor = (min_cursor, max_cursor)
            if cursor in visited:
                break
            visited.add(cursor)
            items, next_min, next_max = self._call(
                "media_comments_v1_chunk", post["id"], min_id=min_cursor, max_id=max_cursor
            )
            sampling["pages"].append({"page": page + 1, "min_id": min_cursor, "max_id": max_cursor,
                                      "next_min_id": next_min, "next_max_id": next_max, "observed_at": self._now()})
            for item in items:
                comment = self._comment(item)
                if comment and len(comments) < self.limits["comments_per_post"]:
                    comments.setdefault(comment["id"], comment)
            post["comments"] = list(comments.values())
            if not items or len(comments) >= self.limits["comments_per_post"]:
                break
            min_cursor, max_cursor = (next_min or "", "") if next_min else ("", next_max or "")
            if not min_cursor and not max_cursor:
                break
        parents = sorted(comments.values(), key=lambda item: item["created_at"] or "", reverse=True)
        for parent in parents[:self.limits["reply_parents_per_post"]]:
            replies = {}
            cursor = None
            visited = set()
            for page in range(self.limits["reply_pages_per_parent"]):
                if cursor in visited:
                    break
                visited.add(cursor)
                items, next_cursor = self._call(
                    "media_comment_replies_chunk", post["id"], parent["id"],
                    max_amount=self.limits["replies_per_parent"] - len(replies), min_id=cursor
                )
                sampling["reply_pages"].append({"parent_comment_id": parent["id"], "page": page + 1,
                                               "min_id": cursor, "next_min_id": next_cursor,
                                               "method": "media_comment_replies_chunk", "observed_at": self._now()})
                for item in items:
                    reply = self._comment(item, parent["id"])
                    if reply and len(replies) < self.limits["replies_per_parent"]:
                        replies.setdefault(reply["id"], reply)
                parent["replies"] = list(replies.values())
                if not items or not next_cursor or len(replies) >= self.limits["replies_per_parent"]:
                    break
                cursor = next_cursor

    def _proposals(self, posts: list[dict], now: datetime) -> list[dict]:
        known = {tag.casefold() for domain in self.config["domains"] for tag in domain["hashtags"]}
        generic = {tag.casefold() for tag in self.config["discovery"]["generic_hashtags"]}
        candidates = defaultdict(lambda: {"post_ids": set(), "domains": set(), "india_evidence": []})
        cutoff = now - timedelta(days=self.limits["recent_days"])
        for post in posts:
            if not post["taken_at"] or not cutoff <= datetime.fromisoformat(post["taken_at"]) <= now:
                continue
            recent = [source for source in post["sources"] if source["source_type"] in ("hashtag_recent", "user_feed")]
            if not recent:
                continue
            for tag in set(re.findall(r"(?<!\w)#(\w+)", post["caption"])):
                tag = tag.casefold()
                if tag in known or tag in generic or len(tag) < 3 or tag.isdigit():
                    continue
                candidate = candidates[tag]
                candidate["post_ids"].add(post["id"])
                candidate["domains"].update(source["domain"] for source in recent)
                candidate["india_evidence"] = post["india_evidence"]
        return [
            {"hashtag": tag, "supporting_post_ids": sorted(value["post_ids"]),
             "domains": sorted(value["domains"]), "india_evidence": value["india_evidence"],
             "status": "proposal_requires_human_review", "auto_collected": False}
            for tag, value in sorted(candidates.items(), key=lambda entry: (-len(entry[1]["post_ids"]), entry[0]))[:20]
        ]

    def collect(self, seed_offset: int = 0) -> dict:
        if self.started:
            raise CollectionAborted("Collector is one-shot; login will not be attempted again.")
        if type(seed_offset) is not int or seed_offset < 0:
            raise ValueError("seed_offset must be a nonnegative integer")
        self.started = True
        started_at = self._now()
        settings = _secure_settings(self.settings_file) if self.settings_file else {}
        session = os.environ.get("INSTAGRAM_SESSION_ID") or settings.get("cookies", {}).get("sessionid") or settings.get("authorization_data", {}).get("sessionid")
        if not isinstance(session, str) or not session:
            raise CollectionAborted("Set INSTAGRAM_SESSION_ID locally or provide a secure settings file containing a sessionid.")
        # Browser cookie stores are URL-encoded (%3A); the private API header needs the raw value.
        session = unquote(session)
        if self.client is None:
            self.client = _FailClosedClient(settings, self._before_request)
        if self._call("login_by_sessionid", session) is not True:
            raise CollectionAborted("Session login failed; no second login attempted.")
        seeds = []
        domains = self.config["domains"]
        for index in range(max((len(domain["hashtags"]) for domain in domains), default=0)):
            for domain in domains:
                if index < len(domain["hashtags"]):
                    seeds.append({"type": "hashtag", "query": domain["hashtags"][index], "domain": domain["name"]})
                if index == 0:
                    seeds.extend(dict(panel, type="user", domain=domain["name"])
                                 for panel in domain.get("panels", []) if panel.get("enabled") is True)
        offset = seed_offset % len(seeds) if seeds else 0
        selected = (seeds[offset:] + seeds[:offset])[:self.limits["max_seeds"]]
        posts = {}
        processed = 0
        exhausted = False
        try:
            for seed in selected:
                accepted = 0
                profile = None
                if seed["type"] == "user":
                    profile = self._call("user_info_by_username_v1", seed["query"])
                    if not self._is_public(profile):
                        self.source_observations.append({"source_type": "user_feed", "query": seed["query"],
                                                         "status": "skipped_private_or_unknown", "observed_at": self._now()})
                        processed += 1
                        continue
                modes = [("feed", self.limits["posts_per_seed"])] if profile is not None else [
                    ("recent", self.limits["recent_posts_per_seed"]), ("top", self.limits["top_posts_per_seed"])
                ]
                for mode, cap in modes:
                    cap = min(cap, self.limits["posts_per_seed"] - accepted)
                    if cap <= 0:
                        continue
                    if profile is not None:
                        method = "user_medias_paginated_v1"
                        items, cursor = self._call(method, str(_get(profile, "pk")), amount=cap, end_cursor="")
                    else:
                        method = "hashtag_medias_v1_chunk"
                        items, cursor = self._call(method, seed["query"], max_amount=cap, tab_key=mode, max_id=None)
                    source = {"source_type": "user_feed" if profile is not None else f"hashtag_{mode}",
                              "query": seed["query"], "method": method, "domain": seed["domain"],
                              "observed_at": self._now(), "next_cursor": cursor, "india_evidence": self._evidence(seed)}
                    self.source_observations.append(dict(source, returned_count=len(items), status="observed"))
                    for media in items[:cap]:
                        if not (_get(media, "pk") or _get(media, "id")) or not self._is_public(_get(media, "user")):
                            continue
                        post = self._post(media, deepcopy(source))
                        if post["id"] in posts:
                            posts[post["id"]]["sources"].append(post["sources"][0])
                            posts[post["id"]]["india_evidence"].append(source["india_evidence"])
                        else:
                            posts[post["id"]] = post
                        accepted += 1
                processed += 1
            now = datetime.fromisoformat(self._now())
            recent = sorted(
                (post for post in posts.values() if post["taken_at"] and
                 now - timedelta(days=self.limits["recent_days"]) <= datetime.fromisoformat(post["taken_at"]) <= now and
                 any(source["source_type"] != "hashtag_top" for source in post["sources"])),
                key=lambda post: post["taken_at"], reverse=True,
            )
            low = sorted(recent, key=lambda post: (post["like_count"] is None, post["like_count"] or 0))
            sampled = set()
            for index in range(len(recent)):
                for pool, reason in ((recent, "recent_public_post"), (low, "low_engagement_recent_public_post")):
                    post = pool[index]
                    if post["id"] in sampled or len(sampled) >= self.limits["comment_posts"]:
                        continue
                    sampled.add(post["id"])
                    self._comments(post, reason)
        except BudgetExhausted:
            exhausted = True
            for post in posts.values():
                if post["comments_sampling"]["status"] == "sampled":
                    post["comments_sampling"]["status"] = "possibly_truncated_by_run_budget"
        finished_at = self._now()
        return {
            "posts": list(posts.values()), "source_observations": self.source_observations,
            "run": {"status": "budget_exhausted" if exhausted else "completed", "started_at": started_at,
                    "finished_at": finished_at, "method_calls_including_login": self.calls,
                    "private_request_attempts": self.private_requests if isinstance(self.client, _FailClosedClient) else None,
                    "request_budget": self.limits["request_budget"], "selected_seeds": len(selected),
                    "completed_seeds": processed, "next_seed_offset": (offset + max(processed, 1)) % len(seeds) if seeds else 0,
                    "budget_note": "Caps client method calls (login included) and private request attempts independently. The library may internally issue extra HTTP requests; this is not an exact wire-request guarantee."},
            "discovery": {"candidates": self._proposals(list(posts.values()), datetime.fromisoformat(finished_at)),
                          "exploration_fraction": self.config["discovery"]["exploration_fraction"],
                          "proposed_exploration_seed_slots": math.floor(self.limits["max_seeds"] * self.config["discovery"]["exploration_fraction"]),
                          "policy": "Review proposals before adding seeds; rotate next_seed_offset across runs for domain coverage. No automatic exploration requests."},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in bounded instagrapi collection of public content.")
    parser.add_argument("--config", type=Path, default=Path("config/discovery.json"))
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/instagrapi"))
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--collect", action="store_true", help="Explicitly permit login and Instagram requests.")
    args = parser.parse_args()
    if not args.collect:
        parser.error("No network access without explicit --collect. This command performs session login when enabled.")
    try:
        with args.config.open() as stream:
            config = json.load(stream)
        result = InstagrapiCollector(config, settings_file=args.settings_file).collect(args.seed_offset)
    except (CollectionAborted, BudgetExhausted) as error:
        raise SystemExit(str(error)) from None
    except (ValueError, KeyError, OSError, TypeError):
        raise SystemExit("Collection aborted. Check local configuration/session permissions; no retry or output was performed.") from None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "posts.json").write_text(json.dumps(result["posts"], indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "collection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["run"], indent=2))


if __name__ == "__main__":
    main()
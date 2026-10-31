"""Public India trend signals: Google Trends (searches), Google News (coverage), X trends (debate), Reddit (discussion).

Every fetch is independent; an unreachable source is recorded in `availability`, never fatal.
"""

from __future__ import annotations

import html
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Sec-CH-UA": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}
HEADERS = BROWSER_HEADERS

REDDIT_HEADERS = {
    "Host": "www.reddit.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Connection": "keep-alive",
    "Cookie": """edgebucket=AoQG1CTI7ePkpnsjQj; loid=00000000004c82x0fk.2.1565431000685.Z0FBQUFBQnF3bTJVT1J5Q1BDMXo4c2RhZXc1ZmhxaEthWEVwa1pXWlF4OWRFOXhUREhwT1Y3T0RzZm5RNHZKN3ZVV0JIT3RtOU11VEFUZzE5ejdlMDZpdzJJeGh1RFJyOFg5ZDg3OUFxSkdLWURMMUl5SjVYODdyN05yQllnNHd4V184US1Pa2hsNGo; session_tracker=rkgqlnrgopnneragrn.0.1791127191428.Z0FBQUFBQnF3bTZZNUJTR096RU1HM0hnWVNNNlZUN3I5c3lxajM1WEpTZGMycjhBRnBtQWJlQ2p2VzdzaE5NNGxsRmtORU1iSmZjWjNzNkNOODNCTXlUSlM0MVBjQmRaU0ZEMlRCQ1dUMFRjZlM3Zll1UjJEeDRCWEJQQ1FzcVk0c09qWEtVeERSSEk; csrf_token=4c08ff642e145c307bb232dbc9aafe14; csv=2; ads_cookie=1; reddit_supported_media_codecs=video/avc%2Cvideo/vp9; g_state={"i_l":2,"i_ll":1791126932491,"i_b":"ncICelz88H41YbWF+YU6XRGmdRqM3aHLnBNfKET5+AA","i_e":{"enable_itp_optimization":24},"i_et":1791126932491,"i_p":1791213313861}; token_v2=eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpzS3dsMnlsV0VtMjVmcXhwTU40cWY4MXE2OWFFdWFyMnpLMUdhVGxjdWNZIiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL3d3dy5yZWRkaXQuY29tIiwic3ViIjoidXNlciIsImV4cCI6MTc5MTIxMzMzMi4wNDk3MjYsImlhdCI6MTc5MTEyNjkzMi4wNDk3MjYsImp0aSI6Ilk0MTN5UnMtbDJOcENuUWdwSlYxVWFzb3Jkc3FkZyIsImNpZCI6IjBSLVdBTWh1b28tTXlRIiwibGlkIjoidDJfNGM4MngwZmsiLCJhaWQiOiJ0Ml80YzgyeDBmayIsImF0IjoxLCJsY2EiOjE1NjU0MzEwMDA2ODUsInNjcCI6ImVKeGtrVkhPM0RBSWhPX0NzMF9RcTFSVjVkaGtGelUyRWVDczl2WVZXWk9OOUw4TjhHVU1rOS1RcTJJbGctUktNRmRJZ0MzVEJnbDJJdS1Yd3FPZlJPODhlc0dHM1JRU0ZNRmE2WlRUWTkweUNTUjRraHJMR3hKUXhXNWtMamM2c09XZUh3Z0pHdGZDZmFYSHBVMW9HY2JpZG8xcldEV3VHMDlxN3RXNGRyYnB3dmJFLUdabnRTa1Z0X1dqVFBLNlV2a1VMX3JuTjdXM2p1VzdfUzUwWk1PR3F2bUIzcGxKQ080c2Zycm13OTlUazFGc0NGYTE5M2FTT3BaMjNxNWowU0swT0hkODF2UFhaakl1cDJsNVp2dDExeThodzF1ZTF6RHFBR2FzTVk4eXhwRjF6Szg2Z01ZVkpSdExFTjlHSUZjdWZfMlhZTDgyX1RtNGZlS0g3MGJjOVliZm1vRjZMa0djLWlWa0NIXy1Ed0FRMWVUUiIsInJjaWQiOiItVk85RTlTWEluNTBHZ2k3dDhVOUF5NE0tdG9Cdk9ZM2xiRUx5Q3lDNy1nIiwiZmxvIjoyfQ.ZQSOKbEx2H78CYZF07hG6vdFt0rxHNYwD24Osljr6Wp0k9z5pSYsxvTj_X17fBmkYP2nTflucWGarq64pb34MhJOWmDWQ41PN_k9TbEGWszsusKY1yyMxxhvxOLHPAW2vWYYzfeq3PEum4tidkC-a0Ni7tQXFnHVWFrbSfhxMTZvfJM72KHUmKLsDSdwgpyQaFlxnaz7YTURYlMVjjZnZ86IfhQttzOIelv3D1fiR50vlI2UTuFvvUfx2s11O9n6avUleUzKcJSGD7VrnIkE8N4kalx5NrcTI4oQmuUzVHR0xguzo2Kpyr2SL1Rhf5wCQfrVIL_t7nfRDyF8D54sXw; reddit_session=eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpsVFdYNlFVUEloWktaRG1rR0pVd1gvdWNFK01BSjBYRE12RU1kNzVxTXQ4IiwidHlwIjoiSldUIn0.eyJzdWIiOiJ0Ml80YzgyeDBmayIsImV4cCI6MTgwNjc2NTM0My44MzUyNjEsImlhdCI6MTc5MTEyNjk0My44MzUyNjEsImp0aSI6InBTNnlLVjhtaXVoazRfeU9FNEkySnFUelRVZ1ZhQSIsImF0IjoxLCJjaWQiOiJjb29raWUiLCJsY2EiOjE1NjU0MzEwMDA2ODUsInNjcCI6ImVKd0FBZ0Q5XzF0ZEF3QUJGUUM1IiwiZmxvIjoyLCJhbXIiOlsicHdkIl19.akuoMOaoIyYnUCbiUsimm4DFwQvCnqkC5c9_2fhn7fy8bD0gS6z2zMKPFcMpmgs9j2x1IZdHRPZ_XNX3vtCUgo4uNIBxXAAqAuFvNvBw478otwkD_ACBW1ji3z3grbCDIiwpvRXohjVeND8czSOah93-uAqbnA0AEETQ_Sr4i05Qo2h-lBFpMVIiBUA_qSjWClbSIHr925mtGfYSdfkxKmDmn3XJ4K9qNAJDEiluypWyRL1v8JUL6nOvx9vkvVwI-8jTICmEpouLyIfO_9TF3ExG_LYL0X_fzNaOuaQQbPrLRzmVzKDflxUrMnPjDTbgPfsvpjhCBgAsCnI10wZRNQ; reddit_chat_view=closed; pc=0l""",
    "Sec-GPC": "1",
    "Priority": "u=0, i",
}



HT = "{https://trends.google.com/trending/rss}"
NEWS_LANGUAGES = ("en-IN", "hi", "ta", "te", "bn", "mr", "ml", "kn", "gu", "pa")
NEWS_SECTIONS = ("NATION", "BUSINESS", "TECHNOLOGY", "SCIENCE", "HEALTH")
NEWS_SEARCHES = (
    "sparks outrage",
    "protest",
    "policy reform",
    "inflation",
    "rbi monetary policy",
    "supreme court",
    "union budget",
    "election commission",
    "economic growth",
)
REDDIT_SUBS = (
    "india",
    "IndiaSpeaks",
    "unitedstatesofindia",
    "IndianModerate",
    "indiadiscussion",
    "geopoliticsIndia",
    "IndiaInvestments",
    "IndianStockMarket",
    "IndianStreetBets",
)


def _time(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _traffic(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"([\d,]+)\s*([KkMm]?)", value.replace("+", ""))
    if not match:
        return None
    number = int(match.group(1).replace(",", ""))
    return number * {"k": 1_000, "m": 1_000_000}.get(match.group(2).lower(), 1)


def google_trends(geo: str = "IN") -> list[dict]:
    response = requests.get(f"https://trends.google.com/trending/rss?geo={geo}", headers=BROWSER_HEADERS, timeout=20)
    response.raise_for_status()
    signals = []
    for item in ET.fromstring(response.content).iter("item"):
        news = [{"title": html.unescape(n.findtext(f"{HT}news_item_title") or ""),
                 "source": n.findtext(f"{HT}news_item_source"), "url": n.findtext(f"{HT}news_item_url")}
                for n in item.findall(f"{HT}news_item")]
        signals.append({
            "source": "google_trends", "feed": f"trends_{geo}", "title": item.findtext("title") or "",
            "published_at": _time(item.findtext("pubDate")), "traffic": _traffic(item.findtext(f"{HT}approx_traffic")),
            "related": [entry["title"] for entry in news if entry["title"]], "url": news[0]["url"] if news else None,
            "publisher": news[0]["source"] if news else None,
        })
    return signals


def google_news(url: str, feed: str, language: str) -> list[dict]:
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=20)
    response.raise_for_status()
    signals = []
    for item in ET.fromstring(response.content).iter("item"):
        title = html.unescape(item.findtext("title") or "")
        publisher = item.findtext("source")
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(publisher) - 3]
        related = [html.unescape(text) for text in re.findall(r"<a [^>]*>(.*?)</a>", html.unescape(item.findtext("description") or ""))]
        signals.append({"source": "google_news", "feed": feed, "title": title, "language": language,
                        "published_at": _time(item.findtext("pubDate")), "publisher": publisher,
                        "url": item.findtext("link"), "related": [text for text in related if text != title][:4],
                        "cluster_size": max(1, len(related))})
    return signals


def _flatten_reddit_comments(children: list[dict], limit: int) -> list[dict]:
    comments = []
    for child in children:
        if len(comments) >= limit or child.get("kind") != "t1":
            continue
        comment = child.get("data", {})
        body = comment.get("body") or ""
        replies = comment.get("replies")
        reply_children = replies.get("data", {}).get("children", []) if isinstance(replies, dict) else []
        reply_count = sum(reply.get("kind") == "t1" for reply in reply_children)
        if body and body not in {"[deleted]", "[removed]"}:
            created = comment.get("created_utc")
            comments.append({
                "id": comment.get("id"),
                "parent_id": comment.get("parent_id"),
                "body": body,
                "score": comment.get("score"),
                "reply_count": reply_count,
                "depth": comment.get("depth"),
                "created_at": datetime.fromtimestamp(created, timezone.utc).isoformat() if created else None,
                "url": "https://www.reddit.com" + comment.get("permalink", ""),
            })
        if reply_children and len(comments) < limit:
            comments.extend(_flatten_reddit_comments(reply_children, limit - len(comments)))
    return comments[:limit]


def _reddit_retry_delay(response: requests.Response, retry_number: int, backoff_seconds: float) -> float:
    retry_after = response.headers.get("Retry-After")
    server_delay = None
    if retry_after:
        try:
            server_delay = max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                server_delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    exponential_delay = backoff_seconds * (2 ** (retry_number - 1))
    return max(server_delay or 0.0, exponential_delay)


def _reddit_get(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, int | str],
    headers: dict[str, str],
    retries: int,
    retry_backoff_seconds: float,
) -> requests.Response:
    for attempt in range(retries + 1):
        response = session.get(url, params=params, headers=headers, timeout=20)
        if response.status_code not in {403, 429}:
            response.raise_for_status()
            return response
        if attempt == retries:
            response.raise_for_status()
        delay = _reddit_retry_delay(response, attempt + 1, retry_backoff_seconds)
        print(f"Reddit HTTP {response.status_code}; retry {attempt + 1}/{retries} in {delay:.1f}s", flush=True)
        response.close()
        time.sleep(delay)
    raise RuntimeError("Reddit retry loop ended unexpectedly")


def reddit(
    subreddit: str,
    limit: int = 40,
    comment_limit: int = 0,
    comment_depth: int = 2,
    request_delay_seconds: float = 0,
    sort: str = "hot",
    time_filter: str = "day",
    retries: int = 2,
    retry_backoff_seconds: float = 5.0,
) -> list[dict]:
    """Fetch hot, new, or time-filtered top posts with optional comment samples."""
    if sort not in {"hot", "new", "top", "rising", "controversial"}:
        raise ValueError("sort must be 'hot', 'new', 'top', 'rising', or 'controversial'")
    if time_filter not in {"hour", "day", "week", "month", "year", "all"}:
        raise ValueError("time_filter must be hour, day, week, month, year, or all")
    if not 0 <= retries <= 5:
        raise ValueError("retries must be between 0 and 5")
    if retry_backoff_seconds < 0.5:
        raise ValueError("retry_backoff_seconds must be at least 0.5")
    if os.environ.get("API_KEY"):
        return reddit_apidirect(subreddit, limit, comment_limit, comment_depth, sort, time_filter)
    if sort in {"rising", "controversial"}:
        raise ValueError(f"sort='{sort}' requires API_KEY; the scraper fallback does not support it")
    post_limit = min(max(limit, 1), 100)
    sampled_comment_limit = min(max(comment_limit, 0), 100)
    headers = REDDIT_HEADERS
    session = requests.Session()
    listing_params = {"limit": post_limit}
    if sort == "top":
        listing_params["t"] = time_filter
    listing_name = f"top_{time_filter}" if sort == "top" else sort
    response = _reddit_get(
        session,
        f"https://www.reddit.com/r/{subreddit}/{sort}.json",
        params=listing_params,
        headers=headers,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    response.raise_for_status()
    signals = []
    for child in response.json().get("data", {}).get("children", []):
        post = child.get("data", {})
        if post.get("stickied"):
            continue
        permalink = post.get("permalink", "")
        comments = []
        comment_status = "not_requested"
        if sampled_comment_limit and post.get("num_comments", 0):
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            try:
                comment_response = _reddit_get(
                    session,
                    f"https://www.reddit.com{permalink}.json",
                    params={"limit": sampled_comment_limit, "depth": max(1, comment_depth), "sort": "top"},
                    headers=headers,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                comment_response.raise_for_status()
                payload = comment_response.json()
                children = payload[1].get("data", {}).get("children", []) if len(payload) > 1 else []
                comments = _flatten_reddit_comments(children, sampled_comment_limit)
                comment_status = "fetched"
            except requests.HTTPError as error:
                if getattr(error.response, "status_code", None) in {403, 429}:
                    raise
                comment_status = f"unavailable:{type(error).__name__}"
            except (IndexError, KeyError, TypeError, ValueError, requests.RequestException) as error:
                comment_status = f"unavailable:{type(error).__name__}"
        elif sampled_comment_limit:
            comment_status = "no_comments"

        signals.append({
            "source": "reddit",
            "feed": f"r/{subreddit}/{listing_name}",
            "listing_sort": sort,
            "listing_time_filter": time_filter if sort == "top" else None,
            "post_id": post.get("id"),
            "title": post.get("title", ""),
            "published_at": datetime.fromtimestamp(post.get("created_utc", 0), timezone.utc).isoformat(),
            "engagement": {
                "score": post.get("score"),
                "comments": post.get("num_comments"),
                "upvote_ratio": post.get("upvote_ratio"),
                "awards": post.get("total_awards_received"),
                "crossposts": post.get("num_crossposts"),
            },
            "comment_sample": {
                "requested": sampled_comment_limit,
                "retrieved": len(comments),
                "status": comment_status,
            },
            "comment_interactions": {
                "score_total": sum(comment.get("score") or 0 for comment in comments),
                "replies_total": sum(comment.get("reply_count") or 0 for comment in comments),
            },
            "comments": comments,
            "url": "https://www.reddit.com" + permalink,
        })
    return signals


APIDIRECT_REDDIT_URL = "https://apidirect.io/v1/reddit/subreddit/posts"
APIDIRECT_COMMENTS_URL = "https://apidirect.io/v1/reddit/post/comments"
APIDIRECT_POSTED_AGO = {"hour": "1h", "day": "24h", "week": "7d", "month": "30d", "year": "12m", "all": "all"}
APIDIRECT_SORT = {"hot": "hot", "new": "most_recent", "top": "top", "rising": "rising", "controversial": "controversial"}


def _apidirect_time(value: str | None) -> str | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat()


def _apidirect_comments(post_id: str, limit: int, depth: int, api_key: str) -> list[dict]:
    response = requests.get(
        APIDIRECT_COMMENTS_URL,
        params={"post_id": post_id, "sort_by": "top", "pages": min(max(math.ceil(limit / 50), 1), 80)},
        headers={"X-API-Key": api_key},
        timeout=60,
    )
    response.raise_for_status()
    kept = [c for c in response.json().get("comments", [])
            if (c.get("depth") or 0) < max(1, depth) and c.get("text") and c["text"] not in {"[deleted]", "[removed]"}][:limit]
    return [{
        "id": c.get("comment_id"),
        "parent_id": c.get("parent_id"),
        "body": c["text"],
        "score": c.get("upvotes"),
        "reply_count": sum(other.get("parent_id") == c.get("comment_id") for other in kept),
        "depth": c.get("depth"),
        "created_at": _apidirect_time(c.get("date")),
        "url": c.get("url", ""),
    } for c in kept]


def reddit_apidirect(subreddit: str, limit: int = 40, comment_limit: int = 0, comment_depth: int = 2,
                     sort: str = "hot", time_filter: str = "day") -> list[dict]:
    """Reddit posts and comment samples via API Direct."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is not set")
    post_limit = min(max(limit, 1), 100)
    sampled_comment_limit = min(max(comment_limit, 0), 100)
    params: dict[str, int | str] = {"subreddit": subreddit, "sort_by": APIDIRECT_SORT[sort],
                                    "pages": min(max(math.ceil(post_limit / 25), 1), 40)}
    windowed = sort in {"top", "controversial"}
    if windowed:
        params["posted_ago"] = APIDIRECT_POSTED_AGO[time_filter]
    response = requests.get(APIDIRECT_REDDIT_URL, params=params, headers={"X-API-Key": api_key}, timeout=60)
    response.raise_for_status()
    listing_name = f"{sort}_{time_filter}" if windowed else sort
    signals = []
    for post in response.json().get("posts", [])[:post_limit]:
        if post.get("stickied"):
            continue
        comments: list[dict] = []
        if sampled_comment_limit and post.get("comments"):
            try:
                comments = _apidirect_comments(post["post_id"], sampled_comment_limit, comment_depth, api_key)
                comment_status = "fetched"
            except requests.RequestException as error:
                comment_status = f"unavailable:{type(error).__name__}"
        else:
            comment_status = "no_comments" if sampled_comment_limit else "not_requested"
        signals.append({
            "source": "reddit",
            "feed": f"r/{subreddit}/{listing_name}",
            "listing_sort": sort,
            "listing_time_filter": time_filter if windowed else None,
            "post_id": post.get("post_id"),
            "title": post.get("title", ""),
            "published_at": _apidirect_time(post.get("date")),
            "engagement": {
                "score": post.get("upvotes"),
                "comments": post.get("comments"),
                "upvote_ratio": post.get("upvote_ratio"),
                "awards": None,
                "crossposts": post.get("crossposts"),
            },
            "comment_sample": {"requested": sampled_comment_limit, "retrieved": len(comments), "status": comment_status},
            "comment_interactions": {
                "score_total": sum(c.get("score") or 0 for c in comments),
                "replies_total": sum(c.get("reply_count") or 0 for c in comments),
            },
            "comments": comments,
            "url": post.get("url", ""),
        })
    return signals


def x_trends(country: str = "india", slots: int = 24) -> list[dict]:
    """X (Twitter) trending terms for the last ~24 hourly snapshots (trends24.in), with rank and persistence."""
    response = requests.get(f"https://trends24.in/{country}/", headers=BROWSER_HEADERS, timeout=20)
    response.raise_for_status()
    response.encoding = "utf-8"  # trends24 sends no charset; requests would fall back to ISO-8859-1 and mangle Hindi
    link = re.compile(r'<a[^>]+href="https?://(?:twitter|x)\.com/search\?q=[^"]*"[^>]*>(.*?)</a>', re.S)
    terms: dict[str, dict] = {}
    blocks = [block.split("</ol>")[0] for block in re.split(r"<ol\b", response.text)[1:]][:slots]
    for slot, block in enumerate(blocks):
        for rank, match in enumerate(link.finditer(block), 1):
            term = html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
            if term:
                entry = terms.setdefault(term.casefold(), {"title": term, "slots": set(), "rank": rank})
                entry["slots"].add(slot)
                entry["rank"] = min(entry["rank"], rank)
    reference = datetime.now(timezone.utc)
    return [{"source": "x_trends", "feed": f"x_trends_{country}", "title": entry["title"],
             "published_at": (reference - timedelta(hours=max(entry["slots"]))).isoformat(),
             "x_rank": entry["rank"], "x_hours": len(entry["slots"]), "x_latest": 0 in entry["slots"],
             "url": "https://x.com/search?q=" + requests.utils.quote(entry["title"])}
            for entry in terms.values()]


def collect_signals(include_reddit: bool = True, include_google: bool = True, include_x: bool = True) -> tuple[list[dict], dict]:
    jobs = [("trends_IN", lambda: google_trends("IN"))] if include_google else []
    for language in NEWS_LANGUAGES if include_google else ():
        code = language.split("-")[0]
        jobs.append((f"news_{code}", lambda language=language: google_news(
            f"https://news.google.com/rss?hl={language}&gl=IN&ceid=IN:{language.split('-')[0] if language != 'en-IN' else 'en'}",
            f"news_{language}", language.split("-")[0])))
    for section in NEWS_SECTIONS if include_google else ():
        jobs.append((f"news_{section.lower()}", lambda section=section: google_news(
            f"https://news.google.com/rss/headlines/section/topic/{section}?hl=en-IN&gl=IN&ceid=IN:en",
            f"news_{section.lower()}", "en")))
    for query in NEWS_SEARCHES if include_google else ():
        jobs.append((f"news_search_{query}", lambda query=query: google_news(
            f"https://news.google.com/rss/search?q={requests.utils.quote(query)}+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            f"news_search:{query}", "en")))
    if include_x:
        jobs.append(("x_trends_india", lambda: x_trends("india")))
    if include_reddit:
        for subreddit in REDDIT_SUBS:
            jobs.append((f"reddit_{subreddit}", lambda subreddit=subreddit: reddit(subreddit)))
    signals, availability = [], {}
    for name, job in jobs:
        try:
            items = job()
            availability[name] = len(items)
            signals.extend(items)
        except Exception as error:
            availability[name] = f"unavailable: {type(error).__name__}"
    seen, unique = set(), []
    for signal in signals:
        key = (signal["source"], signal["title"].casefold())
        if signal["title"] and key not in seen:
            seen.add(key)
            unique.append(signal)
    for index, signal in enumerate(unique):
        signal["signal_id"] = f"s{index}"
    return unique, availability

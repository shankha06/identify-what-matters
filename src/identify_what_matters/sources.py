"""Public India trend signals: Google Trends (searches), Google News (coverage), X trends (debate), Reddit (discussion).

Every fetch is independent; an unreachable source is recorded in `availability`, never fatal.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

HEADERS = {
    "Host": "www.reddit.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:155.0) Gecko/20100101 Firefox/155.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
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
    "TE": "trailers"
}



HT = "{https://trends.google.com/trending/rss}"
NEWS_LANGUAGES = ("en-IN", "hi", "ta", "te", "bn", "mr", "ml", "kn", "gu", "pa")
NEWS_SECTIONS = ("NATION", "BUSINESS", "ENTERTAINMENT", "SPORTS", "TECHNOLOGY", "HEALTH", "SCIENCE")
NEWS_SEARCHES = ("viral video", "trending on social media", "netizens react", "sparks outrage", "meme")
REDDIT_SUBS = ("india", "IndiaSpeaks", "indiasocial", "bollywood", "Cricket", "developersIndia", "IndianStockMarket")


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
    response = requests.get(f"https://trends.google.com/trending/rss?geo={geo}", headers=HEADERS, timeout=20)
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
    response = requests.get(url, headers=HEADERS, timeout=20)
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


def reddit(subreddit: str) -> list[dict]:
    """Scrape subreddit posts using YARS with random user agent session and exponential backoff."""
    import sys
    from pathlib import Path
    yars_path = Path(__file__).resolve().parents[2] / "YARS" / "src"
    if str(yars_path) not in sys.path:
        sys.path.insert(0, str(yars_path))

    try:
        from yars.sessions import RandomUserAgentSession
        session = RandomUserAgentSession()
    except Exception:
        session = requests.Session()

    response = session.get(
        f"https://www.reddit.com/r/{subreddit}/hot.json?limit=40",
        headers={**HEADERS},
        timeout=15,
    )
    response.raise_for_status()
    signals = []
    for child in response.json().get("data", {}).get("children", []):
        post = child.get("data", {})
        if post.get("stickied"):
            continue
        signals.append({"source": "reddit", "feed": f"r/{subreddit}", "title": post.get("title", ""),
                        "published_at": datetime.fromtimestamp(post.get("created_utc", 0), timezone.utc).isoformat(),
                        "engagement": {"score": post.get("score"), "comments": post.get("num_comments")},
                        "url": "https://www.reddit.com" + post.get("permalink", "")})
    return signals


def x_trends(country: str = "india", slots: int = 24) -> list[dict]:
    """X (Twitter) trending terms for the last ~24 hourly snapshots (trends24.in), with rank and persistence."""
    response = requests.get(f"https://trends24.in/{country}/", headers=HEADERS, timeout=20)
    response.raise_for_status()
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

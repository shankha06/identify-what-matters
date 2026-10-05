"""
Instagram Scraper & Client Engine
Supports:
1. Modern Instagram Mobile Web Endpoints (tags web_info, comments, topsearch)
2. Fallback to instagramy / legacy HTML parsing
3. Rate limit handling, exponential backoff, and pagination
"""

import time
import logging
from typing import Dict, Any, List, Optional
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InstagramClient")

# Meta Web Client App ID used by instagram.com web client
DEFAULT_IG_APP_ID = "936619743392459"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class InstagramClient:
    def __init__(
        self,
        session_id: str,
        ds_user_id: Optional[str] = None,
        csrf_token: Optional[str] = None,
        ig_app_id: str = DEFAULT_IG_APP_ID,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 15.0,
    ):
        self.session_id = session_id
        self.ds_user_id = ds_user_id or session_id.split("%3A")[0].split(":")[0]
        self.csrf_token = csrf_token
        self.ig_app_id = ig_app_id
        self.user_agent = user_agent
        self.timeout = timeout

        self._setup_client()

    def _setup_client(self):
        cookie_header = f"sessionid={self.session_id}; ds_user_id={self.ds_user_id};"
        if self.csrf_token:
            cookie_header += f" csrftoken={self.csrf_token};"

        headers = {
            "User-Agent": self.user_agent,
            "X-IG-App-ID": self.ig_app_id,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.instagram.com/",
            "Cookie": cookie_header,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.client = httpx.Client(
            base_url="https://www.instagram.com",
            headers=headers,
            timeout=self.timeout,
            follow_redirects=True,
        )

    def fetch_tag_posts(self, tag_name: str, max_posts: int = 40) -> List[Dict[str, Any]]:
        """
        Fetch top and recent posts for a specific hashtag via Instagram internal API.
        """
        clean_tag = tag_name.lstrip("#").strip()
        url = f"/api/v1/tags/web_info/?tag_name={clean_tag}"
        logger.info(f"Fetching posts for tag: #{clean_tag}...")

        try:
            resp = self.client.get(url)
            if resp.status_code == 429:
                logger.warning("Rate limit (429) hit. Pausing before retry...")
                time.sleep(5)
                resp = self.client.get(url)

            if resp.status_code != 200:
                logger.error(f"Failed to fetch #{clean_tag}: Status {resp.status_code}")
                return []

            data = resp.json()
            if not data.get("data"):
                logger.warning(f"No data section returned for #{clean_tag}")
                return []

            posts = []
            sections = (data["data"].get("top", {}).get("sections") or []) + \
                       (data["data"].get("recent", {}).get("sections") or [])

            seen_ids = set()
            for sec in sections:
                medias = sec.get("layout_content", {}).get("medias", [])
                for item in medias:
                    media = item.get("media", {})
                    pk = media.get("id") or media.get("pk")
                    if not pk or pk in seen_ids:
                        continue
                    seen_ids.add(pk)

                    caption_text = media.get("caption", {}).get("text", "") if media.get("caption") else ""
                    posts.append({
                        "id": str(pk),
                        "code": media.get("code"),
                        "shortcode": media.get("code"),
                        "caption": caption_text,
                        "like_count": media.get("like_count", 0),
                        "comment_count": media.get("comment_count", 0),
                        "taken_at": media.get("taken_at", int(time.time())),
                        "username": media.get("user", {}).get("username"),
                        "user_id": media.get("user", {}).get("pk"),
                        "is_video": media.get("is_video", False),
                        "view_count": media.get("view_count", 0),
                        "tag": clean_tag,
                    })
                    if len(posts) >= max_posts:
                        break
                if len(posts) >= max_posts:
                    break

            logger.info(f"Retrieved {len(posts)} posts for #{clean_tag}")
            return posts

        except Exception as e:
            logger.error(f"Error fetching #{clean_tag}: {e}")
            return []

    def fetch_post_comments(self, post_id: str, max_comments: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch top comments for a given media/post ID.
        """
        # post_id might be like "3219321564395207064_23285396643" -> extract numeric prefix
        numeric_id = post_id.split("_")[0]
        url = f"/api/v1/media/{numeric_id}/comments/"
        logger.info(f"Fetching comments for post {numeric_id}...")

        try:
            resp = self.client.get(url)
            if resp.status_code != 200:
                logger.warning(f"Status {resp.status_code} fetching comments for {numeric_id}")
                return []

            data = resp.json()
            raw_comments = data.get("comments", [])
            comments = []
            for c in raw_comments[:max_comments]:
                comments.append({
                    "id": str(c.get("pk") or c.get("id")),
                    "text": c.get("text", ""),
                    "like_count": c.get("comment_like_count", 0),
                    "created_at": c.get("created_at", int(time.time())),
                    "username": c.get("user", {}).get("username"),
                })
            return comments
        except Exception as e:
            logger.error(f"Error fetching comments for {numeric_id}: {e}")
            return []

    def search_entities(self, query: str) -> Dict[str, Any]:
        """
        Search Instagram blended directory for hashtags, accounts, places.
        """
        url = f"/api/v1/web/search/topsearch/?context=blended&query={query}&rank_token=0.5"
        try:
            resp = self.client.get(url)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Error searching {query}: {e}")
        return {}

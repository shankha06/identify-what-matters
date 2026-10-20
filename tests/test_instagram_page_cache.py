import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from identify_what_matters import instagram_access as ia


class _StubClient:
    def __init__(self):
        self.page_calls: list = []

    async def media_comments_v2(self, id, page_id=None):
        self.page_calls.append(page_id)
        if page_id is None:
            return {"next_page_id": "p2",
                    "response": {"comment_count": 7,
                                 "comments": [{"pk": 1, "text": "a", "child_comment_count": 0,
                                               "comment_like_count": 5, "user": {"username": "u"}}]}}
        return {"response": {"comment_count": 7,
                             "comments": [{"pk": 2, "text": "b", "child_comment_count": 0,
                                           "user": {"username": "v"}}]}}

    async def media_comments_replies_v2(self, media_id, comment_id):
        return {"child_comments": []}


class PageCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for patch in (mock.patch.object(ia, "TOTALS_CACHE_PATH", root / "totals.json"),
                      mock.patch.object(ia, "REQUEST_CACHE_PATH", root / "requests.json"),
                      mock.patch.dict(os.environ, {"HIKERAPI_TOKEN": "test"})):
            patch.start()
            self.addCleanup(patch.stop)
        self.transport = ia.HikerTransport(delay_seconds=0.0, token="test", user_id_cache=root / "ids.json",
                                           feed_cache=root / "feeds.json")
        self.client = _StubClient()
        self.transport.client = self.client

    def test_comment_total_then_comments_reuses_first_page(self):
        async def scenario():
            total = await self.transport.comment_total("m1", datetime.now(timezone.utc).isoformat())
            result = await self.transport.comments("m1", max_pages=2, max_reply_threads=0)
            return total, result

        total, result = asyncio.run(scenario())
        self.assertEqual(total, 7)
        self.assertEqual(self.client.page_calls, [None, "p2"])
        self.assertEqual([c["id"] for c in result["comments"]], ["1", "2"])

    def test_fresh_total_makes_no_request(self):
        async def scenario():
            now = datetime.now(timezone.utc).isoformat()
            await self.transport.comment_total("m1", now)
            await self.transport.comment_total("m1", now)

        asyncio.run(scenario())
        self.assertEqual(self.client.page_calls, [None])

    def test_comment_total_reuses_cached_first_page_from_comments(self):
        async def scenario():
            await self.transport.comments("m1", max_pages=2, max_reply_threads=0)
            return await self.transport.comment_total("m1", datetime.now(timezone.utc).isoformat())

        total = asyncio.run(scenario())
        self.assertEqual(total, 7)
        self.assertEqual(self.client.page_calls, [None, "p2"])


if __name__ == "__main__":
    unittest.main()

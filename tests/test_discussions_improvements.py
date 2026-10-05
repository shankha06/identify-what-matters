import unittest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from identify_what_matters.trends import History, GENERIC_HASHTAGS, probe
from identify_what_matters.discussions import panel_scores, discover_discussions


class TestDiscussionsImprovements(unittest.TestCase):

    def test_generic_hashtags_filtered(self):
        self.assertIn("viral", GENERIC_HASHTAGS)
        self.assertIn("reels", GENERIC_HASHTAGS)
        self.assertIn("trendingindia", GENERIC_HASHTAGS)
        self.assertNotIn("karanaujla", GENERIC_HASHTAGS)
        self.assertNotIn("cskvsmi", GENERIC_HASHTAGS)

    def test_account_baselines_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_trends.sqlite"
            history = History(db_path)
            self.assertEqual(history.load_baselines(), {})

            ref = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
            # Create dummy feed for an account with settled posts (> 6 hours old)
            feeds = {
                "test_page": [
                    {"code": f"c{i}", "taken_at": (ref - timedelta(hours=10)).isoformat(),
                     "pinned": False, "paid": False, "counts_hidden": False,
                     "like_count": 1000 + i * 100, "comment_count": 50 + i * 5}
                    for i in range(5)
                ]
            }
            history.update_baselines(feeds, ref)
            baselines = history.load_baselines()
            self.assertIn("test_page", baselines)
            base = baselines["test_page"]
            self.assertEqual(base["sample_count"], 5)
            self.assertGreater(base["median_likes"], 0)
            self.assertGreater(base["median_comments"], 0)

            # Test using baselines in panel_scores
            scores = panel_scores(feeds, ref, baselines=baselines)
            self.assertEqual(len(scores), 5)
            self.assertIn("heat", scores[0]["panel"])

    def test_short_caption_fallback(self):
        # Verify that discover_discussions handles posts with short captions enriched by comments
        ref = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
        bundle = {
            "feeds": {
                "meme_page": [
                    {
                        "code": "m1",
                        "taken_at": (ref - timedelta(hours=2)).isoformat(),
                        "caption": "💔💔",
                        "author_id": "author_1",
                        "like_count": 5000,
                        "comment_count": 250,
                        "pinned": False,
                        "paid": False,
                        "counts_hidden": False,
                        "panel": {"group": "memes", "heat": 2.5, "ritual": False},
                        "comments": [
                            {"text": "Rohit Sharma retirement announcement reaction", "like_count": 120},
                            {"text": "Cant believe this is happening", "like_count": 80}
                        ]
                    },
                    {
                        "code": "m2",
                        "taken_at": (ref - timedelta(hours=2)).isoformat(),
                        "caption": "Massive W for Indian cricket fans today",
                        "author_id": "author_2",
                        "like_count": 3000,
                        "comment_count": 180,
                        "pinned": False,
                        "paid": False,
                        "counts_hidden": False,
                        "panel": {"group": "cricket", "heat": 2.0, "ritual": False},
                        "comments": [
                            {"text": "Rohit Sharma sensational captaincy", "like_count": 95}
                        ]
                    },
                    {
                        "code": "m3",
                        "taken_at": (ref - timedelta(hours=1)).isoformat(),
                        "caption": "End of an era for Indian cricket Rohit Sharma",
                        "author_id": "author_3",
                        "like_count": 4500,
                        "comment_count": 300,
                        "pinned": False,
                        "paid": False,
                        "counts_hidden": False,
                        "panel": {"group": "news", "heat": 2.2, "ritual": False},
                        "comments": [
                            {"text": "Legend will be missed forever", "like_count": 150}
                        ]
                    }
                ]
            },
            "terms": {},
            "seeds": {},
            "probes": {},
            "comments": {}
        }

        class MockLLM:
            def json(self, prompt, system, validate=None):
                res = {"clusters": [{"i": 0, "is_topic": True, "title": "Rohit Sharma Cricket Discussion",
                                      "category": "sports", "kind": "sports_moment", "entities": ["Rohit Sharma"]}]}
                return validate(res) if validate else res

            def map(self, fn, items):
                return [fn(item) for item in items]

        mock_llm = MockLLM()
        discussions = discover_discussions(bundle, [], mock_llm, ref)
        self.assertEqual(len(discussions), 1)
        self.assertEqual(discussions[0]["title"], "Rohit Sharma Cricket Discussion")


if __name__ == "__main__":
    unittest.main()

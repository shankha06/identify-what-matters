import unittest
from datetime import datetime, timezone, timedelta

from identify_what_matters.proactive_radar import (
    calculate_velocity_signals,
    extract_emerging_entities,
    identify_proactive_candidates,
    generate_proactive_probe_queries,
    classify_lifecycle_trajectory
)


class TestProactiveRadar(unittest.TestCase):

    def setUp(self):
        self.ref = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)

    def test_calculate_velocity_signals_normal(self):
        post = {
            "code": "norm1",
            "taken_at": (self.ref - timedelta(hours=10)).isoformat(),
            "comment_count": 30,
            "like_count": 1000,
        }
        baseline = {"median_comments": 40.0, "median_likes": 1200.0, "median_ratio": 0.03}
        vel = calculate_velocity_signals(post, self.ref, baseline)
        self.assertFalse(vel["is_flame"])
        self.assertAlmostEqual(vel["comment_velocity"], 3.0, places=1)

    def test_calculate_velocity_signals_flame_shock(self):
        # A fresh post (2 hours old) with huge comments and ratio surge
        post = {
            "code": "flame1",
            "taken_at": (self.ref - timedelta(hours=2)).isoformat(),
            "comment_count": 600,
            "like_count": 2500,
        }
        baseline = {"median_comments": 50.0, "median_likes": 2000.0, "median_ratio": 0.025}
        vel = calculate_velocity_signals(post, self.ref, baseline)
        self.assertTrue(vel["is_flame"])
        self.assertGreater(vel["comment_velocity"], 100.0)
        self.assertGreater(vel["ratio_deviation"], 4.0)

    def test_extract_emerging_entities(self):
        comments = [
            {"text": "Rohit Sharma retirement announcement is shock #RohitSharma", "like_count": 50},
            {"text": "Why Rohit Sharma taking this step now? #RohitSharma #Cricket", "like_count": 20},
            {"text": "Rohit Sharma will be missed forever legend", "like_count": 15},
            {"text": "Bhai kya sach me Rohit Sharma retired?", "like_count": 5},
        ]
        entities = extract_emerging_entities(comments, min_occurrences=2)
        entity_names = [e["entity"] for e in entities]
        self.assertTrue(any("rohit sharma" in name.lower() for name in entity_names))
        self.assertIn("#rohitsharma", entity_names)

    def test_identify_proactive_candidates_and_query_generation(self):
        bundle = {
            "feeds": {
                "meme_account": [
                    {
                        "code": "p1",
                        "taken_at": (self.ref - timedelta(hours=1)).isoformat(),
                        "comment_count": 450,
                        "like_count": 1800,
                        "caption": "Wait for the end 💀",
                        "panel_group": "memes_commentary",
                        "comments": [
                            {"text": "Bhadhohi hospital incident is heartbreaking", "like_count": 45},
                            {"text": "Strict action needed in Bhadhohi hospital case", "like_count": 30},
                            {"text": "Bhadhohi hospital staff must be arrested #BhadohiNews", "like_count": 25},
                        ]
                    }
                ]
            },
            "comments": {}
        }
        baselines = {"meme_account": {"median_comments": 40.0, "median_likes": 1500.0, "median_ratio": 0.025}}
        candidates = identify_proactive_candidates(bundle, self.ref, baselines)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["post_code"], "p1")

        queries = generate_proactive_probe_queries(candidates, max_queries=5)
        self.assertGreaterEqual(len(queries), 1)
        self.assertTrue(any("bhadohi" in q.lower() for q in queries))

    def test_classify_lifecycle_trajectory(self):
        # Spark topic: zero news publishers, seen < 6 hours ago
        topic_spark = {
            "publishers": 0,
            "first_seen": (self.ref - timedelta(hours=3)).isoformat(),
            "instagram_metrics": {"panel_accounts": 2, "comment_like_ratio": 0.05},
            "brief": {"debate": {"is_debate": True}}
        }
        self.assertEqual(classify_lifecycle_trajectory(topic_spark, self.ref), "spark")

        # Peak debate topic: high comment:like ratio, active dispute
        topic_debate = {
            "publishers": 1,
            "first_seen": (self.ref - timedelta(hours=15)).isoformat(),
            "instagram_metrics": {"panel_accounts": 3, "comment_like_ratio": 0.06},
            "brief": {"debate": {"is_debate": True}}
        }
        self.assertEqual(classify_lifecycle_trajectory(topic_debate, self.ref), "peak_debate")

        # Saturated topic: covered widely by news publishers
        topic_sat = {
            "publishers": 6,
            "first_seen": (self.ref - timedelta(hours=20)).isoformat(),
            "instagram_metrics": {"panel_accounts": 2, "comment_like_ratio": 0.02},
            "brief": {"debate": {"is_debate": False}}
        }
        self.assertEqual(classify_lifecycle_trajectory(topic_sat, self.ref), "saturated")


if __name__ == "__main__":
    unittest.main()

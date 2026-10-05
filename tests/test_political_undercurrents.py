import unittest
from datetime import datetime
from identify_what_matters.proactive_radar import (
    calculate_velocity_signals,
    identify_proactive_candidates,
    generate_proactive_probe_queries,
    classify_lifecycle_trajectory,
    POLITICAL_CIVIC_PATTERNS
)


class TestPoliticalUndercurrents(unittest.TestCase):
    def test_political_regex(self):
        self.assertTrue(POLITICAL_CIVIC_PATTERNS.search("Rahul Gandhi demands CEC resignation"))
        self.assertTrue(POLITICAL_CIVIC_PATTERNS.search("CJP protest in Mumbai Shivaji Park"))
        self.assertTrue(POLITICAL_CIVIC_PATTERNS.search("Supreme Court hearing on reservation and caste census"))
        self.assertTrue(POLITICAL_CIVIC_PATTERNS.search("FIR against Manish Kashyap over Bihar bridge"))
        self.assertFalse(POLITICAL_CIVIC_PATTERNS.search("Delicious chocolate cake recipe"))

    def test_political_velocity_boost(self):
        now = datetime(2026, 10, 3, 12, 0)
        pol_item = {
            "taken_at": "2026-10-03T10:00:00",
            "comment_count": 50,
            "like_count": 500,
            "caption": "Rahul Gandhi speaks out on CEC Gyanesh Kumar controversy #politics"
        }
        generic_item = {
            "taken_at": "2026-10-03T10:00:00",
            "comment_count": 50,
            "like_count": 500,
            "caption": "Enjoying beautiful sunset vibes at the beach"
        }
        v_pol = calculate_velocity_signals(pol_item, now)
        v_gen = calculate_velocity_signals(generic_item, now)
        self.assertTrue(v_pol["is_political"])
        self.assertFalse(v_gen["is_political"])
        self.assertGreater(v_pol["proactive_score"], v_gen["proactive_score"])

    def test_flame_entity_extraction_and_probes(self):
        now = datetime(2026, 10, 3, 12, 0)
        bundle = {
            "feeds": {
                "news_sentinel": [
                    {
                        "code": "XYZ123",
                        "pk": "999888",
                        "taken_at": "2026-10-03T10:30:00",
                        "comment_count": 300,
                        "like_count": 1000,
                        "caption": "Massive protest against Election Commission at Jantar Mantar #CECResign #VoteChor",
                        "panel_group": "political_commentary"
                    }
                ]
            },
            "comments": {
                "999888": {
                    "comments": [
                        {"text": "Gyanesh Kumar must resign immediately #CECResign", "like_count": 50},
                        {"text": "EVM controversy again, democracy in danger", "like_count": 40},
                        {"text": "Rahul Gandhi raised the right question", "like_count": 30},
                    ]
                }
            }
        }
        candidates = identify_proactive_candidates(bundle, now)
        self.assertGreater(len(candidates), 0)
        queries = generate_proactive_probe_queries(candidates, max_queries=5)
        self.assertTrue(any("cec" in q.lower() or "gyanesh" in q.lower() or "resign" in q.lower() or "votechor" in q.lower() for q in queries))


if __name__ == "__main__":
    unittest.main()

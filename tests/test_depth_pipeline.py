import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from identify_what_matters.account_graph import astroturf_summary, build_account_graph
from identify_what_matters.trends import (
    History,
    instagram_metrics,
    lead_lag,
    lead_lag_summary,
    publisher_first,
    publisher_pickup,
    stage_history,
    status,
)

REF = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def iso(hours_before: float) -> str:
    return (REF - timedelta(hours=hours_before)).isoformat()


class LeadLagTest(unittest.TestCase):
    def test_instagram_first_has_positive_lead(self):
        topic = {"instagram": {"media": [{"code": "a", "taken_at": iso(6)}]}, "earliest": iso(2), "pickup_at": iso(1)}
        result = lead_lag(topic)
        self.assertEqual(result["verdict"], "instagram_first")
        self.assertEqual(result["lead_vs_news_hours"], 4.0)
        self.assertEqual(result["lead_vs_pickup_hours"], 5.0)

    def test_news_first_has_negative_lead(self):
        topic = {"instagram": {"media": [{"code": "a", "taken_at": iso(6)}]}, "earliest": iso(10)}
        result = lead_lag(topic)
        self.assertEqual(result["verdict"], "news_first")
        self.assertEqual(result["lead_vs_news_hours"], -4.0)

    def test_within_an_hour_is_simultaneous(self):
        topic = {"instagram": {"media": [{"code": "a", "taken_at": iso(6)}]}, "earliest": iso(5.5)}
        self.assertEqual(lead_lag(topic)["verdict"], "simultaneous")

    def test_missing_news_or_posts(self):
        self.assertEqual(lead_lag({"instagram": {"media": [{"code": "a", "taken_at": iso(2)}]}})["verdict"], "instagram_only")
        self.assertEqual(lead_lag({"instagram": {"media": []}, "earliest": iso(2)})["verdict"], "no_instagram_evidence")

    def test_summary_counts_verdicts_and_median(self):
        topics = [{"lead": lead_lag({"instagram": {"media": [{"code": "a", "taken_at": iso(6)}]}, "earliest": iso(2)})},
                  {"lead": lead_lag({"instagram": {"media": [{"code": "b", "taken_at": iso(6)}]}, "earliest": iso(10)})},
                  {}]
        summary = lead_lag_summary(topics)
        self.assertEqual(summary["compared_with_news"], 2)
        self.assertEqual(summary["median_lead_vs_news_hours"], 0.0)
        self.assertEqual(summary["verdicts"]["not_measured"], 1)


class PublisherTimingTest(unittest.TestCase):
    def test_publisher_first_keeps_earliest_per_outlet(self):
        members = [{"publisher": "A", "published_at": iso(5)}, {"publisher": "A", "published_at": iso(9)},
                   {"publisher": "B", "published_at": iso(7)}, {"publisher": None, "published_at": iso(1)}]
        first = publisher_first(members)
        self.assertEqual(first, {"A": iso(9), "B": iso(7)})

    def test_pickup_is_third_outlet_or_none(self):
        self.assertIsNone(publisher_pickup({"A": iso(5), "B": iso(4)}))
        self.assertEqual(publisher_pickup({"A": iso(5), "B": iso(4), "C": iso(3)}), iso(3))


class StageHistoryTest(unittest.TestCase):
    def test_transitions_and_hours_in_current_stage(self):
        timeline = [{"at": iso(48), "stage": "simmering"}, {"at": iso(24), "stage": "breaking_out"}]
        result = stage_history(timeline, "breaking_out", REF)
        self.assertEqual(result["runs_observed"], 3)
        self.assertEqual(result["transitions"], [{"from": "simmering", "to": "breaking_out", "at": iso(24)}])
        self.assertEqual(result["stage_hours"], 24.0)

    def test_new_stage_adds_current_transition(self):
        timeline = [{"at": iso(24), "stage": "breaking_out"}]
        result = stage_history(timeline, "peak", REF)
        self.assertEqual(result["transitions"][-1]["from"], "breaking_out")
        self.assertEqual(result["transitions"][-1]["to"], "peak")
        self.assertEqual(result["stage_hours"], 0.0)


def metrics_topic(previous_score=None, stage=None):
    topic = {"first_seen": REF.isoformat(), "earliest": None, "title": "t",
             "instagram_metrics": {"posts_24h": 0, "recency_ratio": 0.0, "discussion_level": "low",
                                   "hashtag_on_topic_share": None, "hashtag_counts": {}},
             "scores": {"trend_score": 2.0}, "breakout": {"stage": stage or "simmering"}}
    if previous_score is not None:
        topic["previous"] = {"trend_score": previous_score, "at": iso(3), "hashtag_counts": {}}
    return topic


class StatusSettledTest(unittest.TestCase):
    def test_settled_without_fall_stays_settled(self):
        self.assertEqual(status(metrics_topic(stage="settled"), REF), "settled")

    def test_settled_after_steep_fall_is_fading(self):
        self.assertEqual(status(metrics_topic(previous_score=3.0, stage="settled"), REF), "fading")


class CommentCoverageTest(unittest.TestCase):
    def test_coverage_is_scored_share_of_platform_comments(self):
        media = [{"code": "a", "author_id": "u1", "taken_at": iso(2), "comment_count": 100, "like_count": 50,
                  "play_count": 0, "url": "https://example.test/a", "query": "kw",
                  "comments": [{"text": "t", "like_count": 1}] * 20}]
        metrics = instagram_metrics({"instagram": {"media": media}}, REF)
        self.assertEqual(metrics["comments_scored"], 20)
        self.assertEqual(metrics["comments_platform"], 100)
        self.assertEqual(metrics["comment_coverage"], 0.2)

    def test_coverage_is_none_without_platform_count(self):
        media = [{"code": "a", "author_id": "u1", "taken_at": iso(2), "comment_count": 0, "like_count": 0,
                  "play_count": 0, "url": "https://example.test/a", "query": "kw", "comments": []}]
        self.assertIsNone(instagram_metrics({"instagram": {"media": media}}, REF)["comment_coverage"])


def amplifier_topic(topic_id: str, accounts: list[str]) -> dict:
    media = [{"code": f"{topic_id}-post", "author_id": accounts[0], "author": accounts[0],
              "comments": [{"author_id": a, "author": a, "text": "same line"} for a in accounts[1:]]}]
    return {"topic_id": topic_id, "title": f"Topic {topic_id}", "instagram": {"media": media}}


class AccountGraphTest(unittest.TestCase):
    def test_cross_topic_accounts_and_coordination_pair(self):
        topics = [amplifier_topic("T1", ["bot1", "bot2", "bot3", "fan"]),
                  amplifier_topic("T2", ["bot1", "bot2", "bot3", "other"]),
                  amplifier_topic("T3", ["bot2", "bot1", "bot3", "third"])]
        graph = build_account_graph(topics)
        labels = {row["label"] for row in graph["accounts"]}
        self.assertEqual(labels, {"bot1", "bot2", "bot3"})
        pairs = [set(p["accounts"]) for p in graph["coordination_pairs"]]
        self.assertIn({"bot1", "bot2"}, pairs)
        self.assertEqual(graph["coordination_pairs"][0]["shared_topics"], 3)
        self.assertTrue(graph["topic_links"])
        self.assertEqual(graph["topic_links"][0]["shared_accounts"], 3)

    def test_single_topic_has_no_cross_accounts(self):
        graph = build_account_graph([amplifier_topic("T1", ["a", "b"])])
        self.assertEqual(graph["cross_topic_accounts"], 0)
        self.assertEqual(graph["accounts"], [])


class FakeDetector:
    def __init__(self):
        self.seen = None

    def analyze_topic_astroturf(self, title, comments, debate_sides=None):
        self.seen = comments
        return {"astroturf_index": 42, "bot_share_pct": 12.5, "bot_comment_count": 3,
                "total_comments_audited": len(comments), "template_clusters": [{}],
                "coordination_status": "watch", "side_bot_bias": {},
                "verdict_summary": "Some template repetition."}


class AstroturfSummaryTest(unittest.TestCase):
    def test_returns_none_below_minimum_comments(self):
        topic = amplifier_topic("T1", ["a", "b", "c"])
        self.assertIsNone(astroturf_summary(topic, detector=FakeDetector()))

    def test_summarises_detector_output_and_copies_comments(self):
        accounts = ["acct"] + [f"u{i}" for i in range(25)]
        topic = amplifier_topic("T1", accounts)
        detector = FakeDetector()
        summary = astroturf_summary(topic, detector=detector)
        self.assertEqual(summary["astroturf_index"], 42)
        self.assertEqual(summary["total_comments_audited"], 25)
        self.assertEqual(summary["template_clusters"], 1)
        detector.seen[0]["text"] = "mutated"
        self.assertEqual(topic["instagram"]["media"][0]["comments"][0]["text"], "same line")


class HistoryTimelineTest(unittest.TestCase):
    def test_timeline_returns_earlier_observations_oldest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory) / "trends.sqlite")
            topic = {"topic_id": "T1", "title": "x", "category": "politics", "first_seen": iso(48),
                     "scores": {"trend_score": 2.0, "instagram_heat": 1.0, "news_heat": 0.5},
                     "_vector": np.zeros(3, dtype=np.float32), "breakout": {"stage": "simmering"},
                     "publishers": 2, "instagram_metrics": {"hashtag_counts": {"x": 5}}}
            history.save("run1", REF - timedelta(hours=24), [topic], {})
            topic["breakout"] = {"stage": "breaking_out"}
            history.save("run2", REF, [topic], {})
            points = history.timeline("T1")
            self.assertEqual([p["stage"] for p in points], ["simmering", "breaking_out"])
            self.assertEqual(points[0]["publishers"], 2)
            self.assertEqual(points[0]["hashtag_counts"], {"x": 5})
            history.db.close()


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import datetime, timedelta, timezone

from identify_what_matters.hiker_client import CollectionAborted, ItemUnavailable, _HikerClient
from identify_what_matters.proactive_radar import breakout_assessment
from identify_what_matters.trends import measured_opinion

REF = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def post(code, author, hours, comments, likes, over=1.0, lift=1.0):
    return {"code": code, "author_id": author, "taken_at": (REF - timedelta(hours=hours)).isoformat(),
            "comment_count": comments, "like_count": likes,
            "panel": {"account": author, "over_comments": over, "ratio_lift": lift}}


class BreakoutAssessmentTest(unittest.TestCase):
    def test_young_overperformer_without_press_is_pre_breakout(self):
        topic = {"publishers": 0, "instagram": {"media": [post("a", "u1", 2, 180, 2000, over=5.2, lift=3.1)]}}
        result = breakout_assessment(topic, REF)
        self.assertEqual(result["stage"], "pre_breakout")
        self.assertIn("no news coverage yet", result["reasons"])

    def test_spread_across_accounts_in_six_hours_is_breaking_out(self):
        media = [post(c, f"u{i}", 1 + i, 90, 3000) for i, c in enumerate("abc")]
        result = breakout_assessment({"publishers": 1, "instagram": {"media": media}}, REF)
        self.assertEqual(result["stage"], "breaking_out")

    def test_wide_press_is_saturated_and_scored_lower(self):
        media = [post("a", "u1", 2, 180, 2000, over=5.2, lift=3.1)]
        quiet = breakout_assessment({"publishers": 0, "instagram": {"media": media}}, REF)
        covered = breakout_assessment({"publishers": 6, "instagram": {"media": media}}, REF)
        self.assertEqual(covered["stage"], "saturated")
        self.assertLess(covered["score"], quiet["score"])

    def test_no_new_posts_in_a_day_is_settled(self):
        media = [post("a", "u1", 30, 50, 900, over=1, lift=1)]
        self.assertEqual(breakout_assessment({"publishers": 0, "instagram": {"media": media}}, REF)["stage"], "settled")

    def test_multiday_active_debate_is_simmering_not_settled(self):
        # A topic active over 4 days (e.g. 28h, 70h, 96h) with multiple accounts and active comments
        media = [post("a", "u1", 28, 200, 3000), post("b", "u2", 70, 150, 2500), post("c", "u3", 96, 120, 1800)]
        res = breakout_assessment({"publishers": 0, "instagram": {"media": media}}, REF)
        self.assertEqual(res["stage"], "simmering")
        self.assertGreater(res["score"], 0)

    def test_brief_on_topic_filter_is_respected(self):
        media = [post("a", "u1", 2, 180, 2000, over=5.2, lift=3.1), post("b", "u2", 60, 10, 100)]
        topic = {"publishers": 0, "brief": {"on_topic_codes": ["b"]}, "instagram": {"media": media}}
        self.assertEqual(breakout_assessment(topic, REF)["stage"], "settled")


class MeasuredOpinionTest(unittest.TestCase):
    def test_like_weighted_side_shares_and_top_quotes(self):
        pool = [{"text": "Resign now, this is a cover up", "like_count": 900},
                {"text": "Protest is drama before elections", "like_count": 20},
                {"text": "Buy followers here", "like_count": 0},
                {"text": "lol both sides same", "like_count": 5}]
        result = {"debate": {"sides": [{"position": "Official must resign"}, {"position": "Protest is political drama"}]},
                  "comment_labels": [{"c": 0, "tone": "critical", "side": 0},
                                     {"c": 1, "tone": "critical", "side": 1, "sarcastic": True},
                                     {"c": 2, "tone": "neutral", "side": None, "off_topic": True},
                                     {"c": 3, "tone": "humorous", "side": None},
                                     {"c": 99, "tone": "critical", "side": 0}]}
        opinion = measured_opinion(result, pool)
        self.assertEqual(opinion["comments"], 3)
        self.assertEqual(opinion["sarcastic"], 1)
        self.assertGreater(opinion["sides"][0]["share"], opinion["sides"][1]["share"])
        self.assertEqual(opinion["sides"][0]["top_comment"], "Resign now, this is a cover up")
        self.assertAlmostEqual(sum(opinion["tone"].values()), 1.0, places=1)
        self.assertGreater(opinion["unaligned_share"], 0)

    def test_missing_labels_returns_none(self):
        self.assertIsNone(measured_opinion({"comment_labels": []}, [{"text": "x"}]))


class ItemErrorTest(unittest.TestCase):
    def _client(self, responses):
        client = _HikerClient.__new__(_HikerClient)
        client.failure, client.item_error_streak, client.before_request = None, 0, lambda: None
        queue = list(responses)

        def fake_request(self_, method, path, **kwargs):
            return queue.pop(0)
        return client, fake_request

    def test_not_found_skips_without_aborting(self):
        from hikerapi.base import BaseSyncClient
        client, fake = self._client([{"exc_type": "UserNotFound", "detail": "x"}, {"ok": True}])
        original = BaseSyncClient._request
        BaseSyncClient._request = fake
        try:
            with self.assertRaises(ItemUnavailable):
                client._request("get", "/v1/user/by/username")
            self.assertIsNone(client.failure)
            self.assertEqual(client._request("get", "/v1/user/by/username"), {"ok": True})
            self.assertEqual(client.item_error_streak, 0)
        finally:
            BaseSyncClient._request = original

    def test_streak_of_item_errors_and_unclassified_errors_abort(self):
        from hikerapi.base import BaseSyncClient
        client, fake = self._client([{"exc_type": "UserNotFound"}] * 4)
        original = BaseSyncClient._request
        BaseSyncClient._request = fake
        try:
            for _ in range(3):
                with self.assertRaises(ItemUnavailable):
                    client._request("get", "/v1/user/by/username")
            with self.assertRaises(CollectionAborted):
                client._request("get", "/v1/user/by/username")
            self.assertIsNotNone(client.failure)
        finally:
            BaseSyncClient._request = original
        client, fake = self._client([{"exc_type": "RateLimitError", "detail": "slow down"}])
        BaseSyncClient._request = fake
        try:
            with self.assertRaises(CollectionAborted):
                client._request("get", "/v1/user/by/username")
        finally:
            BaseSyncClient._request = original


if __name__ == "__main__":
    unittest.main()

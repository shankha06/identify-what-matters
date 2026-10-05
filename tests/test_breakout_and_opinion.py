import unittest
from datetime import datetime, timedelta, timezone

from instagrapi.exceptions import ChallengeRequired, PrivateAccount

from identify_what_matters.instagrapi_client import CollectionAborted, ItemUnavailable, _FailClosedClient
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
        media = [post("a", "u1", 30, 500, 9000, over=4, lift=2)]
        self.assertEqual(breakout_assessment({"publishers": 0, "instagram": {"media": media}}, REF)["stage"], "settled")

    def test_brief_on_topic_filter_is_respected(self):
        media = [post("a", "u1", 2, 180, 2000, over=5.2, lift=3.1), post("b", "u2", 40, 10, 100)]
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
    def _client(self, errors):
        client = _FailClosedClient.__new__(_FailClosedClient)
        client.failure, client.item_error_streak, client.before_request = None, 0, lambda: None
        client.raw_medias, client.raw_users, client.last_json = {}, {}, {}
        queue = list(errors)

        def fake_send(self_, endpoint, **kwargs):
            error = queue.pop(0)
            if error:
                raise error
            return {"ok": True}
        return client, fake_send

    def test_private_account_skips_without_aborting(self):
        from instagrapi import Client
        client, fake = self._client([PrivateAccount("Not authorized to view user"), None])
        original = Client._send_private_request
        Client._send_private_request = fake
        try:
            with self.assertRaises(ItemUnavailable):
                client._send_private_request("feed/user/1/")
            self.assertIsNone(client.failure)
            self.assertEqual(client._send_private_request("feed/user/2/"), {"ok": True})
            self.assertEqual(client.item_error_streak, 0)
        finally:
            Client._send_private_request = original

    def test_streak_of_item_errors_and_challenges_abort(self):
        from instagrapi import Client
        client, fake = self._client([PrivateAccount("x")] * 4)
        original = Client._send_private_request
        Client._send_private_request = fake
        try:
            for _ in range(3):
                with self.assertRaises(ItemUnavailable):
                    client._send_private_request("feed/user/1/")
            with self.assertRaises(CollectionAborted):
                client._send_private_request("feed/user/1/")
            self.assertIsNotNone(client.failure)
        finally:
            Client._send_private_request = original
        client, fake = self._client([ChallengeRequired("challenge")])
        Client._send_private_request = fake
        try:
            with self.assertRaises(CollectionAborted):
                client._send_private_request("feed/user/1/")
        finally:
            Client._send_private_request = original


if __name__ == "__main__":
    unittest.main()

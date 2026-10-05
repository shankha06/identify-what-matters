import unittest
from identify_what_matters import stance


class TestNoiseFilterAndClustering(unittest.TestCase):
    def test_filter_comment_noise_spam_and_emojis(self):
        comments = [
            "This is a genuine substantive discussion comment about the event.",
            "DM for promo and paid promotions 100% genuine profits",
            "🔥🔥🔥",
            "check out my new crypto trading profile link in bio",
            "k",
            "संसद में विपक्ष ने जो सवाल उठाया वो सही है।"
        ]
        results = stance.filter_comment_noise(comments)
        self.assertEqual(len(results), 6)
        self.assertFalse(results[0]["is_noise"])
        self.assertTrue(results[1]["is_noise"])
        self.assertEqual(results[1]["reason"], "spam_pattern")
        self.assertTrue(results[2]["is_noise"])
        self.assertEqual(results[2]["reason"], "emoji_only")
        self.assertTrue(results[3]["is_noise"])
        self.assertEqual(results[3]["reason"], "spam_pattern")
        self.assertTrue(results[4]["is_noise"])
        self.assertEqual(results[4]["reason"], "too_short")
        self.assertFalse(results[5]["is_noise"])

    def test_cluster_arguments_with_noise_suppression(self):
        context = "Protest against electoral roll verification procedures."
        positions = [
            "The protest is legitimate and demands democratic accountability.",
            "The protest is politically motivated and lacks justification."
        ]
        comments = [
            {"text": "We demand complete transparency from the commission!", "like_count": 50},
            {"text": "Completely fake protest created by paid workers.", "like_count": 35},
            {"text": "DM for promo follow back 100% genuine profit", "like_count": 999},
            {"text": "🔥🔥🔥", "like_count": 200},
            {"text": "Good point raised by the citizens.", "like_count": 10}
        ]
        res = stance.cluster_arguments(context, positions, comments)
        self.assertEqual(res["clean_comments"], 3)
        self.assertEqual(res["noise_comments"], 2)
        self.assertEqual(len(res["sides"]), 2)
        # Ensure high-like spam comment did not distort side arguments or quotes
        for side in res["sides"]:
            self.assertNotIn("promo", side.get("representative_comment", "").lower())
            self.assertNotIn("🔥", side.get("representative_comment", "").lower())


if __name__ == "__main__":
    unittest.main()

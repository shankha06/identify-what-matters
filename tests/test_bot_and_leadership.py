import unittest
from identify_what_matters.bot_detector import (
    bot_detector,
    classify_template_lean,
    compute_comment_repetition_clusters
)
from identify_what_matters.strategic_leadership import (
    compute_narrative_penetration_rate,
    compute_persuadability_matrix,
    compute_policy_gap_detector,
    compute_opposition_vulnerabilities,
    compute_influencer_alignment,
    generate_action_recommendations
)


class TestBotAndLeadership(unittest.TestCase):

    def test_classify_template_lean(self):
        # Opposition template
        anti_res = classify_template_lean("Narendra Modi resign button ✅")
        self.assertIn("Anti-Incumbent", anti_res["synthetic_lean"])
        self.assertEqual(anti_res["lean_category"], "anti_incumbent")

        # Pro-incumbent template
        pro_res = classify_template_lean("Har Har Mahadev ❤️🙏🛐")
        self.assertIn("Pro-Incumbent", pro_res["synthetic_lean"])
        self.assertEqual(pro_res["lean_category"], "pro_incumbent")

    def test_bot_audit_sentiment_split(self):
        comments = [
            {"text": "Narendra Modi resign button ✅ please share", "author_id": "u1"},
            {"text": "Narendra Modi resign button ✅ please share", "author_id": "u2"},
            {"text": "Narendra Modi resign button ✅ please share", "author_id": "u3"},
            {"text": "Har Har Mahadev ❤️🙏🛐 Jai Shri Ram", "author_id": "u4"},
            {"text": "Har Har Mahadev ❤️🙏🛐 Jai Shri Ram", "author_id": "u5"},
            {"text": "The flyover repair in our ward was done on time and looks solid.", "author_id": "u6",
             "user": {"media_count": 80, "follower_count": 320}},
            {"text": "Hoping that public bus services improve further this monsoon.", "author_id": "u7",
             "user": {"media_count": 50, "follower_count": 210}},
        ]
        audit = bot_detector.analyze_topic_astroturf("Civic Transit & Politics", comments)

        self.assertIn("organic_sentiment_mix", audit)
        self.assertIn("synthetic_sentiment_mix", audit)
        self.assertIn("sentiment_divergence", audit)
        self.assertIn("bot_adjusted_stance", audit)
        self.assertIn("organic_vs_synthetic_ratio", audit)

        # Check clusters contain synthetic_lean
        self.assertGreaterEqual(len(audit["template_clusters"]), 1)
        cl0 = audit["template_clusters"][0]
        self.assertIn("synthetic_lean", cl0)
        self.assertIn("dominant_sentiment", cl0)

    def test_narrative_penetration_rate(self):
        dummy_topics = [
            {
                "id": "t1",
                "title": "Voter deletion controversy in district rolls",
                "is_political": True,
                "issue_cluster": "elections",
                "undercurrent_score": 7.5,
                "political_pulse": {
                    "opp_favor_pct": 65.0,
                    "bjp_favor_pct": 20.0,
                    "neutral_pct": 15.0,
                },
                "undercurrent": {
                    "framing_reaction_gap": 0.40
                }
            }
        ]
        alerts = compute_narrative_penetration_rate(dummy_topics)
        self.assertEqual(len(alerts), 1)
        self.assertGreaterEqual(alerts[0]["npr_score"], 30.0)
        self.assertIn("penetrated", alerts[0]["alert_text"])

    def test_persuadability_matrix(self):
        dummy_topics = [
            {"id": "t1", "is_political": True, "issue_cluster": "employment", "undercurrent_score": 8.0},
            {"id": "t2", "is_political": True, "issue_cluster": "education", "undercurrent_score": 7.0},
        ]
        matrix = compute_persuadability_matrix(dummy_topics)
        self.assertGreaterEqual(len(matrix), 3)
        self.assertIn("persuadability_score", matrix[0])
        self.assertIn("responsive_policy_lever", matrix[0])

    def test_policy_gap_detector(self):
        dummy_topics = [
            {"id": "t1", "is_political": True, "issue_cluster": "employment", "undercurrent_score": 9.0, "ig": {"comments_7d": 5000}},
            {"id": "t2", "is_political": True, "issue_cluster": "infrastructure", "undercurrent_score": 3.0, "ig": {"comments_7d": 100}},
        ]
        gaps = compute_policy_gap_detector(dummy_topics)
        self.assertGreaterEqual(len(gaps), 5)
        self.assertIn("gap_delta", gaps[0])
        self.assertIn("recommended_injection", gaps[0])

    def test_action_recommendations_generation(self):
        dummy_topics = [
            {"id": "t1", "title": "Paper leak protest", "is_political": True, "issue_cluster": "education", "undercurrent_score": 8.5,
             "political_pulse": {"opp_favor_pct": 60.0, "bjp_favor_pct": 25.0, "neutral_pct": 15.0},
             "stance": {"critical": 0.70}}
        ]
        early_warnings = [
            {
                "threat_axis": "Campus Agitation & Exam Integrity",
                "priority": "CRITICAL",
                "demographic_trait": "18-24 Gen Z Youth",
                "avg_undercurrent": 8.5,
                "immediate_actions": ["Deploy education ministry town hall and publish verified exam audit."]
            }
        ]
        panel_posts = [
            {"account": "influencer_delhi", "likes": 50000, "heat": 3.2, "group": "creator_commentary", "tone": "critical", "lean": "Pro-opposition"}
        ]
        recs = generate_action_recommendations(dummy_topics, early_warnings, panel_posts)
        self.assertGreaterEqual(len(recs), 2)
        self.assertIn("priority", recs[0])
        self.assertIn("trigger_event", recs[0])
        self.assertIn("target_audience", recs[0])
        self.assertIn("recommended_action", recs[0])


if __name__ == "__main__":
    unittest.main()

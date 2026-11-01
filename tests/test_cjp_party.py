import unittest

from identify_what_matters.political_classifier import ENTITY_TARGET_PATTERNS, IS_POLITICAL_RE


class TestCJPParty(unittest.TestCase):
    """CJP = Cockroach Janta Party. Bare acronym 'CJP' is intentionally NOT matched (ambiguous)."""

    def setUp(self):
        self.pattern = ENTITY_TARGET_PATTERNS["CJP"]

    def test_entity_pattern_matches_full_name_and_hashtag_forms(self):
        self.assertTrue(self.pattern.search("Cockroach Janta Party rally in Mumbai"))
        self.assertTrue(self.pattern.search("cockroachjantaparty volunteers join today"))
        self.assertTrue(self.pattern.search("#CockroachIsBack trending again"))

    def test_entity_pattern_ignores_unrelated_text(self):
        self.assertFalse(self.pattern.search("Delicious chocolate cake recipe"))
        self.assertFalse(self.pattern.search("BJP rally in Patna"))

    def test_cjp_counts_as_political_topic(self):
        self.assertTrue(IS_POLITICAL_RE.search("Cockroach Janta Party protest march"))


if __name__ == "__main__":
    unittest.main()

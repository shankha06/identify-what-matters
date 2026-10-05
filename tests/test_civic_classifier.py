import unittest
from identify_what_matters.civic_classifier import CivicClassifier


class TestCivicClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = CivicClassifier(
            model_path="artifacts/civic-models/civic-me5-base",
            device="cpu",
        )

    def test_civic_detection(self):
        civic_texts = [
            "Protests at Jantar Mantar demanding CEC resignation over election concerns.",
            "Punjab farmers union plans rail roko agitation for MSP law guarantee.",
            "Supreme Court bench hears petition challenging citizenship amendment rules.",
        ]
        results = self.classifier.predict(civic_texts)
        for r in results:
            self.assertTrue(r["is_civic"])
            self.assertGreater(r["score"], 0.70)

    def test_non_civic_rejection(self):
        non_civic_texts = [
            "Chocolate cake recipe with simple 3 ingredients at home.",
            "Virat Kohli hits century in IPL match against Kolkata Knight Riders.",
            "DM for paid promotion and telegram crypto trading signals 100% profit.",
        ]
        results = self.classifier.predict(non_civic_texts)
        for r in results:
            self.assertFalse(r["is_civic"])
            self.assertLess(r["score"], 0.30)


if __name__ == "__main__":
    unittest.main()

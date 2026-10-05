import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from identify_what_matters import stance_corpus


def discussion(key):
    sides = [0, 0, 0, 1, 1, 1, 2, None, 0, 1, 2, None, 0, 1, None, None]
    comments = [{"text": f"{key} comment number {i} about the roll deletion", "language": "en" if i % 2 else "hinglish",
                 "side": side, "tone": "critical", "sarcastic": i == 3, "off_topic": i == 15}
                for i, side in enumerate(sides)]
    return {"key": key, "caption": "The state voter roll lost lakhs of names overnight, officials say duplicates.",
            "question": "Were the deletions legitimate?", "positions": ["Illegal purge", "Routine clean-up", "Wait for audit"],
            "comments": comments}


class StanceCorpusTest(unittest.TestCase):
    def test_validate_and_build_pairs(self):
        accepted = stance_corpus.validate({"discussions": [discussion("d0"), discussion("d1")]},
                                          "elections_and_voter_rolls", 6)
        self.assertEqual([d["split"] for d in accepted], ["dev", "dev"])
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "calls").mkdir()
            (Path(root) / "calls" / "x.json").write_text(json.dumps({"discussions": accepted}))
            stance_corpus.build(SimpleNamespace(output=root, seed=1))
            rows = [json.loads(l) for l in (Path(root) / "dataset" / "stance_dev.jsonl").read_text().splitlines()]
            tone = (Path(root) / "dataset" / "tone_dev.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 2 * 15 * 3)
        self.assertEqual(len(tone), 2 * 16)
        self.assertTrue(all(r["label"] == int(r["gold_side"] == r["side_index"]) for r in rows))

    def test_rejects_identifiers_and_unbalanced_sides(self):
        bad = discussion("d0")
        bad["comments"][0]["text"] = "follow @someone for more"
        with self.assertRaises(ValueError):
            stance_corpus.validate({"discussions": [bad, discussion("d1")]}, "elections_and_voter_rolls", 0)
        lopsided = discussion("d0")
        for c in lopsided["comments"]:
            c["side"] = 0
        with self.assertRaises(ValueError):
            stance_corpus.validate({"discussions": [lopsided, discussion("d1")]}, "elections_and_voter_rolls", 0)


if __name__ == "__main__":
    unittest.main()

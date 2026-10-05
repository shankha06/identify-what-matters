from datetime import datetime, timezone
import tempfile
from pathlib import Path
import unittest

from identify_what_matters.utils import (
    get_torch_device,
    hours_since,
    now,
    now_iso,
    parse_json_payload,
    parse_time,
    write_json,
    write_rows,
)


class UtilsTests(unittest.TestCase):
    def test_parse_json_payload_clean(self):
        self.assertEqual(parse_json_payload('{"key": "value"}'), {"key": "value"})

    def test_parse_json_payload_with_fences_and_whitespace(self):
        text = 'Here is the result:\n```json\n{"items": [1, 2, 3]}\n```\nHope that helps!'
        self.assertEqual(parse_json_payload(text), {"items": [1, 2, 3]})

    def test_parse_json_payload_array(self):
        self.assertEqual(parse_json_payload('[{"id": 1}, {"id": 2}]'), [{"id": 1}, {"id": 2}])

    def test_parse_json_payload_invalid(self):
        with self.assertRaises(ValueError):
            parse_json_payload("No json here at all")

    def test_parse_time_and_hours_since(self):
        ref = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)
        past_iso = "2026-10-05T06:00:00Z"
        self.assertAlmostEqual(hours_since(past_iso, ref), 6.0)

        # Min hours clamp
        self.assertAlmostEqual(hours_since(ref, ref, min_hours=0.1), 0.1)

        # None input
        self.assertIsNone(hours_since(None, ref))
        self.assertIsNone(parse_time(None))

    def test_write_json_and_write_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / "sub" / "test.json"
            write_json(json_file, {"hello": "world"})
            self.assertTrue(json_file.exists())
            import json
            self.assertEqual(json.loads(json_file.read_text()), {"hello": "world"})

            jsonl_file = Path(tmp_dir) / "sub" / "test.jsonl"
            write_rows(jsonl_file, [{"a": 1}, {"b": 2}])
            lines = [json.loads(l) for l in jsonl_file.read_text().strip().split("\n")]
            self.assertEqual(lines, [{"a": 1}, {"b": 2}])

    def test_get_torch_device(self):
        dev = get_torch_device()
        self.assertIn(dev, ("cuda", "mps", "cpu"))


if __name__ == "__main__":
    unittest.main()

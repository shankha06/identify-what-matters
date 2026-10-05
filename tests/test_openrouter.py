import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from identify_what_matters.llm import LLM
from identify_what_matters.openrouter import OpenRouterClient
from identify_what_matters.utils import Budget


class OpenRouterClientTests(unittest.TestCase):
    def test_reasoning_details_are_forwarded_and_usage_is_combined(self):
        reasoning_details = [{"type": "reasoning", "summary": "internal reasoning"}]
        first = Mock()
        first.json.return_value = {
            "choices": [{"message": {"content": "draft", "reasoning_details": reasoning_details}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
        second = Mock()
        second.json.return_value = {
            "choices": [{"message": {"content": "final"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 6},
        }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("identify_what_matters.openrouter.requests.post", side_effect=[first, second]) as post:
                result = OpenRouterClient().generate("prompt", "system")

        self.assertEqual(result.text, "final")
        self.assertEqual((result.input_tokens, result.output_tokens), (25, 10))
        self.assertEqual(post.call_count, 2)
        import json
        first_payload = json.loads(post.call_args_list[0].kwargs["data"])
        second_payload = json.loads(post.call_args_list[1].kwargs["data"])
        second_messages = second_payload["messages"]
        self.assertEqual(first_payload["model"], "openai/gpt-6.1-sol")
        self.assertEqual(first_payload["reasoning"], {"enabled": True})
        self.assertEqual(second_messages[-2]["reasoning_details"], reasoning_details)

    def test_budget_counts_both_provider_requests(self):
        state = {"calls": 0, "usage": []}
        budget = Budget(state, max_calls=3, max_output_tokens=100)

        self.assertTrue(budget.reserve(60, calls=2))
        self.assertEqual(state["calls"], 2)
        self.assertFalse(budget.reserve(20, calls=2))

    def test_main_llm_budget_requires_two_requests(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                llm = LLM(cache_dir=cache_dir, max_calls=1)

            with self.assertRaisesRegex(RuntimeError, "call budget 1 exhausted"):
                llm.json("prompt", "system")
            self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()

import unittest

from wheatley.tools.parser import parse_tool_calls


class ToolParserTests(unittest.TestCase):
    def test_parse_tool_calls_json(self):
        calls = parse_tool_calls(
            '{"tool_calls":[{"name":"get_time","arguments":{}}]}'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_time")

    def test_parse_fenced_json(self):
        calls = parse_tool_calls(
            'Sure.\n```json\n{"tool":"set_eye_expression","args":{"expression":"thinking"}}\n```'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments["expression"], "thinking")

    def test_ignores_plain_text(self):
        self.assertEqual(parse_tool_calls("hello there"), [])


if __name__ == "__main__":
    unittest.main()


import unittest

from research_agent.schema import report_response_format
from research_agent.tools import SEARCH_SOURCES_TOOL


class ContractTests(unittest.TestCase):
    def test_responses_contracts_are_strict_json_schema_objects(self):
        report_format = report_response_format()

        self.assertEqual(report_format["type"], "json_schema")
        self.assertTrue(report_format["strict"])
        report_schema = report_format["schema"]
        self.assertFalse(report_schema["additionalProperties"])
        self.assertIn("key_findings", report_schema["required"])
        self.assertFalse(
            report_schema["$defs"]["Finding"]["additionalProperties"]
        )

    def test_search_tool_is_strict_and_requires_both_arguments(self):
        self.assertTrue(SEARCH_SOURCES_TOOL["strict"])
        self.assertEqual(
            SEARCH_SOURCES_TOOL["parameters"]["required"],
            ["query", "max_results"],
        )

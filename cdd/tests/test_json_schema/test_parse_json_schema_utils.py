"""
Tests for the Intermediate Representation produced by the JSON schema parser
"""

from copy import deepcopy
from unittest import TestCase

import cdd.json_schema.utils.parse_utils
from cdd.tests.mocks.json_schema import server_error_schema
from cdd.tests.utils_for_tests import unittest_main


class TestParseJsonSchemaUtils(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    """

    def test_json_schema_property_to_param_anyOf(self) -> None:
        """
        Tests that `json_schema_property_to_param` works with `anyOf`
        """

        mock = "address", {
            "anyOf": [{"$ref": "#/components/schemas/address"}],
            "doc": "The customer's address.",
            "nullable": True,
        }

        res = cdd.json_schema.utils.parse_utils.json_schema_property_to_param(
            mock, {mock[0]: False}
        )

        self.assertEqual(res[0], "address")
        self.assertDictEqual(
            res[1], {"typ": "Optional[Address]", "doc": mock[1]["doc"]}
        )

    def test_json_schema_property_to_param_ref(self) -> None:
        """
        Tests that `json_schema_property_to_param` works with `$ref` as type
        """

        mock = "tax", {"$ref": "#/components/schemas/customer_tax"}

        res = cdd.json_schema.utils.parse_utils.json_schema_property_to_param(mock, {})

        self.assertEqual(res[0], "tax")
        self.assertDictEqual(
            res[1], {"doc": "[FK(CustomerTax)]", "typ": "Optional[CustomerTax]"}
        )

    def test_json_schema_property_to_param_default_none(self) -> None:
        """
        Tests that `json_schema_property_to_param` works with `$ref` as type
        """

        mock = (lambda k: (k, deepcopy(server_error_schema["properties"][k])))(
            "error_description"
        )
        mock[1]["default"] = None

        res = cdd.json_schema.utils.parse_utils.json_schema_property_to_param(
            mock, {mock[0]: True}
        )

        self.assertEqual(res[0], mock[0])
        self.assertDictEqual(res[1], mock[1])

        res = cdd.json_schema.utils.parse_utils.json_schema_property_to_param(mock, {})
        self.assertEqual(res[0], mock[0])
        self.assertDictEqual(res[1], mock[1])


unittest_main()

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

    IR is a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    """

    def test_json_schema_property_to_param_anyOf(self) -> None:
        """
        Tests that `json_schema_property_to_param` works with `anyOf`
        """

        mock = "address", {
            "anyOf": [{"$ref": "#/components/schemas/address"}],
            "doc": "The customer's address.",
            "nullable": True,
        }  # type: tuple[str, dict]

        res = cdd.json_schema.utils.parse_utils.json_schema_property_to_param(
            mock, {mock[0]: False}
        )  # type: tuple[str, dict]

        self.assertEqual(res[0], "address")
        self.assertDictEqual(
            res[1], {"typ": "Optional[Address]", "doc": mock[1]["doc"]}
        )

    def test_json_schema_property_to_param_ref(self) -> None:
        """
        Tests that `json_schema_property_to_param` works with `$ref` as type
        """

        mock = "tax", {
            "$ref": "#/components/schemas/customer_tax"
        }  # type: tuple[str, dict]

        res = cdd.json_schema.utils.parse_utils.json_schema_property_to_param(
            mock, {}
        )  # type: tuple[str, dict]

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
        )  # type: tuple[str, dict]

        self.assertEqual(res[0], mock[0])
        self.assertDictEqual(res[1], mock[1])

        res = cdd.json_schema.utils.parse_utils.json_schema_property_to_param(
            mock, {}
        )  # type: tuple[str, dict]
        self.assertEqual(res[0], mock[0])
        self.assertDictEqual(res[1], mock[1])

    def test_json_schema_property_to_param_removes_string_from_anyOf(self) -> None:
        """Tests that `json_schema_property_to_param` removes `string` from `anyOf`"""
        param = ("foo", {"anyOf": ["string", "can"], "typ": ["string", "can", "haz"]})
        cdd.json_schema.utils.parse_utils.json_schema_property_to_param(param, {})
        self.assertDictEqual(param[1], {"typ": "Optional[can]"})


unittest_main()

"""
Tests for the Intermediate Representation produced by the JSON schema parser
"""

from unittest import TestCase

import cdd.json_schema.emit
import cdd.json_schema.parse
from cdd.tests.mocks.ir import intermediate_repr_no_default_sql_doc
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.utils_for_tests import unittest_main


class TestParseJsonSchema(TestCase):
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

    def test_from_json_schema(self) -> None:
        """
        Tests that `parse.json_schema` produces `intermediate_repr_no_default_sql_doc` properly
        """
        self.assertDictEqual(
            cdd.json_schema.parse.json_schema(config_schema),
            intermediate_repr_no_default_sql_doc,
        )


unittest_main()

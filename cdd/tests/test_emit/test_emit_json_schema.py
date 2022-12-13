"""
Tests for `cdd.emit.json_schema`
"""

from copy import deepcopy
from json import load
from operator import itemgetter
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase

import cdd.emit.argparse_function
import cdd.emit.class_
import cdd.emit.docstring
import cdd.emit.file
import cdd.emit.function
import cdd.emit.json_schema
import cdd.emit.sqlalchemy
import cdd.parse.argparse_function
import cdd.parse.function
from cdd.gen_utils import get_input_mapping_from_path
from cdd.tests.mocks.ir import intermediate_repr_no_default_sql_doc
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.utils_for_tests import unittest_main


class TestEmitJsonSchema(TestCase):
    """Tests emission"""

    def test_to_json_schema(self) -> None:
        """
        Tests that `emit.json_schema` with `intermediate_repr_no_default_doc` produces `config_schema`
        """
        gen_config_schema = cdd.emit.json_schema.json_schema(
            deepcopy(intermediate_repr_no_default_sql_doc),
            "https://offscale.io/config.schema.json",
            emit_original_whitespace=True,
        )
        self.assertEqual(
            *map(itemgetter("description"), (gen_config_schema, config_schema))
        )
        self.assertDictEqual(
            gen_config_schema,
            config_schema,
        )

    def test_to_json_schema_file(self) -> None:
        """
        Tests that `emit.json_schema` with `intermediate_repr_no_default_doc` produces `config_schema`
        """
        with TemporaryDirectory() as temp_dir:
            temp_file = path.join(temp_dir, "foo{}py".format(path.extsep))
            cdd.emit.json_schema.json_schema_file(
                get_input_mapping_from_path("function", "cdd.tests", "test_gen_utils"),
                temp_file,
            )
            with open(temp_file, "rt") as f:
                temp_json_schema = load(f)
            self.assertDictEqual(
                temp_json_schema,
                {
                    "$id": "https://offscale.io/f.schema.json",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "description": ":rtype: ```str```",
                    "properties": {"s": {"description": "str", "type": "string"}},
                    "required": ["s"],
                    "type": "object",
                },
            )

    maxDiff = None


unittest_main()

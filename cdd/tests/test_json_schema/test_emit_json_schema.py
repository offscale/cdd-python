"""
Tests for `cdd.emit.json_schema`
"""

from copy import deepcopy
from json import load
from operator import itemgetter
from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase

import cdd.argparse_function.emit
import cdd.argparse_function.parse
import cdd.class_.emit
import cdd.docstring.emit
import cdd.function.emit
import cdd.function.parse
import cdd.json_schema.emit
import cdd.shared.emit.file
import cdd.sqlalchemy.emit
from cdd.compound.gen_utils import get_input_mapping_from_path
from cdd.tests.mocks.ir import intermediate_repr_no_default_sql_doc
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.utils_for_tests import unittest_main


class TestEmitJsonSchema(TestCase):
    """Tests emission"""

    def test_to_json_schema(self) -> None:
        """
        Tests that `emit.json_schema` with `intermediate_repr_no_default_doc` produces `config_schema`
        """
        gen_config_schema = cdd.json_schema.emit.json_schema(
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
            cdd.json_schema.emit.json_schema_file(
                get_input_mapping_from_path(
                    "function", "cdd.tests.test_compound", "test_gen_utils"
                ),
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


unittest_main()

"""
Tests for `cdd.emit.json_schema`
"""

from copy import deepcopy
from operator import itemgetter
from unittest import TestCase

import cdd.emit.argparse_function
import cdd.emit.class_
import cdd.emit.docstring
import cdd.emit.file
import cdd.emit.function
import cdd.emit.json_schema
import cdd.emit.sqlalchemy
import cdd.parse.argparse_function
from cdd.tests.mocks.ir import intermediate_repr_no_default_sql_doc
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.utils_for_tests import unittest_main


class TestEmitClass(TestCase):
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


unittest_main()

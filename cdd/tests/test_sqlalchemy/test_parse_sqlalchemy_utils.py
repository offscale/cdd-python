"""
Tests for the utils that is used by the SQLalchemy parsers
"""

from copy import deepcopy
from unittest import TestCase

from cdd.sqlalchemy.utils.parse_utils import (
    column_call_name_manipulator,
    column_call_to_param,
    get_pk_and_type,
    get_table_name,
)
from cdd.tests.mocks.ir import intermediate_repr_node_pk
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.mocks.sqlalchemy import (
    config_decl_base_ast,
    dataset_primary_key_column_assign,
    node_fk_call,
)
from cdd.tests.utils_for_tests import unittest_main


class TestParseSqlAlchemyUtils(TestCase):
    """
    Tests the SQLalchemy parser utilities
    """

    def test_column_call_to_param_pk(self) -> None:
        """
        Tests that `parse.sqlalchemy.utils.column_call_to_param` works with PK
        """

        gold_name, gold_param = (
            lambda _name: (
                _name,
                {
                    "default": config_schema["properties"][_name]["default"],
                    "typ": "str",
                    "doc": config_schema["properties"][_name]["description"],
                    "x_typ": {"sql": {"type": "String"}},
                },
            )
        )("dataset_name")
        gen_name, gen_param = column_call_to_param(
            column_call_name_manipulator(
                deepcopy(dataset_primary_key_column_assign.value), "add", gold_name
            )
        )
        self.assertEqual(gold_name, gen_name)
        self.assertDictEqual(gold_param, gen_param)

    def test_column_call_to_param_fk(self) -> None:
        """
        Tests that `parse.sqlalchemy.utils.column_call_to_param` works with FK
        """
        gen_name, gen_param = column_call_to_param(deepcopy(node_fk_call))
        gold_name, gold_param = (
            lambda _name: (_name, deepcopy(intermediate_repr_node_pk["params"][_name]))
        )("primary_element")
        self.assertEqual(gold_name, gen_name)
        self.assertDictEqual(gold_param, gen_param)

    def test_get_pk_and_type(self) -> None:
        """
        Tests get_pk_and_type
        """
        self.assertEqual(
            get_pk_and_type(config_decl_base_ast), ("dataset_name", "String")
        )
        no_pk = deepcopy(config_decl_base_ast)
        del no_pk.body[2]
        self.assertIsNone(get_pk_and_type(no_pk))

    def test_get_table_name(self) -> None:
        """
        Tests `get_table_name`
        """
        self.assertEqual(get_table_name(config_decl_base_ast), "config_tbl")
        no_table_name = deepcopy(config_decl_base_ast)
        del no_table_name.body[1]
        self.assertEqual(get_table_name(no_table_name), "Config")


unittest_main()

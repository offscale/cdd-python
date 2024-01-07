"""
Tests for `cdd.emit.utils.json_schema_utils`
"""

from ast import Dict, List, Load, Set, Tuple
from copy import deepcopy
from typing import List as TList
from unittest import TestCase

from cdd.json_schema.utils.emit_utils import param2json_schema_property
from cdd.shared.ast_utils import set_value
from cdd.shared.types import IntermediateRepr
from cdd.tests.mocks.ir import intermediate_repr_no_default_doc
from cdd.tests.utils_for_tests import unittest_main


class TestEmitJsonSchemaUtils(TestCase):
    """Tests cdd.emit.utils.json_schema_utils"""

    required: TList[str] = []
    name: str = "param"
    default_ = tuple(range(5))
    default_elts = list(map(set_value, default_))

    def test_param2json_schema_property_default_property(self) -> None:
        """
        Tests that `param2json_schema_property` with different default types
        """

        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {
                        "default": Set(
                            elts=self.default_elts,
                            ctx=Load(),
                            expr=None,
                        )
                    },
                ),
                required=self.required,
            ),
            (self.name, {"default": set(self.default_)}),
        )

        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {
                        "default": Tuple(
                            elts=self.default_elts,
                            ctx=Load(),
                            expr=None,
                            lineno=None,
                            col_offset=None,
                        )
                    },
                ),
                required=self.required,
            ),
            (self.name, {"default": self.default_}),
        )

        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {"default": List(elts=self.default_elts, ctx=Load(), expr=None)},
                ),
                required=self.required,
            ),
            (self.name, {"default": list(self.default_)}),
        )

        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {"default": None},
                ),
                required=self.required,
            ),
            (self.name, {}),
        )

        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {
                        "default": Dict(
                            keys=[set_value("Foo")],
                            values=[set_value("bar")],
                            expr=None,
                        )
                    },
                ),
                required=self.required,
            ),
            (self.name, {"default": {"Foo": "bar"}}),
        )

    def test_param2json_schema_property_choices(self) -> None:
        """
        Tests that `param2json_schema_property` with different choices as `Set`
        """
        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {
                        "choices": Set(
                            elts=self.default_elts,
                            ctx=Load(),
                            expr=None,
                        )
                    },
                ),
                required=self.required,
            ),
            (self.name, {"pattern": "0|1|2|3|4"}),
        )

    def test_param2json_schema_property_type_datetime(self) -> None:
        """
        Tests that `param2json_schema_property` with different type as `datetime`
        """
        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {"typ": "datetime"},
                ),
                required=self.required,
            ),
            (self.name, {"type": "string", "format": "date-time"}),
        )

    def test_param2json_schema_property_type_boolean(self) -> None:
        """
        Tests that `param2json_schema_property` with different type as `datetime`
        """
        ir: IntermediateRepr = deepcopy(intermediate_repr_no_default_doc)
        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    {"typ": "bool"},
                ),
                required=self.required,
            ),
            (self.name, {"type": "boolean"}),
        )

        required: TList[str] = []
        self.assertEqual(
            param2json_schema_property(
                (
                    self.name,
                    ir["params"]["as_numpy"],
                ),
                required=required,
            ),
            (
                self.name,
                {
                    "description": ir["params"]["as_numpy"]["description"],
                    "type": "boolean",
                },
            ),
        )
        self.assertListEqual(required, [])


unittest_main()

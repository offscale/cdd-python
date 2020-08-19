"""
Tests for docstring parsing
"""
from copy import deepcopy
from unittest import TestCase

import doctrans.emit
import doctrans.emitter_utils
from doctrans.rest_docstring_parser import parse_docstring, _parse_line
from doctrans.tests.mocks.docstrings import (
    docstring_str,
    intermediate_repr,
    docstring_str_no_default_doc,
    intermediate_repr_no_default_doc_or_prop,
    intermediate_repr_no_default_doc,
)
from doctrans.tests.utils_for_tests import unittest_main


class TestMarshallDocstring(TestCase):
    """
    Tests whether docstrings are parsed out—and emitted—correctly
    """

    def test_ir_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `intermediate_repr`
              from `docstring_str` """
        self.assertDictEqual(
            parse_docstring(docstring_str), intermediate_repr_no_default_doc
        )

    def test_intermediate_repr_no_default_doc_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_str_no_default_doc`
              from `docstring_str_no_default_doc` """
        self.assertDictEqual(
            parse_docstring(docstring_str_no_default_doc, emit_default_doc=False),
            intermediate_repr_no_default_doc_or_prop,
        )

    def test_ir_equality_fails(self) -> None:
        """
        Tests whether `parse_docstring` produces `intermediate_repr`
              from and incorrect docstring """
        with self.assertRaises(AssertionError) as cte:
            parse_docstring(docstring_str.replace(":type K", ":type notOK"))
        self.assertEqual("'K' != 'notOK'", cte.exception.__str__())

    def test_ir2docstring(self) -> None:
        """ Tests whether `to_docstring` produces `docstring_str` from `intermediate_repr` """
        self.assertEqual(
            doctrans.emitter_utils.to_docstring(
                deepcopy(intermediate_repr),
                indent_level=0,
                emit_types=True,
                emit_default_doc=True,
                emit_separating_tab=False,
            ).strip(),
            docstring_str.strip(),
        )

    def test_docstring_line_parsing(self) -> None:
        """ Tests that the line parsing function works properly """
        lines = ":param foo: bar", " additional description", "type foo: str"

        default, docs, typ = None, [], []
        for line in lines:
            default = _parse_line(
                line, "foo", default, docs, typ, emit_default_doc=True
            )
        self.assertIsNone(default)
        self.assertListEqual(
            docs, [":param foo: bar", " additional description", "type foo: str"]
        )
        self.assertEqual(len(typ), 0)

        default, docs, typ = None, [], ["append_here"]
        for line in lines:
            default = _parse_line(
                line, "foo", default, docs, typ, emit_default_doc=True
            )
        self.assertIsNone(default)
        self.assertEqual(len(docs), 0)
        self.assertListEqual(
            typ,
            [
                "append_here",
                ":param foo: bar",
                " additional description",
                "type foo: str",
            ],
        )


unittest_main()

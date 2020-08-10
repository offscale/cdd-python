"""
Tests for docstring parsing
"""
from copy import deepcopy
from unittest import TestCase

from doctrans import parse
from doctrans.rest_docstring_parser import parse_docstring, _parse_line
from doctrans.tests.mocks.docstrings import (
    docstring_str,
    docstring_structure,
    docstring_str_no_default_doc,
    docstring_structure_no_default_doc_or_prop,
    docstring_structure_no_default_doc,
)
from doctrans.tests.utils_for_tests import unittest_main


class TestMarshallDocstring(TestCase):
    """
    Tests whether docstrings are parsed out—and emitted—correctly
    """

    def test_docstring_struct_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_structure`
              from `docstring_str` """
        self.assertDictEqual(
            parse_docstring(docstring_str), docstring_structure_no_default_doc
        )

    def test_docstring_structure_no_default_doc_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_str_no_default_doc`
              from `docstring_str_no_default_doc` """
        self.assertDictEqual(
            parse_docstring(docstring_str_no_default_doc, emit_default_doc=False),
            docstring_structure_no_default_doc_or_prop,
        )

    def test_docstring_struct_equality_fails(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_structure`
              from and incorrect docstring """
        with self.assertRaises(AssertionError) as cte:
            parse_docstring(docstring_str.replace(":type K", ":type notOK"))
        self.assertEqual("'K' != 'notOK'", cte.exception.__str__())

    def test_docstring_struct2docstring(self) -> None:
        """
        Tests whether `docstring_structure2docstring` produces `docstring_str`
              from `docstring_structure` """
        self.assertEqual(
            parse.to_docstring(
                deepcopy(docstring_structure),
                indent_level=0,
                emit_types=True,
                emit_default_doc=True,
                emit_separating_tab=False,
            ),
            docstring_str,
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

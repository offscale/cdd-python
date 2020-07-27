"""
Tests for docstring parsing
"""
from copy import deepcopy
from unittest import TestCase

from doctrans import docstring_struct
from doctrans.rest_docstring_parser import parse_docstring
from doctrans.tests.mocks.docstrings import docstring_str, docstring_structure, docstring_str_no_default_doc, \
    docstring_structure_no_default_doc
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
            parse_docstring(docstring_str),
            docstring_structure
        )

    def test_docstring_structure_no_default_doc_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_str_no_default_doc`
              from `docstring_str_no_default_doc` """
        self.assertDictEqual(
            parse_docstring(docstring_str_no_default_doc, emit_default_doc=False),
            docstring_structure_no_default_doc
        )

    def test_docstring_struct_equality_fails(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_structure`
              from and incorrect docstring """
        with self.assertRaises(AssertionError) as cte:
            parse_docstring(docstring_str.replace(':type K', ':type notOK'))
        self.assertEqual('\'K\' != \'notOK\'', cte.exception.__str__())

    def test_docstring_struct2docstring(self) -> None:
        """
        Tests whether `docstring_structure2docstring` produces `docstring_str`
              from `docstring_structure` """
        self.assertEqual(
            docstring_struct.to_docstring(deepcopy(docstring_structure), indent_level=0,
                                          emit_types=True, emit_default_doc=True,
                                          emit_separating_tab=False),
            docstring_str
        )


unittest_main()

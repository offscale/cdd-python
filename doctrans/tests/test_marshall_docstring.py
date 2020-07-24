"""
Tests for docstring parsing
"""
from copy import deepcopy
from unittest import TestCase, main as unittest_main

from doctrans.docstring_structure_utils import docstring_structure2docstring
from doctrans.rest_docstring_parser import parse_docstring
from doctrans.tests.mocks.docstrings import docstring_str, docstring_structure, docstring_str_no_default_doc, \
    docstring_structure_no_default_doc


class TestMarshallDocstring(TestCase):
    """
    Tests whether docstrings are parsed out—and emitted—correctly
    """
    maxDiff = 5555

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
            docstring_structure2docstring(deepcopy(docstring_structure), indent_level=0,
                                          emit_types=True, emit_default_doc=True,
                                          emit_separating_tab=False),
            docstring_str
        )


if __name__ == '__main__':
    unittest_main()

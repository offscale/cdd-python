"""
Tests for docstring parsing
"""
from unittest import TestCase, main as unittest_main

from doctrans.rest_docstring_parser import parse_docstring
from doctrans.tests.mocks.docstrings import docstring_str, docstring_structure


class TestParseDocstring(TestCase):
    """
    Tests whether docstrings are parsed out correctly
    """

    def test_docstring_struct_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_structure`
              from `docstring_str` """
        self.assertDictEqual(
            parse_docstring(docstring_str),
            docstring_structure
        )

    def test_docstring_struct_equality_fails(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_structure`
              from and incorrect docstring """
        with self.assertRaises(AssertionError) as cte:
            parse_docstring(docstring_str.replace(':type K', ':type notOK'))
        self.assertEqual('\'K\' != \'notOK\'', cte.exception.__str__())


if __name__ == '__main__':
    unittest_main()

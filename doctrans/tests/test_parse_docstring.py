from unittest import TestCase, main as unittest_main

from doctrans.info import parse_docstring
from doctrans.tests.mocks.docstrings import docstring_str, docstring_structure


class TestParseDocstring(TestCase):
    def test_docstring_struct_equality(self) -> None:
        self.assertDictEqual(
            parse_docstring(docstring_str),
            docstring_structure
        )

    def test_docstring_struct_equality_fails(self) -> None:
        with self.assertRaises(AssertionError) as cte:
            parse_docstring(docstring_str.replace(':type K', ':type notOK'))
        self.assertEqual('\'K\' != \'notOK\'', cte.exception.__str__())


if __name__ == '__main__':
    unittest_main()

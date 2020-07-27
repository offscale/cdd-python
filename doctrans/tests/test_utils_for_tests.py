"""
Tests for docstring parsing
"""
from io import StringIO
from unittest import TestCase
from unittest.mock import patch, MagicMock

from doctrans.tests.utils_for_tests import unittest_main


class TestUtilsForTests(TestCase):
    """
    Tests whether docstrings are parsed out—and emitted—correctly
    """

    def test_unittest_main(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_structure`
              from `docstring_str` """
        self.assertEqual(
            type(unittest_main).__name__, 'function'
        )
        self.assertIsNone(unittest_main())
        argparse_mock = MagicMock()
        with patch('doctrans.tests.utils_for_tests.__name__', '__main__'
                   ), patch('sys.stderr', new_callable=StringIO
                            ), self.assertRaises(SystemExit) as e:
            import doctrans.tests.utils_for_tests
            doctrans.tests.utils_for_tests.unittest_main()

        self.assertIsInstance(e.exception.code, bool)
        self.assertIsNone(argparse_mock.call_args)
        self.assertIsNone(doctrans.tests.utils_for_tests.unittest_main())


unittest_main()

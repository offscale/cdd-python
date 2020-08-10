"""
Tests for docstring parsing
"""
from ast import Module
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
        Tests whether `unittest_main` is called when `__name__ == '__main__'`
        """
        self.assertEqual(type(unittest_main).__name__, "function")
        self.assertIsNone(unittest_main())
        argparse_mock = MagicMock()
        with patch("doctrans.tests.utils_for_tests.__name__", "__main__"), patch(
            "sys.stderr", new_callable=StringIO
        ), self.assertRaises(SystemExit) as e:
            import doctrans.tests.utils_for_tests

            doctrans.tests.utils_for_tests.unittest_main()

        self.assertIsInstance(e.exception.code, bool)
        self.assertIsNone(argparse_mock.call_args)
        self.assertIsNone(doctrans.tests.utils_for_tests.unittest_main())

    def test_run_ast_test(self) -> None:
        """
        Tests whether `run_ast_test` correct avoids running the AST comparison dependent on Python version
        """

        def count_true(value, msg):
            """ Version of `self.assertTrue` which also keeps count

            :param value: Potentially `True`
            :type value: ```Union[Literal[True], Any]```

            :param msg: Message to raise in error
            :type msg: ```str```
            """
            assert value is True, msg
            count_true.i += 1

        count_true.i = 0

        assert_true = self.assertTrue
        module = Module(body=[], type_ignores=[])
        with patch("platform.python_version_tuple", lambda: ("3", "7")):
            import doctrans.tests.utils_for_tests

            self.assertTrue = count_true
            doctrans.tests.utils_for_tests.run_ast_test(self, module, module)
            self.assertEqual(count_true.i, 0)

        with patch("platform.python_version_tuple", lambda: ("3", "8")):
            import doctrans.tests.utils_for_tests

            self.assertTrue = count_true
            doctrans.tests.utils_for_tests.run_ast_test(self, module, module)
            self.assertEqual(count_true.i, 1)

        with patch("platform.python_version_tuple", lambda: ("3", "9")):
            import doctrans.tests.utils_for_tests

            self.assertTrue = count_true
            doctrans.tests.utils_for_tests.run_ast_test(self, module, module)
            self.assertEqual(count_true.i, 2)

        self.assertTrue = assert_true


unittest_main()

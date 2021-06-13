"""
Tests for docstring parsing
"""

from ast import Module
from collections import namedtuple
from io import StringIO
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, patch

from cdd.pure_utils import PY_GTE_3_8
from cdd.tests.mocks.classes import tensorboard_doc_str, tensorboard_doc_str_no_args_str
from cdd.tests.utils_for_tests import remove_args_from_docstring, unittest_main


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
        with patch("cdd.tests.utils_for_tests.__name__", "__main__"), patch(
            "sys.stderr", new_callable=StringIO
        ), self.assertRaises(SystemExit) as e:
            import cdd.tests.utils_for_tests

            cdd.tests.utils_for_tests.unittest_main()

        self.assertIsInstance(e.exception.code, bool)
        self.assertIsNone(argparse_mock.call_args)
        self.assertIsNone(cdd.tests.utils_for_tests.unittest_main())

    def test_run_ast_test(self) -> None:
        """
        Tests whether `run_ast_test` correct avoids running the AST comparison dependent on Python version
        """

        def assert_true(value, msg=None):
            """Version of `self.assertTrue` which also keeps count

            :param value: Potentially `True`
            :type value: ```Union[Literal[True], Any]```

            :param msg: Message to raise in error
            :type msg: ```Optional[str]```
            """
            TestUtilsForTests.i += TestUtilsForTests.increment
            assert value, msg or "{!r} not truthy".format(value)

        def assert_equal(a, b, msg=None):
            """Version of `self.assertEqual` which also keeps count

            :param a: Any value that can be compared. Compared with `b`.
            :type a: ```Any```

            :param b: Any value that can be compared. Compared with `a`.
            :type b: ```Any```

            :param msg: Message to raise in error
            :type msg: ```Optional[str]```
            """
            TestUtilsForTests.i += TestUtilsForTests.increment
            assert a == b, msg or "{!r} != {!r}".format(a, b)

        TestUtilsForTests.increment = 2 if PY_GTE_3_8 else 1
        TestUtilsForTests.i = 0

        test_case_module: Any = namedtuple("TestCase", ("assertTrue", "assertEqual"))(
            assert_true,
            assert_equal,
        )

        module = Module(body=[], type_ignores=[], stmt=None)

        import cdd.tests.utils_for_tests

        _orig_cdd_tests_utils_for_tests_PY3_8 = cdd.tests.utils_for_tests.PY3_8

        try:
            with patch("sys.version_info", (3, 7)):
                cdd.tests.utils_for_tests.PY_GTE_3_8 = (
                    cdd.tests.utils_for_tests.PY3_8
                ) = False

                cdd.tests.utils_for_tests.run_ast_test(test_case_module, module, module)
                self.assertEqual(TestUtilsForTests.increment * 2, TestUtilsForTests.i)

            with patch("sys.version_info", (3, 8)):
                cdd.tests.utils_for_tests.PY_GTE_3_8 = (
                    cdd.tests.utils_for_tests.PY3_8
                ) = True

                cdd.tests.utils_for_tests.run_ast_test(test_case_module, module, module)
                self.assertEqual(TestUtilsForTests.increment * 4, TestUtilsForTests.i)

            with patch("sys.version_info", (3, 9)):
                cdd.tests.utils_for_tests.PY_GTE_3_8 = (
                    cdd.tests.utils_for_tests.PY3_8
                ) = True

                cdd.tests.utils_for_tests.run_ast_test(test_case_module, module, module)
                self.assertEqual(TestUtilsForTests.increment * 6, TestUtilsForTests.i)
        finally:
            cdd.tests.utils_for_tests.PY3_8 = _orig_cdd_tests_utils_for_tests_PY3_8

    def test_remove_args_from_docstring(self) -> None:
        """
        Test that `remove_args_from_docstring`
         produces `tensorboard_doc_str_no_args_str`
             from `tensorboard_doc_str`
        """
        self.assertEqual(
            remove_args_from_docstring(tensorboard_doc_str),
            tensorboard_doc_str_no_args_str,
        )


unittest_main()

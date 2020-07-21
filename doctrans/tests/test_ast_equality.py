"""
Tests for AST equality
"""

from unittest import TestCase, main as unittest_main

from doctrans.tests.mocks.argparse import argparse_func_str, argparse_func_ast
from doctrans.tests.mocks.classes import class_str, class_ast
from doctrans.tests.utils_for_tests import run_ast_test


class TestAstEquality(TestCase):
    """
    Tests whether the AST generated matches the mocked one expected
    """

    def test_argparse_func(self) -> None:
        """ Tests whether the `argparse_func_str` correctly produces `argparse_func_ast` """
        run_ast_test(self, argparse_func_str, argparse_func_ast)

    def test_class(self) -> None:
        """ Tests whether the `class_str` correctly produces `class_ast` """
        run_ast_test(self, class_str, class_ast)


if __name__ == '__main__':
    unittest_main()

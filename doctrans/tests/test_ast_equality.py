"""
Tests for AST equality
"""
import ast
from unittest import TestCase

from doctrans.tests.mocks.argparse import argparse_func_ast, argparse_func_str
from doctrans.tests.mocks.classes import class_ast, class_str
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main


class TestAstEquality(TestCase):
    """
    Tests whether the AST generated matches the mocked one expected
    """

    def test_argparse_func(self) -> None:
        """ Tests whether the `argparse_func_str` correctly produces `argparse_func_ast` """
        run_ast_test(self, ast.parse(argparse_func_str).body[0], argparse_func_ast)

    def test_class(self) -> None:
        """ Tests whether the `class_str` correctly produces `class_ast` """
        run_ast_test(self, class_str, class_ast)


unittest_main()

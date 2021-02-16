"""
Tests for AST equality
"""
import ast
from unittest import TestCase

from cdd.tests.mocks.argparse import argparse_func_ast, argparse_func_str
from cdd.tests.mocks.classes import class_ast, class_str
from cdd.tests.utils_for_tests import reindent_docstring, run_ast_test, unittest_main


class TestAstEquality(TestCase):
    """
    Tests whether the AST generated matches the mocked one expected
    """

    def test_argparse_func(self) -> None:
        """ Tests whether the `argparse_func_str` correctly produces `argparse_func_ast` """
        run_ast_test(
            self,
            *map(
                reindent_docstring,
                (ast.parse(argparse_func_str).body[0], argparse_func_ast),
            )
        )

    def test_class(self) -> None:
        """ Tests whether the `class_str` correctly produces `class_ast` """
        run_ast_test(self, class_str, class_ast)


unittest_main()

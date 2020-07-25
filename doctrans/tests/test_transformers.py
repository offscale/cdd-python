"""
Tests for marshalling between formats
"""
from ast import FunctionDef
from unittest import TestCase, main as unittest_main

from doctrans.pure_utils import rpartial
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_str
from doctrans.tests.mocks.methods import class_with_method_types_ast
from doctrans.tests.utils_for_tests import run_ast_test
from doctrans.transformers import ArgparseTransform, ClassTransform, DocstringTransform


class TestTransformers(TestCase):
    """ Tests whether conversion between formats works """

    def test_argparse2class(self) -> None:
        """
        Tests whether `to_class` produces `class_ast` given `argparse_func_ast`
        """
        run_ast_test(
            self,
            ArgparseTransform(
                argparse_func_ast,
                inline_types=False,
                emit_default_doc=True
            ).to_class(),
            gold=class_ast
        )

    def test_class2argparse(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_func_ast` given `class_ast`
        """
        run_ast_test(
            self,
            ClassTransform(
                class_ast,
                inline_types=True,
                emit_default_doc=False
            ).to_argparse(),
            gold=argparse_func_ast
        )

    def test_docstring2class(self) -> None:
        """
        Tests whether `to_class` produces `class_ast` given `docstring_str`
        """
        run_ast_test(
            self,
            DocstringTransform(
                docstring_str,
                inline_types=True,
                emit_default_doc=True
            ).to_class(),
            gold=class_ast
        )

    def test_class2docstring(self) -> None:
        """
        Tests whether `to_docstring` produces `docstring_str` given `class_ast`
        """
        self.assertEqual(
            ClassTransform(
                class_ast,
                inline_types=True,
                emit_default_doc=False
            ).to_docstring(),
            docstring_str
        )

    def test_docstring2function(self) -> None:
        """
        Tests whether `to_function` produces method from `class_with_method_types_ast` given `docstring_str`
        """
        run_ast_test(
            self,
            DocstringTransform(
                docstring_str,
                inline_types=True,
                emit_default_doc=False
            ).to_function(
                'method_name',
                function_type='self'
            ),
            gold=next(filter(rpartial(isinstance, FunctionDef),
                             class_with_method_types_ast.body))
        )


if __name__ == '__main__':
    unittest_main()

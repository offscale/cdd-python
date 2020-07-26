"""
Tests for marshalling between formats
"""
import os
from ast import FunctionDef, parse
from sys import version
from unittest import TestCase, main as unittest_main

from meta.asttools import cmp_ast

from doctrans import transformers, docstring_struct
from doctrans.pure_utils import rpartial
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_str, docstring_structure
from doctrans.tests.mocks.methods import class_with_method_types_ast
from doctrans.tests.utils_for_tests import run_ast_test


class TestTransformers(TestCase):
    """ Tests whether conversion between formats works """

    def test_to_class(self) -> None:
        """
        Tests whether `to_class` produces `class_ast` given `argparse_func_ast`
        """
        run_ast_test(
            self,
            transformers.to_class(
                docstring_struct.from_argparse_ast(argparse_func_ast,
                                                   emit_default_doc=True)
            ),
            gold=class_ast
        )

        # Tests whether `to_class` produces `class_ast` given `docstring_str`
        run_ast_test(
            self,
            transformers.to_class(
                docstring_struct.from_docstring(docstring_str)
            ),
            gold=class_ast
        )

    def test_to_argparse(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_func_ast` given `class_ast`
        """
        run_ast_test(
            self,
            transformers.to_argparse(
                docstring_struct.from_class(class_ast, emit_default_doc=False),
                emit_default_doc=False
            ),
            gold=argparse_func_ast
        )

    def test_to_docstring(self) -> None:
        """
        Tests whether `to_docstring` produces `docstring_str` given `class_ast`
        """
        self.assertEqual(
            transformers.to_docstring(
                docstring_struct.from_class(class_ast)
            ),
            docstring_str
        )

    def test_to_numpy_docstring(self) -> None:
        """
        Tests whether `to_docstring` fails when `docstring_format` is 'numpy'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: transformers.to_docstring(docstring_structure,
                                              docstring_format='numpy')
        )

    def test_to_google_docstring(self) -> None:
        """
        Tests whether `to_docstring` fails when `docstring_format` is 'google'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: transformers.to_docstring(docstring_structure,
                                              docstring_format='google')
        )

    def test_to_file(self):
        filename = os.path.join(os.path.dirname(__file__),
                                'delete_me.py')
        try:
            transformers.to_file(
                class_ast,
                filename,
                skip_black=True
            )

            with open(filename, 'rt') as f:
                ugly = f.read()

            os.remove(filename)

            transformers.to_file(
                class_ast,
                filename,
                skip_black=False
            )

            with open(filename, 'rt') as f:
                blacked = f.read()

            self.assertNotEqual(ugly, blacked)
            if version[:3] == '3.8':
                self.assertTrue(cmp_ast(parse(ugly), parse(blacked)),
                                'Ugly AST doesn\'t match blacked AST')

        finally:
            if os.path.isfile(filename):
                os.remove(filename)

    def test_to_function(self) -> None:
        """
        Tests whether `to_function` produces method from `class_with_method_types_ast` given `docstring_str`
        """
        run_ast_test(
            self,
            transformers.to_function(
                docstring_struct.from_docstring(docstring_str,
                                                emit_default_doc=False),
                function_name='method_name',
                function_type='self',
                emit_default_doc=False
            ),
            gold=next(filter(rpartial(isinstance, FunctionDef),
                             class_with_method_types_ast.body))
        )


if __name__ == '__main__':
    unittest_main()

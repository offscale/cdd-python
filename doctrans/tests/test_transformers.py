"""
Tests for marshalling between formats
"""
import os
from ast import FunctionDef, parse
from unittest import TestCase

from meta.asttools import cmp_ast

from doctrans import transformers, docstring_struct
from doctrans.ast_utils import get_function_type
from doctrans.pure_utils import rpartial, PY3_8
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_str, docstring_structure
from doctrans.tests.mocks.methods import class_with_method_types_ast, class_with_method_ast
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main


class TestTransformers(TestCase):
    """ Tests whether conversion between formats works """

    def test_to_class_from_argparse_ast(self) -> None:
        """
        Tests whether `to_class` produces `class_ast` given `argparse_func_ast`
        """
        run_ast_test(
            self,
            transformers.to_class(
                docstring_struct.from_argparse_ast(argparse_func_ast)
            ),
            gold=class_ast
        )

    def test_to_class_from_docstring_str(self) -> None:
        """
        Tests whether `to_class` produces `class_ast` given `docstring_str`
        """
        run_ast_test(
            self,
            transformers.to_class(
                docstring_struct.from_docstring(docstring_str),
            ),
            gold=class_ast
        )

    def test_to_argparse(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_func_ast` given `class_ast`
        """
        self.assertEqual(transformers.to_source(transformers.to_argparse(
            docstring_struct.from_class(class_ast),
            emit_default_doc=False
        )), transformers.to_source(argparse_func_ast))
        run_ast_test(
            self,
            transformers.to_argparse(
                docstring_struct.from_class(class_ast),
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

    def test_to_numpy_docstring_fails(self) -> None:
        """
        Tests whether `to_docstring` fails when `docstring_format` is 'numpy'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: transformers.to_docstring(docstring_structure,
                                              docstring_format='numpy')
        )

    def test_to_google_docstring_fails(self) -> None:
        """
        Tests whether `to_docstring` fails when `docstring_format` is 'google'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: transformers.to_docstring(docstring_structure,
                                              docstring_format='google')
        )

    def test_to_file(self) -> None:
        """
        Tests whether `to_file` constructs a file, and fills it with the right content
        """

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
            # if PY3_8:
            self.assertTrue(cmp_ast(parse(ugly), parse(blacked)),
                            'Ugly AST doesn\'t match blacked AST')

        finally:
            if os.path.isfile(filename):
                os.remove(filename)

    def test_to_function(self) -> None:
        """
        Tests whether `to_function` produces method from `class_with_method_types_ast` given `docstring_str`
        """
        function_def = next(filter(rpartial(isinstance, FunctionDef),
                                   class_with_method_types_ast.body))
        run_ast_test(
            self,
            transformers.to_function(docstring_struct.from_docstring(docstring_str),
                                     function_name=function_def.name,
                                     function_type=get_function_type(function_def),
                                     emit_default_doc=False,
                                     inline_types=True,
                                     emit_separating_tab=PY3_8),
            gold=function_def
        )

    def test_to_function_with_docstring_types(self) -> None:
        """
        Tests that `to_function` can generate a function with types in docstring
        """
        function_def = next(filter(rpartial(isinstance, FunctionDef),
                                   class_with_method_ast.body))
        run_ast_test(
            self,
            transformers.to_function(docstring_struct.from_function(function_def),
                                     function_name=function_def.name,
                                     function_type=get_function_type(function_def),
                                     emit_default_doc=False,
                                     inline_types=False,
                                     indent_level=2,
                                     emit_separating_tab=False),
            gold=function_def
        )

    def test_to_function_with_inline_types(self) -> None:
        """
        Tests that `to_function` can generate a function with inline types
        """
        function_def = next(filter(rpartial(isinstance, FunctionDef),
                                   class_with_method_types_ast.body))
        # transformers.to_file(gen_ast, os.path.join(os.path.dirname(__file__), 'delme.py'))
        run_ast_test(
            self,
            transformers.to_function(docstring_struct.from_function(function_def),
                                     function_name=function_def.name,
                                     function_type=get_function_type(function_def),
                                     emit_default_doc=False,
                                     inline_types=True,
                                     emit_separating_tab=PY3_8),
            gold=function_def
        )


unittest_main()

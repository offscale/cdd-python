"""
Tests for reeducation
"""
from _ast import ClassDef, FunctionDef
from copy import deepcopy
from unittest import TestCase

from doctrans import docstring_struct
from doctrans.conformance import replace_node
from doctrans.pure_utils import rpartial
from doctrans.source_transformer import to_code
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_structure
from doctrans.tests.mocks.methods import (
    class_with_method_types_ast,
)
from doctrans.tests.utils_for_tests import unittest_main


class TestConformance(TestCase):
    """
    Tests must comply. They shall be assimilated.
    """

    def test_ground_truth(self) -> None:
        """ Straight from the ministry. Absolutely. """
        # ground_truth()

    def test_replace_node(self) -> None:
        """ Tests `replace_node` """
        _docstring_structure = deepcopy(docstring_structure)
        same, found = replace_node(
            fun_name="argparse_function",
            from_func=docstring_struct.from_argparse_ast,
            outer_name="set_cli_args",
            inner_name=None,
            outer_node=argparse_func_ast,
            inner_node=None,
            docstring_structure=_docstring_structure,
            typ=FunctionDef,
        )
        self.assertEqual(*map(to_code, (argparse_func_ast, found)))
        self.assertTrue(same)

        same, found = replace_node(
            fun_name="class",
            from_func=docstring_struct.from_class,
            outer_name="ConfigClass",
            inner_name=None,
            outer_node=class_ast,
            inner_node=None,
            docstring_structure=_docstring_structure,
            typ=ClassDef,
        )
        self.assertEqual(*map(to_code, (class_ast, found)))
        self.assertTrue(same)

        function_def = next(
            filter(rpartial(isinstance, FunctionDef), class_with_method_types_ast.body)
        )
        same, found = replace_node(
            fun_name="function",
            from_func=docstring_struct.from_class_with_method,
            outer_name="C",
            inner_name="method_name",
            outer_node=class_with_method_types_ast,
            inner_node=function_def,
            docstring_structure=_docstring_structure,
            typ=FunctionDef,
        )
        self.assertEqual(*map(to_code, (function_def, found)))
        self.assertTrue(same)


unittest_main()

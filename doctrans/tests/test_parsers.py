"""
Tests for the Intermediate Representation produced by the parsers
"""
from ast import FunctionDef
from unittest import TestCase

from doctrans import parse, emit
from doctrans.ast_utils import get_value
from doctrans.pure_utils import tab, PY_GTE_3_8
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import (
    intermediate_repr_no_default_doc,
)
from doctrans.tests.mocks.ir import method_complex_args_variety_ir
from doctrans.tests.mocks.methods import (
    function_adder_ast,
    function_default_complex_default_arg_ast,
    method_complex_args_variety_ast,
    method_complex_args_variety_str,
    function_adder_str,
    function_adder_ir,
)
from doctrans.tests.utils_for_tests import unittest_main, inspectable_compile


class TestParsers(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    """

    def test_from_argparse_ast(self) -> None:
        """
        Tests whether `argparse_ast` produces `intermediate_repr_no_default_doc_or_prop`
              from `argparse_func_ast`"""
        self.assertDictEqual(
            parse.argparse_ast(argparse_func_ast), intermediate_repr_no_default_doc
        )

    def test_from_argparse_ast_empty(self) -> None:
        """
        Tests `argparse_ast` empty condition
        """
        self.assertEqual(
            emit.to_code(
                emit.argparse_function(
                    parse.argparse_ast(
                        FunctionDef(
                            body=[],
                            arguments_args=None,
                            identifier_name=None,
                            stmt=None,
                        )
                    ),
                    emit_default_doc=True,
                )
            ).rstrip("\n"),
            "def set_cli_args(argument_parser):\n"
            '{tab}"""\n'
            "{tab}Set CLI arguments\n\n"
            "{tab}:param argument_parser: argument parser\n"
            "{tab}:type argument_parser: ```ArgumentParser```\n\n"
            "{tab}:return: argument_parser\n"
            "{tab}:rtype: ```ArgumentParser```\n"
            '{tab}"""\n'
            "{tab}argument_parser.description = ''\n"
            "{tab}return argument_parser".format(tab=tab),
        )

    def test_from_class(self) -> None:
        """
        Tests whether `class_` produces `intermediate_repr_no_default_doc`
              from `class_ast`
        """
        ir = parse.class_(class_ast)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(ir, intermediate_repr_no_default_doc)

    def test_from_function(self) -> None:
        """
        Tests that parse.function produces properly
        """
        gen_ir = parse.function(function_default_complex_default_arg_ast)
        gold_ir = {
            "description": "",
            "name": "call_peril",
            "params": [
                {
                    "default": "mnist",
                    "name": "dataset_name",
                    "typ": "str",
                    "doc": None,
                },
                {
                    "default": get_value(
                        function_default_complex_default_arg_ast.args.defaults[1]
                    ),
                    "name": "writer",
                    "typ": None,
                    "doc": None,
                },
            ],
            "returns": None,
            "type": "static",
        }
        self.assertDictEqual(
            gen_ir,
            gold_ir,
        )

    def test_from_function_kw_only(self) -> None:
        """
        Tests that parse.function produces properly from function with only keyword arguments
        """
        self.assertDictEqual(
            parse.function(function_adder_ast),
            function_adder_ir,
        )

    def test_from_function_in_memory(self) -> None:
        """
        Tests that parse.function produces properly from a function in memory of current interpreter
        """

        def foo(a=5, b=6):
            """
            the foo function

            :param a: the a value
            :param b: the b value

            """
            pass

        self.assertIsNone(foo(5, 6))

        ir = parse.function(foo)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            ir,
            {
                "doc": "the foo function",
                "name": "TestParsers.test_from_function_in_memory.<locals>.foo",
                "params": [
                    {"default": 5, "doc": "the a value", "name": "a", "typ": "int"},
                    {"default": 6, "doc": "the b value", "name": "b", "typ": "int"},
                ],
                "returns": None,
                "type": "static",
            },
        )

    maxDiff = None

    def test_from_method_in_memory(self) -> None:
        """
        Tests that `parse.function` produces properly from a function in memory of current interpreter with:
        - kw only args;
        - default args;
        - annotations
        - required;
        - unannotated;
        - splat
        """

        method_complex_args_variety_with_imports_str = (
            "from sys import stdout\n"
            "from {} import Literal\n"
            "{}".format(
                "typing" if PY_GTE_3_8 else "typing_extensions",
                method_complex_args_variety_str,
            )
        )
        call_cliff = getattr(
            inspectable_compile(method_complex_args_variety_with_imports_str),
            "call_cliff",
        )

        ir = parse.function(call_cliff)
        del ir["_internal"]  # Not needed for this test

        # This is a hack because JetBrains wraps stdout
        self.assertIn(
            type(ir["params"][-2]["default"]).__name__,
            frozenset(("FlushingStringIO", "TextIOWrapper")),
        )
        ir["params"][-2]["default"] = "stdout"

        # This extra typ is removed, for now. TODO: Update AST-level parser to set types when defaults are given.
        for i in -2, 3:
            del ir["params"][i]["typ"]

        self.assertDictEqual(
            ir,
            method_complex_args_variety_ir,
        )

    def test_from_method_in_memory_return_complex(self) -> None:
        """
        Tests that `parse.function` produces properly from a function in memory of current interpreter with:
        - complex return type
        - kwonly args
        """

        method_complex_args_variety_with_imports_str = (
            "from sys import stdout\n"
            "from {} import Literal\n"
            "{}".format(
                "typing" if PY_GTE_3_8 else "typing_extensions",
                function_adder_str,
            )
        )
        add_6_5 = getattr(
            inspectable_compile(method_complex_args_variety_with_imports_str),
            "add_6_5",
        )

        ir = parse.function(add_6_5)
        del ir["_internal"]  # Not needed for this test

        self.assertDictEqual(
            ir,
            function_adder_ir,
        )

    def test_from_method_complex_args_variety(self) -> None:
        """
        Tests that `parse.function` produces correctly with:
        - kw only args;
        - default args;
        - annotations
        - required;
        - unannotated;
        - splat
        """
        self.assertDictEqual(
            parse.function(method_complex_args_variety_ast),
            method_complex_args_variety_ir,
        )

    def test_from_class_in_memory(self) -> None:
        """
        Tests that parse.class produces properly from a `class` in memory of current interpreter
        """

        class A(object):
            """A is one boring class"""

        ir = parse.class_(A)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            ir,
            {
                "doc": "A is one boring class",
                "name": "TestParsers.test_from_class_in_memory.<locals>.A",
                "params": [],
                "returns": None,
            },
        )


unittest_main()

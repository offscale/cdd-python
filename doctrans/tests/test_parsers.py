"""
Tests for the Intermediate Representation produced by the parsers
"""
from ast import FunctionDef
from unittest import TestCase

from docstring_parser import rest

from doctrans import parse, emit
from doctrans.ast_utils import get_value
from doctrans.emitter_utils import to_docstring
from doctrans.pure_utils import tab
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import (
    docstring_str,
    intermediate_repr_no_default_doc,
)
from doctrans.tests.mocks.methods import (
    function_adder_ast,
    function_default_complex_default_arg_ast,
    method_complex_args_variety_ast,
)
from doctrans.tests.utils_for_tests import unittest_main


class TestParsers(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
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
        self.assertDictEqual(parse.class_(class_ast), intermediate_repr_no_default_doc)

    def test_from_docstring(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_str`"""
        ir, returns = parse.docstring(docstring_str, return_tuple=True)
        self.assertTrue(returns)
        self.assertDictEqual(ir, intermediate_repr_no_default_doc)

    def test_to_docstring_fails(self) -> None:
        """
        Tests docstring failure conditions
        """
        self.assertRaises(
            NotImplementedError,
            lambda: to_docstring(
                intermediate_repr_no_default_doc, docstring_format="numpy"
            ),
        )

    def test_from_docstring_parser(self) -> None:
        """
        Tests if it can convert from the 3rd-party libraries format to this one
        """
        self.assertDictEqual(
            parse.docstring_parser(
                rest.parse(
                    "[Summary]\n\n"
                    ":param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]\n"
                    ":type [ParamName]: [ParamType](, optional)\n\n"
                    ":raises [ErrorType]: [ErrorDescription]\n\n"
                    ":return: [ReturnDescription]\n"
                    ":rtype: [ReturnType]\n"
                )
            ),
            {
                "params": [
                    {
                        "default": "[DefaultParamVal]",
                        "doc": "[ParamDescription]",
                        "name": "[ParamName]",
                    }
                ],
                "raises": [
                    {
                        "doc": "[ErrorDescription]",
                        "name": "raises",
                        "typ": "[ErrorType]",
                    }
                ],
                "returns": {"doc": "[ReturnDescription]", "name": "return_type"},
                "short_description": "[Summary]",
            },
        )

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

    def test_from_function_kw_only(self):
        """
        Tests that parse.function produces properly from function with only keyword arguments
        """
        self.assertDictEqual(
            parse.function(function_adder_ast),
            {
                "long_description": "",
                "name": "add_6_5",
                "params": [
                    {"default": 6, "doc": "first param", "name": "a", "typ": "int"},
                    {"default": 5, "doc": "second param", "name": "b", "typ": "int"},
                ],
                "returns": {
                    "default": "operator.add(a, b)",
                    "doc": "Aggregated summation of `a` and `b`.",
                    "name": "return_type",
                    "typ": "int",
                },
                "short_description": "",
                "type": "static",
            },
        )

    def test_from_function_actual(self):
        """
        Tests that parse.function produces properly from an actual function
        """

        def foo(a=5, b=6):
            pass

        self.assertDictEqual(
            parse.function(foo),
            {
                "long_description": None,
                "name": "TestParsers.test_from_function_actual.<locals>.foo",
                "params": [
                    {"default": 5, "name": "a", "typ": "int"},
                    {"default": 6, "name": "b", "typ": "int"},
                ],
                "returns": None,
            },
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
            {
                "long_description": "",
                "name": "call_cliff",
                "params": [
                    {"doc": "name of dataset.", "name": "dataset_name"},
                    {"doc": "Convert to numpy ndarrays", "name": "as_numpy"},
                    {
                        "doc": "backend engine, e.g., `np` or `tf`.",
                        "name": "K",
                        "typ": "Literal['np', 'tf']",
                    },
                    {
                        "default": "~/tensorflow_datasets",
                        "doc": "directory to look for models in.",
                        "name": "tfds_dir",
                    },
                    {
                        "default": "stdout",
                        "doc": "IO object to write out to",
                        "name": "writer",
                    },
                    {
                        "doc": "additional keyword arguments",
                        "name": "kwargs",
                        "typ": "dict",
                    },
                ],
                "returns": {
                    "default": "K",
                    "doc": "backend engine",
                    "name": "return_type",
                    "typ": "Literal['np', 'tf']",
                },
                "short_description": "Call cliff",
                "type": "self",
            },
        )


unittest_main()

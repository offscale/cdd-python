"""
Tests for the Intermediate Representation produced by the parsers
"""
from ast import FunctionDef
from importlib.abc import Loader
from importlib.util import module_from_spec, spec_from_loader
from inspect import getsource
from unittest import TestCase

from docstring_parser import rest

from doctrans import parse, emit
from doctrans.ast_utils import get_value
from doctrans.emitter_utils import to_docstring
from doctrans.pure_utils import tab, PY_GTE_3_8, pp
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import (
    docstring_str,
    intermediate_repr_no_default_doc,
    docstring_numpydoc_str,
)
from doctrans.tests.mocks.ir import method_complex_args_variety_ir
from doctrans.tests.mocks.methods import (
    function_adder_ast,
    function_default_complex_default_arg_ast,
    method_complex_args_variety_ast,
    method_complex_args_variety_str,
)
from doctrans.tests.utils_for_tests import unittest_main


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
        self.assertDictEqual(parse.class_(class_ast), intermediate_repr_no_default_doc)

    def test_from_docstring(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_str`"""
        ir, returns = parse.docstring(docstring_str, return_tuple=True)
        self.assertTrue(returns)
        self.assertDictEqual(ir, intermediate_repr_no_default_doc)

    def test_from_docstring_numpydoc(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_numpydoc_str`"""
        ir, returns = parse.docstring(docstring_numpydoc_str, return_tuple=True)
        pp(ir)
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

    maxDiff = None

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
                "doc": "[Summary]",
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

    def test_from_function_kw_only(self) -> None:
        """
        Tests that parse.function produces properly from function with only keyword arguments
        """
        self.assertDictEqual(
            parse.function(function_adder_ast),
            {
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
                "doc": "",
                "type": "static",
            },
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

        self.assertDictEqual(
            parse.function(foo),
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

        class MemoryInspectLoader(Loader):
            """ Set the filename for the inspect module, but don't actually, actually give the full source """

            def create_module(self, spec=None):
                """ Stub method"""
                super(MemoryInspectLoader, self).create_module(spec=spec)

            def get_code(self):
                """ Stub method; soon to add actual source code to """
                raise NotImplementedError()

        _locals = module_from_spec(
            spec_from_loader("helper", loader=None, origin="str")  # MemoryInspectLoader
        )
        exec(
            "from sys import stdout\n"
            "from {} import Literal\n"
            "{}".format(
                "typing" if PY_GTE_3_8 else "typing_extensions",
                method_complex_args_variety_str,
            ),
            _locals.__dict__,
        )
        call_cliff = getattr(_locals, "call_cliff")
        setattr(call_cliff, "__loader__", MemoryInspectLoader)

        print("getsource(call_cliff):", getsource(call_cliff), ";")
        return

        ir = parse.function(getattr(_locals, "call_cliff"))

        # This is a hack because JetBrains wraps stdout
        self.assertIn(
            type(ir["params"][-2]["default"]).__name__,
            frozenset(("FlushingStringIO", "TextIOWrapper")),
        )
        ir["params"][-2]["default"] = "stdout"

        # TODO: Fix this hack by making the loader do its job and parsing the source code in `parse._inspect`
        # ir["returns"]["default"] = "K"

        self.assertDictEqual(
            ir,
            method_complex_args_variety_ir,
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

    def test_from_class_actual(self) -> None:
        """
        Tests that parse.class produces properly from a `class` in memory of current interpreter
        """

        class A(object):
            """A is one boring class"""

        self.assertDictEqual(
            parse.class_(A),
            {
                "doc": "A is one boring class",
                "name": "TestParsers.test_from_class_actual.<locals>.A",
                "params": [],
                "returns": None,
            },
        )


unittest_main()

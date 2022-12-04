"""
Tests for the Intermediate Representation produced by the function parser
"""

from collections import OrderedDict
from copy import deepcopy
from unittest import TestCase

import cdd.emit.argparse_function
import cdd.parse.class_
import cdd.parse.function
import cdd.parse.json_schema
from cdd.ast_utils import get_value
from cdd.pure_utils import PY_GTE_3_8, paren_wrap_code
from cdd.tests.mocks.ir import (
    function_adder_ir,
    function_google_tf_ops_losses__safe_mean_ir,
    method_complex_args_variety_ir,
)
from cdd.tests.mocks.methods import (
    function_adder_ast,
    function_adder_str,
    function_default_complex_default_arg_ast,
    function_google_tf_ops_losses__safe_mean_ast,
    method_complex_args_variety_ast,
    method_complex_args_variety_str,
)
from cdd.tests.utils_for_tests import inspectable_compile, unittest_main


class TestParseFunction(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    """

    def test_from_function(self) -> None:
        """
        Tests that cdd.parse.function.function produces properly
        """
        gen_ir = cdd.parse.function.function(function_default_complex_default_arg_ast)
        gold_ir = {
            "name": "call_peril",
            "params": OrderedDict(
                (
                    (
                        "dataset_name",
                        {"default": "mnist", "typ": "str"},
                    ),
                    (
                        "writer",
                        {
                            "default": "```{}```".format(
                                paren_wrap_code(
                                    get_value(
                                        function_default_complex_default_arg_ast.args.defaults[
                                            1
                                        ]
                                    )
                                )
                            ),
                        },
                    ),
                )
            ),
            "returns": None,
            "type": "static",
        }

        del gen_ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            gen_ir,
            gold_ir,
        )

    def test_from_function_kw_only(self) -> None:
        """
        Tests that cdd.parse.function.function produces properly from function with only keyword arguments
        """
        gen_ir = cdd.parse.function.function(function_adder_ast, function_type="static")
        del gen_ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            gen_ir,
            function_adder_ir,
        )

    def test_from_function_in_memory(self) -> None:
        """
        Tests that cdd.parse.function.function produces properly from a function in memory of current interpreter
        """

        def foo(a=5, b=6):
            """
            the foo function

            :param a: the a value
            :param b: the b value

            """

        self.assertIsNone(foo(5, 6))

        ir = cdd.parse.function.function(foo)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            ir,
            {
                "doc": "the foo function",
                "name": "TestParseFunction.test_from_function_in_memory.<locals>.foo",
                "params": OrderedDict(
                    (
                        ("a", {"default": 5, "doc": "the a value", "typ": "int"}),
                        ("b", {"default": 6, "doc": "the b value", "typ": "int"}),
                    )
                ),
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

        method_complex_args_variety_with_imports_str = (
            "from sys import stdout\n"
            "from {package} import Literal\n"
            "{body}".format(
                package="typing" if PY_GTE_3_8 else "typing_extensions",
                body=method_complex_args_variety_str,
            )
        )
        call_cliff = getattr(
            inspectable_compile(method_complex_args_variety_with_imports_str),
            "call_cliff",
        )

        ir = cdd.parse.function.function(call_cliff)
        del ir["_internal"]  # Not needed for this test

        # This is a hack because JetBrains wraps stdout
        self.assertIn(
            type(ir["params"]["writer"]["default"]).__name__,
            frozenset(("FlushingStringIO", "TextIOWrapper")),
        )

        # This extra typ is copied, for now. TODO: Update AST-level parser to set types when defaults are given.

        del ir["params"]["writer"]["typ"]
        ir["params"]["writer"]["default"] = "```{}```".format(paren_wrap_code("stdout"))

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
            "from {package} import Literal\n"
            "{body}".format(
                package="typing" if PY_GTE_3_8 else "typing_extensions",
                body=function_adder_str,
            )
        )
        add_6_5 = getattr(
            inspectable_compile(method_complex_args_variety_with_imports_str),
            "add_6_5",
        )

        ir = cdd.parse.function.function(add_6_5)
        del ir["_internal"]  # Not needed for this test

        self.assertDictEqual(
            ir,
            function_adder_ir,
        )

    maxDiff = None

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
        gen_ir = cdd.parse.function.function(method_complex_args_variety_ast)
        del gen_ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            gen_ir,
            method_complex_args_variety_ir,
        )

    def test_from_function_google_tf_ops_losses__safe_mean_ast(self) -> None:
        """Tests IR from function_google_tf_ops_losses__safe_mean_ast"""
        ir = cdd.parse.function.function(function_google_tf_ops_losses__safe_mean_ast)
        _internal = ir.pop("_internal")
        for key in "original_doc_str", "body":
            del _internal[key]
        self.assertDictEqual(
            _internal, {"from_name": "_safe_mean", "from_type": "static"}
        )
        self.assertEqual(
            ir["returns"]["return_type"]["doc"],
            function_google_tf_ops_losses__safe_mean_ir["returns"]["return_type"][
                "doc"
            ],
        )
        self.assertDictEqual(ir, function_google_tf_ops_losses__safe_mean_ir)

        no_body = deepcopy(function_google_tf_ops_losses__safe_mean_ast)
        del no_body.body[1:]
        ir = cdd.parse.function.function(no_body)
        del ir["_internal"]
        gold = deepcopy(function_google_tf_ops_losses__safe_mean_ir)
        gold["returns"]["return_type"] = {
            "doc": function_google_tf_ops_losses__safe_mean_ir["returns"][
                "return_type"
            ]["doc"]
        }
        self.assertDictEqual(ir, gold)


unittest_main()

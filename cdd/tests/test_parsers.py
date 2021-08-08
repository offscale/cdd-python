"""
Tests for the Intermediate Representation produced by the parsers
"""

import ast
from ast import FunctionDef
from collections import OrderedDict
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from cdd import emit, parse
from cdd.ast_utils import RewriteAtQuery, get_value
from cdd.pure_utils import PY_GTE_3_8, paren_wrap_code, tab
from cdd.tests.mocks.argparse import argparse_func_ast
from cdd.tests.mocks.classes import (
    class_ast,
    class_google_tf_tensorboard_ast,
    class_google_tf_tensorboard_str,
    class_torch_nn_l1loss_ast,
    class_torch_nn_l1loss_str,
    class_torch_nn_one_cycle_lr_ast,
    class_torch_nn_one_cycle_lr_str,
)
from cdd.tests.mocks.ir import (
    class_google_tf_tensorboard_ir,
    class_torch_nn_l1loss_ir,
    class_torch_nn_one_cycle_lr_ir,
    docstring_google_tf_adadelta_function_ir,
    function_adder_ir,
    intermediate_repr_no_default_doc,
    intermediate_repr_no_default_sql_doc,
    method_complex_args_variety_ir,
)
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.mocks.methods import (
    docstring_google_tf_adadelta_function_str,
    function_adder_ast,
    function_adder_str,
    function_default_complex_default_arg_ast,
    method_complex_args_variety_ast,
    method_complex_args_variety_str,
)
from cdd.tests.mocks.sqlalchemy import (
    config_decl_base_ast,
    config_decl_base_str,
    config_tbl_ast,
    config_tbl_str,
)
from cdd.tests.utils_for_tests import inspectable_compile, run_ast_test, unittest_main


class TestParsers(TestCase):
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

    maxDiff = None

    def test_from_argparse_ast(self) -> None:
        """
        Tests whether `argparse_ast` produces `intermediate_repr_no_default_doc`
              from `argparse_func_ast`"""
        ir = parse.argparse_ast(argparse_func_ast)
        del ir["_internal"]  # Not needed for this test
        _intermediate_repr_no_default_doc = deepcopy(intermediate_repr_no_default_doc)
        _intermediate_repr_no_default_doc["name"] = "set_cli_args"
        self.assertDictEqual(ir, _intermediate_repr_no_default_doc)

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
                            name=None,
                            arguments_args=None,
                            identifier_name=None,
                            stmt=None,
                        )
                    ),
                    emit_default_doc=True,
                )
            ).rstrip("\n"),
            "def set_cli_args(argument_parser):\n{tab}{body}".format(
                tab=tab,
                body=tab.join(
                    (
                        '"""\n',
                        "Set CLI arguments\n\n",
                        ":param argument_parser: argument parser\n",
                        ":type argument_parser: ```ArgumentParser```\n\n",
                        ":returns: argument_parser\n",
                        ":rtype: ```ArgumentParser```\n",
                        '"""\n',
                        "argument_parser.description = ''\n",
                        "return argument_parser",
                    )
                ),
            ),
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
        Tests that parse.function produces properly from function with only keyword arguments
        """
        gen_ir = parse.function(function_adder_ast, function_type="static")
        del gen_ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            gen_ir,
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

        self.assertIsNone(foo(5, 6))

        ir = parse.function(foo)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            ir,
            {
                "doc": "the foo function",
                "name": "TestParsers.test_from_function_in_memory.<locals>.foo",
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

        ir = parse.function(call_cliff)
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
        gen_ir = parse.function(method_complex_args_variety_ast)
        del gen_ir["_internal"]  # Not needed for this test
        self.assertDictEqual(
            gen_ir,
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
                "params": OrderedDict(),
                "returns": None,
            },
        )

    def test_from_adadelta_class_in_memory(self) -> None:
        """
        Tests that parse.class produces properly from a `class` in memory of current interpreter
        """
        Adadelta = getattr(
            inspectable_compile(docstring_google_tf_adadelta_function_str),
            "Adadelta",
        )
        ir = parse.class_(Adadelta)
        del ir["_internal"]
        # self.assertDictEqual(ir, docstring_google_tf_adadelta_ir)
        self.assertDictEqual(
            ir,
            docstring_google_tf_adadelta_function_ir,
        )

    def test_from_class_and_function(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults
        """

        # Sanity check
        run_ast_test(
            self,
            class_google_tf_tensorboard_ast,
            gold=ast.parse(class_google_tf_tensorboard_str).body[0],
        )

        parsed_ir = parse.class_(
            class_google_tf_tensorboard_ast,
            merge_inner_function="__init__",
            infer_type=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertDictEqual(parsed_ir, class_google_tf_tensorboard_ir)

    def test_from_class_and_function_in_memory(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults
        """

        parsed_ir = parse.class_(
            RewriteAtQuery,
            merge_inner_function="__init__",
            infer_type=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertDictEqual(
            parsed_ir,
            {
                "doc": "Replace the node at query with given node",
                "name": "RewriteAtQuery",
                "params": OrderedDict(
                    (
                        (
                            "search",
                            {
                                "doc": "Search query, e.g., "
                                "['node_name', 'function_name', "
                                "'arg_name']",
                                "typ": "List[str]",
                            },
                        ),
                        (
                            "replacement_node",
                            {"doc": "Node to replace this search", "typ": "AST"},
                        ),
                        (
                            "replaced",
                            {
                                "doc": "Whether a node has been replaced "
                                "(only replaces first "
                                "occurrence)"
                            },
                        ),
                    )
                ),
                "returns": None,
            },
        )

    def test_from_class_and_function_torch(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults, given a PyTorch loss class
        """

        # Sanity check
        run_ast_test(
            self,
            class_torch_nn_l1loss_ast,
            gold=ast.parse(class_torch_nn_l1loss_str).body[0],
        )

        parsed_ir = parse.class_(
            class_torch_nn_l1loss_ast,
            merge_inner_function="__init__",
            infer_type=True,
            parse_original_whitespace=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertDictEqual(parsed_ir, class_torch_nn_l1loss_ir)

    def test_from_class_torch_nn_one_cycle_lr(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults, given a PyTorch loss LR scheduler class
        """

        # Sanity check
        run_ast_test(
            self,
            class_torch_nn_one_cycle_lr_ast,
            gold=ast.parse(class_torch_nn_one_cycle_lr_str).body[0],
        )

        parsed_ir = parse.class_(
            class_torch_nn_one_cycle_lr_ast,
            merge_inner_function="__init__",
            infer_type=True,
        )

        del parsed_ir["_internal"]  # Not needed for this test

        self.assertEqual(parsed_ir["params"]["last_epoch"]["typ"], "int")
        self.assertDictEqual(parsed_ir, class_torch_nn_one_cycle_lr_ir)

    def test__class_from_memory(self) -> None:
        """
        Tests that the parser can combine the outer class docstring + structure
        with the inner function parameter defaults, given a PyTorch loss LR scheduler class
        """

        class A(object):
            """A is one boring class"""

        with patch("inspect.getsourcefile", lambda _: None):
            ir = parse._class_from_memory(A, A.__name__, False, False, False)
        self.assertDictEqual(
            ir,
            {
                "doc": "A is one boring class",
                "name": "A",
                "params": OrderedDict(),
                "returns": None,
            },
        )

    def test_from_json_schema(self) -> None:
        """
        Tests that `parse.json_schema` produces `intermediate_repr_no_default_sql_doc` properly
        """
        self.assertDictEqual(
            parse.json_schema(config_schema), intermediate_repr_no_default_sql_doc
        )

    def test_from_sqlalchemy_table(self) -> None:
        """
        Tests that `parse.sqlalchemy_table` produces `intermediate_repr_no_default_sql_doc` properly
        """

        # Sanity check
        run_ast_test(
            self,
            config_tbl_ast,
            gold=ast.parse(config_tbl_str).body[0],
        )

        for variant in (
            config_tbl_str,
            config_tbl_str.replace("config_tbl =", "config_tbl: Table =", 1),
            config_tbl_str.replace("config_tbl =", "", 1).lstrip(),
        ):
            ir = parse.sqlalchemy_table(ast.parse(variant).body[0])
            self.assertEqual(ir["name"], "config_tbl")
            ir["name"] = None
            self.assertDictEqual(ir, intermediate_repr_no_default_sql_doc)

    def test_from_sqlalchemy(self) -> None:
        """
        Tests that `parse.sqlalchemy` produces `intermediate_repr_no_default_sql_doc` properly
        """

        # Sanity check
        run_ast_test(
            self,
            config_decl_base_ast,
            gold=ast.parse(config_decl_base_str).body[0],
        )

        ir = parse.sqlalchemy(config_decl_base_ast)
        self.assertEqual(ir["name"], "config_tbl")
        ir["name"] = None
        self.assertDictEqual(ir, intermediate_repr_no_default_sql_doc)


unittest_main()

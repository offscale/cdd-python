""" Tests for emitter_utils """

from ast import Attribute, Call, Expr, Load, Name, Subscript, keyword
from collections import OrderedDict
from copy import deepcopy
from operator import itemgetter
from unittest import TestCase

from cdd.ast_utils import get_value, set_value
from cdd.emit.utils.argparse_function_utils import parse_out_param
from cdd.emit.utils.docstring_utils import interpolate_defaults
from cdd.emit.utils.sqlalchemy_utils import param_to_sqlalchemy_column_call
from cdd.tests.mocks.argparse import argparse_add_argument_ast, argparse_func_ast
from cdd.tests.mocks.ir import intermediate_repr
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestEmitterUtils(TestCase):
    """Test class for emitter_utils"""

    def test_param_to_sqlalchemy_column_call_when_sql_constraints(self) -> None:
        """Tests that with SQL constraints the SQLalchemy column is correctly generated"""
        run_ast_test(
            self,
            param_to_sqlalchemy_column_call(
                (
                    "foo",
                    {
                        "doc": "",
                        "typ": "str",
                        "x_typ": {"sql": {"constraints": {"indexed": True}}},
                    },
                ),
                include_name=False,
            ),
            gold=Call(
                func=Name(id="Column", ctx=Load()),
                args=[Name(id="String", ctx=Load())],
                keywords=[keyword(arg="indexed", value=set_value(True))],
            ),
        )

    def test_parse_out_param(self) -> None:
        """Test that parse_out_param parses out the right dict"""
        # Sanity check
        self.assertEqual(
            get_value(argparse_func_ast.body[5].value.args[0]), "--as_numpy"
        )

        self.assertDictEqual(
            *map(
                itemgetter("as_numpy"),
                (
                    OrderedDict((parse_out_param(argparse_func_ast.body[5]),)),
                    intermediate_repr["params"],
                ),
            )
        )

    def test_parse_out_param_default(self) -> None:
        """Test that parse_out_param sets default when required and unset"""

        self.assertDictEqual(
            parse_out_param(argparse_add_argument_ast)[1],
            {"default": 0, "doc": None, "typ": "int"},
        )

    def test_parse_out_param_fails(self) -> None:
        """Test that parse_out_param throws NotImplementedError when unsupported type given"""
        self.assertRaises(
            NotImplementedError,
            lambda: parse_out_param(
                Expr(
                    Call(
                        args=[set_value("--num")],
                        func=Attribute(
                            Name("argument_parser", Load()),
                            "add_argument",
                            Load(),
                        ),
                        keywords=[
                            keyword(
                                arg="type",
                                value=Subscript(
                                    expr_context_ctx=None,
                                    expr_slice=None,
                                    expr_value=None,
                                ),
                                identifier=None,
                            ),
                            keyword(
                                arg="required",
                                value=set_value(True),
                                identifier=None,
                            ),
                        ],
                        expr=None,
                        expr_func=None,
                    )
                )
            ),
        )

    def test_interpolate_defaults(self) -> None:
        """Test that interpolate_defaults corrects sets the default property"""
        param = "K", deepcopy(intermediate_repr["params"]["K"])
        param_with_correct_default = deepcopy(param[1])
        del param[1]["default"]
        self.assertDictEqual(interpolate_defaults(param)[1], param_with_correct_default)


unittest_main()

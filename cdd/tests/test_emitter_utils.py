""" Tests for emitter_utils """
from ast import Attribute, Call, Expr, Load, Name, Subscript, keyword
from copy import deepcopy
from unittest import TestCase

from cdd.ast_utils import set_value
from cdd.emitter_utils import interpolate_defaults, parse_out_param
from cdd.pure_utils import rpartial
from cdd.tests.mocks.argparse import argparse_add_argument_ast, argparse_func_ast
from cdd.tests.mocks.ir import intermediate_repr
from cdd.tests.utils_for_tests import unittest_main


class TestEmitterUtils(TestCase):
    """ Test class for emitter_utils """

    def test_parse_out_param(self) -> None:
        """ Test that parse_out_param parses out the right dict """
        self.assertDictEqual(
            parse_out_param(
                next(filter(rpartial(isinstance, Expr), argparse_func_ast.body[::-1]))
            )[1],
            # Last element:
            intermediate_repr["params"]["data_loader_kwargs"],
        )

    def test_parse_out_param_default(self) -> None:
        """ Test that parse_out_param sets default when required and unset """

        self.assertDictEqual(
            parse_out_param(argparse_add_argument_ast)[1],
            {"default": 0, "doc": None, "typ": "int"},
        )

    def test_parse_out_param_fails(self) -> None:
        """ Test that parse_out_param throws NotImplementedError when unsupported type given """
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
        """ Test that interpolate_defaults corrects sets the default property """
        param = "K", deepcopy(intermediate_repr["params"]["K"])
        param_with_correct_default = deepcopy(param[1])
        del param[1]["default"]
        self.assertDictEqual(interpolate_defaults(param)[1], param_with_correct_default)


unittest_main()

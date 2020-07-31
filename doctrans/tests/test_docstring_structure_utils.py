""" Tests for docstring_structure_utils """
from ast import Expr, Call, Constant, Attribute, Load, Name, keyword, Subscript
from copy import deepcopy
from unittest import TestCase

from doctrans.docstring_structure_utils import parse_out_param, interpolate_defaults
from doctrans.pure_utils import rpartial
from doctrans.tests.mocks.argparse import argparse_func_ast, argparse_add_argument_ast
from doctrans.tests.mocks.docstrings import docstring_structure
from doctrans.tests.utils_for_tests import unittest_main


class TestDocstringStructureUtils(TestCase):
    """ Test class for docstring_structure_utils """

    def test_parse_out_param(self) -> None:
        """ Test that parse_out_param parses out the right dict """
        self.assertDictEqual(
            parse_out_param(next(filter(rpartial(isinstance, Expr),
                                        argparse_func_ast.body[::-1]))),
            docstring_structure['params'][-1]
        )

    def test_parse_out_param_default(self) -> None:
        """ Test that parse_out_param sets default when required and unset """

        self.assertDictEqual(
            parse_out_param(argparse_add_argument_ast),
            {'default': 0,
             'doc': None,
             'name': 'num',
             'typ': 'int'}
        )

    def test_parse_out_param_fails(self) -> None:
        """ Test that parse_out_param throws NotImplementedError when unsupported type given """
        self.assertRaises(
            NotImplementedError,
            lambda: parse_out_param(
                Expr(value=Call(args=[Constant(kind=None,
                                               value='--num')],
                                func=Attribute(attr='add_argument',
                                               ctx=Load(),
                                               value=Name(ctx=Load(),
                                                          id='argument_parser')),
                                keywords=[keyword(arg='type',
                                                  value=Subscript()),
                                          keyword(arg='required',
                                                  value=Constant(kind=None,
                                                                 value=True))]))
            )
        )

    def test_interpolate_defaults(self) -> None:
        """ Test that interpolate_defaults corrects sets the default property """
        param = deepcopy(docstring_structure['params'][2])
        param_with_correct_default = deepcopy(param)
        del param['default']
        self.assertDictEqual(
            interpolate_defaults(param),
            param_with_correct_default
        )


unittest_main()

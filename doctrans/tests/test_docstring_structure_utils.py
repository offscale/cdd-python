""" Tests for docstring_structure_utils """
from ast import Expr
from copy import deepcopy
from unittest import TestCase, main as unittest_main

from doctrans.docstring_structure_utils import parse_out_param, interpolate_defaults
from doctrans.pure_utils import rpartial
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.docstrings import docstring_structure


class TestDocstringStructureUtils(TestCase):
    """ Test class for docstring_structure_utils """

    def test_parse_out_param(self) -> None:
        """ Test that parse_out_param parses out the right dict """
        self.assertDictEqual(
            parse_out_param(next(filter(rpartial(isinstance, Expr),
                                        argparse_func_ast.body[::-1]))),
            docstring_structure['params'][-1]
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


if __name__ == '__main__':
    unittest_main()

"""
Tests for the Intermediate Representation produced by the argparse parser
"""

from ast import FunctionDef
from copy import deepcopy
from unittest import TestCase

import cdd.argparse_function.emit
import cdd.argparse_function.parse
from cdd.shared.pure_utils import tab
from cdd.shared.source_transformer import to_code
from cdd.shared.types import IntermediateRepr
from cdd.tests.mocks.argparse import argparse_func_ast
from cdd.tests.mocks.ir import intermediate_repr_no_default_doc
from cdd.tests.utils_for_tests import unittest_main


class TestParseArgparse(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    """

    def test_from_argparse_ast(self) -> None:
        """
        Tests whether `argparse_ast` produces `intermediate_repr_no_default_doc`
              from `argparse_func_ast`"""
        ir: IntermediateRepr = cdd.argparse_function.parse.argparse_ast(
            argparse_func_ast
        )
        del ir["_internal"]  # Not needed for this test
        _intermediate_repr_no_default_doc = deepcopy(intermediate_repr_no_default_doc)
        _intermediate_repr_no_default_doc["name"] = "set_cli_args"
        self.assertDictEqual(ir, _intermediate_repr_no_default_doc)

    def test_from_argparse_ast_empty(self) -> None:
        """
        Tests `argparse_ast` empty condition
        """
        self.assertEqual(
            to_code(
                cdd.argparse_function.emit.argparse_function(
                    cdd.argparse_function.parse.argparse_ast(
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
            "def set_cli_args(argument_parser):\n"
            "{tab}{body}".format(
                tab=tab,
                body=tab.join(
                    (
                        '"""\n',
                        "Set CLI arguments\n",
                        "\n",
                        ":param argument_parser: argument parser\n",
                        ":type argument_parser: ```ArgumentParser```\n",
                        "\n",
                        ":return: argument_parser\n",
                        ":rtype: ```ArgumentParser```\n",
                        '"""\n',
                        "argument_parser.description = ''\n",
                        "return argument_parser",
                    )
                ),
            ),
        )


unittest_main()

"""
Tests for the Intermediate Representation produced by the `pydantic` parser
"""

import ast
from unittest import TestCase

import cdd.pydantic.parse
from cdd.shared.types import IntermediateRepr
from cdd.tests.mocks.ir import pydantic_ir
from cdd.tests.mocks.pydantic import pydantic_class_cls_def, pydantic_class_str
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestParsePydantic(TestCase):
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

    def test_from_pydantic_roundtrip(self) -> None:
        """
        Tests pydantic roundtrip of mocks
        """
        run_ast_test(
            self, ast.parse(pydantic_class_str).body[0], pydantic_class_cls_def
        )

    def test_from_pydantic(self) -> None:
        """
        Tests whether `pydantic` produces `pydantic_ir`
              from `pydantic_class_cls_def`
        """
        ir: IntermediateRepr = cdd.pydantic.parse.pydantic(pydantic_class_cls_def)
        del ir["_internal"]  # Not needed for this test
        self.assertDictEqual(ir, pydantic_ir)


unittest_main()

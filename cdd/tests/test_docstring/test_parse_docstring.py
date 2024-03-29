"""
Tests for the Intermediate Representation produced by the parsers
"""

from collections import OrderedDict
from unittest import TestCase

import cdd.docstring.parse
from cdd.shared.types import IntermediateRepr
from cdd.tests.mocks.docstrings import (
    docstring_keras_rmsprop_class_str,
    docstring_keras_rmsprop_method_str,
    docstring_reduction_v2_str,
)
from cdd.tests.mocks.ir import (
    docstring_keras_rmsprop_class_ir,
    docstring_keras_rmsprop_method_ir,
)
from cdd.tests.utils_for_tests import unittest_main


class TestParseDocstring(TestCase):
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

    def test_from_docstring_docstring_reduction_v2_str(self) -> None:
        """
        Test that the non-matching docstring doesn't fill out params
        """
        ir: IntermediateRepr = cdd.docstring.parse.docstring(docstring_reduction_v2_str)
        self.assertEqual(ir["params"], OrderedDict())
        self.assertEqual(ir["returns"], None)

    def test_from_docstring_keras_rmsprop_class_str(self) -> None:
        """Tests IR from docstring_keras_rmsprop_class_str"""

        self.assertDictEqual(
            cdd.docstring.parse.docstring(docstring_keras_rmsprop_class_str),
            docstring_keras_rmsprop_class_ir,
        )

    def test_from_docstring_keras_rmsprop_class_method_str(self) -> None:
        """Tests IR from docstring_keras_rmsprop_method_str"""

        self.assertDictEqual(
            cdd.docstring.parse.docstring(docstring_keras_rmsprop_method_str),
            docstring_keras_rmsprop_method_ir,
        )


unittest_main()

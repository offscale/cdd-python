""" Tests for cst_utils """

from unittest import TestCase

from cdd.cst_utils import (
    DictExprStatement,
    GenExprStatement,
    ListCompStatement,
    SetExprStatement,
    UnchangingLine,
    infer_cst_type,
)
from cdd.tests.utils_for_tests import unittest_main


class TestCstUtils(TestCase):
    """Test class for cst_utils"""

    def test_infer_cst_type(self) -> None:
        """Test that `infer_cst_type` can infer the right CST type"""
        for input_str, expected_type in (
            ("foo", UnchangingLine),
            ("()", GenExprStatement),
            ("[i for i in ()]", ListCompStatement),
            ("{i for i in ()}", SetExprStatement),
            ("{i:i for i in ()}", DictExprStatement),
        ):
            self.assertEqual(
                expected_type,
                infer_cst_type(
                    input_str,
                    words=tuple(filter(None, map(str.strip, input_str.split(" ")))),
                ),
            )


unittest_main()

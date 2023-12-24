"""
Tests for source_transformer
"""

from ast import FunctionDef, Pass, arguments
from unittest import TestCase

from cdd.shared.pure_utils import tab
from cdd.shared.source_transformer import to_code
from cdd.tests.utils_for_tests import unittest_main


class TestSourceTransformer(TestCase):
    """
    Tests for source_transformer
    """

    def test_to_code(self) -> None:
        """
        Tests to_source in Python 3.9 and < 3.9
        """
        func_def: FunctionDef = FunctionDef(
            name="funcy",
            args=arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=[Pass()],
            decorator_list=[],
            type_params=[],
            lineno=None,
            returns=None,
        )

        self.assertEqual(
            to_code(func_def).rstrip("\n"),
            "def funcy():\n" "{tab}pass".format(tab=tab),
        )


unittest_main()

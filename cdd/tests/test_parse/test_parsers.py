"""
Test for the `cdd.parse` module
"""

from os import path
from unittest import TestCase

from cdd.shared.parse import PARSERS
from cdd.shared.pure_utils import all_dunder_for_module
from cdd.tests.utils_for_tests import unittest_main


class TestParsers(TestCase):
    """
    Tests the `cdd.parse` module magic `__all__`
    """

    def test_parsers_root(self) -> None:
        """Confirm that emitter names are up-to-date"""
        self.assertListEqual(
            PARSERS,
            all_dunder_for_module(
                path.dirname(path.dirname(path.dirname(__file__))),
                (
                    "sqlalchemy_hybrid",
                    "sqlalchemy_table",
                ),
            ),
        )


unittest_main()

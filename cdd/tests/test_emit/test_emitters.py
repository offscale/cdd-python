"""
Test for the `cdd.emit` module
"""

from os import path
from unittest import TestCase

from cdd.shared.emit import EMITTERS
from cdd.shared.pure_utils import all_dunder_for_module
from cdd.tests.utils_for_tests import unittest_main


class TestEmitters(TestCase):
    """
    Tests the `cdd.emit` module magic `__all__`
    """

    def test_emitters_root(self) -> None:
        """Confirm that emitter names are up-to-date"""
        self.assertListEqual(
            EMITTERS,
            all_dunder_for_module(
                path.dirname(path.dirname(path.dirname(__file__))),
                (
                    "sqlalchemy_hybrid",
                    "sqlalchemy_table",
                ),
            ),
        )


unittest_main()

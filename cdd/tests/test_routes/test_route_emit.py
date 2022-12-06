"""
Tests `cdd.routes.emit` EMITTERS
"""

from os import path
from unittest import TestCase

from cdd.pure_utils import all_dunder_for_module
from cdd.routes.emit import EMITTERS
from cdd.tests.utils_for_tests import unittest_main


class TestRoutesEmit(TestCase):
    """Tests `cdd.routes.emit`"""

    def test_routes_emit_root(self) -> None:
        """Confirm that route emitter names are up-to-date"""
        self.assertListEqual(
            EMITTERS, all_dunder_for_module(path.join("routes", "emit"), tuple())
        )


unittest_main()

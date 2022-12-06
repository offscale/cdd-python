"""
Tests `cdd.routes.parse` PARSERS
"""

from os import path
from unittest import TestCase

from cdd.pure_utils import all_dunder_for_module
from cdd.routes.parse import PARSERS
from cdd.tests.utils_for_tests import unittest_main


class TestRoutesParse(TestCase):
    """Tests `cdd.routes.parse`"""

    def test_routes_parse_root(self) -> None:
        """Confirm that route parser names are up-to-date"""
        self.assertListEqual(
            PARSERS, all_dunder_for_module(path.join("routes", "parse"), tuple())
        )


unittest_main()

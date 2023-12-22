"""
Tests `cdd.routes.parse` PARSERS
"""

from itertools import filterfalse
from operator import itemgetter
from os import listdir, path
from unittest import TestCase

from cdd.routes.parse import PARSERS
from cdd.tests.utils_for_tests import unittest_main


class TestRoutesParse(TestCase):
    """Tests `cdd.routes.parse`"""

    def test_routes_parse_root(self) -> None:
        """Confirm that route parser names are up-to-date"""
        module_directory: str = path.join(
            path.dirname(path.dirname(path.dirname(__file__))), "routes", "parse"
        )
        self.assertListEqual(
            PARSERS,
            # all_dunder_for_module(module_directory, iter(()))
            sorted(
                filterfalse(
                    lambda name: name.startswith("_") or name.endswith("_utils"),
                    map(itemgetter(0), map(path.splitext, listdir(module_directory))),
                )
            ),
        )


unittest_main()

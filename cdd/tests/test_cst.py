""" Tests for cst """

from os import extsep, path
from unittest import TestCase

from cdd.cst import cst_parse
from cdd.tests.mocks.cst import cstify_cst
from cdd.tests.utils_for_tests import unittest_main


class TestCst(TestCase):
    """Test class for cst"""

    def test_cstify_file(self) -> None:
        """Tests that `handle_multiline_comment` can do its namesake"""
        with open(
            path.join(
                path.dirname(__file__),
                "mocks",
                "cstify{extsep}py".format(extsep=extsep),
            ),
            "rt",
        ) as f:
            cst = cst_parse(f.read())

        self.assertTupleEqual(cst, cstify_cst)


unittest_main()

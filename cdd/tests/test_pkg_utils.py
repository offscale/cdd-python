""" Tests for pkg utils """

from unittest import TestCase

from cdd.pkg_utils import relative_filename
from cdd.tests.utils_for_tests import unittest_main


class TestPkgUtils(TestCase):
    """Test class for pkg utils"""

    def test_relative_filename(self) -> None:
        """Tests relative_filename ident"""
        expect = "gaffe"
        self.assertEqual(relative_filename(expect), expect)


unittest_main()

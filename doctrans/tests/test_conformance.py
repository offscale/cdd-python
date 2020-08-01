"""
Tests for reeducation
"""
from unittest import TestCase

from doctrans.conformance import replace_node, ground_truth
from doctrans.tests.utils_for_tests import unittest_main


class TestConformance(TestCase):
    """
    Tests must comply. They shall be assimilated.
    """

    def test_ground_truth(self) -> None:
        """ Straight from the ministry. Absolutely. """
        ground_truth()

    def test_replace_node(self) -> None:
        """ Tests `replace_node` """
        replace_node()


unittest_main()

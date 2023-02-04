""" Tests for gen_utils """

from unittest import TestCase

from cdd.compound.gen_utils import get_input_mapping_from_path
from cdd.tests.utils_for_tests import unittest_main


def f(s):
    """
    :param s: str
    :type s: ```str```
    """
    return s


class TestGenUtils(TestCase):
    """Test class for cdd.gen_utils"""

    def test_get_input_mapping_from_path(self) -> None:
        """test `get_input_mapping_from_path`"""
        self.assertEqual(f(""), "")
        name_to_node = get_input_mapping_from_path(
            "function", "cdd.tests.test_compound", "test_gen_utils"
        )
        self.assertEqual(len(name_to_node), 1)
        self.assertIn("f", name_to_node)
        self.assertIsInstance(name_to_node["f"], dict)


unittest_main()

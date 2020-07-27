""" Tests for pure utils """
from unittest import TestCase

from doctrans.pure_utils import rpartial, pp, tab, simple_types, identity
from doctrans.tests.utils_for_tests import unittest_main


class TestPureUtils(TestCase):
    """ Test class for pure utils """

    def test_pp(self) -> None:
        """ Test that pp is from the right module """
        self.assertEqual(pp.__module__, 'pprint')

    def test_tab(self) -> None:
        """ Test that tab is of right length """
        self.assertEqual(tab, '    ')

    def test_simple_types(self) -> None:
        """ Tests that simple types only includes int,str,float,bool with right default values """
        self.assertDictEqual(simple_types, {'int': 0, float: .0, 'str': '', 'bool': False})

    def test_rpartial(self) -> None:
        """ Test that rpartial works as advertised """
        self.assertTrue(rpartial(isinstance, str)(''))
        self.assertFalse(rpartial(isinstance, str)(0))

    def test_identity(self) -> None:
        """ Tests that ident returns itself """
        self.assertEqual(identity(''), '')
        self.assertFalse(identity(False))
        self.assertTrue(identity(True))
        self.assertIsNone(identity(None))


unittest_main()

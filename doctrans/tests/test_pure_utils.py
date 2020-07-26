""" Tests for pure utils """
from unittest import TestCase, main as unittest_main

from doctrans.pure_utils import rpartial, pp, tab, simple_types


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


if __name__ == '__main__':
    unittest_main()

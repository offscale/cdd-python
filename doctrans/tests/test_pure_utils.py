""" Tests for pure utils """
from unittest import TestCase

from doctrans.pure_utils import (
    rpartial,
    pp,
    tab,
    simple_types,
    identity,
    pluralise,
    strip_split,
    sanitise,
    quote,
    update_d,
)
from doctrans.tests.utils_for_tests import unittest_main


class TestPureUtils(TestCase):
    """ Test class for pure utils """

    def test_pp(self) -> None:
        """ Test that pp is from the right module """
        self.assertEqual(pp.__module__, "pprint")

    def test_tab(self) -> None:
        """ Test that tab is of right length """
        self.assertEqual(tab, "    ")

    def test_simple_types(self) -> None:
        """ Tests that simple types only includes int,str,float,bool with right default values """
        self.assertDictEqual(
            simple_types, {"int": 0, float: 0.0, "str": "", "bool": False}
        )

    def test_rpartial(self) -> None:
        """ Test that rpartial works as advertised """
        self.assertTrue(rpartial(isinstance, str)(""))
        self.assertFalse(rpartial(isinstance, str)(0))

    def test_identity(self) -> None:
        """ Tests that ident returns itself """
        self.assertEqual(identity(""), "")
        self.assertFalse(identity(False))
        self.assertTrue(identity(True))
        self.assertIsNone(identity(None))

    def test_pluralises(self) -> None:
        """ Tests that pluralises pluralises """
        self.assertEqual(pluralise(""), "")
        self.assertEqual(pluralise("goose"), "geese")
        self.assertEqual(pluralise("dolly"), "dollies")
        self.assertEqual(pluralise("genius"), "genii")
        self.assertEqual(pluralise("pass"), "passes")
        self.assertEqual(pluralise("zero"), "zeros")
        self.assertEqual(pluralise("casino"), "casinos")
        self.assertEqual(pluralise("hero"), "heroes")
        self.assertEqual(pluralise("church"), "churches")
        self.assertEqual(pluralise("x"), "xs")
        self.assertEqual(pluralise("ant"), "ants")
        self.assertEqual(pluralise("car"), "cars")
        self.assertEqual(pluralise("wish"), "wishes")
        self.assertEqual(pluralise("morphosis"), "morphosises")
        self.assertEqual(pluralise("s"), "ss")

    def test_sanitise(self) -> None:
        """ Tests sanity """
        self.assertEqual(sanitise("class"), "class_")

    def test_strip_split(self) -> None:
        """ Tests that strip_split works on separated input and separator free input """
        self.assertTupleEqual(tuple(strip_split("foo.bar", ".")), ("foo", "bar"))
        self.assertTupleEqual(tuple(strip_split("foo", " ")), ("foo",))

    def test_quote(self) -> None:
        """ Tests quote edge cases """
        self.assertEqual(quote(""), "")
        self.assertIsNone(quote(None))
        self.assertEqual(quote('""'), '""')
        self.assertEqual(quote("''"), "''")
        self.assertEqual(quote('"foo"'), '"foo"')
        self.assertEqual(quote("'bar'"), "'bar'")
        self.assertEqual(quote("haz"), '"haz"')

    def test_update_d(self):
        """ Tests inline update dict """
        d = {}
        u = {"a": 5}
        self.assertDictEqual(update_d(d, u), u)
        self.assertDictEqual(d, u)
        del d

        d = {}
        self.assertDictEqual(update_d(d, **u), u)
        self.assertDictEqual(d, u)


unittest_main()

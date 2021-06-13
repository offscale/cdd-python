""" Tests for pure utils """

import unittest
from functools import partial
from itertools import zip_longest
from unittest import TestCase

from cdd.pure_utils import (
    assert_equal,
    blockwise,
    deindent,
    diff,
    get_module,
    identity,
    location_within,
    lstrip_namespace,
    multiline,
    pluralise,
    pp,
    quote,
    rpartial,
    sanitise,
    set_attr,
    set_item,
    simple_types,
    strip_split,
    tab,
    update_d,
)
from cdd.tests.utils_for_tests import unittest_main


class TestPureUtils(TestCase):
    """Test class for pure utils"""

    def test_blockwise(self) -> None:
        """Tests that blockwise produces the expected output"""
        self.assertIsInstance(blockwise(iter(())), zip_longest)
        self.assertTupleEqual(tuple(blockwise(iter(()))), tuple())
        self.assertTupleEqual(tuple(blockwise("ABC")), (("A", "B"), ("C", None)))
        self.assertTupleEqual(tuple(blockwise("ABCD")), (("A", "B"), ("C", "D")))

    def test_pp(self) -> None:
        """Test that pp is from the right module"""
        self.assertEqual(pp.__module__, "pprint")

    def test_tab(self) -> None:
        """Test that tab is of right length"""
        self.assertEqual(tab, "    ")

    def test_simple_types(self) -> None:
        """Tests that simple types only includes int,str,float,bool with right default values"""
        self.assertDictEqual(
            simple_types,
            {
                None: None,
                "int": 0,
                "float": 0.0,
                "complex": 0j,
                "str": "",
                "bool": False,
            },
        )

    def test_rpartial(self) -> None:
        """Test that rpartial works as advertised"""
        self.assertTrue(rpartial(isinstance, str)(""))
        self.assertFalse(rpartial(isinstance, str)(0))

    def test_identity(self) -> None:
        """Tests that ident returns itself"""
        self.assertEqual(identity(""), "")
        self.assertFalse(identity(False))
        self.assertTrue(identity(True))
        self.assertIsNone(identity(None))

    def test_location_within0(self) -> None:
        """Tests `location_within` responds with correct `start_idx`, `end_idx` and `found` elements"""

        mock_str = "foocanhaz"

        can_res = 3, 6, "can"
        none_res = -1, -1, None

        self.assertTupleEqual(location_within(mock_str, ("can",)), can_res)
        self.assertTupleEqual(location_within(mock_str, ("bar",)), none_res)
        self.assertTupleEqual(location_within(mock_str, ("bar", "can")), can_res)
        self.assertTupleEqual(location_within(mock_str, ("br", "can")), can_res)
        self.assertTupleEqual(
            location_within(mock_str, ("bar", "con", "bon")), none_res
        )

    def test_location_within1(self) -> None:
        """Tests `location_within` responds with correct `start_idx`, `end_idx` and `found` elements"""

        none_res = -1, -1, None

        self.assertTupleEqual(location_within("a", ("bar", "con", "bon")), none_res)
        self.assertTupleEqual(location_within("can", ("can",)), (0, 3, "can"))
        self.assertTupleEqual(
            location_within(map(str, range(10)), map(str, range(10, 20))), none_res
        )

    def test_multiline(self) -> None:
        """Tests that `multiline` multilines"""
        self.assertEqual(
            """123456789_\n123456789_\n123456789_\n123456789""",
            """123456789_\n""" """123456789_\n""" """123456789_\n""" """123456789""",
        )
        self.assertEqual(
            multiline(
                """123456789_\n""" """123456789_\n""" """123456789_\n""" """123456789"""
            ),
            tab.join(
                (
                    "'123456789_' \\\n",
                    "'123456789_' \\\n",
                    "'123456789_' \\\n",
                    "'123456789'",
                )
            ),
        )

    def test_pluralises(self) -> None:
        """Tests that pluralises pluralises"""
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
        """Tests sanity"""
        self.assertEqual(sanitise("class"), "class_")

    def test_set_attr(self) -> None:
        """Tests `set_attr`"""

        class Att(object):
            """Mock class for `test_set_attr`"""

        self.assertEqual(set_attr(Att, "bar", 5).bar, 5)

    def test_set_item(self) -> None:
        """Tests `set_item`"""
        self.assertEqual(set_item({}, "foo", "haz")["foo"], "haz")

    def test_strip_split(self) -> None:
        """Tests that strip_split works on separated input and separator free input"""
        self.assertTupleEqual(tuple(strip_split("foo.bar", ".")), ("foo", "bar"))
        self.assertTupleEqual(tuple(strip_split("foo", " ")), ("foo",))

    def test_quote(self) -> None:
        """Tests quote edge cases"""
        self.assertEqual(quote(""), "")
        self.assertIsNone(quote(None))
        self.assertEqual(quote('""'), '""')
        self.assertEqual(quote("''"), "''")
        self.assertEqual(quote('"foo"'), '"foo"')
        self.assertEqual(quote("'bar'"), "'bar'")
        self.assertEqual(quote("haz"), '"haz"')

    def test_update_d(self) -> None:
        """Tests inline update dict"""
        d = {}
        u = {"a": 5}
        self.assertDictEqual(update_d(d, u), u)
        self.assertDictEqual(d, u)
        del d

        d = {}
        self.assertDictEqual(update_d(d, **u), u)
        self.assertDictEqual(d, u)

    def test_lstrip_namespace(self) -> None:
        """Tests that `lstrip_namespace` gives correct results"""
        self.assertEqual(lstrip_namespace("AAaBB", ("A", "a")), "BB")

    def test_diff(self) -> None:
        """Tests that `diff` gives correct results"""
        lstrip_l = partial(diff, op=str.lstrip)
        self.assertTupleEqual(lstrip_l("A"), (0, "A"))
        self.assertTupleEqual(lstrip_l(""), (0, ""))
        self.assertTupleEqual(lstrip_l(" A"), (1, "A"))
        self.assertTupleEqual(diff("AB", op=rpartial(str.lstrip, "A")), (1, "B"))

    def test_get_module(self) -> None:
        """Tests that `get_module` works as advertised"""
        self.assertIs(get_module("unittest"), unittest)
        self.assertEqual(
            get_module(
                "test_get_module",
                extra_symbols={"test_get_module": self.test_get_module},
            ).__file__,
            __file__,
        )
        self.assertRaises(
            ModuleNotFoundError,
            lambda: get_module("FFDSF", extra_symbols={"F": "A"}),
        )
        self.assertIs(get_module("get_module"), get_module)

    def test_assert_eq(self) -> None:
        """Basic tests to confirm same functionality as unittest.AssertEqual"""
        with self.assertRaises(AssertionError) as cm:
            assert_equal(5, 6)
        self.assertEqual(cm.exception.args[0].strip(), "5 != 6")

        self.assertTrue(assert_equal(5, 5))

    def test_deindent(self) -> None:
        """Test that deindent deindents"""
        self.assertEqual(deindent("    foo\tbar\n\tcanhaz"), "foo\tbar\ncanhaz")


unittest_main()

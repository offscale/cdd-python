""" Tests for parser_utils """
from collections import OrderedDict
from copy import deepcopy
from unittest import TestCase

from cdd.parser_utils import _join_non_none, ir_merge
from cdd.tests.utils_for_tests import unittest_main


class TestParserUtils(TestCase):
    """ Test class for parser_utils """

    def test_ir_merge_empty(self) -> None:
        """ Tests for `ir_merge` when both are empty """
        target = {"params": OrderedDict(), "returns": None}
        other = {"params": OrderedDict(), "returns": None}
        self.assertDictEqual(
            ir_merge(target, other),
            target,
        )

    def test_ir_merge_other_empty(self) -> None:
        """ Tests for `ir_merge` when only non-target is empty """
        target = {
            "params": OrderedDict(
                (("something", {}),),
            ),
            "returns": None,
        }
        other = {"params": OrderedDict(), "returns": None}
        self.assertDictEqual(
            ir_merge(target, other),
            target,
        )

    def test_ir_merge_same_len(self) -> None:
        """ Tests for `ir_merge` when target and non-target have same size """
        target = {
            "params": OrderedDict(
                (("something", {"typ": "str"}),),
            ),
            "returns": None,
        }
        other = {
            "params": OrderedDict(
                (("something", {"doc": "neat"}),),
            ),
            "returns": None,
        }
        self.assertDictEqual(
            ir_merge(deepcopy(target), other),
            {
                "params": OrderedDict(
                    (("something", {"doc": "neat", "typ": "str"}),),
                ),
                "returns": None,
            },
        )

    def test_ir_merge_same_len_returns(self) -> None:
        """ Tests for `ir_merge` when target and non-target have same size and a return """
        target = {
            "params": OrderedDict(),
            "returns": OrderedDict(
                (
                    (
                        "return_type",
                        {"typ": "str"},
                    ),
                )
            ),
        }
        other = {
            "params": OrderedDict(),
            "returns": OrderedDict(
                (
                    (
                        "return_type",
                        {"doc": "so stringy"},
                    ),
                )
            ),
        }
        self.assertDictEqual(
            ir_merge(deepcopy(target), other),
            {
                "params": OrderedDict(),
                "returns": OrderedDict(
                    (("return_type", {"typ": "str", "doc": "so stringy"}),)
                ),
            },
        )

    def test__join_non_none_returns_early(self) -> None:
        """ Tests that `_join_non_none` returns early """
        empty_str_dict = {"": ""}
        self.assertDictEqual(
            _join_non_none(primacy={}, other=empty_str_dict), empty_str_dict
        )
        self.assertDictEqual(
            _join_non_none(primacy=empty_str_dict, other={}), empty_str_dict
        )


unittest_main()

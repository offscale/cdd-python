""" Tests for parser_utils """

from collections import OrderedDict
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from cdd.ast_utils import set_value
from cdd.parser_utils import _join_non_none, get_source, infer, ir_merge
from cdd.tests.mocks import imports_header
from cdd.tests.mocks.argparse import argparse_func_ast, argparse_func_str
from cdd.tests.mocks.classes import class_ast, class_str
from cdd.tests.mocks.methods import (
    function_default_complex_default_arg_ast,
    method_complex_args_variety_str,
)
from cdd.tests.mocks.sqlalchemy import (
    config_decl_base_ast,
    config_decl_base_str,
    config_tbl_ast,
    sqlalchemy_imports_str,
)
from cdd.tests.utils_for_tests import inspectable_compile, unittest_main


class TestParserUtils(TestCase):
    """Test class for parser_utils"""

    def test_ir_merge_empty(self) -> None:
        """Tests for `ir_merge` when both are empty"""
        target = {"params": OrderedDict(), "returns": None}
        other = {"params": OrderedDict(), "returns": None}
        self.assertDictEqual(
            ir_merge(target, other),
            target,
        )

    def test_ir_merge_other_empty(self) -> None:
        """Tests for `ir_merge` when only non-target is empty"""
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
        """Tests for `ir_merge` when target and non-target have same size"""
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
        """Tests for `ir_merge` when target and non-target have same size and a return"""
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
        """Tests that `_join_non_none` returns early"""
        empty_str_dict = {"": ""}
        self.assertDictEqual(
            _join_non_none(primacy={}, other=empty_str_dict), empty_str_dict
        )
        self.assertDictEqual(
            _join_non_none(primacy=empty_str_dict, other={}), empty_str_dict
        )

    def test_infer_argparse_ast(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `argparse_ast`
        """
        self.assertEqual(infer(argparse_func_ast), "argparse_ast")

    def test_infer_memory_argparse_ast(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `argparse_ast`
        """
        set_cli_args = getattr(
            inspectable_compile(argparse_func_str),
            "set_cli_args",
        )
        self.assertEqual(infer(set_cli_args), "argparse_ast")

    def test_infer_docstring(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `docstring`
        """
        self.assertEqual(infer(""), "docstring")
        self.assertEqual(infer(set_value("")), "docstring")

    def test_infer_class(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `class_`
        """
        self.assertEqual(infer(class_ast), "class_")

    def test_infer_memory_class(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `class_`
        """
        set_cli_args = getattr(
            inspectable_compile(imports_header + class_str),
            "ConfigClass",
        )
        self.assertEqual(infer(set_cli_args), "class_")

    def test_infer_function(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `function`
        """
        self.assertEqual(infer(function_default_complex_default_arg_ast), "function")

    def test_infer_memory_function(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `function`
        """
        call_cliff = getattr(
            inspectable_compile(
                "\n".join(
                    (imports_header, "stdout = None", method_complex_args_variety_str)
                )
            ),
            "call_cliff",
        )
        self.assertEqual(infer(call_cliff), "function")

    def test_infer_sqlalchemy_table(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `sqlalchemy_table`
        """
        self.assertEqual(infer(config_tbl_ast), "sqlalchemy_table")

    # def test_infer_memory_sqlalchemy_table(self) -> None:
    #     """
    #     Test `infer` can figure out the right parser name when its expected to be `sqlalchemy_table`
    #     """
    #     config_tbl = getattr(
    #         inspectable_compile("\n".join((
    #             sqlalchemy_imports_str,
    #             "metadata = None", config_tbl_str,
    #         ))),
    #         "config_tbl",
    #     )
    #     self.assertEqual(infer(config_tbl), "sqlalchemy_table")

    def test_infer_sqlalchemy(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `sqlalchemy`
        """
        self.assertEqual(infer(config_decl_base_ast), "sqlalchemy")

    def test_infer_memory_sqlalchemy(self) -> None:
        """
        Test `infer` can figure out the right parser name when its expected to be `sqlalchemy`
        """
        Config = getattr(
            inspectable_compile(
                "\n".join(
                    (sqlalchemy_imports_str, "Base = object", config_decl_base_str)
                )
            ),
            "Config",
        )
        self.assertEqual(infer(Config), "sqlalchemy")

    def test_get_source_raises(self):
        """Tests that `get_source` raises an exception"""
        with self.assertRaises(TypeError):
            get_source(None)

        def raise_os_error(_):
            """raise_OSError"""
            raise OSError

        with patch("inspect.getsourcelines", raise_os_error), self.assertRaises(
            OSError
        ):
            get_source(min)


unittest_main()

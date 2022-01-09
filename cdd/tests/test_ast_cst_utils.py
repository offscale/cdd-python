""" Tests for ast_cst_utils """

from ast import ClassDef, Expr, FunctionDef, walk
from copy import deepcopy
from io import StringIO
from os import path
from os.path import extsep
from unittest import TestCase
from unittest.mock import patch

from cdd.ast_cst_utils import (
    Delta,
    find_cst_at_ast,
    maybe_replace_doc_str_in_function_or_class,
)
from cdd.ast_utils import set_value
from cdd.cst_utils import ClassDefinitionStart, FunctionDefinitionStart
from cdd.pure_utils import rpartial
from cdd.source_transformer import ast_parse
from cdd.tests.mocks.cst import cstify_cst
from cdd.tests.utils_for_tests import unittest_main


class TestAstCstUtils(TestCase):
    """Test class for cst_utils"""

    def setUp(self) -> None:
        """
        Initialise vars useful for multiple tests
        """
        with open(
            path.join(
                path.dirname(__file__),
                "mocks",
                "cstify{extsep}py".format(extsep=extsep),
            ),
            "rt",
        ) as f:
            self.ast_mod = ast_parse(f.read(), skip_docstring_remit=True)

        self.func_node = next(
            filter(
                lambda func: func.name == "add1",
                filter(rpartial(isinstance, FunctionDef), walk(self.ast_mod)),
            )
        )

    def test_find_cst_at_ast(self) -> None:
        """Test that `find_cst_at_ast` can find the CST for one function"""
        cst_idx, cst_node = find_cst_at_ast(cstify_cst, self.func_node)
        self.assertIsNotNone(cst_idx)
        self.assertIsNotNone(cst_node)
        self.assertIsInstance(cst_node, FunctionDefinitionStart)

    def test_find_cst_at_ast_finds_all_functions(self) -> None:
        """Test that `find_cst_at_ast` can find the CST for all functions"""
        funcs = tuple(
            filter(rpartial(isinstance, FunctionDef), walk(self.ast_mod)),
        )
        for func in funcs:
            cst_idx, cst_node = find_cst_at_ast(cstify_cst, func)
            self.assertIsNotNone(cst_idx)
            self.assertIsNotNone(cst_node, "{name} not found".format(name=func.name))
            self.assertIsInstance(cst_node, FunctionDefinitionStart)

    def test_find_cst_at_ast_finds_class(self) -> None:
        """Test that `find_cst_at_ast` can find the CST for all functions"""
        class_def = next(
            filter(rpartial(isinstance, ClassDef), walk(self.ast_mod)),
        )
        cst_idx, cst_node = find_cst_at_ast(cstify_cst, class_def)
        self.assertIsNotNone(cst_idx)
        self.assertIsNotNone(cst_node)
        self.assertIsInstance(cst_node, ClassDefinitionStart)

    def test_find_cst_at_ast_errors_on_module(self) -> None:
        """Test that `find_cst_at_ast` fails to find the CST for `ast.Module`"""
        with patch("cdd.ast_cst_utils.stderr", new_callable=StringIO) as e:
            find_cst_at_ast(cstify_cst, self.ast_mod)

        self.assertEqual(e.getvalue(), "`Module` not implemented\n")

    def test_maybe_replace_doc_str_in_function_or_class_replaced(self) -> None:
        """tests test_maybe_replace_doc_str_in_function_or_class succeeds in replacing"""
        cst_list = list(deepcopy(cstify_cst))
        cst_idx, cst_node = find_cst_at_ast(cst_list, self.func_node)
        self.assertIsNotNone(cst_node)
        existing_doc_str = (
            "\n"
            '        """\n'
            "        :param foo: a foo\n"
            "        :type foo: ```int```\n"
            "\n"
            "        :return: foo + 1\n"
            "        :rtype: ```int```\n"
            '        """'
        )
        self.assertEqual(cst_list[cst_idx + 1].value, existing_doc_str)
        new_doc_str = "Rewritten docstring"
        func_node = deepcopy(self.func_node)
        func_node.body[0] = Expr(set_value(new_doc_str))
        self.assertEqual(
            maybe_replace_doc_str_in_function_or_class(func_node, cst_idx, cst_list),
            Delta.replaced,
        )
        self.assertEqual(cst_list[cst_idx + 1].value.strip()[3:-3], new_doc_str)

    def test_maybe_replace_doc_str_in_function_or_class_added(self) -> None:
        """tests test_maybe_replace_doc_str_in_function_or_class succeeds in adding"""
        cst_list = list(deepcopy(cstify_cst))
        cst_idx, cst_node = find_cst_at_ast(cst_list, self.func_node)
        self.assertIsNotNone(cst_node)
        del cst_list[cst_idx + 1]  # Remove docstr
        new_doc_str = "New docstring"
        func_node = deepcopy(self.func_node)
        func_node.body[0] = Expr(set_value(new_doc_str))
        self.assertEqual(
            maybe_replace_doc_str_in_function_or_class(func_node, cst_idx, cst_list),
            Delta.added,
        )
        self.assertEqual(cst_list[cst_idx + 1].value.strip()[3:-3], new_doc_str)

    def test_maybe_replace_doc_str_in_function_or_class_removed(self) -> None:
        """tests test_maybe_replace_doc_str_in_function_or_class succeeds in removing"""
        cst_list = list(deepcopy(cstify_cst))
        cst_idx, cst_node = find_cst_at_ast(cst_list, self.func_node)
        self.assertIsNotNone(cst_node)
        # del cst_list[cst_idx + 1]  # Remove docstr
        func_node = deepcopy(self.func_node)
        del func_node.body[0]  # Remove other [new] docstr
        with patch("cdd.ast_cst_utils.get_doc_str", lambda _: None):
            # Ignore existing docstr like structure^
            self.assertEqual(
                maybe_replace_doc_str_in_function_or_class(
                    func_node, cst_idx, cst_list
                ),
                Delta.removed,
            )
        self.assertFalse(cst_list[cst_idx + 1].is_docstr)


unittest_main()

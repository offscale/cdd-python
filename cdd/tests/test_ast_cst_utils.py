""" Tests for ast_cst_utils """

from ast import ClassDef, Expr, FunctionDef, Load, Name, Pass, walk
from copy import deepcopy
from io import StringIO
from operator import attrgetter
from os import path
from os.path import extsep
from unittest import TestCase
from unittest.mock import MagicMock, patch

from cdd.ast_cst_utils import (
    Delta,
    debug_doctrans,
    find_cst_at_ast,
    maybe_replace_doc_str_in_function_or_class,
    maybe_replace_function_args,
    maybe_replace_function_return_type,
)
from cdd.ast_utils import set_value
from cdd.cst_utils import (
    ClassDefinitionStart,
    FunctionDefinitionStart,
    TripleQuoted,
    reindent_block_with_pass_body,
)
from cdd.pure_utils import identity, rpartial
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

    def maybe_replace_doc_str_in_function_or_class_test(
        self, existing_doc_str, new_doc_str, delta
    ) -> None:
        """
        Helper for writing `maybe_replace_doc_str_in_function_or_class` tests

        :param existing_doc_str: Existing doc_str (delete if None else replace)
        :type existing_doc_str: ```Optional[str]```

        :param new_doc_str: doc_str expected to end up with (delete if None else replace)
        :type new_doc_str: ```AST```

        :param delta: Delta value indicating what changed (if anything)
        :type delta: ```Delta```
        """

        cst_list = list(deepcopy(cstify_cst))
        cst_idx, cst_node = find_cst_at_ast(cst_list, self.func_node)
        self.assertIsNotNone(cst_node)
        if existing_doc_str is None:
            del cst_list[cst_idx + 1]
        else:
            cst_list[cst_idx + 1] = TripleQuoted(
                is_double_q=cst_list[cst_idx + 1].is_double_q,
                is_docstr=cst_list[cst_idx + 1].is_docstr,
                value=existing_doc_str,
                line_no_start=cst_list[cst_idx + 1].line_no_start,
                line_no_end=cst_list[cst_idx + 1].line_no_end,
            )

        func_node = deepcopy(self.func_node)
        func_node.body[0] = (
            Pass() if new_doc_str is None else Expr(set_value(new_doc_str))
        )

        with patch("cdd.ast_cst_utils.debug_doctrans", identity):
            self.assertEqual(
                delta,
                maybe_replace_doc_str_in_function_or_class(
                    func_node, cst_idx, cst_list
                ),
            )

        if (
            isinstance(cst_list[cst_idx + 1], TripleQuoted)
            and cst_list[cst_idx + 1].is_docstr
        ):
            self.assertEqual(
                cst_list[cst_idx + 1].value.strip()[3:-3].strip(), new_doc_str
            )

    def test_maybe_replace_doc_str_in_function_or_class_replaced(self) -> None:
        """tests test_maybe_replace_doc_str_in_function_or_class succeeds in replacing"""

        self.maybe_replace_doc_str_in_function_or_class_test(
            existing_doc_str=(
                "\n"
                '        """\n'
                "        :param foo: a foo\n"
                "        :type foo: ```int```\n"
                "\n"
                "        :return: foo + 1\n"
                "        :rtype: ```int```\n"
                '        """'
            ),
            new_doc_str="Rewritten docstring",
            delta=Delta.replaced,
        )

    def test_maybe_replace_doc_str_in_function_or_class_added(self) -> None:
        """tests test_maybe_replace_doc_str_in_function_or_class succeeds in adding"""

        self.maybe_replace_doc_str_in_function_or_class_test(
            existing_doc_str=None,
            new_doc_str="New docstring",
            delta=Delta.added,
        )

    def test_maybe_replace_doc_str_in_function_or_class_removed(self) -> None:
        """tests test_maybe_replace_doc_str_in_function_or_class succeeds in removing"""

        self.maybe_replace_doc_str_in_function_or_class_test(
            existing_doc_str="Remove this",
            new_doc_str=None,
            delta=Delta.removed,
        )

    def maybe_replace_function_return_type_test(
        self,
        ast_node_to_find,
        ast_node_to_make,
        cur_ast_node,
        cur_cst_value,
        before,
        after,
        delta,
    ) -> None:
        """
        Helper for writing `maybe_replace_function_return_type` tests

        :param ast_node_to_find: AST node to find
        :type ast_node_to_find: ```AST```

        :param ast_node_to_make: AST node to end up with. If None
        :type ast_node_to_make: ```AST```

        :param cur_ast_node: AST parse of the CST node that currently exists where the `ast_node_to_find` is
        :type cur_ast_node: ```AST```

        :param cur_cst_value: Create a new FunctionDefinitionStart with this. If provided will replace existing.
        :type cur_cst_value: ```Optional[str]```

        :param before: Function prototype before function is run
        :type before: ```str```

        :param after: Function prototype after function is run
        :type after: ```str```

        :param delta: Delta value indicating what changed (if anything)
        :type delta: ```Delta```
        """

        cst_list = list(deepcopy(cstify_cst))
        cst_idx, cst_node = find_cst_at_ast(cst_list, ast_node_to_find)
        self.assertIsNotNone(cst_node)
        if cur_cst_value is not None:
            cst_list[cst_idx] = FunctionDefinitionStart(
                line_no_start=cst_list[cst_idx].line_no_start,
                line_no_end=cst_list[cst_idx].line_no_end,
                name=cst_list[cst_idx].name,
                value=cur_cst_value,
            )
        self.assertEqual(
            "".join(map(attrgetter("value"), cst_list[cst_idx : cst_idx + 1])), before
        )
        with patch("cdd.ast_cst_utils.debug_doctrans", identity):
            self.assertEqual(
                maybe_replace_function_return_type(
                    new_node=ast_node_to_make,
                    cst_idx=cst_idx,
                    cst_list=cst_list,
                    cur_ast_node=cur_ast_node,
                ),
                delta,
            )
        self.assertEqual(
            "".join(map(attrgetter("value"), cst_list[cst_idx : cst_idx + 1])), after
        )

    def test_maybe_replace_function_return_type_adds(self) -> None:
        """
        Tests that `maybe_replace_function_return_type` adds return type
        """

        before = "\n\n    @staticmethod\n    def add1(foo):"
        after = "\n\n    @staticmethod\n    def add1(foo) -> int:"

        new_func_node = deepcopy(self.func_node)
        new_func_node.returns = Name("int", Load())

        self.maybe_replace_function_return_type_test(
            ast_node_to_find=self.func_node,
            ast_node_to_make=new_func_node,
            cur_ast_node=ast_parse(
                "{func_start} pass".format(func_start=before.strip().replace("  ", "")),
                skip_annotate=True,
                skip_docstring_remit=True,
            ).body[0],
            cur_cst_value=None,
            before=before,
            after=after,
            delta=Delta.added,
        )

    def test_maybe_replace_function_return_type_removes(self):
        """
        Tests that `maybe_replace_function_return_type` removes return type
        """

        before = "\n\n    @staticmethod\n    def add1(foo) -> int:"
        after = "\n\n    @staticmethod\n    def add1(foo):"

        self.maybe_replace_function_return_type_test(
            ast_node_to_find=self.func_node,
            ast_node_to_make=self.func_node,
            cur_ast_node=ast_parse(
                "{func_start} pass".format(func_start=before.strip().replace("  ", "")),
                skip_annotate=True,
                skip_docstring_remit=True,
            ).body[0],
            cur_cst_value=before,
            before=before,
            after=after,
            delta=Delta.removed,
        )

    def test_maybe_replace_function_return_type_replaces(self):
        """
        Tests that `maybe_replace_function_return_type` replaces return type
        """

        before = "\n\n    @staticmethod\n    def add1(foo) -> int:"
        after = "\n\n    @staticmethod\n    def add1(foo) -> float:"

        new_func_node = deepcopy(self.func_node)
        new_func_node.returns = Name("float", Load())

        self.maybe_replace_function_return_type_test(
            ast_node_to_find=self.func_node,
            ast_node_to_make=new_func_node,
            cur_ast_node=ast_parse(
                "{func_start} pass".format(func_start=before.strip().replace("  ", "")),
                skip_annotate=True,
                skip_docstring_remit=True,
            ).body[0],
            cur_cst_value=before,
            before=before,
            after=after,
            delta=Delta.replaced,
        )

    def test_maybe_replace_function_args_nop(self):
        """
        Tests that `maybe_replace_function_args` does nothing on equal args
        """

        func_src = "\n\n    @staticmethod\n    def add1(foo: int) -> int:"

        func_node = ast_parse(
            "{func_start} pass".format(func_start=func_src.strip().replace("  ", "")),
            skip_annotate=True,
            skip_docstring_remit=True,
        ).body[0]

        cst_list = list(deepcopy(cstify_cst))
        cst_idx, cst_node = find_cst_at_ast(cst_list, self.func_node)
        self.assertIsNotNone(cst_node)
        cst_list[cst_idx] = FunctionDefinitionStart(
            line_no_start=cst_list[cst_idx].line_no_start,
            line_no_end=cst_list[cst_idx].line_no_end,
            name=cst_list[cst_idx].name,
            value=func_src,
        )
        self.assertEqual(
            maybe_replace_function_args(
                new_node=func_node,
                cur_ast_node=func_node,
                cst_idx=cst_idx,
                cst_list=cst_list,
            ),
            Delta.nop,
        )
        self.assertEqual(
            "".join(map(attrgetter("value"), cst_list[cst_idx : cst_idx + 1])), func_src
        )

    def maybe_replace_function_args_test(self, before, after, delta):
        """
        Helper for writing `maybe_replace_function_args` tests

        :param before: Function prototype before function is run
        :type before: ```str```

        :param after: Function prototype after function is run
        :type after: ```str```

        :param delta: Delta value indicating what changed (if anything)
        :type delta: ```Delta```
        """

        new_node = ast_parse(
            reindent_block_with_pass_body(after),
            skip_annotate=True,
            skip_docstring_remit=True,
        ).body[0]

        cur_ast_node = ast_parse(
            reindent_block_with_pass_body(before),
            skip_annotate=True,
            skip_docstring_remit=True,
        ).body[0]

        cst_list = list(deepcopy(cstify_cst))
        cst_idx, cst_node = find_cst_at_ast(cst_list, self.func_node)
        self.assertIsNotNone(cst_node)
        cst_list[cst_idx] = FunctionDefinitionStart(
            line_no_start=cst_list[cst_idx].line_no_start,
            line_no_end=cst_list[cst_idx].line_no_end,
            name=cst_list[cst_idx].name,
            value=before,
        )
        with patch("cdd.ast_cst_utils.debug_doctrans", identity):
            self.assertEqual(
                maybe_replace_function_args(
                    new_node=new_node,
                    cur_ast_node=cur_ast_node,
                    cst_idx=cst_idx,
                    cst_list=cst_list,
                ),
                delta,
            )
        self.assertEqual(
            after, "".join(map(attrgetter("value"), cst_list[cst_idx : cst_idx + 1]))
        )

    def test_maybe_replace_function_args_added(self):
        """
        Tests that `maybe_replace_function_args` adds to args (adds type annotations)
        """

        self.maybe_replace_function_args_test(
            before="\n\n    @staticmethod\n    def add1(foo) -> int:",
            after="\n\n    @staticmethod\n    def add1(foo: int) -> int:",
            delta=Delta.added,
        )

    def test_maybe_replace_function_args_removed(self):
        """
        Tests that `maybe_replace_function_args` removes from args (removes type annotations)
        """

        self.maybe_replace_function_args_test(
            before="\n\n    @staticmethod\n    def add1(foo: int) -> int:",
            after="\n\n    @staticmethod\n    def add1(foo) -> int:",
            delta=Delta.removed,
        )

    def test_maybe_replace_function_args_replaced(self):
        """
        Tests that `maybe_replace_function_args` replaces to args (replaces type annotations)
        """

        self.maybe_replace_function_args_test(
            before="\n\n    @staticmethod\n    def add1(foo: int) -> int:",
            after="\n\n    @staticmethod\n    def add1(foo: float) -> int:",
            delta=Delta.replaced,
        )

    def test_debug_doctrans(self) -> None:
        """
        Test that the `print` function is called the right number of times from  `debug_doctrans`' invocation
        """
        with patch("cdd.ast_cst_utils.print", new_callable=MagicMock()) as print_func:
            debug_doctrans(Delta.added, "", "", "")
        self.assertEqual(print_func.call_count, 1)


unittest_main()

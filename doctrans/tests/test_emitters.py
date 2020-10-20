"""
Tests for marshalling between formats
"""
import ast
import os
from ast import FunctionDef
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest import TestCase

from doctrans import emit, parse
from doctrans.ast_utils import (
    get_function_type,
    find_in_ast,
    annotate_ancestry,
)
from doctrans.pure_utils import rpartial, reindent, tab, deindent, PY3_8, PY_GTE_3_9
from doctrans.tests.mocks.argparse import (
    argparse_func_ast,
    argparse_func_with_body_ast,
    argparse_func_nargs_ast,
)
from doctrans.tests.mocks.classes import class_ast, class_nargs_ast
from doctrans.tests.mocks.docstrings import docstring_str, intermediate_repr
from doctrans.tests.mocks.methods import (
    class_with_method_types_ast,
    class_with_method_ast,
    class_with_method_and_body_types_ast,
    class_with_method_types_str,
)
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main
from meta.asttools import cmp_ast


class TestEmitters(TestCase):
    """ Tests whether conversion between formats works """

    def test_to_class_from_argparse_ast(self) -> None:
        """
        Tests whether `class_` produces `class_ast` given `argparse_func_ast`
        """
        a = parse.argparse_ast(argparse_func_ast)
        gen_ast = emit.class_(a)
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=class_ast,
        )

    def test_to_class_from_docstring_str(self) -> None:
        """
        Tests whether `class_` produces `class_ast` given `docstring_str`
        """
        run_ast_test(
            self,
            emit.class_(
                parse.docstring(docstring_str),
            ),
            gold=class_ast,
        )

    def test_to_argparse(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_func_ast` given `class_ast`
        """
        run_ast_test(
            self,
            emit.argparse_function(
                parse.class_(class_ast),
                emit_default_doc=False,
                emit_default_doc_in_return=False,
            ),
            gold=argparse_func_ast,
        )

    def test_to_argparse_func_nargs(self) -> None:
        run_ast_test(
            self,
            emit.argparse_function(
                parse.class_(class_nargs_ast),
                emit_default_doc=False,
                emit_default_doc_in_return=False,
                function_name="set_cli_nargs",
            ),
            gold=argparse_func_nargs_ast,
        )

    def test_to_docstring(self) -> None:
        """
        Tests whether `docstring` produces `docstring_str` given `class_ast`
        """
        self.assertEqual(
            emit.docstring(parse.class_(class_ast)),
            docstring_str,
        )

    def test_to_numpy_docstring_fails(self) -> None:
        """
        Tests whether `docstring` fails when `docstring_format` is 'numpy'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: emit.docstring(intermediate_repr, docstring_format="numpy"),
        )

    def test_to_google_docstring_fails(self) -> None:
        """
        Tests whether `docstring` fails when `docstring_format` is 'google'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: emit.docstring(intermediate_repr, docstring_format="google"),
        )

    def test_to_file(self) -> None:
        """
        Tests whether `file` constructs a file, and fills it with the right content
        """

        with TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "delete_me.py")
            try:
                emit.file(class_ast, filename, skip_black=True)

                with open(filename, "rt") as f:
                    ugly = f.read()

                os.remove(filename)

                emit.file(class_ast, filename, skip_black=False)

                with open(filename, "rt") as f:
                    blacked = f.read()

                self.assertNotEqual(ugly, blacked)
                # if PY3_8:
                self.assertTrue(
                    cmp_ast(ast.parse(ugly), ast.parse(blacked)),
                    "Ugly AST doesn't match blacked AST",
                )

            finally:
                if os.path.isfile(filename):
                    os.remove(filename)

    def test_to_function(self) -> None:
        """
        Tests whether `function` produces method from `class_with_method_types_ast` given `docstring_str`
        """

        function_def = deepcopy(
            next(
                filter(
                    rpartial(isinstance, FunctionDef), class_with_method_types_ast.body
                )
            )
        )
        # Reindent docstring
        function_def.body[0].value.value = "\n{tab}{docstring}\n{tab}".format(
            tab=tab,
            docstring=reindent(
                deindent(ast.get_docstring(function_def)),
            ),
        )

        function_name = function_def.name
        function_type = get_function_type(function_def)

        gen_ast = emit.function(
            parse.docstring(docstring_str),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            inline_types=True,
            emit_separating_tab=PY3_8,
            indent_level=0,
            emit_as_kwonlyargs=False,
        )

        gen_ast.body[0].value.value = "\n{tab}{docstring}".format(
            tab=tab, docstring=reindent(deindent(ast.get_docstring(gen_ast)))
        )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_to_function_with_docstring_types(self) -> None:
        """
        Tests that `function` can generate a function_def with types in docstring
        """
        function_def = deepcopy(
            next(filter(rpartial(isinstance, FunctionDef), class_with_method_ast.body))
        )
        # Reindent docstring
        function_def.body[0].value.value = "\n{tab}{docstring}\n{tab}".format(
            tab=tab, docstring=reindent(ast.get_docstring(function_def))
        )

        gen_ast = emit.function(
            parse.function(function_def),
            function_name=function_def.name,
            function_type=get_function_type(function_def),
            emit_default_doc=False,
            inline_types=False,
            indent_level=1,
            emit_separating_tab=True,
            emit_as_kwonlyargs=False,
        )
        gen_ast.body[0].value.value = "\n{tab}{docstring}\n{tab}".format(
            tab=tab, docstring=reindent(ast.get_docstring(gen_ast))
        )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_to_function_with_inline_types(self) -> None:
        """
        Tests that `function` can generate a function_def with inline types
        """
        function_def = deepcopy(
            next(
                filter(
                    rpartial(isinstance, FunctionDef), class_with_method_types_ast.body
                )
            )
        )
        function_name = function_def.name
        function_type = get_function_type(function_def)

        gen_ast = emit.function(
            intermediate_repr=parse.function(
                function_def=function_def,
                function_name=function_name,
                function_type=function_type,
            ),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            inline_types=True,
            emit_separating_tab=True,
            indent_level=1,
            emit_as_kwonlyargs=False,
        )
        # emit.file(gen_ast, os.path.join(os.path.dirname(__file__), 'delme.py'), mode='wt')
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_to_function_emit_as_kwonlyargs(self) -> None:
        """
        Tests whether `function` produces method with keyword only arguments
        """

        function_def = deepcopy(
            next(
                filter(
                    rpartial(isinstance, FunctionDef),
                    ast.parse(class_with_method_types_str.replace("self", "self, *"))
                    .body[0]
                    .body,
                )
            )
        )
        # Reindent docstring
        function_def.body[0].value.value = "\n{tab}{docstring}\n{tab}".format(
            tab=tab,
            docstring=reindent(
                deindent(ast.get_docstring(function_def)),
            ),
        )

        function_name = function_def.name
        function_type = get_function_type(function_def)

        gen_ast = emit.function(
            parse.docstring(docstring_str),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            inline_types=True,
            emit_separating_tab=PY3_8,
            indent_level=0,
            emit_as_kwonlyargs=True,
        )

        gen_ast.body[0].value.value = "\n{tab}{docstring}".format(
            tab=tab, docstring=reindent(deindent(ast.get_docstring(gen_ast)))
        )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_from_class_with_body_in_method_to_method_with_body(self) -> None:
        """ Tests if this can make the roundtrip from a full function to a full function """
        annotate_ancestry(class_with_method_and_body_types_ast)

        function_def = next(
            filter(
                rpartial(isinstance, FunctionDef),
                class_with_method_and_body_types_ast.body,
            )
        )
        # Reindent docstring
        function_def.body[0].value.value = "\n{tab}{docstring}\n{tab}".format(
            tab=tab, docstring=reindent(ast.get_docstring(function_def))
        )

        gen_ast = emit.function(
            parse.function(
                find_in_ast(
                    "C.function_name".split("."),
                    class_with_method_and_body_types_ast,
                ),
            ),
            emit_default_doc=False,
            function_name="function_name",
            function_type="self",
            indent_level=1,
            emit_separating_tab=True,
            emit_as_kwonlyargs=False,
        )

        # emit.file(gen_ast, os.path.join(os.path.dirname(__file__), "delme.py"), mode="wt")

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
            run_cmp_ast=PY_GTE_3_9,
        )

    def test_from_argparse_with_extra_body_to_argparse_with_extra_body(self) -> None:
        """ Tests if this can make the roundtrip from a full argparse function to a argparse full function """

        run_ast_test(
            self,
            emit.argparse_function(
                parse.argparse_ast(argparse_func_with_body_ast),
                emit_default_doc=False,
                emit_default_doc_in_return=False,
            ),
            argparse_func_with_body_ast,
        )


unittest_main()

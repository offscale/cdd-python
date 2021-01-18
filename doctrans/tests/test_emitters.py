"""
Tests for marshalling between formats
"""
import ast
import os
from ast import FunctionDef
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest import TestCase

from meta.asttools import cmp_ast

from doctrans import emit, parse
from doctrans.ast_utils import annotate_ancestry, find_in_ast, get_function_type
from doctrans.pure_utils import PY3_8, deindent, reindent, rpartial, tab
from doctrans.tests.mocks.argparse import (
    argparse_func_action_append_ast,
    argparse_func_ast,
    argparse_func_torch_nn_l1loss_ast,
    argparse_func_with_body_ast,
    argparse_function_google_tf_tensorboard_ast,
)
from doctrans.tests.mocks.classes import (
    class_ast,
    class_google_tf_tensorboard_ast,
    class_nargs_ast,
    class_squared_hinge_config_ast,
)
from doctrans.tests.mocks.docstrings import docstring_no_default_str, docstring_str
from doctrans.tests.mocks.ir import class_torch_nn_l1loss_ir, intermediate_repr
from doctrans.tests.mocks.methods import (
    class_with_method_and_body_types_ast,
    class_with_method_ast,
    class_with_method_str,
    class_with_method_types_ast,
    class_with_method_types_str,
    function_google_tf_squared_hinge_str,
)
from doctrans.tests.utils_for_tests import run_ast_test, unittest_main


class TestEmitters(TestCase):
    """ Tests whether conversion between formats works """

    def test_to_class_from_argparse_ast(self) -> None:
        """
        Tests whether `class_` produces `class_ast` given `argparse_func_ast`
        """
        run_ast_test(
            self,
            gen_ast=emit.class_(
                parse.argparse_ast(argparse_func_ast), emit_default_doc=True
            ),
            gold=class_ast,
        )

    def test_to_class_from_argparse_action_append_ast(self) -> None:
        """
        Tests whether a class from an argparse function with `nargs` set
        """
        run_ast_test(
            self,
            emit.class_(
                parse.argparse_ast(argparse_func_action_append_ast),
            ),
            gold=class_nargs_ast,
        )

    def test_to_class_from_docstring_str(self) -> None:
        """
        Tests whether `class_` produces `class_ast` given `docstring_str`
        """
        run_ast_test(
            self,
            emit.class_(
                parse.docstring(docstring_str, emit_default_doc=True),
                emit_default_doc=True,
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
        """
        Tests whether an argparse function is generated with `action="append"` set properly
        """
        run_ast_test(
            self,
            gen_ast=emit.argparse_function(
                parse.class_(class_nargs_ast),
                emit_default_doc=False,
                emit_default_doc_in_return=False,
                function_name="set_cli_action_append",
            ),
            gold=argparse_func_action_append_ast,
        )

    def test_to_argparse_google_tf_tensorboard(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_function_google_tf_tensorboard_ast`
                                    given `class_google_tf_tensorboard_ast`
        """
        ir = parse.class_(
            class_google_tf_tensorboard_ast, merge_inner_function="__init__"
        )
        run_ast_test(
            self,
            emit.argparse_function(
                ir,
                emit_default_doc=False,
                emit_default_doc_in_return=False,
            ),
            gold=argparse_function_google_tf_tensorboard_ast,
        )

    def test_to_docstring(self) -> None:
        """
        Tests whether `docstring` produces `docstring_str` given `class_ast`
        """
        self.assertEqual(
            emit.docstring(parse.class_(class_ast), emit_default_doc=True),
            docstring_str,
        )

    def test_to_docstring_emit_default_doc_false(self) -> None:
        """
        Tests whether `docstring` produces `docstring_str` given `class_ast`
        """
        ir = parse.class_(class_ast)
        self.assertEqual(
            emit.docstring(ir, emit_default_doc=False),
            docstring_no_default_str,
        )

    maxDiff = None

    def test_to_numpy_docstring_fails(self) -> None:
        """
        Tests whether `docstring` fails when `docstring_format` is 'numpy'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: emit.docstring(
                deepcopy(intermediate_repr), docstring_format="numpy"
            ),
        )

    def test_to_google_docstring_fails(self) -> None:
        """
        Tests whether `docstring` fails when `docstring_format` is 'google'
        """
        self.assertRaises(
            NotImplementedError,
            lambda: emit.docstring(
                deepcopy(intermediate_repr), docstring_format="google"
            ),
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
                deindent(ast.get_docstring(function_def).translate("\n\t")),
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

        # Sanity check
        run_ast_test(
            self,
            class_with_method_ast,
            gold=ast.parse(class_with_method_str).body[0],
        )

        function_def = deepcopy(
            next(filter(rpartial(isinstance, FunctionDef), class_with_method_ast.body))
        )
        # Reindent docstring
        function_def.body[0].value.value = "\n{tab}{docstring}\n{tab}".format(
            tab=tab, docstring=reindent(ast.get_docstring(function_def))
        )

        ir = parse.function(function_def)
        gen_ast = emit.function(
            ir,
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

        ir = parse.function(
            find_in_ast(
                "C.function_name".split("."),
                class_with_method_and_body_types_ast,
            ),
        )
        gen_ast = emit.function(
            ir,
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
        )

    def test_from_function_google_tf_squared_hinge_str_to_class(self) -> None:
        """
        Tests that `emit.function` produces correctly with:
        - __call__
        """
        gen_ast = emit.class_(
            parse.function(
                ast.parse(function_google_tf_squared_hinge_str).body[0],
                infer_type=True,
            ),
            class_name="SquaredHingeConfig",
            emit_call=True,
            emit_default_doc=True,
        )
        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=class_squared_hinge_config_ast,
        )

    def test_from_argparse_with_extra_body_to_argparse_with_extra_body(self) -> None:
        """ Tests if this can make the roundtrip from a full argparse function to a argparse full function """

        ir = parse.argparse_ast(argparse_func_with_body_ast)
        func = emit.argparse_function(
            ir,
            emit_default_doc=False,
            emit_default_doc_in_return=False,
        )
        run_ast_test(
            self,
            func,
            argparse_func_with_body_ast,
        )

    def test_from_torch_ir_to_argparse(self) -> None:
        """ Tests if emission of class from torch IR is as expected """

        func = emit.argparse_function(
            deepcopy(class_torch_nn_l1loss_ir),
            emit_default_doc=False,
            emit_default_doc_in_return=False,
        )
        run_ast_test(
            self,
            func,
            argparse_func_torch_nn_l1loss_ast,
        )


unittest_main()

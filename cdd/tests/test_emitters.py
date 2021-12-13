"""
Tests for marshalling between formats
"""

import ast
import os
from ast import Expr, FunctionDef, arguments
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from os.path import extsep
from platform import system
from sys import modules
from tempfile import TemporaryDirectory
from textwrap import indent
from unittest import TestCase, skipIf

from cdd import emit, parse
from cdd.ast_utils import (
    annotate_ancestry,
    cmp_ast,
    find_in_ast,
    get_function_type,
    set_value,
)
from cdd.pure_utils import none_types, omit_whitespace, reindent, rpartial, tab
from cdd.tests.mocks.argparse import (
    argparse_func_action_append_ast,
    argparse_func_ast,
    argparse_func_torch_nn_l1loss_ast,
    argparse_func_with_body_ast,
    argparse_function_google_tf_tensorboard_ast,
)
from cdd.tests.mocks.classes import (
    class_ast,
    class_google_tf_tensorboard_ast,
    class_nargs_ast,
    class_squared_hinge_config_ast,
)
from cdd.tests.mocks.docstrings import (
    docstring_google_str,
    docstring_google_tf_ops_losses__safe_mean_str,
    docstring_header_str,
    docstring_no_default_str,
    docstring_no_nl_str,
    docstring_no_type_no_default_tpl_str,
    docstring_numpydoc_str,
    docstring_str,
)
from cdd.tests.mocks.ir import (
    class_torch_nn_l1loss_ir,
    function_google_tf_ops_losses__safe_mean_ir,
    intermediate_repr,
    intermediate_repr_no_default_sql_doc,
)
from cdd.tests.mocks.json_schema import config_schema
from cdd.tests.mocks.methods import (
    class_with_method_and_body_types_ast,
    class_with_method_ast,
    class_with_method_str,
    class_with_method_types_ast,
    class_with_method_types_str,
    function_google_tf_ops_losses__safe_mean_ast,
    function_google_tf_squared_hinge_str,
)
from cdd.tests.mocks.sqlalchemy import config_decl_base_ast, config_tbl_ast
from cdd.tests.utils_for_tests import reindent_docstring, run_ast_test, unittest_main


class TestEmitters(TestCase):
    """Tests whether conversion between formats works"""

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
                parse.docstring(docstring_no_nl_str, emit_default_doc=True),
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
            gen_ast=emit.argparse_function(
                parse.class_(class_ast),
                emit_default_doc=False,
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
                function_name="set_cli_action_append",
            ),
            gold=argparse_func_action_append_ast,
        )

    def test_to_argparse_google_tf_tensorboard(self) -> None:
        """
        Tests whether `to_argparse` produces `argparse_function_google_tf_tensorboard_ast`
                                    given `class_google_tf_tensorboard_ast`
        """
        run_ast_test(
            self,
            gen_ast=emit.argparse_function(
                parse.class_(
                    class_google_tf_tensorboard_ast, merge_inner_function="__init__"
                ),
                emit_default_doc=False,
                word_wrap=False,
            ),
            gold=argparse_function_google_tf_tensorboard_ast,
        )

    def test_to_docstring(self) -> None:
        """
        Tests whether `docstring` produces indented `docstring_str` given `class_ast`
        """
        self.assertEqual(
            emit.docstring(parse.class_(class_ast), emit_default_doc=True),
            reindent(docstring_str, 1),
        )

    def test_to_docstring_emit_default_doc_false(self) -> None:
        """
        Tests whether `docstring` produces `docstring_str` given `class_ast`
        """
        ir = parse.class_(class_ast)
        self.assertEqual(
            emit.docstring(ir, emit_default_doc=False),
            reindent(docstring_no_default_str, 1),
        )

    def test_to_numpy_docstring(self) -> None:
        """
        Tests whether `docstring` produces `docstring_numpydoc_str` when `docstring_format` is 'numpydoc'
        """
        self.assertEqual(
            docstring_numpydoc_str,
            emit.docstring(deepcopy(intermediate_repr), docstring_format="numpydoc"),
        )

    def test_to_google_docstring(self) -> None:
        """
        Tests whether `docstring` produces `docstring_google_str` when `docstring_format` is 'google'
        """
        self.assertEqual(
            docstring_google_str,
            emit.docstring(deepcopy(intermediate_repr), docstring_format="google"),
        )

    def test_to_google_docstring_no_types(self) -> None:
        """
        Tests whether a Google docstring is correctly generated sans types
        """

        self.assertEqual(
            *map(
                omit_whitespace,
                (
                    docstring_google_tf_ops_losses__safe_mean_str,
                    emit.docstring(
                        deepcopy(function_google_tf_ops_losses__safe_mean_ir),
                        docstring_format="google",
                        emit_original_whitespace=True,
                        emit_default_doc=False,
                        word_wrap=True,
                    ),
                ),
            )
        )

    def test_to_docstring_use_original_when_whitespace_only_changes(self) -> None:
        """
        Tests whether original docstring is used when whitespace only changes are made
        """

        self.assertEqual(
            *map(
                partial(ast.get_docstring, clean=True),
                map(
                    lambda doc_str: FunctionDef(
                        name="_",
                        args=arguments(
                            posonlyargs=[],
                            args=[],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                            vararg=None,
                            kwarg=None,
                        ),
                        body=[Expr(doc_str)],
                        decorator_list=[],
                        lineno=None,
                        returns=None,
                    ),
                    (
                        emit.docstring(
                            parse.function(
                                function_google_tf_ops_losses__safe_mean_ast
                            ),
                            docstring_format="google",
                            emit_original_whitespace=True,
                            emit_default_doc=False,
                        ),
                        docstring_google_tf_ops_losses__safe_mean_str,
                    ),
                ),
            )
        )

    def test_to_file(self) -> None:
        """
        Tests whether `file` constructs a file, and fills it with the right content
        """

        with TemporaryDirectory() as tempdir:
            filename = os.path.join(
                tempdir, "delete_me{extsep}py".format(extsep=extsep)
            )
            try:
                emit.file(class_ast, filename, skip_black=True)

                with open(filename, "rt") as f:
                    ugly = f.read()

                os.remove(filename)

                emit.file(class_ast, filename, skip_black=False)

                with open(filename, "rt") as f:
                    blacked = f.read()

                self.assertNotEqual(ugly + "" if "black" in modules else "\t", blacked)
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

        function_def = reindent_docstring(
            deepcopy(
                next(
                    filter(
                        rpartial(isinstance, FunctionDef),
                        class_with_method_types_ast.body,
                    )
                )
            )
        )

        function_name = function_def.name
        function_type = get_function_type(function_def)

        gen_ast = emit.function(
            parse.docstring(docstring_str),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            type_annotations=True,
            emit_separating_tab=True,
            indent_level=1,
            emit_as_kwonlyargs=False,
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

        function_def = reindent_docstring(
            deepcopy(
                next(
                    filter(
                        rpartial(isinstance, FunctionDef), class_with_method_ast.body
                    )
                )
            )
        )

        ir = parse.function(function_def)
        gen_ast = reindent_docstring(
            emit.function(
                ir,
                function_name=function_def.name,
                function_type=get_function_type(function_def),
                emit_default_doc=False,
                type_annotations=False,
                indent_level=1,
                emit_separating_tab=True,
                emit_as_kwonlyargs=False,
                word_wrap=False,
            )
        )

        run_ast_test(self, gen_ast=gen_ast, gold=function_def)

    def test_to_function_with_type_annotations(self) -> None:
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
        function_def.body[0].value = set_value(
            "\n{tab}{ds}{tab}".format(
                tab=tab,
                ds=indent(
                    docstring_no_type_no_default_tpl_str.format(
                        header_doc_str=docstring_header_str
                    ),
                    prefix=tab,
                    predicate=lambda _: _,
                ).lstrip(),
            )
        )

        gen_ast = emit.function(
            parse.function(
                function_def,
                function_name=function_name,
                function_type=function_type,
            ),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            type_annotations=True,
            emit_separating_tab=True,
            indent_level=0,
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
        function_def = reindent_docstring(
            deepcopy(
                next(
                    filter(
                        rpartial(isinstance, FunctionDef),
                        ast.parse(
                            class_with_method_types_str.replace("self", "self, *")
                        )
                        .body[0]
                        .body,
                    )
                )
            )
        )
        function_name = function_def.name
        function_type = get_function_type(function_def)

        gen_ast = emit.function(
            parse.docstring(docstring_str),
            function_name=function_name,
            function_type=function_type,
            emit_default_doc=False,
            type_annotations=True,
            emit_separating_tab=True,
            indent_level=1,
            emit_as_kwonlyargs=True,
        )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=function_def,
        )

    def test_from_class_with_body_in_method_to_method_with_body(self) -> None:
        """Tests if this can make the roundtrip from a full function to a full function"""
        annotate_ancestry(class_with_method_and_body_types_ast)

        function_def = reindent_docstring(
            next(
                filter(
                    rpartial(isinstance, FunctionDef),
                    class_with_method_and_body_types_ast.body,
                )
            )
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
            indent_level=0,
            emit_separating_tab=True,
            emit_as_kwonlyargs=False,
        )

        # emit.file(gen_ast, os.path.join(os.path.dirname(__file__),
        #           "delme{extsep}py".format(extsep=extsep)), mode="wt")

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

        gold_ir = parse.class_(class_squared_hinge_config_ast)
        gold_ir.update(
            {
                key: OrderedDict(
                    (
                        (
                            name,
                            {
                                k: v
                                for k, v in _param.items()
                                if k != "typ"
                                and (k != "default" or v not in none_types)
                            },
                        )
                        for name, _param in gold_ir[key].items()
                    )
                )
                for key in ("params", "returns")
            }
        )

        gen_ir = parse.function(
            ast.parse(function_google_tf_squared_hinge_str).body[0],
            infer_type=True,
            word_wrap=False,
        )
        self.assertEqual(
            *map(lambda ir: ir["returns"]["return_type"]["doc"], (gen_ir, gold_ir))
        )
        # print('#gen_ir')
        # pp(gen_ir)
        # print('#gold_ir')
        # pp(gold_ir)

        run_ast_test(
            self,
            *map(
                partial(
                    emit.class_,
                    class_name="SquaredHingeConfig",
                    emit_call=True,
                    emit_default_doc=True,
                    emit_original_whitespace=True,
                    word_wrap=False,
                ),
                (gen_ir, gold_ir),
            )
        )

    def test_from_argparse_with_extra_body_to_argparse_with_extra_body(self) -> None:
        """Tests if this can make the roundtrip from a full argparse function to a argparse full function"""

        ir = parse.argparse_ast(argparse_func_with_body_ast)
        func = emit.argparse_function(ir, emit_default_doc=False, word_wrap=True)
        run_ast_test(
            self, *map(reindent_docstring, (func, argparse_func_with_body_ast))
        )

    def test_from_torch_ir_to_argparse(self) -> None:
        """Tests if emission of class from torch IR is as expected"""

        func = emit.argparse_function(
            deepcopy(class_torch_nn_l1loss_ir),
            emit_default_doc=False,
            wrap_description=False,
            word_wrap=False,
        )
        run_ast_test(
            self,
            func,
            argparse_func_torch_nn_l1loss_ast,
        )

    def test_to_json_schema(self) -> None:
        """
        Tests that `emit.json_schema` with `intermediate_repr_no_default_doc` produces `config_schema`
        """
        self.assertEqual(
            emit.json_schema(
                deepcopy(intermediate_repr_no_default_sql_doc),
                "https://offscale.io/config.schema.json",
                emit_original_whitespace=True,
            )["description"],
            config_schema["description"],
        )
        self.assertDictEqual(
            emit.json_schema(
                deepcopy(intermediate_repr_no_default_sql_doc),
                "https://offscale.io/config.schema.json",
                emit_original_whitespace=True,
            ),
            config_schema,
        )

    def test_to_sqlalchemy_table(self) -> None:
        """
        Tests that `emit.sqlalchemy_table` with `intermediate_repr_no_default_sql_doc` produces `config_tbl_ast`
        """
        run_ast_test(
            self,
            emit.sqlalchemy_table(
                deepcopy(intermediate_repr_no_default_sql_doc), name="config_tbl"
            ),
            gold=config_tbl_ast,
        )

    @skipIf(
        "GITHUB_ACTIONS" in os.environ and system() in frozenset(("Darwin", "Linux")),
        "GitHub Actions fails this test on macOS & Linux (unable to replicate locally)",
    )
    def test_to_sqlalchemy(self) -> None:
        """
        Tests that `emit.sqlalchemy` with `intermediate_repr_no_default_sql_doc` produces `config_tbl_ast`
        """
        system() in frozenset(("Darwin", "Linux")) and print("test_to_sqlalchemy")

        run_ast_test(
            self,
            emit.sqlalchemy(
                deepcopy(intermediate_repr_no_default_sql_doc),
                class_name="Config",
                table_name="config_tbl",
            ),
            gold=config_decl_base_ast,
        )


unittest_main()

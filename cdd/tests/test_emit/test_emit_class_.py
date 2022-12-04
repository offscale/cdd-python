"""
Tests for `cdd.emit.class_`
"""

from ast import FunctionDef
from unittest import TestCase

import cdd.emit.argparse_function
import cdd.emit.class_
import cdd.emit.docstring
import cdd.emit.file
import cdd.emit.function
import cdd.emit.json_schema
import cdd.emit.sqlalchemy
import cdd.parse.argparse_function
import cdd.parse.docstring
import cdd.parse.function
from cdd.ast_utils import annotate_ancestry, find_in_ast
from cdd.pure_utils import rpartial
from cdd.tests.mocks.argparse import argparse_func_action_append_ast, argparse_func_ast
from cdd.tests.mocks.classes import class_ast, class_nargs_ast
from cdd.tests.mocks.docstrings import docstring_no_nl_str
from cdd.tests.mocks.methods import class_with_method_and_body_types_ast
from cdd.tests.utils_for_tests import reindent_docstring, run_ast_test, unittest_main


class TestEmitClass(TestCase):
    """Tests emission"""

    def test_to_class_from_argparse_ast(self) -> None:
        """
        Tests whether `class_` produces `class_ast` given `argparse_func_ast`
        """

        ir = cdd.parse.argparse_function.argparse_ast(argparse_func_ast)
        gen_ast = cdd.emit.class_.class_(
            ir,
            emit_default_doc=True,
            class_name="ConfigClass",
        )

        run_ast_test(
            self,
            gen_ast=gen_ast,
            gold=class_ast,
        )

    def test_to_class_from_argparse_action_append_ast(self) -> None:
        """
        Tests whether a class from an argparse function with `nargs` set
        """
        run_ast_test(
            self,
            cdd.emit.class_.class_(
                cdd.parse.argparse_function.argparse_ast(
                    argparse_func_action_append_ast
                ),
                class_name="ConfigClass",
            ),
            gold=class_nargs_ast,
        )

    def test_to_class_from_docstring_str(self) -> None:
        """
        Tests whether `class_` produces `class_ast` given `docstring_str`
        """
        run_ast_test(
            self,
            cdd.emit.class_.class_(
                cdd.parse.docstring.docstring(
                    docstring_no_nl_str, emit_default_doc=True
                ),
                emit_default_doc=True,
                class_name="ConfigClass",
            ),
            gold=class_ast,
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

        ir = cdd.parse.function.function(
            find_in_ast(
                "C.function_name".split("."),
                class_with_method_and_body_types_ast,
            ),
        )
        gen_ast = cdd.emit.function.function(
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


unittest_main()

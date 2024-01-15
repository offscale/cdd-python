"""
Tests for `cdd.emit.class_`
"""

from ast import FunctionDef
from typing import cast
from unittest import TestCase

import cdd.argparse_function.emit
import cdd.argparse_function.parse
import cdd.class_.emit
import cdd.docstring.emit
import cdd.docstring.parse
import cdd.function.emit
import cdd.function.parse
import cdd.json_schema.emit
import cdd.shared.emit.file
import cdd.sqlalchemy.emit
from cdd.shared.ast_utils import annotate_ancestry, find_in_ast
from cdd.shared.pure_utils import rpartial
from cdd.shared.types import IntermediateRepr
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

        ir: IntermediateRepr = cdd.argparse_function.parse.argparse_ast(
            argparse_func_ast
        )
        gen_ast = cdd.class_.emit.class_(
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
            cdd.class_.emit.class_(
                cdd.argparse_function.parse.argparse_ast(
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
            cdd.class_.emit.class_(
                cdd.docstring.parse.docstring(
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

        function_def: FunctionDef = cast(
            FunctionDef,
            reindent_docstring(
                next(
                    filter(
                        rpartial(isinstance, FunctionDef),
                        class_with_method_and_body_types_ast.body,
                    )
                )
            ),
        )

        ir: IntermediateRepr = cdd.function.parse.function(
            find_in_ast(
                "C.function_name".split("."),
                class_with_method_and_body_types_ast,
            ),
        )
        gen_ast = cdd.function.emit.function(
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

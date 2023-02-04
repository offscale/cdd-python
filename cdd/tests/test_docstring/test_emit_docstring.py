"""
Tests for `cdd.emit.docstring`
"""

import ast
from ast import Expr, FunctionDef, arguments
from copy import deepcopy
from functools import partial
from unittest import TestCase

import cdd.argparse_function.emit
import cdd.argparse_function.parse
import cdd.class_.emit
import cdd.class_.parse
import cdd.docstring.emit
import cdd.function.emit
import cdd.function.parse
import cdd.json_schema.emit
import cdd.shared.emit.file
import cdd.sqlalchemy.emit
from cdd.shared.pure_utils import omit_whitespace, reindent
from cdd.tests.mocks.classes import class_ast
from cdd.tests.mocks.docstrings import (
    docstring_google_str,
    docstring_google_tf_ops_losses__safe_mean_str,
    docstring_no_default_no_nl_str,
    docstring_no_nl_str,
    docstring_numpydoc_str,
)
from cdd.tests.mocks.ir import (
    function_google_tf_ops_losses__safe_mean_ir,
    intermediate_repr,
)
from cdd.tests.mocks.methods import function_google_tf_ops_losses__safe_mean_ast
from cdd.tests.utils_for_tests import unittest_main


class TestEmitDocstring(TestCase):
    """Tests emission"""

    def test_to_docstring(self) -> None:
        """
        Tests whether `docstring` produces indented `docstring_str` given `class_ast`
        """
        self.assertEqual(
            cdd.docstring.emit.docstring(
                cdd.class_.parse.class_(class_ast), emit_default_doc=True
            ),
            reindent(docstring_no_nl_str, 1),
        )

    def test_to_docstring_emit_default_doc_false(self) -> None:
        """
        Tests whether `docstring` produces `docstring_str` given `class_ast`
        """
        ir = cdd.class_.parse.class_(class_ast)
        self.assertEqual(
            cdd.docstring.emit.docstring(ir, emit_default_doc=False),
            reindent(docstring_no_default_no_nl_str, 1),
        )

    def test_to_numpy_docstring(self) -> None:
        """
        Tests whether `docstring` produces `docstring_numpydoc_str` when `docstring_format` is 'numpydoc'
        """
        self.assertEqual(
            docstring_numpydoc_str,
            cdd.docstring.emit.docstring(
                deepcopy(intermediate_repr), docstring_format="numpydoc"
            ),
        )

    def test_to_google_docstring(self) -> None:
        """
        Tests whether `docstring` produces `docstring_google_str` when `docstring_format` is 'google'
        """
        self.assertEqual(
            docstring_google_str,
            cdd.docstring.emit.docstring(
                deepcopy(intermediate_repr), docstring_format="google"
            ),
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
                    cdd.docstring.emit.docstring(
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
                        cdd.docstring.emit.docstring(
                            cdd.function.parse.function(
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


unittest_main()

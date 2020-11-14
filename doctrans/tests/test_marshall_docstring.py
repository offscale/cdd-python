"""
Tests for docstring parsing
"""
from copy import deepcopy
from unittest import TestCase

import doctrans.emit
import doctrans.emitter_utils
from doctrans.docstring_parsers import parse_docstring
from doctrans.tests.mocks.docstrings import (
    docstring_str,
    intermediate_repr,
    docstring_str_no_default_doc,
    intermediate_repr_no_default_doc_or_prop,
    intermediate_repr_no_default_doc,
    docstring_str_extra_colons,
    intermediate_repr_extra_colons,
    docstring_str_only_return_type,
    intermediate_repr_only_return_type,
)
from doctrans.tests.utils_for_tests import unittest_main


class TestMarshallDocstring(TestCase):
    """
    Tests whether docstrings are parsed out—and emitted—correctly
    """

    def test_ir_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `intermediate_repr`
              from `docstring_str`"""
        self.assertDictEqual(
            parse_docstring(docstring_str), intermediate_repr_no_default_doc
        )

    def test_intermediate_repr_no_default_doc_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_str_no_default_doc`
              from `docstring_str_no_default_doc`"""

        self.assertDictEqual(
            parse_docstring(docstring_str_no_default_doc, emit_default_doc=False),
            intermediate_repr_no_default_doc_or_prop,
        )

    def test_intermediate_repr_extra_colons_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_str_no_default_doc`
              from `docstring_str_no_default_doc`"""

        self.assertDictEqual(
            parse_docstring(docstring_str_extra_colons, emit_default_doc=False),
            intermediate_repr_extra_colons,
        )

    def test_intermediate_repr_only_return_type_equality(self) -> None:
        """
        Tests whether `parse_docstring` produces `docstring_str_no_default_doc`
              from `docstring_str_no_default_doc`"""

        self.assertDictEqual(
            parse_docstring(docstring_str_only_return_type, emit_default_doc=False),
            intermediate_repr_only_return_type,
        )

    def test_ir2docstring(self) -> None:
        """ Tests whether `to_docstring` produces `docstring_str` from `intermediate_repr` """
        self.assertEqual(
            doctrans.emitter_utils.to_docstring(
                deepcopy(intermediate_repr),
                indent_level=0,
                emit_types=True,
                emit_default_doc=True,
                emit_separating_tab=False,
            ).strip(),
            docstring_str.strip(),
        )


unittest_main()

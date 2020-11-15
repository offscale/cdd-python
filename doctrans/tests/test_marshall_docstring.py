"""
Tests for docstring parsing
"""

from copy import deepcopy
from unittest import TestCase

from docstring_parser import rest

import doctrans.emit
import doctrans.emitter_utils
from doctrans import parse
from doctrans.docstring_parsers import parse_docstring
from doctrans.emitter_utils import to_docstring
from doctrans.tests.mocks.docstrings import (
    docstring_str,
    intermediate_repr_no_default_doc,
    docstring_numpydoc_str,
    docstring_numpydoc_only_params_str,
    docstring_numpydoc_only_returns_str,
    docstring_numpydoc_only_doc_str,
    docstring_google_str,
)
from doctrans.tests.mocks.docstrings import (
    intermediate_repr,
    docstring_str_no_default_doc,
    intermediate_repr_no_default_doc_or_prop,
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

    def test_from_docstring(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_str`"""
        ir, returns = parse.docstring(docstring_str, return_tuple=True)
        self.assertTrue(returns)
        self.assertDictEqual(ir, intermediate_repr_no_default_doc)

    def test_from_docstring_numpydoc(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_numpydoc_str`"""
        ir, returns = parse.docstring(docstring_numpydoc_str, return_tuple=True)
        self.assertTrue(returns)
        self.assertDictEqual(ir, intermediate_repr_no_default_doc)

    def test_from_docstring_numpydoc_only_params(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_numpydoc_only_params_str`"""
        ir, returns = parse.docstring(
            docstring_numpydoc_only_params_str, return_tuple=True
        )
        self.assertFalse(returns)
        gold = deepcopy(intermediate_repr_no_default_doc)
        del gold["returns"]
        gold.update({"doc": "", "returns": None})
        self.assertDictEqual(ir, gold)

    maxDiff = None

    def test_from_docstring_numpydoc_only_returns(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_numpydoc_only_returns_str`"""
        ir, returns = parse.docstring(
            docstring_numpydoc_only_returns_str, return_tuple=True
        )
        self.assertTrue(returns)
        self.assertDictEqual(
            ir,
            {
                "doc": "",
                "name": None,
                "params": [],
                "returns": intermediate_repr_no_default_doc["returns"],
                "type": "static",
            },
        )

    def test_from_docstring_numpydoc_only_doc_str(self) -> None:
        """
        Tests whether `docstring` produces `intermediate_repr_no_default_doc`
              from `docstring_numpydoc_only_doc_str`"""
        ir, returns = parse.docstring(
            docstring_numpydoc_only_doc_str, return_tuple=True
        )
        self.assertFalse(returns)
        self.assertDictEqual(
            ir,
            {
                "doc": intermediate_repr_no_default_doc["doc"],
                "name": None,
                "params": [],
                "returns": None,
                "type": "static",
            },
        )

    def test_from_docstring_google_fails(self) -> None:
        """
        Tests for coverage. TODO: Actually implement google docstrings"""
        self.assertDictEqual(
            parse_docstring(docstring_google_str),
            intermediate_repr_no_default_doc
        )

    def test_to_docstring_fails(self) -> None:
        """
        Tests docstring failure conditions
        """
        self.assertRaises(
            NotImplementedError,
            lambda: to_docstring(
                intermediate_repr_no_default_doc, docstring_format="numpy"
            ),
        )

    def test_from_docstring_parser(self) -> None:
        """
        Tests if it can convert from the 3rd-party libraries format to this one
        """
        self.assertDictEqual(
            parse.docstring_parser(
                rest.parse(
                    "[Summary]\n\n"
                    ":param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]\n"
                    ":type [ParamName]: [ParamType](, optional)\n\n"
                    ":raises [ErrorType]: [ErrorDescription]\n\n"
                    ":return: [ReturnDescription]\n"
                    ":rtype: [ReturnType]\n"
                )
            ),
            {
                "params": [
                    {
                        "default": "[DefaultParamVal]",
                        "doc": "[ParamDescription]",
                        "name": "[ParamName]",
                    }
                ],
                "raises": [
                    {
                        "doc": "[ErrorDescription]",
                        "name": "raises",
                        "typ": "[ErrorType]",
                    }
                ],
                "returns": {"doc": "[ReturnDescription]", "name": "return_type"},
                "doc": "[Summary]",
            },
        )


unittest_main()

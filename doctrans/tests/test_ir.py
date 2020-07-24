"""
Tests for the Intermediate Representation
"""

from unittest import TestCase, main as unittest_main

from doctrans.docstring_structure_utils import class_def2docstring_structure, argparse_ast2docstring_structure, \
    docstring2docstring_structure, class_with_method2docstring_structure
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_structure, docstring_str, docstring_structure_no_default_doc
from doctrans.tests.mocks.methods import class_with_method_ast


class TestIntermediateRepresentation(TestCase):
    """
    Tests whether the intermediate representation is consistent when parsed from different inputs.

    IR is a dictionary of form:
              {
                  'short_description': ...,
                  'long_description': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    """

    def test_argparse_ast2docstring_structure(self) -> None:
        """
        Tests whether `argparse_ast2docstring_structure` produces `docstring_structure_no_default_doc`
              from `argparse_func_ast` """
        self.assertDictEqual(argparse_ast2docstring_structure(argparse_func_ast, with_default_doc=False),
                             docstring_structure_no_default_doc)

    def test_class_ast2docstring_structure(self) -> None:
        """
        Tests whether `class_def2docstring_structure` produces `docstring_structure`
              from `class_ast` """
        self.assertDictEqual(class_def2docstring_structure(class_ast),
                             docstring_structure)

    def test_class_with_method2docstring_structure(self) -> None:
        """
        Tests whether `class_with_method2docstring_structure` produces `docstring_structure`
              from `class_with_method_ast` """
        self.assertDictEqual(
            class_with_method2docstring_structure(class_with_method_ast, 'method_name',
                                                  with_default_doc=False),
            docstring_structure_no_default_doc
        )

    # Commented out as `ast.parse` isn't extracting the return type in the `def f() -> bool` form.
    '''
    def test_class_with_method2docstring_structure_inline_types(self) -> None:
        """
        Tests whether `class_with_method2docstring_structure` produces `docstring_structure`
              from `class_with_method_types_ast` """
        self.assertDictEqual(
            class_with_method2docstring_structure(class_with_method_types_ast, 'method_name',
                                                  with_default_doc=False),
            docstring_structure
        )
    '''

    def test_docstring2docstring_structure(self) -> None:
        """
        Tests whether `docstring2docstring_structure` produces `docstring_structure`
              from `docstring_str` """
        self.assertDictEqual(docstring2docstring_structure(docstring_str)[0],
                             docstring_structure)


if __name__ == '__main__':
    unittest_main()

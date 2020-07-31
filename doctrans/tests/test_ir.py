"""
Tests for the Intermediate Representation
"""
from ast import FunctionDef
from unittest import TestCase

from docstring_parser.rest import parse

from doctrans import docstring_struct, transformers
from doctrans.docstring_struct import to_docstring
from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_structure, docstring_str, \
    docstring_structure_no_default_doc
from doctrans.tests.mocks.methods import class_with_method_ast, class_with_method_types_ast
from doctrans.tests.utils_for_tests import unittest_main


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

    maxDiff = 55555

    def test_from_argparse_ast(self) -> None:
        """
        Tests whether `from_argparse_ast` produces `docstring_structure_no_default_doc_or_prop`
              from `argparse_func_ast` """
        self.assertDictEqual(docstring_struct.from_argparse_ast(argparse_func_ast),
                             docstring_structure)

    def test_from_argparse_ast_empty(self) -> None:
        """
        Tests `from_argparse_ast` empty condition
        """
        self.assertEqual(
            transformers.to_code(transformers.to_argparse(
                docstring_struct.from_argparse_ast(FunctionDef(body=[])),
                emit_default_doc=True
            )).rstrip('\n'),
            'def set_cli_args(argument_parser):\n    argument_parser.description = \'\''
        )

    def test_from_class(self) -> None:
        """
        Tests whether `from_class` produces `docstring_structure_no_default_doc`
              from `class_ast`
        """
        self.assertDictEqual(docstring_struct.from_class(class_ast),
                             docstring_structure_no_default_doc)

    def test_from_class_with_method(self) -> None:
        """
        Tests whether `from_class_with_method` produces `docstring_structure_no_default_doc`
              from `class_with_method_ast` """
        self.assertDictEqual(
            docstring_struct.from_class_with_method(class_with_method_ast, 'method_name'),
            docstring_structure_no_default_doc
        )

    def test_class_with_method2docstring_structure_inline_types(self) -> None:
        """
        Tests whether `from_class_with_method` produces `docstring_structure_no_default_doc`
              from `class_with_method_types_ast` """
        self.assertDictEqual(
            docstring_struct.from_class_with_method(class_with_method_types_ast, 'method_name'),
            docstring_structure_no_default_doc
        )

    def test_from_docstring(self) -> None:
        """
        Tests whether `from_docstring` produces `docstring_structure_no_default_doc`
              from `docstring_str` """
        _docstring_structure, returns = docstring_struct.from_docstring(docstring_str, return_tuple=True)
        self.assertTrue(returns)
        self.assertDictEqual(_docstring_structure,
                             docstring_structure_no_default_doc)

    def test_to_docstring_fails(self) -> None:
        """
        Tests to_docstring failure conditions
        """
        self.assertRaises(NotImplementedError,
                          lambda: to_docstring(docstring_str,
                                               docstring_format='numpy'))

    def test_from_docstring_parser(self) -> None:
        """
        Tests if it can convert from the 3rd-party libraries format to this one
        """
        self.assertDictEqual(docstring_struct.from_docstring_parser(
            parse('[Summary]\n\n'
                  ':param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]\n'
                  ':type [ParamName]: [ParamType](, optional)\n\n'
                  ':raises [ErrorType]: [ErrorDescription]\n\n'
                  ':return: [ReturnDescription]\n'
                  ':rtype: [ReturnType]\n')
        ),
            {'params': [{'default': '[DefaultParamVal]',
                         'doc': '[ParamDescription]',
                         'name': '[ParamName]'}],
             'raises': [{'doc': '[ErrorDescription]',
                         'name': 'raises',
                         'typ': '[ErrorType]'}],
             'returns': {'doc': '[ReturnDescription]', 'name': 'return_type'},
             'short_description': '[Summary]'}
        )


unittest_main()

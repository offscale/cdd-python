from unittest import TestCase, main as unittest_main

from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_structure, docstring_str
from doctrans.utils import class_ast2docstring_structure, argparse_ast2docstring_structure, \
    docstring2docstring_structure


class TestIntermediateRepresentation(TestCase):
    def test_argparse_ast2docstring_structure(self):
        self.assertDictEqual(argparse_ast2docstring_structure(argparse_func_ast), docstring_structure)

    def test_class_ast2docstring_structure(self):
        self.assertDictEqual(class_ast2docstring_structure(class_ast), docstring_structure)

    def test_docstring2docstring_structure(self):
        self.assertDictEqual(docstring2docstring_structure(docstring_str)[0],
                             docstring_structure)


if __name__ == '__main__':
    unittest_main()

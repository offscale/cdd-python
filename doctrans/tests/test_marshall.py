from unittest import TestCase, main as unittest_main

from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_str
from doctrans.tests.utils_for_tests import run_ast_test
from doctrans.transformers import docstring2class_def, class_def2docstring, ast2argparse, argparse2class


class TestMarshall(TestCase):
    """ Tests whether conversion between formats works """

    def test_argparse2class(self) -> None:
        """
        Tests whether `argparse2class` produces `class_ast` given `argparse_func_ast`
        """
        run_ast_test(self, argparse2class(argparse_func_ast), gold=class_ast)

    def test_ast2argparse(self) -> None:
        """
        Tests whether `ast2argparse` produces `argparse_func_ast` given `class_ast`
        """
        run_ast_test(self, ast2argparse(class_ast, with_default_doc=False), gold=argparse_func_ast)

    def test_ast2docstring(self) -> None:
        """
        Tests whether `class_def2docstring` produces `docstring_str` given `class_ast`
        """
        self.assertEqual(docstring_str, class_def2docstring(class_ast))

    def test_docstring2class_def(self):
        """
        Tests whether `docstring2class_def` produces `class_ast` given `docstring_str`
        """
        # ast2file(gen_ast, os.path.join(os.path.dirname(__file__), 'delme.py'), skip_black=False)
        run_ast_test(self, docstring2class_def(docstring_str), gold=class_ast)


if __name__ == '__main__':
    unittest_main()

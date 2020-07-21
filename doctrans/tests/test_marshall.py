import os
from unittest import TestCase, main as unittest_main

from doctrans.tests.mocks.argparse import argparse_func_ast
from doctrans.tests.mocks.classes import class_ast
from doctrans.tests.mocks.docstrings import docstring_str
from doctrans.tests.utils_for_tests import run_ast_test
from doctrans.transformers import docstring2ast, ast2docstring, ast2argparse, argparse2class, ast2file


class TestMarshall(TestCase):
    maxDiff = 21164

    def test_argparse2class(self) -> None:
        gen_ast = argparse2class(argparse_func_ast)
        run_ast_test(self, gen_ast, gold=class_ast)

    def test_ast2argparse(self) -> None:
        gen_ast = ast2argparse(class_ast, with_default_doc=False)
        run_ast_test(self, gen_ast, gold=argparse_func_ast)

    def test_ast2docstring(self) -> None:
        self.assertEqual(docstring_str, ast2docstring(class_ast))

    def test_docstring2ast(self):
        gen_ast = docstring2ast(docstring_str)
        ast2file(gen_ast, os.path.join(os.path.dirname(__file__), 'delme.py'), skip_black=False)
        run_ast_test(self, gen_ast, gold=class_ast)


if __name__ == '__main__':
    unittest_main()

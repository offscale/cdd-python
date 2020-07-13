from unittest import TestCase, main as unittest_main

from meta.asttools import cmp_ast

from docstring2class.tests.mocks import docstring0, ast_def
from docstring2class.transformers import docstring2ast, ast2file


class TestParseDocstring(TestCase):
    def test_docstring2ast(self):
        gen_ast = docstring2ast(docstring0)
        ast2file(gen_ast, 'delme.py')
        self.assertTrue(cmp_ast(gen_ast, ast_def), 'Generated AST doesn\'t match reference AST')

    # def test_class2docstring(self) -> None: self.assertFalse(True)


if __name__ == '__main__':
    unittest_main()

from sys import version
from unittest import TestCase, main as unittest_main

from meta.asttools import cmp_ast, str_ast

from docstring2class.tests.mocks import docstring0, ast_def
from docstring2class.transformers import docstring2ast, ast2docstring


class TestParseDocstring(TestCase):
    def test_docstring2ast(self):
        gen_ast = docstring2ast(docstring0)
        # ast2file(gen_ast, 'delme.py', skip_black=False)

        self.assertTupleEqual(*tuple(map(lambda ast: tuple(str_ast(ast).split('\n')),
                                         (gen_ast, ast_def.body[0]))))
        if version[:3] == '3.8':
            self.assertTrue(cmp_ast(gen_ast, ast_def.body[0]), 'Generated AST doesn\'t match reference AST')
        else:
            pass
            # self.assertTrue(cmp_ast(gen_ast, parse(cls).body[0]), 'Generated AST doesn\'t match reference AST')

    def test_ast2docstring(self) -> None:
        self.assertEqual(docstring0, ast2docstring(ast_def))


if __name__ == '__main__':
    unittest_main()

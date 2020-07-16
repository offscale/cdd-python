from sys import version
from unittest import TestCase, main as unittest_main

from meta.asttools import cmp_ast, str_ast

from doctrans.tests.mocks import docstring0, ast_def, set_cli_func_ast
from doctrans.transformers import docstring2ast, ast2docstring, ast2argparse, ast2file


class TestParseDocstring(TestCase):
    maxDiff = 19825

    def test_docstring2ast(self):
        gen_ast = docstring2ast(docstring0)
        ast2file(gen_ast, 'delme.py', skip_black=False)
        self.run_ast_test(gen_ast)

    def test_ast2docstring(self) -> None:
        self.assertEqual(docstring0, ast2docstring(ast_def))

    def test_ast2argparse(self) -> None:
        gen_ast = ast2argparse(ast_def)
        self.run_ast_test(gen_ast)
        # ast2file(ast2argparse(ast_def), 'delme.py', skip_black=False)

    def run_ast_test(self, gen_ast):
        self.assertTupleEqual(*tuple(map(lambda ast: tuple(str_ast(ast).split('\n')),
                                         (gen_ast, set_cli_func_ast.body[0]))))
        if version[:3] == '3.8':
            self.assertTrue(cmp_ast(gen_ast, ast_def.body[0]), 'Generated AST doesn\'t match reference AST')


if __name__ == '__main__':
    unittest_main()

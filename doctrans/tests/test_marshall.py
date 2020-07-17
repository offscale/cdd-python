from sys import version
from unittest import TestCase, main as unittest_main

from meta.asttools import cmp_ast, str_ast

from doctrans.tests.mocks import docstring_str, class_ast, argparse_func_ast
from doctrans.transformers import docstring2ast, ast2docstring, ast2argparse


class TestParseDocstring(TestCase):
    def test_docstring2ast(self):
        gen_ast = docstring2ast(docstring_str)
        self.run_ast_test(gen_ast, gold=class_ast)

    def test_ast2docstring(self) -> None:
        self.assertEqual(docstring_str, ast2docstring(class_ast))

    def test_ast2argparse(self) -> None:
        gen_ast = ast2argparse(class_ast)
        # ast2file(gen_ast, path.join(path.dirname(__file__), 'delme.py'), skip_black=False)
        self.run_ast_test(gen_ast, gold=argparse_func_ast)

    def run_ast_test(self, gen_ast, gold):
        self.assertTupleEqual(*tuple(map(lambda ast: tuple(str_ast(ast).split('\n')),
                                         (gen_ast, gold))))
        if version[:3] == '3.8':
            self.assertTrue(cmp_ast(gen_ast, gold), 'Generated AST doesn\'t match reference AST')


if __name__ == '__main__':
    unittest_main()

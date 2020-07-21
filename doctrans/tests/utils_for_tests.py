from ast import parse
from sys import version

from meta.asttools import cmp_ast, str_ast


def run_ast_test(test_case_instance, gen_ast, gold):
    if isinstance(gen_ast, str):
        gen_ast = parse(gen_ast, mode='exec').body[0]

    test_case_instance.assertTupleEqual(*tuple(map(lambda ast: tuple(str_ast(ast).split('\n')),
                                                   (gen_ast, gold))))
    if version[:3] == '3.8':
        test_case_instance.assertTrue(cmp_ast(gen_ast, gold), 'Generated AST doesn\'t match reference AST')

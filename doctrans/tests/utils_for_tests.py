"""
Shared utility function used by many tests
"""

from ast import parse
from sys import version

from meta.asttools import cmp_ast, str_ast


def run_ast_test(test_case_instance, gen_ast, gold):
    """
    Compares `gen_ast` with `gold` standard

    :param test_case_instance: instance of `TestCase`
    :type test_case_instance: ```unittest.TestCase```

    :param gen_ast: generated AST
    :type gen_ast: Union[ast.Module, ast.ClassDef, ast.FunctionDef]

    :param gold: mocked AST
    :type gold: Union[ast.Module, ast.ClassDef, ast.FunctionDef]
    """
    if isinstance(gen_ast, str):
        gen_ast = parse(gen_ast, mode='exec').body[0]

    test_case_instance.assertTupleEqual(*tuple(map(lambda ast: tuple(str_ast(ast).split('\n')),
                                                   (gen_ast, gold))))
    if version[:3] == '3.8':
        test_case_instance.assertTrue(cmp_ast(gen_ast, gold), 'Generated AST doesn\'t match reference AST')

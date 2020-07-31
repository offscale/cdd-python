"""
Shared utility function used by many tests
"""

from ast import parse
from unittest import main

from meta.asttools import cmp_ast

import doctrans.source_transformer


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
        gen_ast = parse(gen_ast, mode="exec").body[0]

    """
    if hasattr(gen_ast, 'body') and len(gen_ast.body) > 0 and hasattr(gen_ast.body, 'value'):
        test_case_instance.assertEqual(get_docstring(gen_ast),
                                       get_docstring(gold))
    """

    test_case_instance.assertEqual(
        *map(doctrans.source_transformer.to_code, (gen_ast, gold))
    )
    # if PY3_8:
    test_case_instance.assertTrue(
        cmp_ast(gen_ast, gold), "Generated AST doesn't match reference AST"
    )


def unittest_main():
    """ Runs unittest.main if __main__ """
    if __name__ == "__main__":
        main()

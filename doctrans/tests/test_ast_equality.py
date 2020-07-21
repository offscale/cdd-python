from unittest import TestCase, main as unittest_main

from doctrans.tests.mocks.argparse import argparse_func_str, argparse_func_ast
from doctrans.tests.mocks.classes import class_str, class_ast
from doctrans.tests.utils_for_tests import run_ast_test


class TestAstEquality(TestCase):
    def test_argparse_func(self) -> None:
        run_ast_test(self, argparse_func_str, argparse_func_ast)

    def test_class(self) -> None:
        run_ast_test(self, class_str, class_ast)


if __name__ == '__main__':
    unittest_main()

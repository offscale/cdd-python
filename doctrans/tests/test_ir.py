from unittest import TestCase, main as unittest_main

from doctrans.tests.mocks import class_ast, docstring_structure, argparse_func_ast
from doctrans.utils import class_ast2docstring_structure, argparse_ast2docstring_structure


class TestIntermediateRepresentation(TestCase):
    maxDiff = 2497

    def test_class_ast2docstring_structure(self):
        self.assertDictEqual(class_ast2docstring_structure(class_ast), docstring_structure)

    def test_argparse_ast2docstring_structure(self):
        self.assertDictEqual(argparse_ast2docstring_structure(argparse_func_ast), docstring_structure)


if __name__ == '__main__':
    unittest_main()

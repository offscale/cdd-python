""" Tests for doctrans_utils """
from unittest import TestCase

from cdd.doctrans_utils import DocTrans, has_inline_types
from cdd.source_transformer import ast_parse, to_code
from cdd.tests.utils_for_tests import run_ast_test, unittest_main


class TestDocTransUtils(TestCase):
    """ Test class for doctrans_utils.py """

    def test_has_inline_types(self) -> None:
        """ Tests has_inline_types """

        self.assertTrue(has_inline_types(ast_parse("a: int = 5")))
        self.assertFalse(has_inline_types(ast_parse("a = 5")))
        self.assertTrue(has_inline_types(ast_parse("def a() -> None: pass")))
        self.assertFalse(has_inline_types(ast_parse("def a(): pass")))

    def test_doctrans(self) -> None:
        """ Tests `DocTrans` """

        original_node = ast_parse(
            "\n\t".join(
                (
                    "def sum(a: int, b: int) -> int:",
                    "res: int = a + b",
                    "return res",
                )
            )
        )
        gold_ast = ast_parse(
            "\n\t".join(
                (
                    "def sum(a, b):",
                    '"""',
                    ":param a:",
                    ":type a: ```int```",
                    "",
                    ":param b:",
                    ":type b: ```int```",
                    "",
                    ":returns:",
                    ":rtype: ```int```" '"""',
                    "res: int = a + b",
                    "return res",
                )
            )
        )
        doc_trans = DocTrans(
            docstring_format="rest", inline_types=False, existing_inline_types=True
        )
        gen_ast = doc_trans.visit(original_node)
        self.assertEqual(*map(to_code, (gen_ast, gold_ast)))
        run_ast_test(self, gen_ast, gold_ast)


unittest_main()

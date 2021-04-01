""" Tests for doctrans_utils """
from ast import (
    Add,
    AnnAssign,
    Assign,
    BinOp,
    Expr,
    FunctionDef,
    Load,
    Module,
    Name,
    Return,
    Store,
    arguments,
)
from unittest import TestCase

from cdd.ast_utils import annotate_ancestry, set_arg, set_value
from cdd.doctrans_utils import DocTrans, has_inline_types
from cdd.pure_utils import tab
from cdd.source_transformer import ast_parse
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

        original_node = Module(
            body=[
                FunctionDef(
                    name="sum",
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            set_arg(arg="a", annotation=Name("int", Load())),
                            set_arg(arg="b", annotation=Name("int", Load())),
                        ],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        vararg=None,
                        kwarg=None,
                    ),
                    body=[
                        AnnAssign(
                            target=Name("res", Store()),
                            annotation=Name("int", Load()),
                            value=BinOp(
                                left=Name("a", Load()),
                                op=Add(),
                                right=Name("b", Load()),
                            ),
                            simple=1,
                        ),
                        Return(value=Name("res", Load())),
                    ],
                    decorator_list=[],
                    lineno=None,
                    returns=Name("int", Load()),
                )
            ],
            type_ignores=[],
        )
        annotate_ancestry(original_node)
        doc_trans = DocTrans(
            docstring_format="rest",
            inline_types=False,
            existing_inline_types=True,
            whole_ast=original_node,
        )

        gen_ast = doc_trans.visit(original_node)

        gold_ast = Module(
            body=[
                FunctionDef(
                    name="sum",
                    args=arguments(
                        posonlyargs=[],
                        args=list(map(set_arg, ("a", "b"))),
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        vararg=None,
                        kwarg=None,
                    ),
                    body=[
                        Expr(
                            value=set_value(
                                "\n{tab}".format(tab=tab)
                                + "\n{tab}".format(tab=tab).join(
                                    (
                                        ":type a: ```int```",
                                        "",
                                        ":type b: ```int```",
                                        "",
                                        ":rtype: ```int```",
                                        "",
                                    )
                                )
                            )
                        ),
                        Assign(
                            targets=[Name("res", Store())],
                            value=BinOp(
                                left=Name("a", Load()),
                                op=Add(),
                                right=Name("b", Load()),
                            ),
                            type_comment=Name("int", Load()),
                            lineno=None,
                        ),
                        Return(value=Name("res", Load())),
                    ],
                    decorator_list=[],
                    lineno=None,
                    returns=None,
                )
            ],
            type_ignores=[],
        )

        run_ast_test(self, gen_ast, gold_ast)


unittest_main()

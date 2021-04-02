"""
Mocks with inline types vs types in docstrings
"""
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
from copy import deepcopy

from cdd.ast_utils import set_arg, set_value
from cdd.pure_utils import tab

assign_with_type_comment = Assign(
    targets=[Name("res", Store())],
    value=BinOp(
        left=Name("a", Load()),
        op=Add(),
        right=Name("b", Load()),
    ),
    type_comment=Name("int", Load()),
    lineno=None,
)
ann_assign_with_annotation = AnnAssign(
    annotation=assign_with_type_comment.type_comment,
    value=assign_with_type_comment.value,
    simple=1,
    target=assign_with_type_comment.targets[0],
    type_comment=None,
    expr=None,
    expr_target=None,
    expr_annotation=None,
    lineno=None,
)

function_type_annotated = Module(
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
                ann_assign_with_annotation,
                Return(value=Name("res", Load())),
            ],
            decorator_list=[],
            lineno=None,
            returns=Name("int", Load()),
        )
    ],
    type_ignores=[],
)

function_type_in_docstring = Module(
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
                assign_with_type_comment,
                Return(value=Name("res", Load())),
            ],
            decorator_list=[],
            lineno=None,
            returns=None,
        )
    ],
    type_ignores=[],
)
function_type_in_docstring_only = deepcopy(function_type_in_docstring)
function_type_in_docstring_only.body[0].body[1].type_comment = None

__all__ = [
    "ann_assign_with_annotation",
    "assign_with_type_comment",
    "function_type_annotated",
    "function_type_in_docstring",
    "function_type_in_docstring_only",
]

"""
Mocks with inline types vs types in docstrings
"""

from ast import (
    Add,
    AnnAssign,
    Assign,
    BinOp,
    ClassDef,
    Expr,
    FunctionDef,
    Load,
    Name,
    Return,
    Store,
    arguments,
)
from copy import deepcopy

from cdd.ast_utils import maybe_type_comment, set_arg, set_value
from cdd.pure_utils import tab
from cdd.tests.mocks.docstrings import docstring_sum_tuple
from cdd.tests.utils_for_tests import reindent_docstring, replace_docstring

_class_doc_str_expr = Expr(
    set_value(
        tab.join(
            (
                "\n",
                "Class mock",
                "\n\n",
                ":cvar a: One swell num",
                "\n\n",
                ":cvar b: Unlucky num",
                "\n",
            )
        )
        + tab
    )
)

_assign_type = Name("int", Load())
assign_with_type_comment = Assign(
    targets=[Name("res", Store())],
    value=BinOp(
        left=Name("a", Load()),
        op=Add(),
        right=Name("b", Load()),
    ),
    type_comment=_assign_type.id,
    lineno=None,
)
ann_assign_with_annotation = AnnAssign(
    annotation=_assign_type,
    value=assign_with_type_comment.value,
    simple=1,
    target=assign_with_type_comment.targets[0],
    expr=None,
    expr_target=None,
    expr_annotation=None,
    lineno=None,
    **maybe_type_comment
)

function_type_annotated = FunctionDef(
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

function_type_in_docstring = FunctionDef(
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
                "\n{tab}{body}".format(
                    tab=tab,
                    body="\n{tab}".format(tab=tab).join(docstring_sum_tuple),
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
function_type_in_docstring_only = deepcopy(function_type_in_docstring)
function_type_in_docstring_only.body[1].type_comment = None

class_with_internal_annotated = ClassDef(
    name="ClassMock",
    bases=tuple(),
    keywords=tuple(),
    decorator_list=[],
    body=[
        _class_doc_str_expr,
        AnnAssign(
            annotation=Name(
                "int",
                Load(),
            ),
            simple=1,
            target=Name("a", Store()),
            value=set_value(5),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
        ),
        AnnAssign(
            annotation=Name(
                "float",
                Load(),
            ),
            simple=1,
            target=Name("b", Store()),
            value=set_value(0.0),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
        ),
        reindent_docstring(function_type_annotated, indent_level=3, smart=False),
    ],
    expr=None,
    identifier_name=None,
)

class_with_internal_type_commented_and_docstring_typed = ClassDef(
    name="ClassMock",
    bases=tuple(),
    keywords=tuple(),
    decorator_list=[],
    body=[
        _class_doc_str_expr,
        Assign(
            targets=[Name("a", Store())],
            value=set_value(5),
            type_comment=Name(
                "int",
                Load(),
            ).id,
            expr=None,
            lineno=None,
        ),
        Assign(
            targets=[Name("b", Store())],
            value=set_value(0.0),
            type_comment=Name(
                "float",
                Load(),
            ).id,
            expr=None,
            lineno=None,
        ),
        replace_docstring(
            deepcopy(function_type_in_docstring),
            "\n{sep}{body}".format(
                sep=tab * 2,
                body="\n{sep}".format(sep=tab * 2).join(docstring_sum_tuple),
            ),
        ),
    ],
    expr=None,
    identifier_name=None,
)

__all__ = [
    "ann_assign_with_annotation",
    "assign_with_type_comment",
    "class_with_internal_annotated",
    "class_with_internal_type_commented_and_docstring_typed",
    "function_type_annotated",
    "function_type_in_docstring",
    "function_type_in_docstring_only",
]

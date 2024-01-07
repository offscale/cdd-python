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

from cdd.shared.ast_utils import maybe_type_comment, set_arg, set_value
from cdd.shared.pure_utils import tab
from cdd.tests.mocks.docstrings import docstring_sum_tuple
from cdd.tests.utils_for_tests import reindent_docstring, replace_docstring

_class_doc_str_expr: Expr = Expr(
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
    ),
    lineno=None,
    col_offset=None,
)

_assign_type: Name = Name("int", Load(), lineno=None, col_offset=None)
assign_with_type_comment: Assign = Assign(
    targets=[Name("res", Store(), lineno=None, col_offset=None)],
    value=BinOp(
        left=Name("a", Load(), lineno=None, col_offset=None),
        op=Add(),
        right=Name("b", Load(), lineno=None, col_offset=None),
    ),
    type_comment=_assign_type.id,
    lineno=None,
)
ann_assign_with_annotation: AnnAssign = AnnAssign(
    annotation=_assign_type,
    value=assign_with_type_comment.value,
    simple=1,
    target=assign_with_type_comment.targets[0],
    expr=None,
    expr_target=None,
    expr_annotation=None,
    lineno=None,
    col_offset=None,
    **maybe_type_comment
)

function_type_annotated: FunctionDef = FunctionDef(
    name="sum",
    args=arguments(
        posonlyargs=[],
        args=[
            set_arg(
                arg="a", annotation=Name("int", Load(), lineno=None, col_offset=None)
            ),
            set_arg(
                arg="b", annotation=Name("int", Load(), lineno=None, col_offset=None)
            ),
        ],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None,
    ),
    body=[
        ann_assign_with_annotation,
        Return(value=Name("res", Load(), lineno=None, col_offset=None)),
    ],
    decorator_list=[],
    type_params=[],
    lineno=None,
    returns=Name("int", Load(), lineno=None, col_offset=None),
)

function_type_in_docstring: FunctionDef = FunctionDef(
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
            ),
            lineno=None,
            col_offset=None,
        ),
        assign_with_type_comment,
        Return(value=Name("res", Load(), lineno=None, col_offset=None)),
    ],
    decorator_list=[],
    type_params=[],
    lineno=None,
    returns=None,
)
function_type_in_docstring_only: FunctionDef = deepcopy(function_type_in_docstring)
function_type_in_docstring_only.body[1].type_comment = None

class_with_internal_annotated: ClassDef = ClassDef(
    name="ClassMock",
    bases=tuple(),
    keywords=tuple(),
    decorator_list=[],
    type_params=[],
    body=[
        _class_doc_str_expr,
        AnnAssign(
            annotation=Name("int", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name("a", Store(), lineno=None, col_offset=None),
            value=set_value(5),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            col_offset=None,
        ),
        AnnAssign(
            annotation=Name("float", Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name("b", Store(), lineno=None, col_offset=None),
            value=set_value(0.0),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            col_offset=None,
        ),
        reindent_docstring(function_type_annotated, indent_level=3, smart=False),
    ],
    expr=None,
    identifier_name=None,
    lineno=None,
    col_offset=None,
)

class_with_internal_type_commented_and_docstring_typed: ClassDef = ClassDef(
    name="ClassMock",
    bases=tuple(),
    keywords=tuple(),
    decorator_list=[],
    type_params=[],
    body=[
        _class_doc_str_expr,
        Assign(
            targets=[Name("a", Store(), lineno=None, col_offset=None)],
            value=set_value(5),
            type_comment=Name(
                "int",
                Load(),
            ).id,
            expr=None,
            lineno=None,
        ),
        Assign(
            targets=[Name("b", Store(), lineno=None, col_offset=None)],
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
    lineno=None,
    col_offset=None,
)

__all__ = [
    "ann_assign_with_annotation",
    "assign_with_type_comment",
    "class_with_internal_annotated",
    "class_with_internal_type_commented_and_docstring_typed",
    "function_type_annotated",
    "function_type_in_docstring",
    "function_type_in_docstring_only",
]  # type: list[str]

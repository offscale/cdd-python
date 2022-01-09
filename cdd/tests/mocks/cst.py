"""Mocks for cst"""

from cdd.cst_utils import (
    AnnAssignment,
    Assignment,
    AugAssignment,
    CallStatement,
    ClassDefinitionStart,
    CommentStatement,
    ElifStatement,
    ElseStatement,
    ExprStatement,
    FalseStatement,
    FromStatement,
    FunctionDefinitionStart,
    IfStatement,
    NoneStatement,
    PassStatement,
    ReturnStatement,
    TripleQuoted,
    TrueStatement,
    UnchangingLine,
)

cstify_cst = (
    CommentStatement(line_no_start=1, line_no_end=1, value="# pragma: no cover"),
    CommentStatement(line_no_start=1, line_no_end=2, value="\n# flake8: noqa"),
    CommentStatement(
        line_no_start=2,
        line_no_end=3,
        value="\n# === Copyright 2022 under CC0",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=False,
        line_no_start=3,
        line_no_end=4,
        value='\n"""Module docstring goes here"""',
    ),
    FromStatement(line_no_start=4, line_no_end=6, value="\n\nfrom operator import add"),
    ClassDefinitionStart(
        line_no_start=6, line_no_end=9, value="\n\n\nclass C(object):", name="C"
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        line_no_start=9,
        line_no_end=10,
        value='\n    """My cls"""',
    ),
    FunctionDefinitionStart(
        line_no_start=10,
        line_no_end=13,
        value="\n" "\n" "    @staticmethod\n" "    def add1(foo):",
        name="add1",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        line_no_start=13,
        line_no_end=20,
        value='\n        """\n'
        "        :param foo: a foo\n"
        "        :type foo: ```int```\n"
        "\n"
        "        :return: foo + 1\n"
        "        :rtype: ```int```\n"
        '        """',
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=False,
        line_no_start=20,
        line_no_end=22,
        value='\n\n        """foo"""',
    ),
    FunctionDefinitionStart(
        line_no_start=22, line_no_end=23, value="\n        def g():", name="g"
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        line_no_start=23,
        line_no_end=23,
        value=' """foo : bar ; can"""',
    ),
    PassStatement(line_no_start=23, line_no_end=23, value="; pass"),
    FunctionDefinitionStart(
        line_no_start=23, line_no_end=25, value="\n\n        def h():", name="h"
    ),
    CommentStatement(line_no_start=25, line_no_end=25, value=" # stuff"),
    PassStatement(line_no_start=25, line_no_end=26, value="\n            pass"),
    FunctionDefinitionStart(
        line_no_start=26,
        line_no_end=29,
        value="\n\n        def adder(a: int,\n                  b: int) -> int:",
        name="adder",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        line_no_start=29,
        line_no_end=36,
        value="\n"
        '            """\n'
        "            :param a: First arg\n"
        "\n"
        "            :param b: Second arg\n"
        "\n"
        "            :return: first + second arg\n"
        '            """',
    ),
    CommentStatement(
        line_no_start=36, line_no_end=37, value="\n            # fmt: off"
    ),
    AnnAssignment(
        line_no_start=37,
        line_no_end=40,
        value="\n            res: \\\n                int \\\n                = a + b",
    ),
    ReturnStatement(line_no_start=40, line_no_end=41, value="\n            return res"),
    Assignment(
        line_no_start=41,
        line_no_end=47,
        value="\n\n        r = (\n            add(foo, 1)\n            or\n            adder(foo, 1)\n        )",
    ),
    IfStatement(line_no_start=47, line_no_end=48, value="\n        if r:"),
    NoneStatement(line_no_start=48, line_no_end=49, value="\n            None"),
    ElifStatement(line_no_start=49, line_no_end=50, value="\n        elif r:"),
    TrueStatement(line_no_start=50, line_no_end=51, value="\n            True"),
    FalseStatement(line_no_start=51, line_no_end=52, value="\n            False"),
    CommentStatement(
        line_no_start=52, line_no_end=53, value="\n            # ([5,5] @ [5,5]) *\\"
    ),
    ExprStatement(
        line_no_start=53,
        line_no_end=55,
        value="\n            -5 / 7 ** 6 + \\\n            6.0 - 6e1 & 1+2.34j",
    ),
    AugAssignment(line_no_start=55, line_no_end=56, value="\n            r <<= 5"),
    CallStatement(line_no_start=56, line_no_end=57, value="\n            print(r)"),
    ElseStatement(line_no_start=57, line_no_end=58, value="\n        else:"),
    PassStatement(line_no_start=58, line_no_end=59, value="\n            pass"),
    CommentStatement(line_no_start=59, line_no_end=60, value="\n        # fmt: on"),
    CommentStatement(
        line_no_start=60,
        line_no_end=61,
        value="\n        # That^ incremented `foo` by 1",
    ),
    ReturnStatement(line_no_start=61, line_no_end=62, value="\n        return r"),
    CommentStatement(
        line_no_start=62,
        line_no_end=65,
        value="\n\n\n# from contextlib import ContextDecorator",
    ),
    CommentStatement(
        line_no_start=65, line_no_end=67, value="\n\n# with ContextDecorator():"
    ),
    CommentStatement(line_no_start=67, line_no_end=68, value="\n#    pass"),
    FunctionDefinitionStart(
        line_no_start=68, line_no_end=71, value="\n\n\ndef f():", name="f"
    ),
    ReturnStatement(line_no_start=71, line_no_end=72, value="\n    return 1"),
    UnchangingLine(line_no_start=72, line_no_end=73, value="\n"),
)


__all__ = ["cstify_cst"]

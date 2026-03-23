"""Mocks for cst"""

from cdd.shared.cst_utils import (
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
    UnchangingLine, CstTypes,
)

cstify_cst = (
    CommentStatement(line_no_start=1, line_no_end=2, value="# pragma: no cover\n"),
    CommentStatement(line_no_start=2, line_no_end=3, value="# flake8: noqa\n"),
    CommentStatement(line_no_start=3, line_no_end=4, value="# fmt: off\n"),
    CommentStatement(line_no_start=4, line_no_end=5, value="# === Copyright 2022 under CC0\n"),
    TripleQuoted(
        is_double_q=True,
        is_docstr=False,
        line_no_start=5,
        line_no_end=6,
        value='"""Module docstring goes here"""\n',
    ),
    UnchangingLine(line_no_start=6, line_no_end=7, value="\n"),
    FromStatement(line_no_start=7, line_no_end=8, value="from operator import add\n"),
    UnchangingLine(line_no_start=8, line_no_end=9, value="\n"),
    UnchangingLine(line_no_start=9, line_no_end=10, value="\n"),
    ClassDefinitionStart(
        line_no_start=10, line_no_end=11, value="class C(object):\n", name="C"
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        line_no_start=11,
        line_no_end=12,
        value='    """My cls"""\n',
    ),
    UnchangingLine(line_no_start=12, line_no_end=13, value="\n"),
    FunctionDefinitionStart(
        line_no_start=13,
        line_no_end=15,
        value="    @staticmethod\n    def add1(foo):\n",
        name="add1",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        line_no_start=15,
        line_no_end=22,
        value='        """\n'
        "        :param foo: a foo\n"
        "        :type foo: ```int```\n"
        "\n"
        "        :return: foo + 1\n"
        "        :rtype: ```int```\n"
        '        """\n',
    ),
    UnchangingLine(line_no_start=22, line_no_end=23, value="\n"),
    TripleQuoted(
        is_double_q=True,
        is_docstr=False,
        line_no_start=23,
        line_no_end=24,
        value='        """foo"""\n',
    ),
    FunctionDefinitionStart(
        line_no_start=24, line_no_end=25, value='        def g(): """foo : bar ; can"""; pass\n',
        name="g",
    ),
    UnchangingLine(line_no_start=25, line_no_end=26, value="\n"),
    FunctionDefinitionStart(
        line_no_start=26, line_no_end=26, value="        def h(): # stuff\n", name="h"
    ),
    PassStatement(line_no_start=27, line_no_end=28, value="            pass\n"),
    UnchangingLine(line_no_start=28, line_no_end=29, value="\n"),
    FunctionDefinitionStart(
        line_no_start=29,
        line_no_end=31,
        value="        def adder(a: int,\n                  b: int) -> int:\n",
        name="adder",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        line_no_start=31,
        line_no_end=38,
        value="            \"\"\"\n"
        "            :param a: First arg\n"
        "\n"
        "            :param b: Second arg\n"
        "\n"
        "            :return: first + second arg\n"
        '            """\n',
    ),
    CommentStatement(line_no_start=38, line_no_end=39, value="            # fmt: off\n"),
    AnnAssignment(
        line_no_start=39,
        line_no_end=42,
        value="            res: \\\n                int \\\n                = a + b\n",
    ),
    ReturnStatement(line_no_start=42, line_no_end=43, value="            return res\n"),
    UnchangingLine(line_no_start=43, line_no_end=44, value="\n"),
    Assignment(
        line_no_start=44,
        line_no_end=49,
        value="        r = (\n"
        "            add(foo, 1)\n"
        "            or\n"
        "            adder(foo, 1)\n"
        "        )\n",
    ),
    IfStatement(line_no_start=49, line_no_end=50, value="        if r:\n"),
    NoneStatement(line_no_start=50, line_no_end=51, value="            None\n"),
    ElifStatement(line_no_start=51, line_no_end=52, value="        elif r:\n"),
    TrueStatement(line_no_start=52, line_no_end=53, value="            True\n"),
    FalseStatement(line_no_start=53, line_no_end=54, value="            False\n"),
    CommentStatement(
        line_no_start=54,
        line_no_end=55,
        value="            # ([5,5] @ [5,5]) *\\\n",
    ),
    ExprStatement(
        line_no_start=55,
        line_no_end=57,
        value="            -5 / 7 ** 6 + \\\n            6.0 - 6e1 & 1+2.34j\n",
    ),
    AugAssignment(line_no_start=57, line_no_end=58, value="            r <<= 5\n"),
    CallStatement(line_no_start=58, line_no_end=59, value="            print(r)\n"),
    ElseStatement(line_no_start=59, line_no_end=60, value="        else:\n"),
    PassStatement(line_no_start=60, line_no_end=61, value="            pass\n"),
    CommentStatement(line_no_start=61, line_no_end=62, value="        # fmt: on\n"),
    CommentStatement(
        line_no_start=62,
        line_no_end=63,
        value="        # That^ incremented `foo` by 1\n",
    ),
    ReturnStatement(line_no_start=63, line_no_end=64, value="        return r\n"),
    UnchangingLine(line_no_start=64, line_no_end=65, value="\n"),
    UnchangingLine(line_no_start=65, line_no_end=66, value="\n"),
    CommentStatement(
        line_no_start=66,
        line_no_end=67,
        value="# from contextlib import ContextDecorator\n",
    ),
    UnchangingLine(line_no_start=67, line_no_end=68, value="\n"),
    CommentStatement(
        line_no_start=68, line_no_end=69, value="# with ContextDecorator():\n"
    ),
    CommentStatement(line_no_start=69, line_no_end=70, value="#    pass\n"),
    UnchangingLine(line_no_start=70, line_no_end=71, value="\n"),
    UnchangingLine(line_no_start=71, line_no_end=72, value="\n"),
    FunctionDefinitionStart(line_no_start=72, line_no_end=73, value="def f():\n", name="f"),
    ReturnStatement(line_no_start=73, line_no_end=74, value="    return 1\n"),
)  # type: tuple[CstTypes, ...]


__all__ = ["cstify_cst"]  # type: list[str]

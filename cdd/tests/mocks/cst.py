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
    CommentStatement(
        line_no_start=0, line_no_end=0, scope=[], value="# Lint as: python2, python3"
    ),
    CommentStatement(
        line_no_start=0, line_no_end=1, scope=[], value="\n# Copyright 2022 under CC0"
    ),
    CommentStatement(
        line_no_start=1,
        line_no_end=2,
        scope=[],
        value="\n# ==============================================================================",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=False,
        scope=[],
        line_no_start=2,
        line_no_end=3,
        value='\n"""Module docstring goes here"""',
    ),
    FromStatement(
        line_no_start=3, line_no_end=5, scope=[], value="\n\nfrom operator import add"
    ),
    ClassDefinitionStart(
        line_no_start=5,
        line_no_end=8,
        scope=[],
        value="\n\n\nclass C(object):",
        name="C",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        scope=["C"],
        line_no_start=8,
        line_no_end=9,
        value='\n    """My cls"""',
    ),
    FunctionDefinitionStart(
        line_no_start=9,
        line_no_end=12,
        scope=["C"],
        value="\n\n    @staticmethod\n" "    def add1(foo):",
        name="add1",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        scope=["C", "add1"],
        line_no_start=12,
        line_no_end=19,
        value='\n        """\n'
        "        :param foo: a foo\n"
        "        :type foo: ```int```\n\n"
        "        :return: foo + 1\n"
        "        :rtype: ```int```\n"
        '        """',
    ),
    FunctionDefinitionStart(
        line_no_start=19,
        line_no_end=22,
        scope=["C", "add1"],
        value="\n\n        def adder(a: int,\n" "                  b: int) -> int:",
        name="adder",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        scope=["C", "add1", "adder"],
        line_no_start=22,
        line_no_end=29,
        value='\n            """\n'
        "            :param a: First arg\n\n"
        "            :param b: Second arg\n\n"
        "            :return: first + second arg\n"
        '            """',
    ),
    CommentStatement(
        line_no_start=29,
        line_no_end=30,
        scope=["C", "add1", "adder"],
        value="\n            # fmt: off",
    ),
    AnnAssignment(
        line_no_start=30,
        line_no_end=33,
        scope=["C", "add1", "adder"],
        value="\n            res: \\\n                int \\\n                = a + b",
    ),
    ReturnStatement(
        line_no_start=33,
        line_no_end=34,
        scope=["C", "add1", "adder"],
        value="\n            return res",
    ),
    Assignment(
        line_no_start=34,
        line_no_end=40,
        scope=["C", "add1", "adder"],
        value="\n\n        r = (\n            add(foo, 1)\n            or\n            adder(foo, 1)\n        )",
    ),
    IfStatement(
        line_no_start=40, line_no_end=41, scope=["C", "add1"], value="\n        if r:"
    ),
    NoneStatement(
        line_no_start=41,
        line_no_end=42,
        scope=["C", "add1"],
        value="\n            None",
    ),
    ElifStatement(
        line_no_start=42, line_no_end=43, scope=["C", "add1"], value="\n        elif r:"
    ),
    TrueStatement(
        line_no_start=43,
        line_no_end=44,
        scope=["C", "add1"],
        value="\n            True",
    ),
    FalseStatement(
        line_no_start=44,
        line_no_end=45,
        scope=["C", "add1"],
        value="\n            False",
    ),
    CommentStatement(
        line_no_start=45,
        line_no_end=46,
        scope=["C", "add1"],
        value="\n            # ([5,5] @ [5,5]) *\\",
    ),
    ExprStatement(
        line_no_start=46,
        line_no_end=48,
        scope=["C", "add1"],
        value="\n            -5 / 7 ** 6 + \\\n            6.0 - 6e1 & 1+2.34j",
    ),
    AugAssignment(
        line_no_start=48,
        line_no_end=49,
        scope=["C", "add1"],
        value="\n            r <<= 5",
    ),
    CallStatement(
        line_no_start=49,
        line_no_end=50,
        scope=["C", "add1"],
        value="\n            print(r)",
    ),
    ElseStatement(
        line_no_start=50, line_no_end=51, scope=["C", "add1"], value="\n        else:"
    ),
    PassStatement(
        line_no_start=51,
        line_no_end=52,
        scope=["C", "add1"],
        value="\n            pass",
    ),
    CommentStatement(
        line_no_start=52,
        line_no_end=53,
        scope=["C", "add1"],
        value="\n        # fmt: on",
    ),
    CommentStatement(
        line_no_start=53,
        line_no_end=54,
        scope=["C", "add1"],
        value="\n        # That^ incremented `foo` by 1",
    ),
    ReturnStatement(
        line_no_start=54,
        line_no_end=55,
        scope=["C", "add1"],
        value="\n        return r",
    ),
    CommentStatement(
        line_no_start=55,
        line_no_end=58,
        scope=["C", "add1"],
        value="\n\n\n# from contextlib import ContextDecorator",
    ),
    CommentStatement(
        line_no_start=58,
        line_no_end=60,
        scope=[],
        value="\n\n# with ContextDecorator():",
    ),
    CommentStatement(line_no_start=60, line_no_end=61, scope=[], value="\n#    pass"),
    FunctionDefinitionStart(
        line_no_start=61, line_no_end=64, scope=[], value="\n\n\ndef f():", name="f"
    ),
    ReturnStatement(
        line_no_start=64, line_no_end=65, scope=["f"], value="\n    return 1"
    ),
    UnchangingLine(line_no_start=65, line_no_end=66, scope=["f"], value="\n"),
)

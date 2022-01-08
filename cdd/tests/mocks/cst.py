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
        value="\n\n    @staticmethod\n    def add1(foo):",
        name="add1",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        scope=["C", "add1"],
        line_no_start=12,
        line_no_end=19,
        value='\n'
              '        """\n'
              '        :param foo: a foo\n'
              '        :type foo: ```int```\n'
              '\n'
              '        :return: foo + 1\n'
              '        :rtype: ```int```\n'
              '        """',
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=False,
        scope=["C", "add1"],
        line_no_start=19,
        line_no_end=21,
        value='\n\n        """foo"""',
    ),
    FunctionDefinitionStart(
        line_no_start=21,
        line_no_end=22,
        scope=["C", "add1"],
        value="\n        def g():",
        name="g",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        scope=["C", "add1", "g"],
        line_no_start=22,
        line_no_end=22,
        value=' """foo : bar ; can"""',
    ),
    PassStatement(
        line_no_start=22, line_no_end=22, scope=["C", "add1", "g"], value="; pass"
    ),
    FunctionDefinitionStart(
        line_no_start=22,
        line_no_end=24,
        scope=[],
        value="\n\n        def h():",
        name="h",
    ),
    CommentStatement(line_no_start=24, line_no_end=24, scope=["h"], value=" # stuff"),
    PassStatement(
        line_no_start=24, line_no_end=25, scope=["h"], value="\n            pass"
    ),
    FunctionDefinitionStart(
        line_no_start=25,
        line_no_end=28,
        scope=["h"],
        value="\n\n        def adder(a: int,\n                  b: int) -> int:",
        name="adder",
    ),
    TripleQuoted(
        is_double_q=True,
        is_docstr=True,
        scope=["h", "adder"],
        line_no_start=28,
        line_no_end=35,
        value='\n            """\n            :param a: First arg\n\n            :param b: Second arg\n\n            :return: first + second arg\n            """',
    ),
    CommentStatement(
        line_no_start=35,
        line_no_end=36,
        scope=["h", "adder"],
        value="\n            # fmt: off",
    ),
    AnnAssignment(
        line_no_start=36,
        line_no_end=39,
        scope=["h", "adder"],
        value="\n            res: \\\n                int \\\n                = a + b",
    ),
    ReturnStatement(
        line_no_start=39,
        line_no_end=40,
        scope=["h", "adder"],
        value="\n            return res",
    ),
    Assignment(
        line_no_start=40,
        line_no_end=46,
        scope=["h", "adder"],
        value="\n\n        r = (\n            add(foo, 1)\n            or\n            adder(foo, 1)\n        )",
    ),
    IfStatement(
        line_no_start=46, line_no_end=47, scope=["h", "adder"], value="\n        if r:"
    ),
    NoneStatement(
        line_no_start=47,
        line_no_end=48,
        scope=["h", "adder"],
        value="\n            None",
    ),
    ElifStatement(
        line_no_start=48,
        line_no_end=49,
        scope=["h", "adder"],
        value="\n        elif r:",
    ),
    TrueStatement(
        line_no_start=49,
        line_no_end=50,
        scope=["h", "adder"],
        value="\n            True",
    ),
    FalseStatement(
        line_no_start=50,
        line_no_end=51,
        scope=["h", "adder"],
        value="\n            False",
    ),
    CommentStatement(
        line_no_start=51,
        line_no_end=52,
        scope=["h", "adder"],
        value="\n            # ([5,5] @ [5,5]) *\\",
    ),
    ExprStatement(
        line_no_start=52,
        line_no_end=54,
        scope=["h", "adder"],
        value="\n            -5 / 7 ** 6 + \\\n            6.0 - 6e1 & 1+2.34j",
    ),
    AugAssignment(
        line_no_start=54,
        line_no_end=55,
        scope=["h", "adder"],
        value="\n            r <<= 5",
    ),
    CallStatement(
        line_no_start=55,
        line_no_end=56,
        scope=["h", "adder"],
        value="\n            print(r)",
    ),
    ElseStatement(
        line_no_start=56, line_no_end=57, scope=["h", "adder"], value="\n        else:"
    ),
    PassStatement(
        line_no_start=57,
        line_no_end=58,
        scope=["h", "adder"],
        value="\n            pass",
    ),
    CommentStatement(
        line_no_start=58,
        line_no_end=59,
        scope=["h", "adder"],
        value="\n        # fmt: on",
    ),
    CommentStatement(
        line_no_start=59,
        line_no_end=60,
        scope=["h", "adder"],
        value="\n        # That^ incremented `foo` by 1",
    ),
    ReturnStatement(
        line_no_start=60,
        line_no_end=61,
        scope=["h", "adder"],
        value="\n        return r",
    ),
    CommentStatement(
        line_no_start=61,
        line_no_end=64,
        scope=["h", "adder"],
        value="\n\n\n# from contextlib import ContextDecorator",
    ),
    CommentStatement(
        line_no_start=64,
        line_no_end=66,
        scope=[],
        value="\n\n# with ContextDecorator():",
    ),
    CommentStatement(line_no_start=66, line_no_end=67, scope=[], value="\n#    pass"),
    FunctionDefinitionStart(
        line_no_start=67, line_no_end=70, scope=[], value="\n\n\ndef f():", name="f"
    ),
    ReturnStatement(
        line_no_start=70, line_no_end=71, scope=["f"], value="\n    return 1"
    ),
    UnchangingLine(line_no_start=71, line_no_end=72, scope=["f"], value="\n"),
)

__all__ = ["cstify_cst"]

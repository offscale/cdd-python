"""
Concrete Syntax Tree utility functions
"""

from collections import OrderedDict, deque, namedtuple
from dataclasses import make_dataclass
from functools import partial, wraps
from keyword import kwlist
from typing import Optional

from cdd.pure_utils import balanced_parentheses, is_triple_quoted, tab

kwset = frozenset(kwlist)
_basic_cst_attributes = "line_no_start", "line_no_end", "value"
UnchangingLine = namedtuple("UnchangingLine", _basic_cst_attributes)
Assignment = namedtuple("Assignment", _basic_cst_attributes)
AnnAssignment = namedtuple("AnnAssignment", _basic_cst_attributes)
AugAssignment = namedtuple("AugAssignment", _basic_cst_attributes)
# Same for `AsyncFunctionDefinition`
FunctionDefinitionStart = namedtuple(
    "FunctionDefinitionStart", (*_basic_cst_attributes, "name")
)
ClassDefinitionStart = namedtuple(
    "ClassDefinitionStart", (*_basic_cst_attributes, "name")
)

# compound_stmt needs all of them to differentiate which indentor is scope-creating
IfStatement = namedtuple("IfStatement", _basic_cst_attributes)
ElifStatement = namedtuple("ElifStatement", _basic_cst_attributes)
ElseStatement = namedtuple("ElseStatement", _basic_cst_attributes)
WithStatement = namedtuple("WithStatement", _basic_cst_attributes)
ForStatement = namedtuple("ForStatement", _basic_cst_attributes)
WhileStatement = namedtuple("WhileStatement", _basic_cst_attributes)
MatchStatement = namedtuple("MatchStatement", _basic_cst_attributes)
CaseStatement = namedtuple("CaseStatement", _basic_cst_attributes)
# | function_def
# | if_stmt
# | class_def
# | with_stmt
# | for_stmt
# | try_stmt
# | while_stmt
# | match_stmt

CommentStatement = namedtuple("CommentStatement", _basic_cst_attributes)
PassStatement = namedtuple("PassStatement", _basic_cst_attributes)
DelStatement = namedtuple("DelStatement", _basic_cst_attributes)
YieldStatement = namedtuple("YieldStatement", _basic_cst_attributes)
BreakStatement = namedtuple("BreakStatement", _basic_cst_attributes)
ContinueStatement = namedtuple("ContinueStatement", _basic_cst_attributes)
GlobalStatement = namedtuple("GlobalStatement", _basic_cst_attributes)
NonlocalStatement = namedtuple("NonlocalStatement", _basic_cst_attributes)
ReturnStatement = namedtuple("ReturnStatement", _basic_cst_attributes)
RaiseStatement = namedtuple("RaiseStatement", _basic_cst_attributes)
TryStatement = namedtuple("TryStatement", _basic_cst_attributes)
CatchStatement = namedtuple("TryStatement", _basic_cst_attributes)
ExceptStatement = namedtuple("ExceptStatement", _basic_cst_attributes)
FinallyStatement = namedtuple("FinallyStatement", _basic_cst_attributes)
FromStatement = namedtuple("FromStatement", _basic_cst_attributes)
ImportStatement = namedtuple("ImportStatement", _basic_cst_attributes)
TrueStatement = namedtuple("TrueStatement", _basic_cst_attributes)
FalseStatement = namedtuple("FalseStatement", _basic_cst_attributes)
NoneStatement = namedtuple("NoneStatement", _basic_cst_attributes)
ExprStatement = namedtuple("ExprStatement", _basic_cst_attributes)
SetExprStatement = namedtuple("SetExprStatement", _basic_cst_attributes)
DictExprStatement = namedtuple("DictExprStatement", _basic_cst_attributes)
GenExprStatement = namedtuple("GenExprStatement", _basic_cst_attributes)
ListCompStatement = namedtuple("ListCompStatement", _basic_cst_attributes)
CallStatement = namedtuple("CallStatement", _basic_cst_attributes)

contains2statement = OrderedDict(
    (
        ("#", CommentStatement),
        ("pass", PassStatement),
        ("del", DelStatement),
        ("yield", YieldStatement),
        ("break", BreakStatement),
        ("continue", ContinueStatement),
        ("global", GlobalStatement),
        ("nonlocal", NonlocalStatement),
        ("return", ReturnStatement),
        ("raise", RaiseStatement),
        ("except", ExceptStatement),
        ("finally", FinallyStatement),
        ("try", TryStatement),
        ("from", FromStatement),
        ("import", ImportStatement),
        ("if", IfStatement),
        ("elif", ElifStatement),
        ("else:", ElseStatement),
        ("with", WithStatement),
        ("for", ForStatement),
        ("while", WhileStatement),
        ("match", MatchStatement),
        ("case", CaseStatement),
        ("True", TrueStatement),
        ("False", FalseStatement),
        ("None", NoneStatement),
    )
)
math_operators = frozenset(
    (
        "@",
        "/",
        "//",
        "*",
        "**",
        "+",
        "-",
        "%",
        "&",
        "|",
        "<<",
        ">>",
        "<",
        ">",
        "==",
        ">=",
        "<=",
        "^",
    )
)

augassign = frozenset(
    ("+=" "-=", "*=", "@=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "**=", "//=")
)
# ':=' could be an augassign I guess
multicontains2statement = (
    (frozenset((":", "=")), AnnAssignment),  # TODO: support `class F: foo: int` also…
    (frozenset(("=",)), Assignment),
)

ast2cst = {
    "Expr": ExprStatement,
    "AnnAssign": AnnAssignment,
    "Assign": Assignment,
    "Constant": ExprStatement,
    "Str": ExprStatement,
    "Num": ExprStatement,
    "ClassDef": ClassDefinitionStart,
    "FunctionDef": FunctionDefinitionStart,
    "AsyncFunctionDef": FunctionDefinitionStart,
    # TODO: Full spec
}

TripleQuoted = make_dataclass(
    "TripleQuoted",
    [
        ("is_double_q", Optional[bool]),
        ("is_docstr", Optional[bool]),
        ("line_no_start", int),
        ("line_no_end", int),
        ("value", str),
    ],
    namespace={},
)


def get_construct_name(words):
    """
    Find the construct name, currently works for:
    - AsyncFunctionDef
    - FunctionDef
    - ClassDef

    :param words: Tuple of words (no whitespace)
    :type words: ```Tuple[str]```

    :return: Name of construct if found else None
    :rtype: ```Optional[str]```
    """
    for idx, word in enumerate(words):
        if len(words) > idx + 1:
            if word == "def":
                return words[idx + 1][: words[idx + 1].find("(")]
            elif word == "class":
                end_idx = (
                    lambda _end_idx: words[idx + 1].find(":")
                    if _end_idx == -1
                    else _end_idx
                )(words[idx + 1].find("("))
                return words[idx + 1][:end_idx]


def cst_scanner(source):
    """
    Reduce source code into chunks useful for parsing.
    These chunks include newlines and the array is one dimensional.

    :param source: Python source code
    :type source: ```str```

    :return: List of scanned source code
    :rtype: ```List[str]```
    """
    scanned, stack = [], []
    for idx, ch in enumerate(source):
        if ch == "\n":  # in frozenset(("\n", ":", ";", '"""', "'''", '#')):
            cst_scan(scanned, stack)
        stack.append(ch)
    cst_scan(scanned, stack)
    if stack:
        scanned.append("".join(stack))
    return scanned


def cst_scan(scanned, stack):
    """
    Checks if what has been scanned (stack) is ready for the `scanned` array
    Add then clear stack if ready else do nothing with both

    :param scanned: List of statements observed
    :type scanned: ```List[str]```

    :param stack: List of characters observed
    :type stack: ```List[str]```
    """
    statement = "".join(stack)
    statement_stripped = statement.strip()

    is_comment = statement_stripped.startswith("#")
    if statement_stripped.endswith("\\"):
        has_triple_quotes = is_other_statement = False
    else:
        has_triple_quotes = is_triple_quoted(statement_stripped)
        is_other_statement = all(
            (
                statement_stripped,
                balanced_parentheses(statement_stripped),
                not statement_stripped.startswith("@")
                or statement_stripped.endswith(":"),
                not statement_stripped.startswith("'''"),
                not statement_stripped.startswith('"""'),
            )
        )

    statement_found = is_comment or has_triple_quotes or is_other_statement
    if statement_found:
        if is_comment:
            scanned.append(statement)
            stack.clear()
        elif is_other_statement:
            expression = []

            def add_and_clear(the_expression_str, expr, scanned_tokens, the_stack):
                """
                Add the found statement and clear the stacks

                :param the_expression_str: Expression string
                :type the_expression_str: ```str```

                :param expr: The expression
                :type expr: ```List[str]```

                :param scanned_tokens: Scanned tokens
                :type scanned_tokens: ```List[str]```

                :param the_stack: The current stack
                :type the_stack: ```List[str]```

                """
                scanned_tokens.append(the_expression_str)
                expr.clear()
                the_stack.clear()

            expression_str = ""
            for idx, ch in enumerate(statement):
                expression.append(ch)
                expression_str = "".join(expression)
                expression_stripped = expression_str.strip()

                if (
                    is_triple_quoted(expression_stripped)
                    or expression_stripped.startswith("#")
                    and expression_str.endswith("\n")
                ):
                    add_and_clear(expression_str, expression, scanned, stack)
                    # Single comment should be picked up
                elif balanced_parentheses(expression_stripped):
                    if (
                        expression_str.endswith("\n")
                        and not expression_stripped.endswith("\\")
                        and (
                            not expression_stripped.endswith(":")
                            or "class" not in expression_stripped
                            and "def" not in expression_stripped
                        )
                        and not expression_str.isspace()
                        and not expression_stripped.startswith("@")
                    ):
                        add_and_clear(expression_str, expression, scanned, stack)
                    else:
                        words = tuple(
                            filter(None, map(str.strip, expression_stripped.split(" ")))
                        )
                        if "def" in words or "class" in words:
                            if words[-1].endswith(":") and balanced_parentheses(
                                expression_stripped
                            ):
                                add_and_clear(
                                    expression_str, expression, scanned, stack
                                )
                        # elif ";"
            if expression:
                add_and_clear(expression_str, expression, scanned, stack)
            # Longest matching pattern should be parsed out but this does shortest^
        else:
            scanned.append(statement)
            stack.clear()


def cst_parser(scanned):
    """
    Checks if what has been scanned (stack) is ready for the `scanned` array
    Add then clear stack if ready else do nothing with both

    :param scanned: List of statements observed
    :type scanned: ```List[str]```

    :return: Parse of scanned statements. One dimensional.
    :rtype: ```Tuple[NamedTuple]```
    """
    state = {
        "acc": 1,
        "prev_node": UnchangingLine(line_no_start=None, line_no_end=None, value=""),
        "parsed": [],
    }
    # return accumulate(scanned, partial(cst_parse_one_node, state=state), initial=[])
    deque(map(partial(cst_parse_one_node, state=state), scanned), maxlen=0)
    return tuple(state["parsed"])


def infer_cst_type(statement_stripped, words):
    """
    Infer the CST type. This is the most important function of the CST parser!

    :param statement_stripped: Original scanned statement minus both ends of whitespace
    :type statement_stripped: ```str```

    :param words: List of whitespace stripped and empty-str removed words from original statement
    :type words: ```List[str]```

    :return: CST type… a NamedTuple with at least ("line_no_start", "line_no_end", "value") attributes
    :rtype: ```NamedTuple```
    """
    if statement_stripped.startswith("["):
        return ListCompStatement
    elif statement_stripped.startswith("{"):
        # Won't work with nested dict inside a SetExpr
        if ":" in statement_stripped:
            return DictExprStatement
        return SetExprStatement
    elif statement_stripped.startswith("("):
        return GenExprStatement

    for word in words:
        if word in contains2statement:
            return contains2statement[word]

    if any(aug_assign in statement_stripped for aug_assign in augassign):
        return AugAssignment

    statement_frozenset = frozenset(statement_stripped)
    for (key, constructor) in multicontains2statement:
        if statement_frozenset & key == key:
            return constructor

    if any(math_operator in statement_stripped for math_operator in math_operators):
        return ExprStatement

    elif "(" in statement_frozenset:
        return CallStatement

    return UnchangingLine  # or: raise NotImplementedError(repr(statement_stripped))


def set_prev_node(function):
    """
    Store the result of the current function run into the `prev_node` attribute

    :param function: The `cst_parse_one_node` function
    :type function: ```Callable[[str, dict], NamedTuple]```

    :return: Wrapped function
    :rtype: ```Callable[[], Callable[[str, dict], NamedTuple]]```
    """

    @wraps(function)
    def wrapper(statement, state):
        """
        Store the result of parsing one statement into a CST node in the `prev_node` key of the `state` dict

        :param statement: Statement that was scanned from Python source code
        :type statement: ```str```

        :param state: Number of lines between runs in `acc` key; `prev_node`
        :type state: ```dict```

        :return: NamedTuple with at least ("line_no_start", "line_no_end", "value") attributes
        :rtype: ```NamedTuple```
        """
        state["prev_node"] = function(statement, state)
        state["parsed"].append(state["prev_node"])
        return state["prev_node"]

    return wrapper


@set_prev_node
def cst_parse_one_node(statement, state):
    """
    Parse one statement into a CST node

    :param statement: Statement that was scanned from Python source code
    :type statement: ```str```

    :param state: Number of lines between runs in `acc` key; `prev_node`; `parsed`
    :type state: ```dict```

    :return: NamedTuple with at least ("line_no_start", "line_no_end", "value") attributes
    :rtype: ```NamedTuple```
    """
    prev_acc = state["acc"]
    state["acc"] += statement.count("\n")

    statement_stripped = statement.strip()

    words = tuple(filter(None, map(str.strip, statement_stripped.split(" "))))

    common_kwargs = dict(
        line_no_start=prev_acc,
        line_no_end=state["acc"],
        value=statement,
    )

    if len(statement_stripped) > 5:
        is_single_quote = (statement_stripped[:3], statement_stripped[-3:]) == (
            "'''",
            "'''",
        )
        is_double_quote = not is_single_quote and (
            statement_stripped[:3],
            statement_stripped[-3:],
        ) == (
            '"""',
            '"""',
        )
        if is_single_quote or is_double_quote:
            return TripleQuoted(
                **common_kwargs,
                is_double_q=is_double_quote,
                is_docstr=isinstance(
                    state["prev_node"], (ClassDefinitionStart, FunctionDefinitionStart)
                ),
            )

    if words:
        if len(words) > 1:
            name = get_construct_name(words)
            if name is not None:
                return (
                    ClassDefinitionStart
                    if words[0] == "class"
                    else FunctionDefinitionStart
                )(**common_kwargs, name=name)

        return infer_cst_type(statement_stripped, words)(**common_kwargs)

    return UnchangingLine(**common_kwargs)


def reindent_block_with_pass_body(s):
    """
    Reindent block (e.g., function definition) and give it a `pass` body

    :param s: Block defining string
    :type s: ```str```

    :return: Reindented string with `pass` body
    :rtype: ```str```
    """
    return "{block_def} pass".format(
        block_def="\n".join(
            map(
                str.lstrip,
                s.split("\n"),
            )
        ).replace(tab, "", 1)
    )


__all__ = [
    "AnnAssignment",
    "Assignment",
    "AugAssignment",
    "CallStatement",
    "ClassDefinitionStart",
    "CommentStatement",
    "ElifStatement",
    "ElseStatement",
    "ExprStatement",
    "FalseStatement",
    "FromStatement",
    "FunctionDefinitionStart",
    "IfStatement",
    "NoneStatement",
    "PassStatement",
    "ReturnStatement",
    "TripleQuoted",
    "TrueStatement",
    "UnchangingLine",
    "ast2cst",
    "cst_parser",
    "cst_scanner",
    "infer_cst_type",
    "kwset",
    "reindent_block_with_pass_body",
]

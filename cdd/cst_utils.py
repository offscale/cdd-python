"""
Concrete Syntax Tree utility functions
"""

from collections import OrderedDict, deque, namedtuple
from copy import deepcopy
from dataclasses import make_dataclass
from functools import partial, wraps
from itertools import takewhile
from keyword import kwlist
from operator import ne
from sys import stderr
from typing import List, Optional

from cdd.ast_utils import get_doc_str
from cdd.pure_utils import (
    balanced_parentheses,
    count_iter_items,
    is_triple_quoted,
    omit_whitespace,
    tab,
)

kwset = frozenset(kwlist)
_basic_cst_attributes = "line_no_start", "line_no_end", "clauses", "scope", "value"
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
        ("scope", List[str]),
        ("clauses", List[str]),
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
        if word == "def":
            return words[idx + 1][: words[idx + 1].find("(")]
        elif word == "class":
            end_idx = (
                lambda _end_idx: words[idx + 1].find(":")
                if _end_idx == -1
                else _end_idx
            )(words[idx + 1].find("("))
            return words[idx + 1][:end_idx]


def maybe_replace_doc_str_in_function_or_class(node, cst_idx, cst_list):
    """
    Maybe replace the doc_str of a function or class

    :param node: AST node
    :type node: ```Union[ClassDef, AsyncFunctionDef, FunctionDef]```

    :param cst_idx: Index of start of function/class in cst_list
    :type cst_idx: ```int```

    :param cst_list: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type cst_list: ```List[NamedTuple]```
    """
    # Maybe replace docstring
    new_doc_str = get_doc_str(node)
    cur_doc_str = cst_list[cst_idx + 1]
    triple_quoted = isinstance(cur_doc_str, TripleQuoted)
    changed = False
    if new_doc_str and not triple_quoted:
        cst_list.insert(
            cst_idx + 1,
            TripleQuoted(
                is_double_q=True,
                is_docstr=True,
                scope=cur_doc_str.scope,
                value=new_doc_str,
                line_no_start=cur_doc_str.line_no_start,
                line_no_end=cur_doc_str.line_no_end,
            ),
        )
        changed = "added"
    elif not new_doc_str and triple_quoted:
        del cst_list[cst_idx + 1]
        changed = "removed"
    else:
        cur_doc_str_only = cur_doc_str.value.strip()[3:-3]
        if ne(*map(omit_whitespace, (cur_doc_str_only, new_doc_str))):
            pre, _, post = cur_doc_str.value.partition(cur_doc_str_only)
            cur_doc_str.value = "{pre}{new_doc_str}{post}".format(
                pre=pre, new_doc_str=new_doc_str, post=post
            )
            changed = "replaced"
    if changed:
        print(changed, " docstr of the `", node.name, "` ", type(node).__name__, sep="")
        # TODO: Redo all subsequent `line_no` `start,end` lines as they're all invalidated now


def find_cst_at_ast(cst_list, node):
    """
    Find (first) CST node matching AST node

    (uses `_location` from `annotate_ancestry`)

    :param cst_list: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type cst_list: ```List[NamedTuple]```

    :param node: AST node
    :type node: ```AST```

    :return: Matching idx and element from cst_list if found else (None, None)
    :rtype: ```Tuple[Optional[int], Optional[NamedTuple]]````
    """
    cst_node_found, cst_node_no = None, None
    node_type = type(node).__name__
    if node_type not in ast2cst:
        print("{node_type} not implemented".format(node_type=node_type), file=stderr)
        return None, None
    cst_type = (ast2cst[node_type]).__name__
    for cst_node_no, cst_node in enumerate(cst_list):
        if (
            type(cst_node).__name__ == cst_type  # `isinstance` doesn't work
            and cst_node.scope == node._location[:-1]
            and cst_node.name == getattr(node, "name", None)
        ):
            cst_node_found = cst_node
            break
    return cst_node_no, cst_node_found


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
            cst_scan(scanned, stack, source[idx + 1] if len(source) > idx + 1 else "")
        stack.append(ch)
    cst_scan(scanned, stack, "")
    if stack:
        scanned.append("".join(stack))
    return scanned


def cst_scan(scanned, stack, peek):
    """
    Checks if what has been scanned (stack) is ready for the `scanned` array
    Add then clear stack if ready else do nothing with both

    :param scanned: List of statements observed
    :type scanned: ```List[str]```

    :param stack: List of characters observed
    :type stack: ```List[str]```

    :param peek: One character ahead of stack
    :type peek: ```str```
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
            expression, statement_stripped_n = [], len(statement_stripped)

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
                # Even with PEG Python is still basically LL(1)
                expression_peek = (
                    statement_stripped[idx + 1]
                    if idx + 1 < statement_stripped_n
                    else ""
                )

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

    :return: Parse of scanned statements. One dimensions but with a `scope` attribute.
    :rtype: ```Tuple[NamedTuple]```
    """
    state = {
        "acc": 0,
        "prev_node": UnchangingLine(
            line_no_start=None, line_no_end=None, value="", clauses=[], scope=[]
        ),
        "scope": [],
        "parsed": [],
    }
    # return accumulate(scanned, partial(cst_parse_one_node, state=state), initial=[])
    deque(map(partial(cst_parse_one_node, state=state), scanned), maxlen=0)
    return tuple(state["parsed"])


def get_scope_clauses(prev_node, statement):
    """
    Get the `scope` and `clauses` of the current `statement` given the `prev_node`

    TODO: Finish this and remove `get_last_node_scope`

    Where `scope` includes the names of any `class` and `def` the `statement` is within; and
      `clauses` is all the types which can create an indent, including: `class`, `def`, `with`, `if`, &etc.

    :param prev_node: NamedTuple with at least ("line_no", "clauses", "scope", "value") attributes
    :type prev_node: ```NamedTuple```

    :param statement: Statement that was scanned from Python source code
    :type statement: ```str```

    :return: scope, clauses
    :rtype: ```Tuple[str,str]```
    """
    scope, clauses = map(
        deepcopy, map(prev_node.__getattribute__, ("scope", "clauses"))
    )

    # These create indents AND new scopes
    if isinstance(prev_node, (ClassDefinitionStart, FunctionDefinitionStart)):
        scope.append(prev_node.name)
        clauses.append(type(prev_node).__name__)
    # These create indents but NOT new scopes
    elif isinstance(
        prev_node,
        (
            IfStatement,
            ElifStatement,
            ElseStatement,
            MatchStatement,
            CaseStatement,
            TryStatement,
            CatchStatement,
            FinallyStatement,
            WithStatement,
            ForStatement,
            WhileStatement,
        ),
    ):
        clauses.append(type(prev_node).__name__)
    else:
        # Scope is reduced by:
        # - Lowering the indent
        # - Having same indent as previous indent-creating node

        prev_indent = count_iter_items(
            takewhile(str.isspace, prev_node.value.lstrip("\n"))
        )
        indent = count_iter_items(takewhile(str.isspace, statement.lstrip("\n")))

        if any(
            (
                # Indent is the same, so same `clauses` and `scope`
                prev_indent == indent,
                # Creating a new scope, don't do anything (yet)
                indent > prev_indent,
                # Attached to previous scope, so don't do anything
                statement.startswith(";"),
            )
        ):
            pass
        else:  # TODO: Remove `clauses` and `scope`

            just = 11
            print(
                "statement:".ljust(just),
                repr(statement),
                " ;\n",
                "prev_node:".ljust(just),
                prev_node,
                " ;\n",
                sep="",
            )

            expected_scope_size = sum(
                map(
                    clauses.count,
                    ("ClassDefinitionStart", "FunctionDefinitionStart"),
                )
            )
    return scope, clauses


def get_last_node_scope(cst_nodes):
    """
    TODO: Remove this and use, in its stead, get_scope_clauses

    Gets the current scope of the last CST node in the list

    TODO: Refactor this by adding `within_types` to every CST node so that the scope-resolution becomes an O(n) op

    :param cst_nodes: Iterable NamedTuple with at least ("line_no", "clauses", "scope", "value") attributes
    :type cst_nodes: ```Iterable[NamedTuple]```
    """
    scope, within_types = [], []
    a = False
    for idx, node in enumerate(cst_nodes):
        indent = count_iter_items(takewhile(str.isspace, node.value.lstrip("\n")))

        # These create indents AND new scopes
        if isinstance(node, (ClassDefinitionStart, FunctionDefinitionStart)):
            scope.append(node.name)
            within_types.append(type(node).__name__)
        # These create indents but NOT new scopes
        elif isinstance(
            node,
            (
                IfStatement,
                ElifStatement,
                ElseStatement,
                MatchStatement,
                CaseStatement,
                TryStatement,
                CatchStatement,
                FinallyStatement,
                WithStatement,
                ForStatement,
                WhileStatement,
            ),
        ):
            within_types.append(type(node).__name__)
        elif isinstance(node, TripleQuoted):
            pass  # Could do weird things with multiline indents so ignore
        elif True:
            print(
                "within_types:",
                within_types,
                ";\n",
                "node.value =:",
                repr(node.value),
                ";\n",
                sep="",
            )
            expected_scope_size = sum(
                map(
                    within_types.count,
                    ("ClassDefinitionStart", "FunctionDefinitionStart"),
                )
            )

        else:
            indent_size = len(tab)
            # subtract the unexpected indent size as it's probably a one liner: `def f(): # pass
            #                                                                        """docstr"""`
            # indent -= indent % indent_size
            depth = indent // indent_size

            # Sanity check: if indent is >= start of line don't delete from `within_type` or `scope`
            previous = []
            for i in range(idx - 1, 0, -1):
                previous.append(cst_nodes[i].value)
                if "\n" in cst_nodes[i].value and cst_nodes[i].value.strip():
                    break
            previous_line = "".join(previous[::-1])
            previous_indent = count_iter_items(
                takewhile(str.isspace, previous_line.lstrip("\n"))
            )

            # if indent > previous_indent:
            #     just = 27
            #     if len(cst_nodes[idx - 1].scope) > len(scope):
            #         # print("previous:".ljust(just), repr(previous_line), ' ;\n',
            #         #       "current:".ljust(just), repr(node.value), ' ;\n',
            #         #       "node:".ljust(just), node, ' ;\n',
            #         #       "cst_nodes[idx - 1].scope:".ljust(just), cst_nodes[idx - 1].scope, ' ;\n',
            #         #       'scope:'.ljust(just), scope, ' ;', sep='')
            #         scope = deepcopy(cst_nodes[idx - 1].scope)
            # else:
            just = 27
            print(
                "previous:".ljust(just),
                repr(previous_line),
                " ;\n",
                "current:".ljust(just),
                repr(node.value),
                " ;\n",
                sep="",
            )
            if depth != len(within_types):
                del within_types[depth:]
            expected_scope_size = sum(
                map(
                    within_types.count,
                    ("ClassDefinitionStart", "FunctionDefinitionStart"),
                )
            )
            if expected_scope_size != len(scope):
                del scope[expected_scope_size:]
    return scope


def infer_cst_type(statement_stripped, words):
    """
    Infer the CST type. This is the most important function of the CST parser!

    :param statement_stripped: Original scanned statement minus both ends of whitespace
    :type statement_stripped: ```str```

    :param words: List of whitespace stripped and empty-str removed words from original statement
    :type words: ```List[str]```

    :return: CST type… a NamedTuple with at least ("line_no", "clauses", "scope", "value") attributes
    :rtype: ```NamedTuple```
    """
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

    elif statement_stripped.startswith("["):
        return ListCompStatement

    elif "(" in statement_frozenset:
        if statement_stripped.startswith("("):
            return GenExprStatement
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

        :param state: Number of lines between runs in `acc` key; `prev_node`; `scope`
        :type state: ```dict```

        :return: NamedTuple with at least ("line_no", "clauses", "scope", "value") attributes
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

    :param state: Number of lines between runs in `acc` key; `prev_node`; `scope`; `parsed`
    :type state: ```dict```

    :return: NamedTuple with at least ("line_no", "clauses", "scope", "value") attributes
    :rtype: ```NamedTuple```
    """
    prev_acc = state["acc"]
    state["acc"] += statement.count("\n")

    statement_stripped = statement.strip()

    words = tuple(filter(None, map(str.strip, statement_stripped.split(" "))))
    scope, clauses = get_scope_clauses(state["prev_node"], statement)

    common_kwargs = dict(
        line_no_start=prev_acc,
        line_no_end=state["acc"],
        scope=scope,
        clauses=clauses,
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
                state["scope"].append(name)
                return (
                    ClassDefinitionStart
                    if words[0] == "class"
                    else FunctionDefinitionStart
                )(**common_kwargs, name=name)

        return infer_cst_type(statement_stripped, words)(**common_kwargs)

    return UnchangingLine(**common_kwargs)


__all__ = [
    "AnnAssignment",
    "Assignment",
    "ClassDefinitionStart",
    "CommentStatement",
    "ElifStatement",
    "FromStatement",
    "FunctionDefinitionStart",
    "IfStatement",
    "TripleQuoted",
    "TripleQuoted",
    "PassStatement",
    "ReturnStatement",
    "UnchangingLine",
    "CallStatement",
    "ElseStatement",
    "AugAssignment",
    "ExprStatement",
    "FalseStatement",
    "TrueStatement",
    "NoneStatement",
    "cst_parser",
    "cst_scanner",
    "kwset",
    "maybe_replace_doc_str_in_function_or_class",
]

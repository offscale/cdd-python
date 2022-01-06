"""
Concrete Syntax Tree utility functions
"""
from collections import OrderedDict, namedtuple, deque
from copy import deepcopy
from dataclasses import make_dataclass
from functools import partial, wraps, reduce
from itertools import takewhile, accumulate
from keyword import kwlist
from operator import ne
from typing import List, Optional

from cdd.ast_utils import get_value
from cdd.pure_utils import balanced_parentheses, count_iter_items, omit_whitespace, tab

kwset = frozenset(kwlist)
_basic_cst_attributes = "line_no_start", "line_no_end", "scope", "value"
UnchangingLine = namedtuple("UnchangingLine", _basic_cst_attributes)
Assignment = namedtuple("Assignment", _basic_cst_attributes)
AnnAssignment = namedtuple("AnnAssignment", _basic_cst_attributes)
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
    )
)

multicontains2statement = (
    (frozenset((":", "=")), AnnAssignment),  # TODO: support `class F: foo: int` also…
    (frozenset(("=",)), Assignment),
)

MultiComment = make_dataclass(
    "MultiComment",
    [
        ("is_double_q", Optional[bool]),
        ("is_docstr", Optional[bool]),
        ("scope", List[str]),
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


def maybe_replace_doc_str_in_function_or_class(node, concrete_lines, cst_node_no):
    """
    Maybe replace the doc_str of a function or class

    :param node: AST node
    :type node: ```Union[ClassDef, AsyncFunctionDef, FunctionDef]```

    :param concrete_lines: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type concrete_lines: ```List[NamedTuple]```

    :param cst_node_no: Index of start of function in concrete_lines
    :type cst_node_no: ```int```
    """
    # Ignore `arg` its `Assign | AnnAssign` are handled later
    if (
        isinstance(concrete_lines[cst_node_no + 1], MultiComment)
        and concrete_lines[cst_node_no + 1].is_docstr
    ):
        # Considering this is looping in forward direction `is_docstr` is either redundant
        # …or this should be refactored to look at `MultiComment`s
        new_doc_str = get_value(get_value(node.body[0]))
        cur_doc_str = concrete_lines[cst_node_no + 1].value.strip()[3:-3]
        if ne(
            *map(
                omit_whitespace,
                (
                    cur_doc_str,
                    new_doc_str,
                ),
            )
        ):
            _indent_idx = count_iter_items(takewhile(str.isspace, cur_doc_str))
            pre, _, post = concrete_lines[cst_node_no + 1].value.partition(cur_doc_str)
            concrete_lines[cst_node_no + 1].value = "{pre}{new_doc_str}{post}".format(
                pre=pre, new_doc_str=new_doc_str, post=post
            )
            print("replaced doc_str at:", node._location, ";")


def find_cst_at_ast(concrete_lines, node):
    """
    Find (first) CST node matching AST node

    (uses `_location` from `annotate_ancestry`)

    :param concrete_lines: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type concrete_lines: ```List[NamedTuple]```

    :param node: AST node
    :type node: ```AST```

    :return: Matching idx and element from concrete_lines if found else (None, None)
    :rtype: ```Tuple[Optional[int], Optional[NamedTuple]]````
    """
    cst_node_found, cst_node_no = None, None
    for cst_node_no, cst_node in enumerate(concrete_lines):
        if (
            cst_node.scope == node._location[:-1]
            and isinstance(
                cst_node,
                (ClassDefinitionStart, FunctionDefinitionStart),
            )
            and cst_node.name == getattr(node, "name", None)
        ):
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
        if ch == "\n":
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
    if not statement.endswith("\\") and any(
        (
            statement_stripped.startswith("#"),
            # not statement_stripped,
            len(statement_stripped) > 5
            and (
                statement_stripped.startswith("'''")
                and statement.endswith("'''")
                or statement_stripped.startswith('"""')
                and statement.endswith('"""')
            ),
            balanced_parentheses(statement_stripped)
            and any(
                (
                    statement_stripped.endswith(":"),
                    len(
                        frozenset(contains2statement.keys())
                        & frozenset(
                            filter(None, map(str.strip, statement_stripped.split(" ")))
                        )
                    )
                    > 0,
                    "=" in statement_stripped,
                )
            )
            # statement_stripped.startswith("")
        )
    ):
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
            line_no_start=None, line_no_end=None, value="", scope=[]
        ),
        "scope": [],
        "parsed": [],
    }
    # return accumulate(scanned, partial(cst_parse_one_node, state=state), initial=[])
    deque(map(partial(cst_parse_one_node, state=state), scanned), maxlen=0)
    return tuple(state["parsed"])


def get_last_node_scope(cst_nodes):
    """
    Gets the current scope of the last CST node in the list

    :param cst_nodes: Iterable NamedTuple with at least ("line_no", "scope", "value") attributes
    :type cst_nodes: ```Iterable[NamedTuple]```
    """
    scope, within_types, last_indent = [], [], 0
    for node in cst_nodes:
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
    else:
        raise NotImplementedError("Count the number of indents and number of `name`d statements to handle reduction of scope")
    return scope


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

        :return: NamedTuple with at least ("line_no", "scope", "value") attributes
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

    :param state: Number of lines between runs in `acc` key; `prev_node`; `scope`
    :type state: ```dict```

    :return: NamedTuple with at least ("line_no", "scope", "value") attributes
    :rtype: ```NamedTuple```
    """
    prev_acc = state["acc"]
    state["acc"] += statement.count("\n")
    if state["acc"] - 1 == prev_acc:
        prev_acc += 1

    statement_stripped = statement.strip()

    words = tuple(filter(None, map(str.strip, statement_stripped.split(" "))))
    indent = max(count_iter_items(takewhile(str.isspace, statement[1:])), 0)
    prev_indent = max(
        count_iter_items(takewhile(str.isspace, state["prev_node"].value[1:])), 0
    )
    indent_size = len(tab)

    # if prev_indent > indent and  state["prev_node"]

    # if prev_indent < indent and indent % indent_size == 0:
    #    current_depth = indent // indent_size
    # print("\ncurrent_depth:", current_depth, ';',
    #       '\nlen(state["scope"]):', len(state["scope"]), ';',
    #       '\nstate["scope"]:', state["scope"], ';')
    # while len(state["scope"]) > current_depth and state["scope"]:
    # print("removing:", state["scope"][-1], "@", repr(statement[1:]), ";\n")
    #    del state["scope"][-1]
    # else:
    #    print("indent:", indent, ";", "\nstatement:", repr(statement), ";")

    # Useful during dev tests, in practice you can have nested non `class_def` | `function_def` `compound_stmt`s…
    # assert indent % 4 != 0 or indent / 4 == len(
    #    state["scope"]
    # ), "scope implied by indent is missing @ {!r}".format(statement)

    common_kwargs = dict(
        line_no_start=prev_acc,
        line_no_end=state["acc"],
        scope=deepcopy(state["scope"]),
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
            return MultiComment(
                **common_kwargs,
                is_double_q=is_double_quote,
                is_docstr=isinstance(
                    state["prev_node"], (ClassDefinitionStart, FunctionDefinitionStart)
                ),
            )

    if len(words) > 1:
        name = get_construct_name(words)
        if name is not None:
            state["scope"].append(name)
            return (
                ClassDefinitionStart if words[0] == "class" else FunctionDefinitionStart
            )(**common_kwargs, name=name)

    if words:
        statement_frozenset = frozenset(statement_stripped)
        for word in words:
            if word in contains2statement:
                return contains2statement[word](**common_kwargs)
        for multistatement in multicontains2statement:
            key, constructor = multistatement
            if statement_frozenset & key == key:
                return constructor(**common_kwargs)

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
    "MultiComment",
    "MultiComment",
    "PassStatement",
    "ReturnStatement",
    "UnchangingLine",
    "cst_parser",
    "cst_scanner",
    "kwset",
    "maybe_replace_doc_str_in_function_or_class",
]

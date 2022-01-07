"""
Concrete Syntax Tree utility functions
"""
from collections import OrderedDict, deque, namedtuple
from dataclasses import make_dataclass
from functools import partial, wraps
from itertools import takewhile
from keyword import kwlist
from operator import ne
from sys import stderr
from typing import List, Optional

from cdd.ast_utils import get_doc_str
from cdd.pure_utils import balanced_parentheses, count_iter_items, omit_whitespace, tab

kwset = frozenset(kwlist)
_basic_cst_attributes = "line_no_start", "line_no_end", "scope", "value"
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
            type(cst_node).__name__ == cst_type
            and cst_node.scope == node._location[:-1]  # `isinstance` doesn't work
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

    if (
        statement_stripped.startswith("#")
        or not statement_stripped.endswith("\\")
        and any(
            (
                all(
                    (
                        len(statement_stripped) > 5,
                        statement_stripped.startswith("'''")
                        and statement.endswith("'''")
                        or statement_stripped.startswith('"""')
                        and statement.endswith('"""'),
                    )
                ),
                all(
                    (
                        statement_stripped,
                        balanced_parentheses(statement_stripped),
                        not statement_stripped.startswith("@")
                        or statement_stripped.endswith(":"),
                        not statement_stripped.startswith("'''"),
                        not statement_stripped.startswith('"""'),
                    )
                ),
            )
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

    TODO: Refactor this by adding `within_types` to every CST node so that the scope-resolution becomes an O(n) op

    :param cst_nodes: Iterable NamedTuple with at least ("line_no", "scope", "value") attributes
    :type cst_nodes: ```Iterable[NamedTuple]```
    """
    scope, within_types = [], []
    for node in cst_nodes:
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
        else:
            indent_size = len(tab)
            assert indent % indent_size == 0, "Indent must be a multiple of tab size"
            depth = indent // indent_size

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

    :return: CST type… a NamedTuple with at least ("line_no", "scope", "value") attributes
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

    :param state: Number of lines between runs in `acc` key; `prev_node`; `scope`; `parsed`
    :type state: ```dict```

    :return: NamedTuple with at least ("line_no", "scope", "value") attributes
    :rtype: ```NamedTuple```
    """
    prev_acc = state["acc"]
    state["acc"] += statement.count("\n")

    statement_stripped = statement.strip()

    words = tuple(filter(None, map(str.strip, statement_stripped.split(" "))))
    scope = get_last_node_scope(state["parsed"])

    common_kwargs = dict(
        line_no_start=prev_acc,
        line_no_end=state["acc"],
        scope=scope,
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

"""
Concrete Syntax Tree utility functions
"""
from collections import namedtuple
from copy import deepcopy
from dataclasses import make_dataclass
from itertools import takewhile
from keyword import kwlist
from operator import ne
from typing import List, Optional

from cdd.ast_utils import get_value
from cdd.pure_utils import count_iter_items, omit_whitespace

kwset = frozenset(kwlist)
UnchangingLine = namedtuple("UnchangingLine", ("line_no", "scope", "value"))
Assignment = namedtuple("Assignment", ("line_no", "scope", "name", "value"))
AnnAssignment = namedtuple("AnnAssignment", ("line_no", "scope", "name", "value"))
# Same for `AsyncFunctionDefinition`
FunctionDefinitionStart = namedtuple(
    "FunctionDefinitionStart", ("line_no", "scope", "name", "value")
)
ClassDefinitionStart = namedtuple(
    "ClassDefinitionStart", ("line_no", "scope", "name", "value")
)
MultiLineComment = make_dataclass(
    "MultiLineComment",
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


def handle_all_other_nodes(line, line_no, line_lstripped, concrete_lines, scope):
    """
    Parse nodes not handled by more specialised parser

    :param line: Current line of Python source code
    :type line: ```str```

    :param line_no: Line number
    :type line_no: ```int```

    :param line_lstripped: `str.lstrip`ped `line`
    :type line_lstripped: ```str```

    :param concrete_lines: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type concrete_lines: ```List[Any]```

    :param scope: Scope of current node
    :type scope: ```List[str]```
    """

    words = tuple(filter(None, map(str.strip, line.split(" "))))
    if len(words) > 1:
        name = get_construct_name(words)
        if name is None:
            if frozenset(words) & kwset:
                concrete_lines.append(UnchangingLine(line_no, deepcopy(scope), line))
            else:
                if ":" in line_lstripped:
                    concrete_lines.append(
                        AnnAssignment(
                            line_no, deepcopy(scope), name=words[0], value=line
                        )
                    )
                elif "=" in line_lstripped:
                    concrete_lines.append(
                        Assignment(line_no, deepcopy(scope), name=words[0], value=line)
                    )
                else:
                    concrete_lines.append(
                        UnchangingLine(line_no, deepcopy(scope), line)
                    )
        else:
            concrete_lines.append(
                (
                    ClassDefinitionStart
                    if words[0] == "class"
                    else FunctionDefinitionStart
                )(line_no, deepcopy(scope), name=name, value=line)
            )
            scope.append(name)
    else:
        concrete_lines.append(UnchangingLine(line_no, deepcopy(scope), line))


def handle_multiline_comment(
    line,
    line_no,
    line_lstripped,
    is_double_q,
    multi_line_comment,
    concrete_lines,
    scope,
):
    """
    Parse multiline comments (including those on one line, like: '''foo''')

    :param line: Current line of Python source code
    :type line: ```str```

    :param line_no: Line number
    :type line_no: ```int```

    :param line_lstripped: `str.lstrip`ped `line`
    :type line_lstripped: ```str```

    :param is_double_q: Whether this is double triple quoted
    :type is_double_q: ```bool```

    :param multi_line_comment: The existing `MultiLineComment` structure
    :type multi_line_comment: ```MultiLineComment```

    :param concrete_lines: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type concrete_lines: ```List[Any]```

    :param scope: Scope of current node
    :type scope: ```List[str]```

    :return: The resulting `MultiLineComment` structure
    :rtype: ```MultiLineComment```
    """
    if multi_line_comment.line_no_start is None:
        multi_line_comment.value = line
        multi_line_comment.is_double_q = is_double_q
        multi_line_comment.is_docstr = line_no > 0 and isinstance(
            concrete_lines[-1], (ClassDefinitionStart, FunctionDefinitionStart)
        )
        multi_line_comment.line_no_start = line_no
        multi_line_comment.scope = deepcopy(scope)
        line_stripped = line_lstripped.rstrip()
        if (
            line_stripped.endswith('"""' if multi_line_comment.is_double_q else "'''")
            and len(line_stripped) > 3
        ):
            multi_line_comment = conclude_multiline_comment(
                concrete_lines, line_no, multi_line_comment
            )
    else:
        multi_line_comment.value += line
        multi_line_comment = conclude_multiline_comment(
            concrete_lines, line_no, multi_line_comment
        )

    return multi_line_comment


def conclude_multiline_comment(concrete_lines, line_no, multi_line_comment):
    """
    Close the multiline comment structure

    :param concrete_lines: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type concrete_lines: ```List[Any]```

    :param line_no: Line the comment ends on
    :type line_no: ```int```

    :param multi_line_comment: The existing `MultiLineComment` structure
    :type multi_line_comment: ```MultiLineComment```

    :return: Empty MultiLineComment
    :rtype: ```MultiLineComment```
    """
    multi_line_comment.line_no_end = line_no
    concrete_lines.append(multi_line_comment)
    multi_line_comment = MultiLineComment(None, None, None, None, None, None)
    return multi_line_comment


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
    if words[0] == "def":
        return words[1][: words[1].find("(")]
    elif words[0] == "async" and words[1] == "def":
        return words[2][: words[2].find("(")]
    elif words[0] == "class":
        end_idx = (lambda _end_idx: words[1].find(":") if _end_idx == -1 else _end_idx)(
            words[1].find("(")
        )
        return words[1][:end_idx]


def whitespace_aware_parse(line_no, line, concrete_lines, multi_line_comment, scope):
    """
    Parse one line from Python source code in a whitespace-maintaining manner forming a CST with `scope` information

    :param line_no: Line number
    :type line_no: ```int```

    :param line: Current line of Python source code
    :type line: ```str```

    :param concrete_lines: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type concrete_lines: ```List[Any]```

    :param multi_line_comment: The existing `MultiLineComment` structure
    :type multi_line_comment: ```MultiLineComment```

    :param scope: Scope of current node
    :type scope: ```List[str]```

    :return: The existing `MultiLineComment` structure
    :rtype: ```MultiLineComment```
    """
    indent = count_iter_items(takewhile(str.isspace, line))
    line_lstripped = line[indent:]
    if indent % 4 == 0:
        while indent / 4 != len(scope):
            del scope[-1]
    # Useful during dev tests, in practice you can have nested non `class_def` | `function_def` `compound_stmt`s…
    assert indent % 4 != 0 or indent / 4 == len(
        scope
    ), "scope implied by indent is missing @ {!r}".format(line)
    is_single_q, is_double_q = map(line_lstripped.startswith, ("'''", '"""'))
    if is_single_q or is_double_q:
        multi_line_comment = handle_multiline_comment(
            line,
            line_no,
            line_lstripped,
            is_double_q,
            multi_line_comment,
            concrete_lines,
            scope,
        )
    elif multi_line_comment.line_no_start is not None:
        multi_line_comment.value += line
        line_stripped = line_lstripped.rstrip()
        quote_mark = '"""' if multi_line_comment.is_double_q else "'''"
        if line_stripped.endswith(quote_mark) and not line_stripped.endswith(
            "\\{quote_mark}".format(quote_mark=quote_mark)
        ):
            multi_line_comment = conclude_multiline_comment(
                concrete_lines, line_no, multi_line_comment
            )
    elif not len(line_lstripped) or line_lstripped.startswith("#"):
        concrete_lines.append(UnchangingLine(line_no, deepcopy(scope), line))
    else:
        handle_all_other_nodes(line, line_no, line_lstripped, concrete_lines, scope)

    return multi_line_comment


def maybe_replace_doc_str_in_function_or_class(node, concrete_lines):
    """
    Maybe replace the doc_str of a function or class

    :param node: AST node
    :type node: ```Union[ClassDef, AsyncFunctionDef, FunctionDef]```

    :param concrete_lines: List of `namedtuple`s with at least ("line_no", "scope", "value") attributes
    :type concrete_lines: ```List[Any]```
    """
    cst_node_found, cst_node_no = None, None
    for cst_node_no, cst_node in enumerate(concrete_lines):
        if (
            cst_node.scope == node._location[:-1]
            and isinstance(
                cst_node,
                (ClassDefinitionStart, FunctionDefinitionStart),
            )
            and cst_node.name == node.name
        ):
            cst_node_found = cst_node
            break
    if cst_node_found is not None:
        # Ignore `arg` its `Assign | AnnAssign` are handled later
        if (
            isinstance(concrete_lines[cst_node_no + 1], MultiLineComment)
            and concrete_lines[cst_node_no + 1].is_docstr
        ):
            # Considering this is looping in forward direction `is_docstr` is either redundant
            # …or this should be refactored to look at `MultiLineComment`s
            new_doc_str = get_value(get_value(node.body[0]))
            if ne(
                *map(
                    omit_whitespace,
                    (
                        concrete_lines[cst_node_no + 1].value,
                        new_doc_str,
                    ),
                )
            ):
                _indent_idx = count_iter_items(
                    takewhile(str.isspace, concrete_lines[cst_node_no + 1].value)
                )
                concrete_lines[
                    cst_node_no + 1
                ].value = "{_indent}{q}{new_doc_str}{q}\n".format(
                    _indent=concrete_lines[cst_node_no + 1].value[:_indent_idx],
                    new_doc_str=new_doc_str,
                    q='"""',
                )
                print("replaced doc_str at:", node._location, ";")

        # pp(cst_node_found)
        # print_ast(_node)


__all__ = [
    "MultiLineComment",
    "kwset",
    "maybe_replace_doc_str_in_function_or_class",
    "whitespace_aware_parse",
]

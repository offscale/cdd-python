"""
Helper to traverse the AST of the input file, extract the docstring out, parse and format to intended style, and emit
"""

from ast import (
    AsyncFunctionDef,
    ClassDef,
    Constant,
    Expr,
    FunctionDef,
    Str,
    fix_missing_locations,
    walk,
)
from collections import deque
from copy import deepcopy
from operator import attrgetter, eq

from cdd.ast_utils import cmp_ast, get_value
from cdd.cst import cst_parse
from cdd.cst_utils import kwset, maybe_replace_doc_str_in_function_or_class
from cdd.doctrans_utils import DocTrans, has_type_annotations
from cdd.pure_utils import omit_whitespace, pp
from cdd.source_transformer import ast_parse, to_code


def node_to_line(node, source_lines):
    """
    :param node: AST node
    :type node: ```AST```

    :param source_lines: Source lines
    :type source_lines: ```List[str]```
    """
    assert node.lineno is not None, "{!r}".format(
        {attr: getattr(node, attr) for attr in dir(node) if not attr.startswith("_")}
    )
    pp({attr: getattr(node, attr) for attr in dir(node) if not attr.startswith("_")})
    if node.lineno > 1 and isinstance(node, (Constant, Str, Expr)):
        prev_line = source_lines[node.lineno - 1]
        prev_line_lstripped = prev_line.lstrip()
        token = prev_line_lstripped.partition(" ")[0]
        if token not in kwset:
            prev_node = ast_parse(
                source_lines[node.lineno - 1],
                skip_annotate=True,
                skip_docstring_remit=True,
            ).body[0]
            a, b = map(get_value, (prev_node, node))
            pp({"a": a, "b": b})
            if isinstance(prev_node, (Constant, Str, Expr)) and eq(
                *map(omit_whitespace, map(get_value, map(get_value, (prev_node, node))))
            ):
                return
        else:
            print("skipping:", repr(get_value(get_value(node))), ";")
    if len(source_lines) > node.lineno:
        source_lines[node.lineno] = to_code(node)
    else:
        source_lines.append(to_code(node))


def all_nodes_to_line(node, source_lines):
    """
    :param node: AST node
    :type node: ```AST```

    :param source_lines: Source lines
    :type source_lines: ```List[str]```
    """
    return deque(
        map(
            lambda _node: (
                all_nodes_to_line if hasattr(_node, "body") else node_to_line
            )(node=_node, source_lines=source_lines),
            node.body,
        ),
        maxlen=0,
    )


def doctrans(filename, docstring_format, type_annotations, no_word_wrap):
    """
    Transform the docstrings found within provided filename to intended docstring_format

    :param filename: Python file to convert docstrings within. Edited in place.
    :type filename: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param type_annotations: True to have type annotations (3.6+), False to place in docstring
    :type type_annotations: ```bool```

    :param no_word_wrap: Whether word-wrap is disabled (on emission).
    :type no_word_wrap: ```Optional[Literal[True]]```
    """
    with open(filename, "rt") as f:
        original_source_lines = f.readlines()
    original_source = "".join(original_source_lines)
    node = ast_parse(original_source, skip_docstring_remit=False)
    original_module = deepcopy(node)

    node = fix_missing_locations(
        DocTrans(
            docstring_format=docstring_format,
            word_wrap=no_word_wrap is None,
            type_annotations=type_annotations,
            existing_type_annotations=has_type_annotations(node),
            whole_ast=original_module,
        ).visit(node)
    )

    if not cmp_ast(node, original_module):
        concrete_lines = cst_parse(original_source_lines)

        # Carefully replace only docstrings, function return annotations, assignment and annotation assignments.
        # Maintaining all other existing whitespace, comments, &etc.

        # Plenty of room for optimisation: probably an inverted index on the concrete_lines to scope is the right idea.
        for _node in walk(node):
            if hasattr(_node, "_location"):
                if isinstance(_node, (AsyncFunctionDef, FunctionDef, ClassDef)):
                    # TODO: Add to the `return` of (AsyncFunctionDef | FunctionDef) if `type_annotations`
                    maybe_replace_doc_str_in_function_or_class(_node, concrete_lines)
                # TODO:
                # elif isinstance(_node, (AnnAssign, Assign)):
                #     print("(AnnAssign | Assign)._location:", _node._location, ";")
                #     print_ast(_node)

        with open(filename, "wt") as f:
            f.writelines(map(attrgetter("value"), concrete_lines))


__all__ = ["doctrans"]

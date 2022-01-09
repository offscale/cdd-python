"""
Utils for working with AST (builtin) and cdd's CST
"""

from enum import Enum
from operator import ne
from sys import stderr

from cdd.ast_utils import get_doc_str
from cdd.cst_utils import TripleQuoted, ast2cst
from cdd.pure_utils import omit_whitespace, tab


def find_cst_at_ast(cst_list, node):
    """
    Find (first) CST node matching AST node

    (uses `_location` from `annotate_ancestry`)

    :param cst_list: List of `namedtuple`s with at least ("line_no_start", "line_no_end", "value") attributes
    :type cst_list: ```List[NamedTuple]```

    :param node: AST node
    :type node: ```AST```

    :return: Matching idx and element from cst_list if found else (None, None)
    :rtype: ```Tuple[Optional[int], Optional[NamedTuple]]````
    """
    cst_node_found, cst_node_no = None, None
    node_type = type(node).__name__
    cst_type = ast2cst.get(node_type, type(None)).__name__
    if cst_type == "NoneType":
        print("`{node_type}` not implemented".format(node_type=node_type), file=stderr)
        return None, None
    for cst_node_no, cst_node in enumerate(cst_list):
        if (
            cst_node.line_no_start <= node.lineno <= cst_node.line_no_end
            # Extra precautions to ensure the wrong node is never replaced:
            and type(cst_node).__name__ == cst_type  # `isinstance` doesn't work
            and getattr(cst_node, "name", None) == getattr(node, "name", None)
        ):
            cst_node_found = cst_node
            break
    return cst_node_no, cst_node_found


class Delta(Enum):
    """
    Maybe Enum for what every `maybe_` function in `ast_cst_utils` can return
    """

    added = 0
    removed = 1
    replaced = 2
    nop = 255


def maybe_replace_doc_str_in_function_or_class(node, cst_idx, cst_list):
    """
    Maybe replace the doc_str of a function or class

    :param node: AST node
    :type node: ```Union[ClassDef, AsyncFunctionDef, FunctionDef]```

    :param cst_idx: Index of start of function/class in cst_list
    :type cst_idx: ```int```

    :param cst_list: List of `namedtuple`s with at least ("line_no_start", "line_no_end", "value") attributes
    :type cst_list: ```List[NamedTuple]```

    :return: Delta value indicating what changed (if anything)
    :rtype: ```Delta```
    """
    new_doc_str = get_doc_str(node)
    cur_doc_str = cst_list[cst_idx + 1]
    existing_doc_str = isinstance(cur_doc_str, TripleQuoted) and cur_doc_str.is_docstr
    changed = Delta.nop
    print(
        "new_doc_str: {!r} ;\n"
        "cur_doc_str: {!r} ;\n"
        "existing_doc_str: {!r} ;".format(new_doc_str, cur_doc_str, existing_doc_str)
    )
    if new_doc_str and not existing_doc_str:
        cst_list.insert(
            cst_idx + 1,
            TripleQuoted(
                is_double_q=True,
                is_docstr=True,
                value='\n{tab}"""{new_doc_str}"""'.format(
                    tab=tab, new_doc_str=new_doc_str
                ),
                line_no_start=cur_doc_str.line_no_start,
                line_no_end=cur_doc_str.line_no_end,
            ),
        )
        changed = Delta.added
    elif not new_doc_str and existing_doc_str:
        del cst_list[cst_idx + 1]
        changed = Delta.removed
    else:
        cur_doc_str_only = cur_doc_str.value.strip()[3:-3]
        if ne(*map(omit_whitespace, (cur_doc_str_only, new_doc_str))):
            pre, _, post = cur_doc_str.value.partition(cur_doc_str_only)
            cur_doc_str.value = "{pre}{new_doc_str}{post}".format(
                pre=pre, new_doc_str=new_doc_str, post=post
            )
            changed = Delta.replaced
    if changed:
        print(
            "{!s} docstr of the `{name}` {typ}".format(
                str(changed), name=node.name, typ=type(node).__name__
            )
        )
        # Subsequent `line_no` `start,end` lines are invalidated. It's necessary to link the CST and AST together.

    return changed


__all__ = [
    "Delta",
    "find_cst_at_ast",
    "maybe_replace_doc_str_in_function_or_class",
]

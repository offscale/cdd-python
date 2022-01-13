"""
Utils for working with AST (builtin) and cdd's CST
"""

from collections import namedtuple
from copy import deepcopy
from enum import Enum
from operator import attrgetter, ne
from sys import stderr

from cdd.ast_utils import cmp_ast, get_doc_str
from cdd.cst_utils import FunctionDefinitionStart, TripleQuoted, UnchangingLine, ast2cst
from cdd.pure_utils import omit_whitespace, tab
from cdd.source_transformer import to_code


def debug_doctrans(changed, affector, name, typ):
    """
    Print debug statement if changed is not nop

    :param changed: Delta value indicating what changed (if anything)
    :type changed: ```Delta```

    :param affector: What is being changed
    :type affector: ```str```

    :param name: Name of what is being changed
    :type name: ```str```

    :param typ: AST type name of what is being changed
    :type typ: ```str```
    """
    if changed is not Delta.nop:
        print(
            "{changed!s}".format(changed=changed).ljust(20),
            "{affector}\t{typ}\t`{name}`".format(affector=affector, typ=typ, name=name),
            sep="",
        )


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
            # Extra precautions to ensure the wrong new_node is never replaced:
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
    new_doc_str = get_doc_str(node) or ""
    cur_node_after_func = (
        cst_list[cst_idx + 1]
        if cst_idx + 1 < len(cst_list)
        else UnchangingLine(0, 0, "")
    )
    existing_doc_str = (
        isinstance(cur_node_after_func, TripleQuoted) and cur_node_after_func.is_docstr
    )
    changed = Delta.nop
    if new_doc_str and not existing_doc_str:
        cst_list.insert(
            cst_idx + 1,
            TripleQuoted(
                is_double_q=True,
                is_docstr=True,
                value='\n{tab}"""{new_doc_str}"""'.format(
                    tab=tab, new_doc_str=new_doc_str
                ),
                line_no_start=cur_node_after_func.line_no_start,
                line_no_end=cur_node_after_func.line_no_end,
            ),
        )
        changed = Delta.added
    elif not new_doc_str and existing_doc_str:
        del cst_list[cst_idx + 1]
        changed = Delta.removed
    # elif not new_doc_str and not existing_doc_str: changed = Delta.nop
    elif new_doc_str and existing_doc_str:
        cur_doc_str_only = cur_node_after_func.value.strip()[3:-3]
        if ne(*map(omit_whitespace, (cur_doc_str_only, new_doc_str))):
            pre, _, post = cur_node_after_func.value.partition(cur_doc_str_only)
            cst_list[cst_idx + 1] = TripleQuoted(
                is_double_q=cst_list[cst_idx + 1].is_double_q,
                is_docstr=cst_list[cst_idx + 1].is_docstr,
                value="{pre}{new_doc_str}{post}".format(
                    pre=pre, new_doc_str=new_doc_str, post=post
                ),
                line_no_start=cur_node_after_func.line_no_start,
                line_no_end=cur_node_after_func.line_no_end,
            )
            changed = Delta.replaced
    if changed is not Delta.nop:
        debug_doctrans(changed, "docstr", node.name, type(node).__name__)
        # Subsequent `line_no` `start,end` lines are invalidated. It's necessary to link the CST and AST together.

    return changed


def maybe_replace_function_return_type(new_node, cur_ast_node, cst_idx, cst_list):
    """
    Maybe replace the function's return type

    :param new_node: AST function node
    :type new_node: ```Union[AsyncFunctionDef, FunctionDef]```

    :param cur_ast_node: AST function node of CST (with fake body)
    :type cur_ast_node: ```AST```

    :param cst_idx: Index of start of function/class in cst_list
    :type cst_idx: ```int```

    :param cst_list: List of `namedtuple`s with at least ("line_no_start", "line_no_end", "value") attributes
    :type cst_list: ```List[NamedTuple]```

    :return: Delta value indicating what changed (if anything)
    :rtype: ```Delta```
    """
    new_node = deepcopy(new_node)
    new_node.body = cur_ast_node.body
    value = None

    def remove_return_typ(statement):
        """
        Remove the return typ

        :param statement: The statement verbatim
        :type statement: ```str```

        :return: The new function prototype
        :rtype: ```str```
        """
        return "{type_less}:".format(
            type_less=statement[: statement.rfind("->")].rstrip()
        )

    def add_return_typ(statement):
        """
        Add the return typ

        :param statement: The statement verbatim
        :type statement: ```str```

        :return: The new function prototype
        :rtype: ```str```
        """
        pre, col, post = statement.rpartition(":")
        return "{pre} -> {return_typ}{col}{post}".format(
            pre=pre,
            return_typ=to_code(new_node.returns).rstrip("\n"),
            col=col,
            post=post,
        )

    if cmp_ast(cur_ast_node.returns, new_node.returns):
        changed = Delta.nop
    elif cur_ast_node.returns and new_node.returns:
        changed = Delta.replaced
        value = add_return_typ(remove_return_typ(cst_list[cst_idx].value))
    elif cur_ast_node.returns and not new_node.returns:
        changed = Delta.removed
        value = remove_return_typ(cst_list[cst_idx].value)
    else:  # not cur_ast_node.returns and new_node.returns:
        changed = Delta.added
        value = add_return_typ(cst_list[cst_idx].value)
    if value is not None:
        cst_list[cst_idx] = FunctionDefinitionStart(
            line_no_start=cst_list[cst_idx].line_no_start,
            line_no_end=cst_list[cst_idx].line_no_end,
            name=cst_list[cst_idx].name,
            value=value,
        )

    if changed is not Delta.nop:
        debug_doctrans(changed, "return_type", new_node.name, type(new_node).__name__)

    return changed


def maybe_replace_function_args(new_node, cur_ast_node, cst_idx, cst_list):
    """
    Maybe replace the doc_str of a function or class

    :param new_node: AST function node
    :type new_node: ```Union[AsyncFunctionDef, FunctionDef]```

    :param cur_ast_node: AST function node of CST (with fake body)
    :type cur_ast_node: ```AST```

    :param cst_idx: Index of start of function/class in cst_list
    :type cst_idx: ```int```

    :param cst_list: List of `namedtuple`s with at least ("line_no_start", "line_no_end", "value") attributes
    :type cst_list: ```List[NamedTuple]```

    :return: Delta value indicating what changed (if anything)
    :rtype: ```Delta```
    """
    new_node = deepcopy(new_node)
    new_node.body = cur_ast_node.body
    Arg = namedtuple("Arg", ("annotation", "arg"))
    if cmp_ast(cur_ast_node.args, new_node.args):
        changed = Delta.nop
    else:
        new_args, cur_args = map(
            lambda args: tuple(map(lambda arg: Arg(arg.annotation, arg.arg), args)),
            map(attrgetter("args.args"), (new_node, cur_ast_node)),
        )

        changed = Delta.nop  # Never nop in this branch
        for i in range(len(cur_args)):
            if cur_args[i].annotation != new_args[i].annotation:
                # Approximation, obviously you could have intermixed annotation and to-be (un)annotated
                if cur_args[i].annotation is None:
                    changed = Delta.added
                elif new_args[i].annotation is None:
                    changed = Delta.removed
                else:
                    changed = Delta.replaced
                break

        pre, returning, post = cst_list[cst_idx].value.rpartition("->")
        left_parens, right_parens = 0, 0
        start_idx, end_idx = pre.rfind("("), pre.rfind(")")
        for start_idx in range(end_idx, 0, -1):
            if pre[start_idx] == ")":
                right_parens += 1
            elif pre[start_idx] == "(":
                left_parens += 1

            if right_parens == left_parens and right_parens > 0:
                break

        cst_list[cst_idx] = FunctionDefinitionStart(
            line_no_start=cst_list[cst_idx].line_no_start,
            line_no_end=cst_list[cst_idx].line_no_end,
            name=cst_list[cst_idx].name,
            # TODO: Handle comments in the middle of args, and match whitespace, and maybe even limit line length
            value="{start}{args}{end}{returning}{post}".format(
                start=pre[: start_idx + 1],
                args=", ".join(
                    "{arg}{annotation}".format(
                        annotation=""
                        if annotation is None
                        else ": {annotation_unparsed}".format(
                            annotation_unparsed=to_code(annotation).rstrip("\n")
                        ),
                        arg=arg,
                    )
                    for annotation, arg in new_args
                ),
                end=pre[end_idx:],
                returning=returning,
                post=post,
            ),
        )

    if changed is not Delta.nop:
        debug_doctrans(changed, "args", new_node.name, type(new_node).__name__)

    return changed


__all__ = [
    "Delta",
    "find_cst_at_ast",
    "maybe_replace_doc_str_in_function_or_class",
    "maybe_replace_function_args",
    "maybe_replace_function_return_type",
]

"""
ast_utils, bunch of helpers for converting input into ast.* input_str
"""
import ast
from ast import (
    AnnAssign,
    Name,
    Load,
    Store,
    Constant,
    Dict,
    Module,
    ClassDef,
    Subscript,
    Tuple,
    Expr,
    Call,
    Attribute,
    keyword,
    parse,
    walk,
    FunctionDef,
    Str,
    NameConstant,
    Assign,
    arg,
    Index,
)
from copy import deepcopy
from functools import partial

from doctrans.defaults_utils import extract_default
from doctrans.pure_utils import simple_types, rpartial, PY3_8


def param2ast(param):
    """
    Converts a param to an AnnAssign

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :return: AST node (AnnAssign)
    :rtype: ```AnnAssign```
    """
    if param["typ"] in simple_types:
        return AnnAssign(
            annotation=Name(ctx=Load(), id=param["typ"]),
            simple=1,
            target=Name(ctx=Store(), id=param["name"]),
            value=set_value(
                kind=None, value=param.get("default") or simple_types[param["typ"]]
            ),
        )
    elif param["typ"] == "dict" or param["typ"].startswith("*"):
        return AnnAssign(
            annotation=Name(ctx=Load(), id="dict"),
            simple=1,
            target=Name(ctx=Store(), id=param["name"]),
            value=Dict(keys=[], values=param.get("default", [])),
        )
    else:
        annotation = ast.parse(param["typ"]).body[0].value

        if param.get("default"):
            default_value = param["default"]
            parsed_default = (
                set_value(value=default_value)
                if isinstance(param["default"], (float, int, str))
                and not isinstance(param["default"], str)
                and not (
                    isinstance(param["default"], str)
                    and param["default"][0] + param["default"][-1]
                    in frozenset(("()", "[]", "{}"))
                )
                else ast.parse(param["default"])
            )

            value = (
                parsed_default.body[0].value
                if hasattr(parsed_default, "body")
                else parsed_default
                if "default" in param
                else Name(ctx=Load(), id=None)
            )
        else:
            value = set_value(None)

        return AnnAssign(
            annotation=annotation,
            simple=1,
            target=Name(ctx=Store(), id=param["name"]),
            value=value,
        )


def find_ast_type(node, node_name=None, of_type=ClassDef):
    """
    Converts an AST to an `ast.ClassDef`

    :param node: Any AST node
    :type node: ```AST```

    :param node_name: Name of AST node. If None, gives first found.
    :type node_name: ```Optional[str]```

    :param of_type: Of which type to find
    :type of_type: ```AST```

    :return: Found AST node
    :rtype: ```AST```
    """
    if isinstance(node, Module):
        it = filter(rpartial(isinstance, of_type), node.body)
        if node_name is not None:
            return next(
                filter(
                    lambda node: hasattr(node, "name") and node.name == node_name, it,
                )
            )
        matching_nodes = tuple(it)
        if len(matching_nodes) > 1:  # We could convert every one I guess?
            raise NotImplementedError()
        elif len(matching_nodes) > 0:
            return matching_nodes[0]
        else:
            raise TypeError("No {!r} in AST".format(type(of_type).__name__))
    elif isinstance(node, ast.AST):
        assert node_name is None or not hasattr(node, "name") or node.name == node_name
        return node
    else:
        raise NotImplementedError(type(node).__name__)


def param2argparse_param(param, emit_default_doc=True):
    """
    Converts a param to an Expr `argparse.add_argument` call

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: `argparse.add_argument` call—with arguments—as an AST node
    :rtype: ```Expr```
    """
    typ, choices, required = "str", None, True
    param.setdefault("typ", "Any")
    if param["typ"] in simple_types:
        typ = param["typ"]
    elif param["typ"] == "dict":
        typ = "loads"
        required = not param["name"].endswith("kwargs")
    elif param["typ"]:
        parsed_type = parse(param["typ"]).body[0]
        for node in walk(parsed_type):
            if isinstance(node, Tuple):
                maybe_choices = tuple(
                    get_value(elt)
                    for elt in node.elts
                    if isinstance(elt, (Constant, Str))
                )
                if len(maybe_choices) == len(node.elts):
                    choices = maybe_choices
            elif isinstance(node, Name):
                if node.id == "Optional":
                    required = False
                elif node.id in simple_types:
                    typ = node.id
                elif node.id not in frozenset(("Union",)):
                    typ = "globals().__getitem__"

    doc, _default = extract_default(param["doc"], emit_default_doc=emit_default_doc)
    default = param.get("default", _default)

    return Expr(
        value=Call(
            args=[set_value(kind=None, value="--{param[name]}".format(param=param))],
            func=Attribute(
                attr="add_argument",
                ctx=Load(),
                value=Name(ctx=Load(), id="argument_parser"),
            ),
            keywords=list(
                filter(
                    None,
                    (
                        keyword(
                            arg="type",
                            value=Attribute(
                                attr="__getitem__",
                                ctx=Load(),
                                value=Call(
                                    args=[],
                                    func=Name(ctx=Load(), id="globals"),
                                    keywords=[],
                                ),
                            )
                            if typ == "globals().__getitem__"
                            else Name(ctx=Load(), id=typ),
                        ),
                        choices
                        if choices is None
                        else keyword(
                            arg="choices",
                            value=Tuple(
                                ctx=Load(),
                                elts=[
                                    set_value(kind=None, value=choice)
                                    for choice in choices
                                ],
                            ),
                        ),
                        keyword(arg="help", value=set_value(kind=None, value=doc)),
                        keyword(
                            arg="required",
                            value=(
                                Constant(kind=None, value=True)
                                if PY3_8
                                else NameConstant(value=True)
                            ),
                        )
                        if required
                        else None,
                        default
                        if default is None
                        else keyword(
                            arg="default", value=set_value(kind=None, value=default)
                        ),
                    ),
                )
            ),
        )
    )


def argparse_param2param(argparse_param):
    """
    Converts a param to an Expr `argparse.add_argument` call

    :param argparse_param: argparse.add_argument` call—with arguments—as an AST node
    :type argparse_param: ```AST.arg```

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    return dict(
        name=argparse_param.arg,
        doc=argparse_param.type_comment,
        **{
            "typ": None
            if argparse_param.annotation is None
            else ast.parse(argparse_param.annotation)
        }
    )


# def needs_quoting(node):
#     """
#     Determine whether the input needs to be quoted
#
#     :param node: AST node
#     :type node: ```Union[AST, AnyStr]```
#
#     :returns: True if input needs quoting
#     :rtype: ```bool```
#     """
#     if isinstance(node, str):
#         if node == "str":
#             return True
#         node = ast.parse(node)
#     elif type(node).__name__ == "_SpecialForm":
#         return False
#     for _node in walk(node):
#         if hasattr(_node, "id") and _node.id == "str":
#             return True
#     return False


def get_function_type(function_def):
    """
    Get the type of the function

    :param function_def: AST node for function definition
    :type function_def: ```FunctionDef```

    :returns: None is a loose function (def f()`), others self-explanatory
    :rtype: ```Optional[Literal['self', 'cls']]```
    """
    assert isinstance(function_def, FunctionDef), "{typ} != FunctionDef".format(
        typ=type(function_def).__name__
    )
    if (
        not hasattr(function_def, "args")
        or function_def.args is None
        or len(function_def.args.args) == 0
    ):
        return None
    elif function_def.args.args[0].arg in frozenset(("self", "cls")):
        return function_def.args.args[0].arg
    return None


def get_value(node):
    """
    Get the value from a Constant or a Str… or anything with a `.value`

    :param node: AST node
    :type node: ```Union[Constant, Str]```

    :returns: Probably a string, but could be any constant value
    :rtype: ```Optional[Union[str, int, float, bool]]```
    """
    if isinstance(node, Str):
        return node.s
    elif isinstance(node, Constant) or hasattr(node, "value"):
        return node.value
    # elif isinstance(node, (Tuple, Name)):  # It used to be Index in Python < 3.9
    else:
        return node


def set_value(value, kind=None, unquote=True):
    """
    Creates a Constant or a Str depending on Python version in use

    :param value: AST node
    :type value: ```Any```

    :param kind: AST node
    :type kind: ```Optional[Any]```

    :param unquote: Whether to unquote the input value
    :type unquote: ```bool```

    :returns: Probably a string, but could be any constant value
    :rtype: ```Union[Constant, Str, NameConstant]```
    """
    if (
        value is not None
        and isinstance(value, str)
        and len(value) > 2
        and value[0] + value[-1] in frozenset(('""', "''"))
    ):
        value = value[1:-1]
    if not PY3_8:
        if isinstance(value, str):
            return Str(s=value)
        elif value is None:
            return NameConstant(value=value)
    return Constant(kind=kind, value=value)


def is_argparse_add_argument(node):
    """
    Checks if AST node is a call to `argument_parser.add_argument`

    :param node: AST node
    :type node: ```AST```

    :returns: Whether the input is the call to `argument_parser.add_argument`
    :rtype: ```bool```
    """
    return (
        isinstance(node, Expr)
        and isinstance(node.value, Call)
        and isinstance(node.value.func, Attribute)
        and isinstance(node.value.func.value, Name)
        and node.value.func.value.id == "argument_parser"
        and node.value.func.attr == "add_argument"
    )


def is_argparse_description(node):
    """
    Checks if AST node is `argument_parser.description`

    :param node: AST node
    :type node: ```AST```

    :returns: Whether the input is the call to `argument_parser.description`
    :rtype: ```bool```
    """
    return (
        isinstance(node, Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], Attribute)
        and node.targets[0].attr == "description"
        and isinstance(node.targets[0].value, Name)
        and node.targets[0].value.id == "argument_parser"
        and isinstance(node.value, (Constant, Str))
    )


def find_in_ast(search, node):
    """
    Find and return the param from within the value

    :param search: Location within AST of property.
       Can be top level like `['a']` for `a=5` or E.g., `['A', 'F']` for `class A: F`, `['f', 'g']` for `def f(g): ...`
    :type search: ```List[str]```

    :param node: AST node (must have a `body`)
    :type node: ```AST```

    :returns: AST node that was found, or None if nothing was found
    :rtype: ```Optional[AST]```
    """
    if len(search) == 0 or hasattr(node, "_location") and node._location == search:
        return node

    child_node, cursor, current_search = node, node.body, deepcopy(search)
    while len(current_search):
        query = current_search.pop(0)
        if (
            len(current_search) == 0
            and hasattr(child_node, "name")
            and child_node.name == query
        ):
            return child_node

        for child_node in cursor:
            if hasattr(child_node, "_location") and child_node._location == search:
                return child_node

            elif isinstance(child_node, FunctionDef):
                if len(current_search):
                    query = current_search.pop(0)
                _cursor = next(
                    filter(
                        lambda idx_arg: idx_arg[1].arg == query,
                        enumerate(child_node.args.args),
                    ),
                    None,
                )
                if _cursor is not None:
                    if len(child_node.args.defaults) > _cursor[0]:
                        setattr(
                            _cursor[1], "default", child_node.args.defaults[_cursor[0]]
                        )
                    cursor = _cursor[1]
                    if len(current_search) == 0:
                        return cursor
            elif (
                isinstance(child_node, AnnAssign)
                and isinstance(child_node.target, Name)
                and child_node.target.id == query
            ):
                return child_node
            elif hasattr(child_node, "name") and child_node.name == query:
                cursor = child_node.body
                break


def annotate_ancestry(node):
    """
    Look to your roots. Find the child; find the parent.
    Sets _location attribute to every child node.

    :param node: AST node. Will be annotated in-place.
    :type node: ```AST```
    """
    node._location = [node.name] if hasattr(node, "name") else []
    parent_location = []
    for _node in ast.walk(node):
        name = [_node.name] if hasattr(_node, "name") else []
        for child_node in ast.iter_child_nodes(_node):
            if hasattr(child_node, "name") and not isinstance(child_node, ast.alias):
                child_node._location = name + [child_node.name]
                parent_location = child_node._location
            elif isinstance(child_node, (Constant, Str)):
                child_node._location = parent_location + [get_value(child_node)]
            elif isinstance(child_node, Assign) and all(
                map(rpartial(isinstance, Name), child_node.targets)
            ):
                for target in child_node.targets:
                    child_node._location = name + [target.id]
            elif isinstance(child_node, AnnAssign) and isinstance(
                child_node.target, Name
            ):
                child_node._location = name + [child_node.target.id]

            if isinstance(child_node, FunctionDef):

                def set_index_and_location(idx_arg):
                    """
                    :param idx_arg: Index and Any; probably out of `enumerate`
                    :type idx_arg: ```Tuple[int, Any]```

                    :returns: Second element, with _idx set with value of first
                    :rtype: ```Any```
                    """
                    idx_arg[1]._idx = idx_arg[0]
                    idx_arg[1]._location = child_node._location + [idx_arg[1].arg]
                    return idx_arg[1]

                child_node.args.args = list(
                    map(
                        set_index_and_location,
                        enumerate(
                            child_node.args.args,
                            -1
                            if len(child_node.args.args) > 0
                            and child_node.args.args[0].arg
                            in frozenset(("self", "cls"))
                            else 0,
                        ),
                    )
                )


class RewriteAtQuery(ast.NodeTransformer):
    """
    Replace the node at query with given node

    :ivar search: Search query, e.g., ['node_name', 'function_name', 'arg_name']
    :ivar replacement_node: Node to replace this search
    :ivar replaced: whether a node has been replaced (only replaces first occurrence)
    """

    def __init__(self, search, replacement_node):
        """
        :param search: Search query, e.g., ['node_name', 'function_name', 'arg_name']
        :type search: ```List[str]```

        :param replacement_node: Node to replace this search
        :type replacement_node: ```ast.AST```
        """
        self.search = search
        self.replacement_node = replacement_node
        self.replaced = False

    def generic_visit(self, node):
        """
        visits the `AST`, if it's the right one, replace it

        :param node: The AST node
        :type node: ```AST```

        :returns: Potentially changed AST node
        :rtype: ```AST```
        """
        if (
            not self.replaced
            and hasattr(node, "_location")
            and node._location == self.search
        ):
            self.replaced = True
            return self.replacement_node
        else:
            return ast.NodeTransformer.generic_visit(self, node)

    def visit_FunctionDef(self, node):
        """
        visits the `FunctionDef`, if it's the right one, replace it

        :param node: FunctionDef
        :type node: ```FunctionDef```

        :returns: Potentially changed FunctionDef
        :rtype: ```FunctionDef```
        """
        if (
            not self.replaced
            and hasattr(node, "_location")
            and node._location == self.search[:-1]
        ):
            if isinstance(self.replacement_node, (AnnAssign, Assign)):
                # Set default
                if isinstance(self.replacement_node, AnnAssign):
                    idx = next(
                        (
                            arg._idx
                            for arg in node.args.args
                            if arg.arg == self.replacement_node.target.id
                            and hasattr(arg, "_idx")
                        ),
                        None,
                    )
                else:
                    idx = next(
                        filter(
                            None,
                            (
                                arg._idx if arg.arg == target.id else None
                                for target in self.replacement_node.targets
                                for arg in node.args.args
                                if hasattr(arg, "_idx")
                            ),
                        ),
                        None,
                    )
                    self.replacement_node = ast.arg(
                        arg=self.replacement_node.targets[0].id,
                        annotation=self.replacement_node.value,
                    )

                if idx is not None and len(node.args.defaults) > idx:
                    new_default = get_value(self.replacement_node)
                    if new_default is not None:
                        node.args.defaults[idx] = new_default

                self.replacement_node = emit_arg(self.replacement_node)

            assert isinstance(
                self.replacement_node, ast.arg
            ), "Expected ast.arg got {!r}".format(type(self.replacement_node).__name__)

            for idx in range(len(node.args.args)):
                if (
                    hasattr(node.args.args[idx], "_location")
                    and node.args.args[idx]._location == self.search
                ):
                    node.args.args[idx] = emit_arg(self.replacement_node)
                    self.replaced = True
                    break

        return node


def emit_ann_assign(node):
    """
    Produce an `AnnAssign` from the input

    :param node: AST node
    :type node: ```AST```

    :returns: Something which parses to the form of `a=5`
    :rtype: ```AnnAssign```
    """
    if isinstance(node, AnnAssign):
        return node
    elif isinstance(node, arg):
        return AnnAssign(
            annotation=node.annotation,
            simple=1,
            target=Name(ctx=Store(), id=node.arg),
            value=node.default if hasattr(node, "default") else None,
            lineno=None,
            col_offset=None,
            end_lineno=None,
            end_col_offset=None,
        )
    else:
        raise NotImplementedError(type(node).__name__)


def emit_arg(node):
    """
    Produce an `arg` from the input

    :param node: AST node
    :type node: ```AST```

    :returns: Something which parses to the form of `a=5`
    :rtype: ```arg```
    """
    if isinstance(node, ast.arg):
        return node
    elif isinstance(node, AnnAssign) and isinstance(node.target, Name):
        return arg(annotation=node.annotation, arg=node.target.id, type_comment=None,)
    elif (
        isinstance(node, Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], Name)
    ):
        return arg(annotation=None, arg=node.targets[0].id, type_comment=None,)
    else:
        raise NotImplementedError(type(node).__name__)


def it2literal(it):
    """
    Convert a collection of constants into a type annotation

    :param it: collection of constants
    :type it: ```Union[Tuple[Union[str, int, float], ...], List[Union[str, int, float], ...]]```

    :return: Subscript Literal for annotation
    :rtype: ```Subscript```
    """
    return Subscript(
        ctx=Load(),
        slice=Index(
            value=Tuple(ctx=Load(), elts=list(map(partial(Constant, kind=None), it)))
            if len(it) > 1
            else set_value(kind=None, value=it[0])
        ),
        value=Name(ctx=Load(), id="Literal"),
    )


__all__ = [
    "annotate_ancestry",
    "RewriteAtQuery",
    "param2ast",
    "find_ast_type",
    "param2argparse_param",
    "find_in_ast",
    "get_function_type",
    "emit_ann_assign",
    "emit_arg",
    "get_value",
    "set_value",
    "is_argparse_add_argument",
    "is_argparse_description",
    "it2literal",
]

"""
ast_utils, bunch of helpers for converting input into ast.* output
"""
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
)

from doctrans.defaults_utils import extract_default
from doctrans.pure_utils import simple_types, rpartial, PY3_8, strip_split


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
                kind=None, value=param.get("default", simple_types[param["typ"]])
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
        annotation = parse(param["typ"]).body[0].value

        if param.get("default") and not determine_quoting(annotation):
            value = (
                parse(param["default"]).body[0].value
                if "default" in param
                else Name(ctx=Load(), id=None)
            )
        else:
            value = set_value(kind=None, value=param.get("default"))

        return AnnAssign(
            annotation=annotation,
            simple=1,
            target=Name(ctx=Store(), id=param["name"]),
            value=value,
        )


def to_class_def(ast, class_name=None):
    """
    Converts an AST to an `ast.ClassDef`

    :param ast: Class AST or Module AST
    :type ast: ```Union[ast.Module, ast.ClassDef]```

    :param class_name: Name of `class`. If None, gives first found.
    :type class_name: ```Optional[str]```

    :return: ClassDef
    :rtype: ```ast.ClassDef```
    """
    if isinstance(ast, Module):
        classes_it = filter(rpartial(isinstance, ClassDef), ast.body)
        if class_name is not None:
            return next(filter(lambda node: node.name == class_name, classes_it,))
        classes = tuple(classes_it)
        if len(classes) > 1:  # We could convert every one I guess?
            raise NotImplementedError()
        elif len(classes) > 0:
            return classes[0]
        else:
            raise TypeError("No ClassDef in AST")
    elif isinstance(ast, ClassDef):
        assert class_name is None or ast.name == class_name
        return ast
    else:
        raise NotImplementedError(type(ast).__name__)


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
    if param["typ"] in simple_types:
        typ = param["typ"]
    elif param["typ"] == "dict":
        typ = "loads"
        required = not param["name"].endswith("kwargs")
    else:
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


def determine_quoting(node):
    """
    Determine whether the input needs to be quoted

    :param node: AST node
    :type node: ```Union[Subscript, Tuple, Name, Attribute]```

    :returns: True if input needs quoting
    :rtype: ```bool```
    """
    if isinstance(node, Subscript) and isinstance(node.value, Name):
        if node.value.id == "Optional":
            return determine_quoting(get_value(node.slice))
        elif node.value.id in frozenset(("Union", "Literal")):
            if all(isinstance(elt, Subscript) for elt in get_value(node.slice).elts):
                return any(determine_quoting(elt) for elt in get_value(node.slice).elts)
            return any(
                (
                    isinstance(elt, Constant)
                    and elt.kind is None
                    and isinstance(elt.value, str)
                    or (isinstance(elt, Str) or elt.id == "str")
                )
                for elt in get_value(node.slice).elts
            )
        elif node.value.id == "Tuple":
            return any(determine_quoting(elt) for elt in get_value(node.slice).elts)
        else:
            raise NotImplementedError(node.value.id)
    elif isinstance(node, Name):
        return node.id == "str"
    elif isinstance(node, Attribute):
        return determine_quoting(node.value)
    else:
        raise NotImplementedError(type(node).__name__)


def get_function_type(function):
    """
    Get the type of the function

    :param function: AST function node
    :type function: ```FunctionDef```

    :returns: None is a loose function (def f()`), others self-explanatory
    :rtype: ```Optional[Literal['self', 'cls']]```
    """
    assert isinstance(function, FunctionDef), "{typ} != FunctionDef".format(
        typ=type(function).__name__
    )
    if function.args is None or len(function.args.args) == 0:
        return None
    elif function.args.args[0].arg in frozenset(("self", "cls")):
        return function.args.args[0].arg
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
    elif isinstance(node, (Tuple, Name)):  # It used to be Index in Python < 3.9
        return node
    else:
        raise NotImplementedError(type(node).__name__)


def set_value(value, kind=None):
    """
    Creates a Constant or a Str depending on Python version in use

    :param value: AST node
    :type value: ```Any```

    :param kind: AST node
    :type kind: ```Optional[Any]```

    :returns: Probably a string, but could be any constant value
    :rtype: ```Union[Constant, Str, NameConstant]```
    """
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


def find_in_ast(query, node):
    """
    Find and return the param from within the value

    :param query: Location within AST of property.
       Can be top level like `a` for `a=5` or E.g., `A.F` for `class A: F`, `f.g` for `def f(g): pass`
    :type query: ```str```

    :param node: AST node (must have a `body`)
    :type node: ```AST```

    :returns: AST node that was found, or None if nothing was found
    :rtype: ```Optional[AST]```
    """
    query_l = list(strip_split(query, "."))
    cursor = node.body
    while len(query_l):
        search = query_l.pop(0)
        if len(query_l) == 0 and hasattr(node, "name") and node.name == search:
            return node

        for node in cursor:
            if isinstance(node, FunctionDef):
                if len(query_l):
                    search = query_l.pop(0)
                _cursor = next(
                    filter(lambda arg: arg.arg == search, node.args.args), None
                )
                if _cursor is not None:
                    cursor = _cursor
                    if len(query_l) == 0:
                        return _cursor
            elif (
                isinstance(node, AnnAssign)
                and isinstance(node.target, Name)
                and node.target.id == search
            ):
                return node
            elif hasattr(node, "name") and node.name == search:
                cursor = node.body
                break

    return None


__all__ = [
    "param2ast",
    "to_class_def",
    "param2argparse_param",
    "determine_quoting",
    "find_in_ast",
    "get_function_type",
    "get_value",
    "set_value",
    "is_argparse_add_argument",
    "is_argparse_description",
]

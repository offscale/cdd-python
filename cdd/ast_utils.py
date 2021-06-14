"""
ast_utils, bunch of helpers for converting input into ast.* input_str
"""

import ast
import pickle
from ast import (
    AST,
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    Attribute,
    Call,
    ClassDef,
    Constant,
    Dict,
    Expr,
    FunctionDef,
    Import,
    ImportFrom,
    Index,
    List,
    Load,
    Module,
    Name,
    NameConstant,
    NodeTransformer,
    Num,
    Store,
    Str,
    Subscript,
    Tuple,
    UnaryOp,
    alias,
    iter_child_nodes,
    keyword,
    walk,
)
from contextlib import suppress
from copy import deepcopy
from importlib import import_module
from inspect import isclass, isfunction
from itertools import chain, filterfalse
from json import dumps
from operator import attrgetter, inv, neg, not_, pos
from sys import version_info

from yaml import safe_dump_all

from cdd.defaults_utils import extract_default, needs_quoting
from cdd.pure_utils import (
    PY_GTE_3_8,
    PY_GTE_3_9,
    code_quoted,
    fill,
    identity,
    none_types,
    paren_wrap_code,
    quote,
    rpartial,
    simple_types,
)

# Was `"globals().__getitem__"`; this type is used for `Any` and any other unhandled

FALLBACK_TYP = "str"

# Was `Attribute(Call(args=[], func=Name("globals", Load()), keywords=[], expr=None, expr_func=None,),
#                "__getitem__", Load(),)`; this type is used for `Any` and any other unhandled (for argparse `type=`)
FALLBACK_ARGPARSE_TYP = Name(
    "str",
    Load(),
)


def param2ast(param):
    """
    Converts a param to an AnnAssign

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :returns: AST node for assignment
    :rtype: ```Union[AnnAssign, Assign]```
    """
    name, _param = param
    del param
    if _param.get("typ") is None and "default" in _param and "[" not in _param:
        _param["typ"] = type(_param["default"]).__name__
    if "default" in _param:
        if isinstance(_param["default"], (Constant, Str)):
            _param["default"] = get_value(_param["default"])
            if _param["default"] in none_types:
                _param["default"] = None
            if _param["typ"] in frozenset(("Constant", "Str", "NamedConstant")):
                _param["typ"] = "object"
        elif _param["default"] == NoneStr:
            _param["default"] = None
    if _param.get("typ") is None:
        return AnnAssign(
            annotation=Name("object", Load()),
            simple=1,
            target=Name(name, Store()),
            value=set_value(_param.get("default")),
            expr=None,
            expr_target=None,
            expr_annotation=None,
        )
        # return Assign(
        #     annotation=None,
        #     simple=1,
        #     targets=[Name(name, Store())],
        #     value=set_value(_param.get("default")),
        #     expr=None,
        #     expr_target=None,
        #     expr_annotation=None,
        #     **maybe_type_comment
        # )
    elif needs_quoting(_param["typ"]):
        return AnnAssign(
            annotation=Name(_param["typ"], Load())
            if _param["typ"] in simple_types
            else get_value(ast.parse(_param["typ"]).body[0]),
            simple=1,
            target=Name(name, Store()),
            value=set_value(
                quote(_param["default"])
                if _param.get("default")
                else simple_types.get(_param["typ"])
            ),
            expr=None,
            expr_target=None,
            expr_annotation=None,
        )
    elif _param["typ"] in simple_types:
        return AnnAssign(
            annotation=Name(_param["typ"], Load()),
            simple=1,
            target=Name(name, Store()),
            value=set_value(
                None
                if _param.get("default") == NoneStr
                else (_param.get("default") or simple_types[_param["typ"]])
            ),
            expr=None,
            expr_target=None,
            expr_annotation=None,
        )
    elif _param["typ"] == "dict" or _param["typ"].startswith("*"):
        return AnnAssign(
            annotation=set_slice(Name("dict", Load())),
            simple=1,
            target=Name(name, Store()),
            value=Dict(keys=[], values=_param.get("default", []), expr=None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
        )
    else:
        return _generic_param2ast((name, _param))


def _generic_param2ast(param):
    """
    Internal function to turn a param into an `AnnAssign`.
    Expected to be used only inside `param2ast`.

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :returns: AST node for assignment
    :rtype: ```AnnAssign```
    """
    name, _param = param
    del param
    from cdd.emitter_utils import ast_parse_fix

    annotation = ast_parse_fix(_param["typ"])
    value = set_value(None)
    if "default" in _param:
        if not code_quoted(_param["default"]) or _param["default"][
            3:-3
        ] not in frozenset(("None", "(None)")):
            try:
                parsed_default = (
                    set_value(_param["default"])
                    if (
                        _param["default"] is None
                        or isinstance(_param["default"], (float, int, str))
                    )
                    and not isinstance(_param["default"], str)
                    and not (
                        isinstance(_param["default"], str)
                        and _param["default"][0] + _param["default"][-1]
                        in frozenset(("()", "[]", "{}"))
                    )
                    else ast.parse(_param["default"])
                )
            except (SyntaxError, TypeError):
                parsed_default = set_value(
                    _param["default"]
                    if code_quoted(_param["default"])
                    else "```{default}```".format(default=_param["default"])
                )

            value = (
                parsed_default.body[0].value
                if hasattr(parsed_default, "body")
                else parsed_default
                if "default" in _param
                else set_value(None)
            )
        # else:
        #     value = set_value(None)
    return AnnAssign(
        annotation=annotation,
        simple=1,
        target=Name(name, Store()),
        value=value,
        expr=None,
        expr_target=None,
        expr_annotation=None,
    )


def find_ast_type(node, node_name=None, of_type=ClassDef):
    """
    Finds first AST node of the given type and possibly name

    :param node: Any AST node
    :type node: ```AST```

    :param node_name: Name of AST node. If None, gives first found.
    :type node_name: ```Optional[str]```

    :param of_type: Of which type to find
    :type of_type: ```AST```

    :returns: Found AST node
    :rtype: ```AST```
    """
    if isinstance(node, Module):
        it = filter(rpartial(isinstance, of_type), node.body)
        if node_name is not None:
            return next(
                filter(
                    lambda e: hasattr(e, "name") and e.name == node_name,
                    it,
                )
            )
        matching_nodes = tuple(it)
        if len(matching_nodes) > 1:  # We could convert every one I guess?
            raise NotImplementedError()
        elif matching_nodes:
            return matching_nodes[0]
        else:
            raise TypeError(
                "No {type_name!r} in AST".format(type_name=type(of_type).__name__)
            )
    elif isinstance(node, AST):
        assert node_name is None or not hasattr(node, "name") or node.name == node_name
        return node
    else:
        raise NotImplementedError(type(node).__name__)


def param2argparse_param(param, word_wrap=True, emit_default_doc=True):
    """
    Converts a param to an Expr `argparse.add_argument` call

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, Dict[str, Any]]```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :returns: `argparse.add_argument` call—with arguments—as an AST node
    :rtype: ```Expr```
    """
    name, _param = param
    del param
    typ, choices, required, action = (
        "str",
        None,
        _param.get("default") is not None,
        None,
    )
    _param.setdefault("typ", "Any")
    action, choices, required, typ, (name, _param) = _resolve_arg(
        action, choices, (name, _param), required, typ
    )
    # is_kwarg = param[0].endswith("kwargs")

    _param.setdefault("doc", "")
    doc, _default = extract_default(_param["doc"], emit_default_doc=emit_default_doc)
    _action, default, _required, _typ = infer_type_and_default(
        action,
        _param.get("default", _default),
        typ,
        required=required
        # _default, _param, action, required, typ#
    )
    if default is None and _param.get("default") == NoneStr:
        required = False
    if _action:
        action = _action
    if _typ is not None:
        typ = _typ
    if typ == "pickle.loads":
        required = False
    elif typ == "str" and action is None:
        typ = None  # Because `str` is default anyway

    return Expr(
        Call(
            args=[set_value("--{name}".format(name=name))],
            func=Attribute(
                Name("argument_parser", Load()),
                "add_argument",
                Load(),
            ),
            keywords=list(
                filter(
                    None,
                    (
                        typ
                        if typ is None
                        else keyword(
                            arg="type",
                            value=FALLBACK_ARGPARSE_TYP
                            if typ == "globals().__getitem__"
                            else Name(typ, Load()),
                            identifier=None,
                        ),
                        choices
                        if choices is None
                        else keyword(
                            arg="choices",
                            value=Tuple(
                                ctx=Load(),
                                elts=list(map(set_value, choices)),
                                expr=None,
                            ),
                            identifier=None,
                        ),
                        action
                        if action is None
                        else keyword(
                            arg="action",
                            value=set_value(action),
                            identifier=None,
                        ),
                        keyword(
                            arg="help",
                            value=set_value((fill if word_wrap else identity)(doc)),
                            identifier=None,
                        )
                        if doc
                        else None,
                        keyword(
                            arg="required",
                            value=set_value(True),
                            identifier=None,
                        )
                        if required is True
                        else None,
                        default
                        if default is None
                        else keyword(
                            arg="default",
                            value=set_value(default),
                            identifier=None,
                        ),
                    ),
                )
            ),
            expr=None,
            expr_func=None,
        )
    )


def _resolve_arg(action, choices, param, required, typ):
    """
    Resolve the arg type, required status, and choices

    :param action: Name of the action
    :type action: ```Optional[str]```

    :param choices: A container of values that should be allowed.
    :type choices: ```Optional[List[str]]```

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param required: Whether to require the argument
    :type required: ```bool```

    :param typ: The type of the argument
    :type typ: ```Optional[str]```

    :returns: action, choices, required, typ, (Name, dict with keys: 'typ', 'doc', 'default')
    :rtype: ```Tuple[Optional[str], Optional[List[str]], bool, Optional[str], Tuple[str, dict]]```
    """
    (name, _param), _required = param, None
    del param
    if isinstance(_param["typ"], str) and _param["typ"].startswith("<class '"):
        _param["typ"] = _param["typ"][len("<class '") : -len("'>")]
    if _param["typ"] in simple_types:
        typ = _param["typ"]
    elif _param["typ"] == "dict" or name.endswith("kwargs"):
        typ, required = "loads", not name.endswith("kwargs")
    elif _param["typ"]:
        from cdd.emitter_utils import ast_parse_fix

        parsed_type = ast_parse_fix(_param["typ"])
        for node in walk(parsed_type):
            _required, action, choices, typ = _parse_node_for_arg(
                _required, action, choices, node, typ
            )
    if _required is None and (typ or "").lower() in frozenset(
        ("str", "complex", "int", "float", "anystr", "list", "tuple", "dict")
    ):
        _required = True
    return (
        action,
        choices,
        required if _required is None else _required,
        typ,
        (name, _param),
    )


def _parse_node_for_arg(_required, action, choices, node, typ):
    """
    Resolve the arg type, required status, and choices

    :param _required: Whether to require the argument
    :type _required: ```bool```

    :param action: Name of the action
    :type action: ```Optional[str]```

    :param choices: A container of values that should be allowed.
    :type choices: ```Optional[List[str]]```

    :param node: AST node
    :type node: ```ast.AST```

    :param typ: The type of the argument
    :type typ: ```Optional[str]```

    :returns: _required, action, choices, typ
    :rtype: ```Tuple[bool, Optional[str], Optional[List[str]], Optional[str]]```
    """
    if isinstance(node, Tuple):
        maybe_choices = tuple(
            get_value(elt) for elt in node.elts if isinstance(elt, (Constant, Str))
        )
        if len(maybe_choices) == len(node.elts):
            choices = maybe_choices
    elif isinstance(node, Name):
        if node.id == "Optional":
            _required = False
        elif node.id in simple_types:
            typ = node.id
        elif node.id not in frozenset(("Union",)):
            typ = FALLBACK_TYP
        if node.id == "List":
            action = "append"
    return _required, action, choices, typ


def func_arg2param(func_arg, default=None):
    """
    Converts a function argument to a param tuple

    :param func_arg: Function argument
    :type func_arg: ```ast.arg```

    :param default: The default value, if None isn't added to returned dict
    :type default: ```Optional[Any]```

    :returns: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    return func_arg.arg, dict(
        doc=getattr(func_arg, "type_comment", None),
        **dict(
            typ=None
            if func_arg.annotation is None
            else _to_code(func_arg.annotation).rstrip("\n"),
            **({} if default is None else {"default": default})
        )
    )


def get_function_type(function_def):
    """
    Get the type of the function

    :param function_def: AST node for function definition
    :type function_def: ```FunctionDef```

    :returns: Type of target, static is static or global method, others just become first arg
    :rtype: ```Literal['self', 'cls', 'static']```
    """
    assert isinstance(function_def, FunctionDef), "{typ} != FunctionDef".format(
        typ=type(function_def).__name__
    )
    if (
        not hasattr(function_def, "args")
        or function_def.args is None
        or not function_def.args.args
    ):
        return "static"
    elif function_def.args.args[0].arg in frozenset(("self", "cls")):
        return function_def.args.args[0].arg
    return "static"


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
    elif isinstance(node, Num):
        return node.n
    elif isinstance(node, Constant) or hasattr(node, "value"):
        value = node.value
        return NoneStr if value is None else value
    # elif isinstance(node, (Tuple, Name)):  # It used to be Index in Python < 3.9
    elif isinstance(node, UnaryOp) and isinstance(
        node.operand, (Str, Num, Constant, NameConstant)
    ):
        return {"USub": neg, "UAdd": pos, "not_": not_, "Invert": inv}[
            type(node.op).__name__
        ](get_value(node.operand))
    elif isinstance(node, Name):
        return node.id
    else:
        return node


def get_at_root(node, types):
    """
    Get the imports from a node

    :param node: AST node with .body, probably an `ast.Module`
    :type node: ```AST```

    :param types: The types to search for (uses in an `isinstance` check)
    :type types: ```Tuple[str,...]````

    :returns: List of imports. Doesn't handle those within a try/except, condition, or not in root scope
    :rtype: ```List[Union[]]```
    """
    assert hasattr(node, "body") and isinstance(node.body, (list, tuple))
    return list(filter(rpartial(isinstance, types), node.body))


def set_value(value, kind=None):
    """
    Creates a `Constant` on Python >= 3.8 otherwise more specific AST type

    :param value: AST node
    :type value: ```Any```

    :param kind: AST node
    :type kind: ```Optional[Any]```

    :returns: Probably a string, but could be any constant value
    :rtype: ```Union[Constant, Num, Str, NameConstant]```
    """
    if (
        value is not None
        and isinstance(value, str)
        and len(value) > 2
        and value[0] + value[-1] in frozenset(('""', "''"))
    ):
        value = value[1:-1]
    return (
        Constant(kind=kind, value=value, constant_value=None, string=None)
        if PY_GTE_3_8
        else (
            Str(s=value, constant_value=None, string=None)
            if isinstance(value, str)
            else Num(n=value, constant_value=None, string=None)
            if not isinstance(value, bool) and isinstance(value, (int, float, complex))
            else NameConstant(value=value, constant_value=None, string=None)
        )
    )


def set_slice(node):
    """
    In Python 3.9 there's a new ast parser (PEG) that no longer wraps things in Index.
    This function handles this issue.

    :param node: An AST node
    :type node: ```ast.AST```

    :returns: Original node, possibly wrapped in an ```Index```
    :rtype: ```Union[ast.AST, Index]```
    """
    return node if PY_GTE_3_9 else Index(node)


def set_arg(arg, annotation=None):
    """
    In Python 3.8 `expr` and `type_comment` need to be set on arg.
    This function handles constructs an `ast.arg` handling that issue.

    :param arg: The argument name
    :type arg: ```Optional[str]```

    :param annotation: The argument's annotation
    :type annotation: ```Optional[ast.AST]```

    :returns: The constructed ```ast.arg```
    :rtype: ```ast.arg```
    """
    return ast.arg(
        arg=arg,
        annotation=annotation,
        **dict(expr=None, **maybe_type_comment) if PY_GTE_3_8 else {}
    )


def set_docstring(doc_str, empty, node):
    """
    Set docstring on node that can have a docstring. If doc_str is empty, the doc_str node is removed.

    :param doc_str: Docstring
    :type doc_str: ```Optional[str]```

    :param empty: Whether the doc_str is empty (micro-optimization)
    :type empty: ```bool```

    :param node: AST node to set the docstring on
    :type node: ```Union[Module, AsyncFunctionDef, FunctionDef, ClassDef]```
    """
    (
        node.body.__setitem__
        if isinstance(node.body[0], Expr)
        and isinstance(get_value(node.body[0].value), str)
        else node.body.insert
    )(
        0,
        Expr(set_value(doc_str)),
    )
    if empty or get_value(node.body[0].value).isspace():
        del node.body[0]


maybe_lineno = {"line_no": None} if PY_GTE_3_8 else {}
maybe_type_comment = {"type_comment": None} if PY_GTE_3_8 else {}


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
        and node.value.func.attr == "add_argument"
        and isinstance(node.value.func.value, Name)
        and node.value.func.value.id == "argument_parser"
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
    if not search or hasattr(node, "_location") and node._location == search:
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

    :returns: Annotated AST node; also `node` arg will be annotated in-place.
    :rtype: ```AST```
    """
    # print("annotating", getattr(node, "name", None))
    node._location = [node.name] if hasattr(node, "name") else []
    parent_location = []
    for _node in walk(node):
        name = [_node.name] if hasattr(_node, "name") else []
        for child_node in iter_child_nodes(_node):
            if hasattr(child_node, "name") and not isinstance(child_node, alias):
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

            if isinstance(child_node, (AsyncFunctionDef, FunctionDef)):

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

                child_node.args.kwonlyargs = list(
                    map(
                        set_index_and_location,
                        enumerate(
                            child_node.args.kwonlyargs,
                            0,
                        ),
                    )
                )
    return node


class RewriteAtQuery(NodeTransformer):
    """
    Replace the node at query with given node

    :ivar search: Search query, e.g., ['node_name', 'function_name', 'arg_name']
    :ivar replacement_node: Node to replace this search
    :ivar replaced: Whether a node has been replaced (only replaces first occurrence)
    """

    def __init__(self, search, replacement_node):
        """
        :param search: Search query, e.g., ['node_name', 'function_name', 'arg_name']
        :type search: ```List[str]```

        :param replacement_node: Node to replace this search
        :type replacement_node: ```AST```
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
            return NodeTransformer.generic_visit(self, node)

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
                            _arg._idx
                            for _arg in node.args.args
                            if _arg.arg == self.replacement_node.target.id
                            and hasattr(_arg, "_idx")
                        ),
                        None,
                    )
                else:
                    idx = next(
                        filter(
                            None,
                            (
                                _arg._idx if _arg.arg == target.id else None
                                for target in self.replacement_node.targets
                                for _arg in node.args.args
                                if hasattr(_arg, "_idx")
                            ),
                        ),
                        None,
                    )
                    self.replacement_node = set_arg(
                        arg=self.replacement_node.targets[0].id,
                        annotation=self.replacement_node.value,
                    )

                if idx is not None and len(node.args.defaults) > idx:
                    new_default = get_value(self.replacement_node)
                    if new_default not in none_types:
                        node.args.defaults[idx] = new_default

                self.replacement_node = emit_arg(self.replacement_node)
            assert isinstance(
                self.replacement_node, ast.arg
            ), "Expected ast.arg got {replacement_node_name!r}".format(
                replacement_node_name=type(self.replacement_node).__name__
            )

            for arg_attr in "args", "kwonlyargs":
                arg_l = getattr(node.args, arg_attr)
                for idx in range(len(arg_l)):
                    if (
                        hasattr(arg_l[idx], "_location")
                        and arg_l[idx]._location == self.search
                    ):
                        arg_l[idx] = emit_arg(self.replacement_node)
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
    elif isinstance(node, ast.arg):
        return AnnAssign(
            annotation=node.annotation,
            simple=1,
            target=Name(node.arg, Store()),
            value=node.default if hasattr(node, "default") else None,
            lineno=None,
            col_offset=None,
            end_lineno=None,
            end_col_offset=None,
            expr=None,
            expr_target=None,
            expr_annotation=None,
        )
    else:
        raise NotImplementedError(type(node).__name__)


def emit_arg(node):
    """
    Produce an `arg` from the input

    :param node: AST node
    :type node: ```AST```

    :returns: Something which parses to the form of `a=5`
    :rtype: ```ast.arg```
    """
    if isinstance(node, ast.arg):
        return node
    elif isinstance(node, AnnAssign) and isinstance(node.target, Name):
        return set_arg(
            annotation=node.annotation,
            arg=node.target.id,
        )
    elif (
        isinstance(node, Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], Name)
    ):
        return set_arg(node.targets[0].id)
    else:
        raise NotImplementedError(type(node).__name__)


def it2literal(it):
    """
    Convert a collection of constants into a type annotation

    :param it: collection of constants
    :type it: ```Union[Tuple[Union[str, int, float], ...], List[Union[str, int, float], ...]]```

    :returns: Subscript Literal for annotation
    :rtype: ```Subscript```
    """
    return Subscript(
        Name("Literal", Load()),
        Index(
            value=Tuple(ctx=Load(), elts=list(map(set_value, it)), expr=None)
            if len(it) > 1
            else set_value(it[0])
        ),
        Load(),
    )


def infer_type_and_default(action, default, typ, required):
    """
    Infer the type string from the default and typ

    :param action: Name of the action
    :type action: ```Optional[str]```

    :param default: Initial default value
    :type default: ```Any```

    :param typ: The type of the argument
    :type typ: ```Optional[str]```

    :param required: Whether to require the argument
    :type required: ```bool```

    :returns: action (e.g., for `argparse.Action`), default, whether its required, inferred type str
    :rtype: ```Tuple[str, Any, bool, str]```
    """
    if code_quoted(default):
        return _infer_type_and_default_from_quoted(action, default, required, typ)
    elif type(default).__name__ in simple_types:
        typ = type(default).__name__
    elif isinstance(default, AST):
        action, default, required, typ = _parse_default_from_ast(
            action, default, required, typ
        )
    elif hasattr(default, "__str__") and str(default) == "<required parameter>":
        # Special type that PyTorch uses & defines
        action, default, required, typ = None, None, True, default.__class__.__name__
    elif isinstance(default, (list, tuple)):
        action, default, required, typ = _infer_type_and_default_for_list_or_tuple(
            action, default, required
        )
    elif isinstance(default, dict):
        typ = "loads"
        try:
            default = dumps(default)
        except TypeError:
            # YAML is more permissive though less concise, but `loads` from yaml is used so this works
            default = safe_dump_all(default)
    elif default is None:
        if "Optional" not in (typ or iter(())) and typ not in frozenset(
            ("Any", "pickle.loads", "loads")
        ):
            typ = None
    elif isinstance(default, type) or isfunction(default) or isclass(default):
        typ, default, required = "pickle.loads", pickle.dumps(default), False
    else:
        raise NotImplementedError(
            "Parsing type {default_type!s}, which contains {default!r}".format(
                default_type=type(default), default=default
            )
        )

    return action, default, required, typ


def _infer_type_and_default_for_list_or_tuple(action, default, required):
    """
    Infer the type string from the default and typ

    :param action: Name of the action
    :type action: ```Optional[str]```

    :param default: Initial default value
    :type default: ```Union[list, tuple]```

    :param required: Whether to require the argument
    :type required: ```bool```

    :returns: action (e.g., for `argparse.Action`), default, whether its required, inferred type str
    :rtype: ```Tuple[Union[Literal["append"], Literal["loads"]], Any, bool, str]```
    """
    if len(default) == 0:
        action, default, required, typ = "append", None, False, None
    elif len(default) == 1:
        action, default, required, typ = (
            "append",
            get_value(default[0]),
            False,
            type(default[0]).__name__,
        )
    else:
        typ, default = "loads", dumps(default)
    return action, default, required, typ


def _infer_type_and_default_from_quoted(action, default, required, typ):
    """
    Internal function to acquire (action, default, required, typ) from code-quoted default

    :param action: Name of the action
    :type action: ```Optional[str]```

    :param default: Initial default value
    :type default: ```ast.AST```

    :param required: Whether to require the argument
    :type required: ```bool```

    :param typ: The type of the argument
    :type typ: ```Optional[str]```

    :returns: action, default, required, typ
    :rtype: ```Tuple[Optional[str], Optional[List[str]], bool, Optional[str]]```
    """
    default = get_value(get_value(ast.parse(default.strip("`")).body[0]))
    # Sometimes `default` is a string like `(-1)`
    if type(default).__name__ not in frozenset(("complex", "int", "float")):
        with suppress(ValueError):
            default = ast.literal_eval(
                default.strip("`") if isinstance(default, str) else default
            )
    return infer_type_and_default(action, default, typ, required=required)


# Should `infer_type_and_default` be folded into this?
def _parse_default_from_ast(action, default, required, typ):
    """
    Internal function to acquire (action, default, required, typ) from AST types

    :param action: Name of the action
    :type action: ```Optional[str]```

    :param default: Initial default value
    :type default: ```ast.AST```

    :param required: Whether to require the argument
    :type required: ```bool```

    :param typ: The type of the argument
    :type typ: ```Optional[str]```

    :returns: action, default, required, typ
    :rtype: ```Tuple[Optional[str], Optional[List[str]], bool, Optional[str]]```
    """
    if isinstance(default, (Constant, Expr, Str, Num)):
        default = get_value(default)
    # if type(default).__name__ in simple_types:
    #    typ, default = type(default).__name__, default
    # else:
    if isinstance(default, (ast.Dict, ast.Tuple)):
        typ, default = "loads", _to_code(default).rstrip("\n")
    elif isinstance(default, (ast.List, ast.Tuple)):
        if len(default.elts) == 0:
            action, default, required, typ = "append", None, False, None
        elif len(default.elts) == 1:
            action, default = "append", get_value(default.elts[0])
            typ = type(default).__name__
        else:
            typ, default = "loads", _to_code(default).rstrip("\n")
    elif default is not None:
        typ, default = None, "```{default}```".format(
            default=paren_wrap_code(_to_code(default).rstrip("\n"))
        )
    # if required is None:
    #    required = "Optional" in (
    #        typ or iter(())
    #    )  # TODO: Work for `Union[None, AnyStr]` and `Any`
    return action, default, required, typ


def parse_to_scalar(node):
    """
    Parse the input to a scalar

    :param node: Any value
    :type node: ```Any```

    :returns: Scalar
    :rtype: ```Union[str, int, float, complex, None]```
    """
    if isinstance(node, (int, float, complex, str, type(None))):
        return node
    elif isinstance(node, (Constant, Expr, Str, Num)):
        return get_value(node)
    elif isinstance(node, ast.AST):
        return _to_code(node).rstrip("\n")
    else:
        raise NotImplementedError(
            "Converting this to scalar: {node!r}".format(node=node)
        )


# Construct from https://docs.sqlalchemy.org/en/13/core/type_basics.html#generic-types
column_type2typ = {
    "BigInteger": "int",
    "Boolean": "bool",
    "Float": "float",
    "Integer": "int",
    "JSON": "Optional[dict]",
    "String": "str",
    "Text": "str",
    "Unicode": "str",
    "UnicodeText": "str",
    "boolean": "bool",
    "dict": "dict",
    "float": "float",
    "int": "int",
    "str": "str",
}

typ2column_type = {v: k for k, v in column_type2typ.items()}
typ2column_type.update(
    {
        "bool": "Boolean",
        "dict": "JSON",
        "float": "Float",
        "int": "Integer",
        "str": "String",
    }
)

# https://json-schema.org/draft/2019-09/json-schema-core.html#rfc.section.4.2.1
json_type2typ = {
    "boolean": "bool",
    "string": "str",
    "object": "dict",
    "array": "list",
    "number": "int",  # <- Actually a problem, maybe `literal_eval` on default then `type()` or just `type(default)`?
    "null": "NoneType",
}
typ2json_type = {v: k for k, v in json_type2typ.items()}


# `to_code` doesn't work due to partially instantiated module
def _to_code(node):
    """
    Convert the AST input to Python source string

    :param node: AST node
    :type node: ```AST```

    :returns: Python source
    :rtype: ```str```
    """

    return (
        getattr(import_module("astor"), "to_source")
        if version_info[:2] < (3, 9)
        else getattr(import_module("ast"), "unparse")
    )(node)


class Undedined:
    """Null class"""


def node_to_dict(node):
    """
    Convert AST node to a dict

    :param node: AST node
    :type node: ```AST```

    :returns: Dict representation
    :rtype: ```dict```
    """
    return {
        attr: (
            lambda val: type(val)(map(get_value, val))
            if isinstance(val, (tuple, list))
            else get_value(val)
        )(getattr(node, attr))
        for attr in dir(node)
        if not attr.startswith("_")
        and not attr.endswith("lineno")
        and not attr.endswith("offset")
    }


def cmp_ast(node0, node1):
    """
    Compare if two nodes are equal. Verbatim stolen from `meta.asttools`.

    :param node0: First node
    :type node0: ```Union[AST, List[AST], Tuple[AST]]```

    :param node1: Second node
    :type node1: ```Union[AST, List[AST], Tuple[AST]]```

    :returns: Whether they are equal (recursive)
    :rtype: ```bool```
    """

    if type(node0) != type(node1):
        return False

    if isinstance(node0, (list, tuple)):
        if len(node0) != len(node1):
            return False

        for left, right in zip(node0, node1):
            if not cmp_ast(left, right):
                return False

    elif isinstance(node0, AST):
        for field in node0._fields:
            left = getattr(node0, field, Undedined)
            right = getattr(node1, field, Undedined)

            if not cmp_ast(left, right):
                return False
    else:
        return node0 == node1

    return True


def to_annotation(typ):
    """
    Converts the typ to an annotation

    :param typ: A string representation of the type to annotate with. Else return give identity.
    :type typ: ```Union[str, AST]```

    :returns: The annotation as a `Name` (usually) or else some more complex type
    :rtype: ```AST```
    """
    if isinstance(typ, AST):
        return typ
    return (
        Name(typ, Load())
        if typ in simple_types
        else get_value(
            (
                lambda parsed: parsed.body[0]
                if getattr(parsed, "body", None)
                else parsed
            )(ast.parse(typ))
        )
    )


def get_ass_where_name(node, name):
    """
    Find all `Assign`/`AnnAssign` in node body where id matches name

    :param node: AST node with a '.body'
    :type node: ```Union[Module, ClassDef, FunctionDef, If, Try, While, With, AsyncFor, AsyncFunctionDef, AsyncWith,
                         ExceptHandler, Expression, For, IfExp, Interactive, Lambda ]```

    :param name: Name to match (matches against `id` field of `Name`)
    :type name: ```str```

    :returns: Generator of all matches names (.value)
    :rtype: ```Generator[Any]```
    """
    return (
        get_value(node)
        for node in node.body
        if isinstance(node, Assign)
        and name
        in frozenset(
            map(attrgetter("id"), filter(rpartial(isinstance, Name), node.targets))
        )
        or isinstance(node, AnnAssign)
        and node.target == name
    )


def del_ass_where_name(node, name):
    """
    Delete all `Assign`/`AnnAssign` in node body where id matches name

    :param node: AST node with a '.body'
    :type node: ```Union[Module, ClassDef, FunctionDef, If, Try, While, With, AsyncFor, AsyncFunctionDef, AsyncWith,
                         ExceptHandler, Expression, For, IfExp, Interactive, Lambda ]```

    :param name: Name to match (matches against `id` field of `Name`)
    :type name: ```str```
    """
    node.body = list(
        filter(
            None,
            (
                None
                if isinstance(_node, Assign)
                and name
                in frozenset(
                    map(
                        attrgetter("id"),
                        filter(rpartial(isinstance, Name), _node.targets),
                    )
                )
                or isinstance(_node, AnnAssign)
                and _node.target == name
                else _node
                for _node in node.body
            ),
        )
    )


def merge_assignment_lists(node, name, unique_sort=True):
    """
    Merge multiple same-name lists within the body of a node into one, e.g., if you have multiple ```__all__```

    :param node: AST node with a '.body'
    :type node: ```Union[Module, ClassDef, FunctionDef, If, Try, While, With, AsyncFor, AsyncFunctionDef, AsyncWith,
                         ExceptHandler, Expression, For, IfExp, Interactive, Lambda ]```

    :param name: Name to match (matches against `id` field of `Name`)
    :type name: ```str```

    :param unique_sort: Whether to ensure its unique + sorted
    :type unique_sort: ```bool```
    """
    asses = tuple(get_ass_where_name(node, name))

    # if len(asses) < 2: return

    # Could extract the `AnnAssign` stuff I suppose…

    del_ass_where_name(node, name)
    elts = chain.from_iterable(
        map(
            attrgetter("elts"),
            asses,
        )
    )
    node.body.append(
        Assign(
            targets=[Name("__all__", Store())],
            value=List(
                ctx=Load(),
                elts=sorted(frozenset(elts), key=get_value)
                if unique_sort
                else list(elts),
                expr=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment
        )
    )


def merge_modules(mod0, mod1, remove_imports_from_second=True):
    """
    Merge modules (removing module docstring)

    :param mod0: Module
    :type mod0: ```Module```

    :param mod1: Module
    :type mod1: ```Module```

    :param remove_imports_from_second: Whether to remove global imports from second module
    :type remove_imports_from_second: ```bool```

    :returns: Merged module (copy)
    :rtype: ```Module```
    """
    mod1_body = (
        mod1.body[1:]
        if mod1.body and isinstance(mod1.body[0], (Str, Constant))
        else mod1.body
    )
    new_mod = deepcopy(mod0)

    new_mod.body += (
        list(
            filterfalse(
                rpartial(isinstance, (ImportFrom, Import)),
                mod1_body,
            )
        )
        if remove_imports_from_second
        else deepcopy(mod1_body)
    )

    return new_mod


NoneStr = "```(None)```" if PY_GTE_3_9 else "```None```"

__all__ = [
    "FALLBACK_ARGPARSE_TYP",
    "FALLBACK_TYP",
    "NoneStr",
    "RewriteAtQuery",
    "annotate_ancestry",
    "cmp_ast",
    "code_quoted",
    "emit_ann_assign",
    "emit_arg",
    "find_ast_type",
    "find_in_ast",
    "func_arg2param",
    "get_at_root",
    "get_function_type",
    "get_value",
    "is_argparse_add_argument",
    "is_argparse_description",
    "it2literal",
    "maybe_type_comment",
    "merge_assignment_lists",
    "merge_modules",
    "node_to_dict",
    "param2argparse_param",
    "param2ast",
    "parse_to_scalar",
    "set_arg",
    "set_docstring",
    "set_slice",
    "set_value",
    "to_annotation",
]

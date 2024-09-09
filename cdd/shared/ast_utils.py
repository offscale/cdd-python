"""
ast_utils, a bunch of helpers for converting input into ast.* input_str
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
    NodeTransformer,
    Set,
    Store,
    Subscript,
    Tuple,
    UnaryOp,
    alias,
    iter_child_nodes,
    keyword,
    walk,
)
from collections import deque, namedtuple
from collections.abc import __all__ as collections_abc__all__
from contextlib import suppress
from copy import deepcopy
from functools import partial
from importlib import import_module
from importlib.util import find_spec
from inspect import isclass, isfunction
from itertools import chain, filterfalse, groupby
from json import dumps
from operator import attrgetter, contains, inv, itemgetter, neg, not_, pos
from os import path
from typing import Callable, FrozenSet, Generator, MutableSet, Optional
from typing import Tuple as TTuple
from typing import __all__ as typing__all__

import cdd.shared.source_transformer
from cdd.shared.defaults_utils import extract_default, needs_quoting
from cdd.shared.pure_utils import (
    PY_GTE_3_8,
    PY_GTE_3_9,
    code_quoted,
    fill,
    find_module_filepath,
    identity,
    none_types,
    paren_wrap_code,
    quote,
    rpartial,
    simple_types,
)

safe_dump_all = (
    getattr(import_module("yaml"), "safe_dump_all")
    if find_spec("yaml") is not None
    else None
)

collections_abc___all__: FrozenSet = frozenset(collections_abc__all__)
del collections_abc__all__

pydantic___all__: FrozenSet = (
    frozenset(
        filterfalse(rpartial(str.startswith, "_"), dir(import_module("pydantic")))
    )
    if find_spec("pydantic") is not None
    else frozenset()
)

sqlalchemy___all__: FrozenSet = (
    frozenset(
        filterfalse(rpartial(str.startswith, "_"), dir(import_module("sqlalchemy")))
    )
    if find_spec("sqlalchemy") is not None
    # Some defaults for tests mostly; from 2.0.25
    else frozenset(
        (
            "ARRAY",
            "AdaptedConnection",
            "Alias",
            "AliasedReturnsRows",
            "Any",
            "AssertionPool",
            "AsyncAdaptedQueuePool",
            "BIGINT",
            "BINARY",
            "BLANK_SCHEMA",
            "BLOB",
            "BOOLEAN",
            "BaseDDLElement",
            "BaseRow",
            "BigInteger",
            "BinaryExpression",
            "BindParameter",
            "BindTyping",
            "Boolean",
            "BooleanClauseList",
            "CHAR",
            "CLOB",
            "CTE",
            "CacheKey",
            "Case",
            "Cast",
            "CheckConstraint",
            "ChunkedIteratorResult",
            "ClauseElement",
            "ClauseList",
            "CollectionAggregate",
            "Column",
            "ColumnClause",
            "ColumnCollection",
            "ColumnDefault",
            "ColumnElement",
            "ColumnExpressionArgument",
            "ColumnOperators",
            "Compiled",
            "CompoundSelect",
            "Computed",
            "Connection",
            "Constraint",
            "CreateEnginePlugin",
            "CursorResult",
            "DATE",
            "DATETIME",
            "DDL",
            "DDLElement",
            "DECIMAL",
            "DOUBLE",
            "DOUBLE_PRECISION",
            "Date",
            "DateTime",
            "DefaultClause",
            "Delete",
            "Dialect",
            "Double",
            "Engine",
            "Enum",
            "ExceptionContext",
            "Executable",
            "ExecutableDDLElement",
            "ExecutionContext",
            "Exists",
            "Extract",
            "FLOAT",
            "FallbackAsyncAdaptedQueuePool",
            "False_",
            "FetchedValue",
            "Float",
            "ForeignKey",
            "ForeignKeyConstraint",
            "FromClause",
            "FromGrouping",
            "FrozenResult",
            "Function",
            "FunctionElement",
            "FunctionFilter",
            "GenerativeSelect",
            "Grouping",
            "HasCTE",
            "HasPrefixes",
            "HasSuffixes",
            "INT",
            "INTEGER",
            "Identity",
            "Index",
            "Insert",
            "Inspector",
            "Integer",
            "Interval",
            "IteratorResult",
            "JSON",
            "Join",
            "LABEL_STYLE_DEFAULT",
            "LABEL_STYLE_DISAMBIGUATE_ONLY",
            "LABEL_STYLE_NONE",
            "LABEL_STYLE_TABLENAME_PLUS_COL",
            "Label",
            "LambdaElement",
            "LargeBinary",
            "Lateral",
            "MappingResult",
            "MergedResult",
            "MetaData",
            "NCHAR",
            "NUMERIC",
            "NVARCHAR",
            "NestedTransaction",
            "NotNullable",
            "Null",
            "NullPool",
            "Nullable",
            "Numeric",
            "Operators",
            "Over",
            "PickleType",
            "Pool",
            "PoolProxiedConnection",
            "PoolResetState",
            "PrimaryKeyConstraint",
            "QueuePool",
            "REAL",
            "ReleaseSavepointClause",
            "Result",
            "ResultProxy",
            "ReturnsRows",
            "RollbackToSavepointClause",
            "RootTransaction",
            "Row",
            "RowMapping",
            "SMALLINT",
            "SQLColumnExpression",
            "SavepointClause",
            "ScalarResult",
            "ScalarSelect",
            "Select",
            "SelectBase",
            "SelectLabelStyle",
            "Selectable",
            "Sequence",
            "SingletonThreadPool",
            "SmallInteger",
            "StatementLambdaElement",
            "StaticPool",
            "String",
            "Subquery",
            "TEXT",
            "TIME",
            "TIMESTAMP",
            "Table",
            "TableClause",
            "TableSample",
            "TableValuedAlias",
            "Text",
            "TextAsFrom",
            "TextClause",
            "TextualSelect",
            "Time",
            "Transaction",
            "True_",
            "TryCast",
            "Tuple",
            "TupleType",
            "TwoPhaseTransaction",
            "TypeClause",
            "TypeCoerce",
            "TypeCompiler",
            "TypeDecorator",
            "URL",
            "UUID",
            "UnaryExpression",
            "Unicode",
            "UnicodeText",
            "UniqueConstraint",
            "Update",
            "UpdateBase",
            "Uuid",
            "VARBINARY",
            "VARCHAR",
            "Values",
            "ValuesBase",
            "Visitable",
            "WithinGroup",
        )
    )
)

typing___all__: FrozenSet = frozenset(typing__all__)
del typing__all__

typing_extensions___all__: FrozenSet = (
    frozenset(getattr(import_module("typing_extensions"), "__all__"))
    if find_spec("typing_extensions") is not None
    else frozenset()
)

DEFAULT_MODULES_TO_ALL = (
    ("typing", typing___all__),
    ("typing_extensions", typing_extensions___all__),
    ("collections.abc", collections_abc___all__),
    ("sqlalchemy", sqlalchemy___all__),
)  # type: tuple[tuple[str, frozenset], ...]

DEFAULT_MODULES_TO_ALL_SQL_FIRST = (
    ("sqlalchemy", sqlalchemy___all__),
    ("typing", typing___all__),
    ("typing_extensions", typing_extensions___all__),
    ("collections.abc", collections_abc___all__),
)  # type: tuple[tuple[str, frozenset], ...]

# Was `"globals().__getitem__"`; this type is used for `Any` and any other unhandled

FALLBACK_TYP: str = "str"

# Was `Attribute(Call(args=[], func=Name("globals", Load(), lineno=None, col_offset=None),
#                     keywords=[], expr=None, expr_func=None,),
#                "__getitem__", Load(),)`; this type is used for `Any` and any other unhandled (for argparse `type=`)
FALLBACK_ARGPARSE_TYP = Name("str", Load(), lineno=None, col_offset=None)

if PY_GTE_3_8:
    from cdd.shared.pure_utils import FakeConstant as Str

    Bytes = NameConstant = Num = Str
else:
    from ast import Bytes, NameConstant, Num, Str


def Dict_to_dict(d):
    """
    Create a `dict` from a `Dict`

    :param d: ast.Dict
    :type d: ```Dict```

    :return: Python dictionary
    :rtype: ```dict```
    """
    return dict(zip(map(get_value, d.keys), map(get_value, d.values)))


def ast_elts_to_container(node, container):
    """
    Convert AST container to Python container

    :param node: AST node with elts attribute
    :type node: ```AST```

    :param container: Python container
    :type container: ```type```

    :return: Python container
    :rtype: ```instanceof container```
    """
    assert hasattr(node, "elts")
    return container(map(get_value, node.elts))


List_to_list = partial(ast_elts_to_container, container=list)
Tuple_to_tuple = partial(ast_elts_to_container, container=tuple)
Set_to_set = partial(ast_elts_to_container, container=set)


def ast_type_to_python_type(node):
    """
    Unparse AST type as Python type

    Implementation notes:
      - this focuses on 'evaluated scalars' that can be represented as JSON
      - think of this as a `get_value` alternative

    :param node: AST node
    :type node: ```Union[Num,Bytes,Str,Constant,Dict,Set,Tuple,List]```

    :rtype: Union[dict,str,int,float,complex,bytes,list,tuple,set]
    """
    assert isinstance(node, AST), "Expected `AST` got `{type_name}`".format(
        type_name=type(node).__name__
    )
    if isinstance(node, Num):
        return node.n
    elif isinstance(node, (Bytes, Str)):
        return node.s
    elif isinstance(node, Constant):
        return node.value
    elif isinstance(node, Dict):
        return Dict_to_dict(node)
    elif isinstance(node, Set):
        return Set_to_set(node)
    elif isinstance(node, Tuple):
        return Tuple_to_tuple(node)
    elif isinstance(node, List):
        return List_to_list(node)
    else:
        raise NotImplementedError(node)


def param2ast(param):
    """
    Converts a param to an AnnAssign

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :return: AST node for assignment
    :rtype: ```Union[AnnAssign, Assign]```
    """
    name, _param = param
    del param

    def get_default_val(val):
        """
        retrieve default val for application to `.value` of `Assign | AnnAssign`

        :param val: value to retrieve default val for
        :type val: ```Optional[str]```

        :return: default val for application to `.value` of `Assign | AnnAssign`
        :rtype: ```Optional[str]```
        """
        return None if val is None else set_value(None if val == NoneStr else val)

    if "default" in _param:
        if isinstance(_param["default"], (Constant, Str, NameConstant, Num)):
            _param["default"] = get_value(_param["default"])
        if _param.get("typ") is None and not getattr(
            _param["default"], "__contains__", iter(())
        )("["):
            _param["typ"] = (
                "Optional[Any]"
                if _param["default"] == NoneStr
                else type(_param["default"]).__name__
            )
        elif _param.get("typ") == "Str":
            _param["typ"] = "str"
        elif _param.get("typ") in frozenset(("Constant", "NameConstant", "Num")):
            _param["typ"] = "object"
    if "typ" in _param and needs_quoting(_param["typ"]):
        default = (
            _param.get("default")
            if _param.get("default") in (None, NoneStr)
            else quote(_param["default"])
        )
        return AnnAssign(
            annotation=(
                Name(_param["typ"], Load(), lineno=None, col_offset=None)
                if _param["typ"] in simple_types
                else get_value(ast.parse(_param["typ"]).body[0])
            ),
            simple=1,
            target=Name(name, Store(), lineno=None, col_offset=None),
            value=get_default_val(default),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        )
    if _param.get("typ") is None:
        return Assign(
            annotation=None,
            simple=1,
            targets=[Name(name, Store(), lineno=None, col_offset=None)],
            value=(lambda val: set_value(val) if val is None else val)(
                get_default_val(_param.get("default"))
            ),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            col_offset=None,
            **maybe_type_comment,
        )
    elif _param["typ"] in simple_types:
        return AnnAssign(
            annotation=Name(_param["typ"], Load(), lineno=None, col_offset=None),
            simple=1,
            target=Name(name, Store(), lineno=None, col_offset=None),
            value=get_default_val(_param.get("default")),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        )
    elif _param["typ"] == "dict" or _param["typ"].startswith("*"):
        return AnnAssign(
            annotation=set_slice(Name("dict", Load(), lineno=None, col_offset=None)),
            simple=1,
            target=Name(name, Store(), lineno=None, col_offset=None),
            value=Dict(keys=[], values=_param.get("default", []), expr=None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            col_offset=None,
            lineno=None,
        )
    else:
        return _generic_param2ast((name, _param))


def _generic_param2ast(param):
    """
    Internal function to turn a param into an `AnnAssign`.
    Expected to be used only inside `param2ast`.

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :return: AST node for assignment
    :rtype: ```AnnAssign```
    """
    name, _param = param
    del param
    from cdd.shared.emit.utils.emitter_utils import ast_parse_fix

    annotation = ast_parse_fix(_param["typ"])
    value = None
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
                else parsed_default if "default" in _param else None
            )
        else:
            value = set_value(None)
    return AnnAssign(
        annotation=annotation,
        simple=1,
        target=Name(name, Store(), lineno=None, col_offset=None),
        expr=None,
        expr_target=None,
        expr_annotation=None,
        col_offset=None,
        value=value,
        lineno=None,
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

    :return: Found AST node
    :rtype: ```AST```
    """
    if isinstance(node, Module):
        it: Optional[Generator[of_type]] = filter(
            rpartial(isinstance, of_type), node.body
        )
        if node_name is not None:
            return next(
                filter(
                    lambda e: hasattr(e, "name") and e.name == node_name,
                    it,
                )
            )
        matching_nodes = tuple(it)  # type: tuple[of_type, ...]
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
    :type param: ```tuple[str, Dict[str, Any]]```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: `argparse.add_argument` call—with arguments—as an AST node
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
        required=required,
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
                Name("argument_parser", Load(), lineno=None, col_offset=None),
                "add_argument",
                Load(),
                lineno=None,
                col_offset=None,
            ),
            keywords=list(
                filter(
                    None,
                    (
                        (
                            typ
                            if typ is None
                            else keyword(
                                arg="type",
                                value=(
                                    FALLBACK_ARGPARSE_TYP
                                    if typ == "globals().__getitem__"
                                    else Name(typ, Load(), lineno=None, col_offset=None)
                                ),
                                identifier=None,
                            )
                        ),
                        (
                            choices
                            if choices is None
                            else keyword(
                                arg="choices",
                                value=Tuple(
                                    ctx=Load(),
                                    elts=list(map(set_value, choices)),
                                    expr=None,
                                    lineno=None,
                                    col_offset=None,
                                ),
                                identifier=None,
                            )
                        ),
                        (
                            action
                            if action is None
                            else keyword(
                                arg="action",
                                value=set_value(action),
                                identifier=None,
                            )
                        ),
                        (
                            keyword(
                                arg="help",
                                value=set_value((fill if word_wrap else identity)(doc)),
                                identifier=None,
                            )
                            if doc
                            else None
                        ),
                        (
                            keyword(
                                arg="required",
                                value=set_value(True),
                                identifier=None,
                            )
                            if required is True
                            else None
                        ),
                        (
                            default
                            if default is None
                            else keyword(
                                arg="default",
                                value=set_value(default),
                                identifier=None,
                            )
                        ),
                    ),
                )
            ),
            expr=None,
            expr_func=None,
            lineno=None,
            col_offset=None,
        ),
        lineno=None,
        col_offset=None,
    )


def _resolve_arg(action, choices, param, required, typ):
    """
    Resolve the arg type, required status, and choices

    :param action: Name of the action
    :type action: ```Optional[str]```

    :param choices: A container of values that should be allowed.
    :type choices: ```Optional[List[str]]```

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :param required: Whether to require the argument
    :type required: ```bool```

    :param typ: The type of the argument
    :type typ: ```Optional[str]```

    :return: action, choices, required, typ, (Name, dict with keys: 'typ', 'doc', 'default')
    :rtype: ```tuple[Optional[str], Optional[list[str]], bool, Optional[str], tuple[str, dict]]```
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
        from cdd.shared.emit.utils.emitter_utils import ast_parse_fix

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

    :return: _required, action, choices, typ
    :rtype: ```tuple[bool, Optional[str], Optional[List[str]], Optional[str]]```
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

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```tuple[str, dict]```
    """
    return func_arg.arg, dict(
        doc=getattr(func_arg, "type_comment", None),
        **dict(
            typ=(
                None
                if func_arg.annotation is None
                else _to_code(func_arg.annotation).rstrip("\n")
            ),
            **({} if default is None else {"default": default}),
        ),
    )


def get_function_type(function_def):
    """
    Get the type of the function

    :param function_def: AST node for function definition
    :type function_def: ```FunctionDef```

    :return: Type of target, static is static or global method, others just become first arg
    :rtype: ```Literal['self', 'cls', 'static']```
    """
    assert isinstance(
        function_def, FunctionDef
    ), "Expected `FunctionDef` got `{type_name}`".format(
        type_name=type(function_def).__name__
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
    :type node: ```Union[Bytes, Constant, Name, Str, UnaryOp]```

    :return: Probably a string, but could be any constant value
    :rtype: ```Optional[Union[str, int, float, bool]]```
    """
    if isinstance(node, (Bytes, Str)):
        return node.s
    elif isinstance(node, Num):
        return node.n
    elif isinstance(node, Constant) or hasattr(node, "value"):
        value = node.value
        return NoneStr if value is None else value
    # elif isinstance(node, (Tuple, Name)):  # It used to be Index in Python < 3.9
    elif isinstance(node, UnaryOp) and isinstance(
        node.operand, (Str, Bytes, Num, Constant, NameConstant)
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
    :type types: ```tuple[str,...]````

    :return: List of imports. Doesn't handle those within a try/except, condition, or not in root scope
    :rtype: ```list[Union[]]```
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

    :return: Probably a string, but could be any constant value
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
            Str(s=value, constant_value=None, string=None, col_offset=None, lineno=None)
            if isinstance(value, str)
            else (
                Num(n=value, constant_value=None, string=None)
                if not isinstance(value, bool)
                and isinstance(value, (int, float, complex))
                else NameConstant(
                    value=value,
                    constant_value=None,
                    string=None,
                    lineno=None,
                    col_offset=None,
                )
            )
        )
    )


def set_slice(node):
    """
    In Python 3.9 there's a new ast parser (PEG) that no longer wraps things in Index.
    This function handles this issue.

    :param node: An AST node
    :type node: ```ast.AST```

    :return: Original node, possibly wrapped in an ```Index```
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

    :return: The constructed ```ast.arg```
    :rtype: ```ast.arg```
    """
    return ast.arg(
        arg=arg,
        annotation=annotation,
        identifier_arg=None,
        lineno=None,
        col_offset=None,
        **dict(expr=None, **maybe_type_comment) if PY_GTE_3_8 else {},
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
        Expr(set_value(doc_str), lineno=None, col_offset=None),
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

    :return: Whether the input is the call to `argument_parser.add_argument`
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

    :return: Whether the input is the call to `argument_parser.description`
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
    :type search: ```list[str]```

    :param node: AST node (must have a `body`)
    :type node: ```AST```

    :return: AST node that was found, or None if nothing was found
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


def annotate_ancestry(node, filename=None):
    """
    Look to your roots. Find the child; find the parent.
    Sets _location and __file__ attributes to every child node.

    :param node: AST node. Will be annotated in-place.
    :type node: ```AST```

    :param filename: Where the node was originally defined. Sets the `__file__` attribute to this.
    :type filename: ```Optional[str]```

    :return: Annotated AST node; also `node` arg will be annotated in-place.
    :rtype: ```AST```
    """
    # print("annotating", getattr(node, "name", None))
    node._location = [node.name] if hasattr(node, "name") else []
    if filename not in (None, "<unknown>") and isinstance(
        node, (AnnAssign, Assign, AsyncFunctionDef, ClassDef, FunctionDef, Module)
    ):
        setattr(node, "__file__", filename)
    parent_location = []
    for _node in walk(node):
        name = [_node.name] if hasattr(_node, "name") else []
        if filename not in (None, "<unknown>") and isinstance(
            _node, (AnnAssign, Assign, AsyncFunctionDef, ClassDef, FunctionDef, Module)
        ):
            setattr(_node, "__file__", filename)
        for child_node in iter_child_nodes(_node):
            if hasattr(child_node, "name") and not isinstance(child_node, alias):
                child_node._location = name + [child_node.name]
                parent_location = child_node._location
            elif isinstance(child_node, (Constant, Str)):
                child_node._location = parent_location + [get_value(child_node)]
            elif isinstance(child_node, Assign) and all(
                map(
                    rpartial(isinstance, Name),
                    child_node.targets,
                )
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
                    :type idx_arg: ```tuple[int, Any]```

                    :return: Second element, with _idx set with value of first
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
                            (
                                -1
                                if len(child_node.args.args) > 0
                                and child_node.args.args[0].arg
                                in frozenset(("self", "cls"))
                                else 0
                            ),
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
        :type search: ```list[str]```

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

        :return: Potentially changed AST node
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

        :return: Potentially changed FunctionDef
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
            ), "Expected `ast.arg` got `{type_name}`".format(
                type_name=type(self.replacement_node).__name__
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

    :return: Something which parses to the form of `a=5`
    :rtype: ```AnnAssign```
    """
    if isinstance(node, AnnAssign):
        return node
    elif isinstance(node, ast.arg):
        return AnnAssign(
            annotation=node.annotation,
            simple=1,
            target=Name(node.arg, Store(), lineno=None, col_offset=None),
            col_offset=getattr(node, "col_offset", None),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
            expr=None,
            expr_target=None,
            expr_annotation=None,
            lineno=None,
            value=node.default if getattr(node, "default", None) is not None else None,
        )
    else:
        raise NotImplementedError(type(node).__name__)


def emit_arg(node):
    """
    Produce an `arg` from the input

    :param node: AST node
    :type node: ```AST```

    :return: Something which parses to the form of `a=5`
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


def construct_module_with_symbols(module, symbols):
    """
    Create a module out of the input module that only contains nodes
    with a name contained in `symbols`

    :param module: Input module
    :type module: ```Module```

    :param symbols: Symbols
    :type symbols: ```FrozenSet[str]```

    :return: Module with only members whose `.name` is in `symbols`
    :rtype: ```Module```
    """
    return Module(
        body=list(
            filter(lambda node: getattr(node, "name", "") in symbols, module.body)
        ),
        type_ignores=[],
        stmt=None,
    )


def it2literal(it):
    """
    Convert a collection of constants into a type annotation

    :param it: collection of constants
    :type it: ```Union[tuple[Union[str, int, float], ...], list[Union[str, int, float], ...]]```

    :return: Subscript Literal for annotation
    :rtype: ```Subscript```
    """
    return Subscript(
        Name("Literal", Load(), lineno=None, col_offset=None),
        Index(
            value=(
                Tuple(
                    ctx=Load(),
                    elts=list(map(set_value, it)),
                    expr=None,
                    lineno=None,
                    col_offset=None,
                )
                if len(it) > 1
                else set_value(it[0])
            )
        ),
        Load(),
        lineno=None,
        col_offset=None,
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

    :return: action (e.g., for `argparse.Action`), default, whether its required, inferred type str
    :rtype: ```tuple[str, Any, bool, str]```
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
    elif isinstance(default, (list, tuple, set)):
        # `set` actually won't marshall back properly as JSON/YAML doesn't support this type :(
        action, default, required, typ = _infer_type_and_default_for_list_or_tuple(
            action, tuple(default) if isinstance(default, set) else default, required
        )
    elif isinstance(default, dict):
        typ = "loads"
        try:
            default = dumps(default)
        except TypeError:
            # YAML is more permissive though less concise, but `loads` from yaml is used so this works
            default = (
                dumps(default, ensure_ascii=False)
                if safe_dump_all is None
                else safe_dump_all(default)
            )
    elif default is None:
        if "Optional" not in (typ or iter(())) and typ not in frozenset(
            ("Any", "pickle.loads", "loads")
        ):
            typ = None
    elif any(
        (
            isinstance(default, type),
            isfunction(default),
            isclass(default),
            contains(
                frozenset(
                    map(
                        "tf.{}".format,
                        (
                            "qint16",
                            "qint16_ref",
                            "qint32",
                            "qint32_ref",
                            "qint8",
                            "qint8_ref",
                            "quint16",
                            "quint16_ref",
                            "quint8",
                            "quint8_ref",
                            "bfloat16",
                            "bool",
                            "complex128",
                            "complex64",
                            "double",
                            "float16",
                            "float32",
                            "float64",
                            "half",
                            "int16",
                            "int32",
                            "int64",
                            "int8",
                            "qint16",
                            "qint32",
                            "qint8",
                            "quint16",
                            "quint8",
                            "resource",
                            "string",
                            "uint16",
                            "uint32",
                            "uint64",
                            "uint8",
                            "variant",
                        ),
                    )
                ),
                repr(default),
            ),
        )
    ):
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

    :return: action (e.g., for `argparse.Action`), default, whether its required, inferred type str
    :rtype: ```tuple[Union[Literal["append"], Literal["loads"]], Any, bool, str]```
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

    :return: action, default, required, typ
    :rtype: ```tuple[Optional[str], Optional[List[str]], bool, Optional[str]]```
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

    :return: action, default, required, typ
    :rtype: ```tuple[Optional[str], Optional[List[str]], bool, Optional[str]]```
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

    :return: Scalar
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


# `to_code` doesn't work due to partially instantiated module
def _to_code(node):
    """
    Convert the AST input to Python source string

    :param node: AST node
    :type node: ```AST```

    :return: Python source
    :rtype: ```str```
    """

    return (
        getattr(import_module("ast"), "unparse")
        if PY_GTE_3_9
        else getattr(import_module("astor"), "to_source")
    )(node)


class Undefined:
    """Null class"""


def node_to_dict(node):
    """
    Convert AST node to a dict

    :param node: AST node
    :type node: ```AST```

    :return: Dict representation
    :rtype: ```dict```
    """
    return {
        attr: (
            lambda val: (
                type(val)(map(get_value, val))
                if isinstance(val, (tuple, list))
                else get_value(val)
            )
        )(getattr(node, attr))
        for attr in dir(node)
        if not attr.startswith("_") and not attr.endswith(("lineno", "offset"))
    }


def cmp_ast(node0, node1):
    """
    Compare if two nodes are equal. Verbatim stolen from `meta.asttools`.

    :param node0: First node
    :type node0: ```Union[AST, List[AST], Tuple[AST]]```

    :param node1: Second node
    :type node1: ```Union[AST, List[AST], Tuple[AST]]```

    :return: Whether they are equal (recursive)
    :rtype: ```bool```
    """

    if type(node0) is not type(node1):
        return False

    if isinstance(node0, (list, tuple)):
        if len(node0) != len(node1):
            return False

        for left, right in zip(node0, node1):
            if not cmp_ast(left, right):
                return False

    elif isinstance(node0, AST):
        for field in node0._fields:
            left = getattr(node0, field, Undefined)
            right = getattr(node1, field, Undefined)

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

    :return: The annotation as a `Name` (usually) or else some more complex type
    :rtype: ```AST```
    """
    if isinstance(typ, AST):
        return typ
    return (
        None
        if typ in none_types
        else (
            Name(typ, Load(), lineno=None, col_offset=None)
            if typ in simple_types
            else get_value(
                (
                    lambda parsed: (
                        parsed.body[0] if getattr(parsed, "body", None) else parsed
                    )
                )(ast.parse(typ))
            )
        )
    )


def to_type_comment(node):
    """
    Convert annotation to a type comment

    :param node: AST node with a '.annotation' or Name or str
    :type node: ```Union[Name, str, AnnAssign, arg, arguments]```

    :return: type_comment
    :rtype: ```str```
    """
    return (
        node.id
        if isinstance(node, Name)
        else (
            node
            if isinstance(node, str) or not hasattr(node, "annotation")
            else _to_code(node.annotation).strip()
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

    :return: Generator of all matches names (.value)
    :rtype: ```Generator[Union[Assign, AnnAssign]]```
    """
    return (
        get_value(_node)
        for _node in node.body
        if isinstance(_node, Assign)
        and name
        in frozenset(
            map(attrgetter("id"), filter(rpartial(isinstance, Name), _node.targets))
        )
        or isinstance(_node, AnnAssign)
        and get_value(_node.target) == name
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
                )
                for _node in node.body
            ),
        )
    )


def get_doc_str(node):
    """
    Similar to `ast.get_docstring` except never `clean`s and returns `None` on failure rather than raising

    :param node: AST node
    :type node: ```AST```

    :return: Docstring if found else None
    :rtype: ```Optional[str]```
    """
    if isinstance(node, (ClassDef, FunctionDef)) and isinstance(node.body[0], Expr):
        val = get_value(node.body[0])
        if isinstance(val, (Constant, Str)):
            return get_value(val)


def get_names(node):
    """
    Get name(s) from:
    - Assign targets
    - AnnAssign target
    - Function, AsyncFunction, ClassDef

    :param node: AST node
    :type node: ```Union[Assign, AnnAssign, Function, AsyncFunctionDef, ClassDef]```

    :return: All top-level symbols (except those within try/except and if/elif/else blocks)
    :rtype: ```Generator[str]```
    """
    if isinstance(node, Assign) and all(map(rpartial(isinstance, Name), node.targets)):
        return map(attrgetter("id"), node.targets)
    elif isinstance(node, AnnAssign) and isinstance(node.target, Name):
        return iter((node.target.id,))
    elif isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef)):
        return iter((node.name,))
    return iter(())


def module_to_all(module_or_filepath):
    """
    From input, create (("module_name", {"symbol0", "symbol1"}),)

    :param module_or_filepath: Module or filepath
    :type module_or_filepath: ```Union[str, Module]```

    :return: `__all__` from module (if present) else all symbols in module
    :rtype: ```List[str]```
    """
    assert isinstance(module_or_filepath, (str, Module))
    if not path.exists(module_or_filepath):
        module_or_filepath = find_module_filepath(module_or_filepath)

    with open(module_or_filepath, "rt") as f:
        module_or_filepath: Module = cdd.shared.source_transformer.ast_parse(
            f.read(), filename=module_or_filepath
        )

    module_or_filepath: Module = module_or_filepath

    # If exists, construct `list[str]` version of `__all__`
    all_ = list(
        map(
            get_value,
            chain.from_iterable(
                map(
                    attrgetter("elts"),
                    map(get_value, get_ass_where_name(module_or_filepath, "__all__")),
                )
            ),
        )
    )

    return (
        all_
        if all_
        else list(chain.from_iterable(map(get_names, module_or_filepath.body)))
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
    elts = map(
        get_value,
        chain.from_iterable(
            map(
                attrgetter("elts"),
                asses,
            )
        ),
    )
    node.body.append(
        Assign(
            targets=[Name("__all__", Store(), lineno=None, col_offset=None)],
            value=List(
                ctx=Load(),
                elts=list(
                    map(set_value, (sorted(frozenset(elts)) if unique_sort else elts))
                ),
                expr=None,
            ),
            expr=None,
            lineno=None,
            **maybe_type_comment,
        )
    )


def merge_modules(mod0, mod1, remove_imports_from_second=True, deduplicate_names=False):
    """
    Merge modules (removing module docstring from mod1)

    :param mod0: Module
    :type mod0: ```Module```

    :param mod1: Module
    :type mod1: ```Module```

    :param remove_imports_from_second: Whether to remove global imports from second module
    :type remove_imports_from_second: ```bool```

    :param deduplicate_names: Whether to deduplicate names; names can be function|class|AnnAssign|Assign name
    :type deduplicate_names: ```bool```

    :return: Merged module (copy)
    :rtype: ```Module```
    """
    mod1_body = (
        mod1.body[1:]
        if mod1.body and isinstance(get_value(mod1.body[0]), (Str, Constant))
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
    # if deduplicate_names:
    #
    #     def unique_nodes(node):
    #         """
    #         :param node: AST node
    #         :type node: ```AST```
    #
    #         :return: node if name is in `seen` set else None; with side-effect of adding to `seen`
    #         :rtype: ```bool```
    #         """
    #
    #         def side_effect_ret(name):
    #             """
    #             :param name: Name
    #             :type name: ```str```
    #
    #             :return: node if name is in `seen` set else None; with side-effect of adding to `seen`
    #             :rtype: ```bool```
    #             """
    #             if name in seen:
    #                 return None
    #             else:
    #                 seen.add(node.name)
    #                 return node
    #
    #         if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
    #             return side_effect_ret(node.name)
    #         elif isinstance(node, AnnAssign):
    #             return side_effect_ret(get_value(node.target))
    #         elif isinstance(node, Assign):
    #             return any(
    #                 filter(
    #                     lambda target: side_effect_ret(get_value(target)), node.targets
    #                 )
    #             )
    #         else:
    #             return node
    #
    #     seen = set()
    #     new_mod.body = list(filter(None, map(unique_nodes, new_mod.body)))

    return new_mod


def optimise_imports(imports):
    """
    Optimise imports involves:
    - Deduplication of module names
    - Deduplication of symbols import from module names

    For more complicated set-ups I recommend use of:
    - autoflake --remove-all-unused-imports
    - isort

    :param imports: `ImportFrom` nodes
    :type imports: ```Iterable[ImportFrom]```

    :return: `ImportFrom` nodes
    :rtype: ```list[ImportFrom]```
    """
    seen_pair = set()
    return [
        ImportFrom(
            module=module_name,
            level=sym[0].level,
            names=list(
                map(
                    lambda al: alias(
                        name=al.name,
                        asname=al.asname,
                        identifier=None,
                        identifier_name=None,
                    ),
                    sym[1],
                )
            ),
        )
        for module_name, symbols in map(
            lambda key_group: (
                key_group[0],
                filter(
                    None,
                    map(
                        lambda node: (
                            lambda filtered: (
                                (
                                    namedtuple("_", ("level",))(node.level),
                                    filtered,
                                )
                                if filtered
                                else None
                            )
                        )(
                            tuple(
                                filter(
                                    None,
                                    map(
                                        lambda name_asname_key: (
                                            None
                                            if name_asname_key.key in seen_pair
                                            else (
                                                seen_pair.add(name_asname_key.key)
                                                or namedtuple("_", ("name", "asname"))(
                                                    name_asname_key.name,
                                                    name_asname_key.asname,
                                                )
                                            )
                                        ),
                                        map(
                                            lambda _alias: namedtuple(
                                                "_", ("name", "asname", "key")
                                            )(
                                                _alias.name,
                                                _alias.asname,
                                                "{}{}{}".format(
                                                    key_group[0],
                                                    _alias.name,
                                                    _alias.asname,
                                                ),
                                            ),
                                            node.names,
                                        ),
                                    ),
                                )
                            )
                        ),
                        key_group[1],
                    ),
                ),
            ),
            groupby(
                sorted(
                    imports,
                    key=attrgetter("module"),
                ),
                key=attrgetter("module"),
            ),
        )
        for sym in symbols
    ]


def get_types(node):
    """
    :param node:
    :type node: ```AST|str```

    :rtype: ```Generator[str]```
    """
    if node is None:
        return iter(())
    elif isinstance(node, str):
        return iter((node,))
    elif isinstance(node, Name):
        return iter((node.id,))
    elif isinstance(node, Subscript):
        assert isinstance(node.value, Name), type(node.value)
        if isinstance(node.slice, Name):
            return iter((node.value.id, node.slice.id))
        elif isinstance(node.slice, Tuple):
            return chain.from_iterable(
                (
                    (node.value.id,),
                    (
                        iter(())
                        if node.value.id == "Literal"
                        else map(get_value, map(get_value, node.slice.elts))
                    ),
                )
            )


def infer_imports(module, modules_to_all=DEFAULT_MODULES_TO_ALL):
    """
    Infer imports from AST nodes (Name|.annotation|.type_comment); in order; these:
      - typing
      - typing_extensions
      - collections.abc
      - sqlalchemy
      - pydantic

    :param module: Module, ClassDef, FunctionDef, AsyncFunctionDef, Assign, AnnAssign
    :type module: ```Union[Module, ClassDef, FunctionDef, AsyncFunctionDef, Assign, AnnAssign]```

    :param modules_to_all: Tuple of module_name to __all__ of module; (str) to FrozenSet[str]
    :type modules_to_all: ```tuple[tuple[str, frozenset], ...]```

    :return: List of imports
    :rtype: ```Optional[Tuple[Union[Import, ImportFrom], ...]]```
    """
    if isinstance(module, (ClassDef, FunctionDef, AsyncFunctionDef, Assign, AnnAssign)):
        module: Module = Module(body=[module], type_ignores=[], stmt=None)
    assert isinstance(module, Module), "Expected `Module` got `{type_name}`".format(
        type_name=type(module).__name__
    )

    def node_to_importable_name(node):
        """
        :param node: AST node
        :type node: ```AST```

        :return: importable typename or None
        :rtype: ```Optional[str]```
        """
        if getattr(node, "type_comment", None) is not None:
            return (
                node.type_comment
                if node.type_comment in simple_types
                else get_value(
                    get_value(get_value(ast.parse(node.type_comment).body[0]))
                )
            )
        elif getattr(node, "annotation", None) is not None:
            node = node  # type: Union[AnnAssign, arg]
            return node.annotation  # cast(node, Union[AnnAssign, arg])
        elif isinstance(node, Name):
            return node.id
        else:
            return None

    _symbol_to_import: Callable[[str], Optional[TTuple[str, str]]] = partial(
        symbol_to_import, modules_to_all=modules_to_all
    )

    # Lots of room for optimisation here; but its probably NP-hard:
    imports = tuple(
        map(
            lambda mod_names: ImportFrom(
                module=mod_names[0],
                names=list(
                    map(
                        lambda name: alias(
                            name,
                            None,
                            identifier=None,
                            identifier_name=None,
                        ),
                        sorted(frozenset(map(itemgetter(0), mod_names[1]))),
                    ),
                ),
                level=0,
                identifier=None,
            ),
            groupby(
                sorted(
                    filter(
                        None,
                        map(
                            # Because there are duplicate names, centralise all import resolution here and order them
                            _symbol_to_import,
                            sorted(
                                frozenset(
                                    chain.from_iterable(
                                        filter(
                                            None,
                                            map(
                                                get_types,
                                                filter(
                                                    None,
                                                    map(
                                                        node_to_importable_name,
                                                        ast.walk(module),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                )
                            ),
                        ),
                    ),
                    key=itemgetter(1),
                ),
                key=itemgetter(1),
            ),
        )
    )  # type: tuple[ImportFrom, ...]

    # cdd.sqlalchemy.utils.parse_utils.imports_from(sqlalchemy_class_or_assigns)
    return imports if imports else None


def symbol_to_import(
    symbol,
    modules_to_all,
):
    """
    Resolve symbol to module

    :param symbol: symbol to look for within various modules
    :type symbol: ```str```

    :param modules_to_all: Tuple of module_name to __all__ of module; (str) to FrozenSet[str]
    :type modules_to_all: ```tuple[tuple[str, frozenset], ...]```

    :return: (symbol, module) if name in module else None
    :rtype: ```Optional[Tuple[str, str]]```
    """
    return next(
        ((symbol, module) for (module, all_) in modules_to_all if symbol in all_), None
    )


def deduplicate_sorted_imports(module):
    """
    Deduplicate sorted imports. NOTE: for a more extensive solution use isort or ruff.

    :param module: Module
    :type module: ```Module```

    :return: Module but with duplicate import entries in first import block removed
    :rtype: ```Module```
    """
    assert isinstance(module, Module), "Expected `Module` got `{}`".format(
        type(module).__name__
    )
    fst_import_idx: Optional[int] = next(
        map(
            itemgetter(0),
            filter(
                lambda idx_node: isinstance(idx_node[1], (ImportFrom, Import)),
                enumerate(module.body),
            ),
        ),
        None,
    )
    if fst_import_idx is None:
        return module
    lst_import_idx: Optional[int] = next(
        iter(
            deque(
                map(
                    itemgetter(0),
                    filter(
                        lambda idx_node: isinstance(idx_node[1], (ImportFrom, Import)),
                        enumerate(module.body, fst_import_idx),
                    ),
                ),
                maxlen=1,
            )
        ),
        None,
    )
    name_seen: MutableSet[str] = set()

    module.body = (
        module.body[:fst_import_idx]
        + list(
            filter(
                attrgetter("names"),
                (
                    # TODO: Infer `level`
                    ImportFrom(
                        module=name,
                        names=list(
                            filter(
                                lambda _alias: (
                                    lambda key: (
                                        False
                                        if key in name_seen
                                        else (name_seen.add(key) or True)
                                    )
                                )(
                                    "<name={!r}, alias.name={!r}, alias.asname={!r}>".format(
                                        name, _alias.name, _alias.asname
                                    )
                                ),
                                sorted(
                                    chain.from_iterable(
                                        map(attrgetter("names"), import_from_nodes)
                                    ),
                                    key=attrgetter("name"),
                                ),
                            )
                        ),
                        level=0,  # import_from_nodes[0].level
                        identifier=None,
                    )
                    for name, import_from_nodes in groupby(
                        module.body[fst_import_idx:lst_import_idx],
                        key=attrgetter("module"),
                    )
                ),
            )
        )
        + module.body[lst_import_idx:]
    )
    name_seen.clear()
    return module


def deduplicate(it):
    """
    Deduplicate an iterable

    :param it: An iterable|collection with hashable elements
    :type it: ```Union[Iterable, Generator, List, Tuple, Set, FrozenSet]```

    :return: Deduplicated iterable of the input
    :rtype: ```Iterable```
    """
    seen = set()
    return (seen.add(e) or e for e in it if e not in seen)


NoneStr = "```(None)```" if PY_GTE_3_9 else "```None```"

__all__ = [
    "DEFAULT_MODULES_TO_ALL",
    "DEFAULT_MODULES_TO_ALL_SQL_FIRST",
    "Dict_to_dict",
    "FALLBACK_ARGPARSE_TYP",
    "FALLBACK_TYP",
    "List_to_list",
    "NoneStr",
    "RewriteAtQuery",
    "Set_to_set",
    "Tuple_to_tuple",
    "_parse_default_from_ast",
    "annotate_ancestry",
    "ast_type_to_python_type",
    "cmp_ast",
    "code_quoted",
    "construct_module_with_symbols",
    "deduplicate",
    "deduplicate_sorted_imports",
    "del_ass_where_name",
    "emit_ann_assign",
    "emit_arg",
    "find_ast_type",
    "find_in_ast",
    "func_arg2param",
    "get_ass_where_name",
    "get_at_root",
    "get_doc_str",
    "get_function_type",
    "get_value",
    "infer_imports",
    "infer_type_and_default",
    "is_argparse_add_argument",
    "is_argparse_description",
    "it2literal",
    "maybe_type_comment",
    "merge_assignment_lists",
    "merge_modules",
    "node_to_dict",
    "optimise_imports",
    "param2argparse_param",
    "param2ast",
    "parse_to_scalar",
    "set_arg",
    "set_docstring",
    "set_slice",
    "set_value",
    "to_annotation",
    "to_type_comment",
]

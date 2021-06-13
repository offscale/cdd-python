"""
Functions which help the functions within the parser module
"""

import ast
from ast import (
    AnnAssign,
    Assign,
    Call,
    ClassDef,
    FunctionDef,
    Module,
    Name,
    Return,
    Tuple,
)
from collections import OrderedDict
from functools import partial
from inspect import _empty, getsource
from itertools import chain
from operator import attrgetter, eq, itemgetter
from types import FunctionType

from cdd.ast_utils import NoneStr, column_type2typ, get_value, json_type2typ
from cdd.pure_utils import lstrip_namespace, none_types, rpartial
from cdd.source_transformer import to_code

lstrip_typings = partial(lstrip_namespace, namespaces=("typings.", "_extensions."))


def ir_merge(target, other):
    """
    Merge two intermediate_repr (IR) together. It doesn't do a `target.update(other)`,
     instead it carefully merges `params` and `returns`

    :param target: The IR to use the values of. These values take precedence. IR is a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type target: ```dict```

    :param other: The IR to update. IR is a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type other: ```dict```

    :returns: IR of updated target. `target` is also updated in-place, and the memory of `other` is used.
    :rtype: ```dict```
    """
    if not target["params"]:
        target["params"] = other["params"]
    elif other["params"]:
        target_params, other_params = map(itemgetter("params"), (target, other))

        for name in other_params.keys() & target_params.keys():
            if not target_params[name].get("doc") and other_params[name].get("doc"):
                target_params[name]["doc"] = other_params[name]["doc"]
            if target_params[name].get("typ") is None and other_params[name].get("typ"):
                target_params[name]["typ"] = other_params[name]["typ"]
            if (
                target_params[name].get("default") in none_types
                and "default" in other_params[name]
                and other_params[name]["default"]
                not in frozenset((None, "None", "(None)"))
            ):
                target_params[name]["default"] = other_params[name]["default"]

        for name in other_params.keys() - target_params.keys():
            target_params[name] = other_params[name]

        target["params"] = target_params

    if "return_type" not in (target.get("returns") or iter(())):
        target["returns"] = other["returns"]
    elif other["returns"]:
        target["returns"]["return_type"] = _join_non_none(
            target["returns"]["return_type"], other["returns"]["return_type"]
        )
    # if "return_type" in target.get("params", frozenset()):
    #     target["returns"]["return_type"] = _join_non_none(
    #         target["returns"]["return_type"], target["params"].pop("return_type")
    #     )

    other_internal = other.get("_internal", {})
    if other_internal.get("body"):
        if "_internal" in target:
            # Merging internal bodies would be a bad idea IMHO
            target["_internal"].update(other_internal)
        else:
            target["_internal"] = other_internal

    return target


def _join_non_none(primacy, other):
    """
    Set all properties on `primacy` parameter, if not already non None, to the non None property on other

    :param primacy: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type primacy: ```dict```

    :param other: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type other: ```dict```

    :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    if not primacy:
        return other
    elif not other:
        return primacy
    all_keys = frozenset(chain.from_iterable((primacy.keys(), other.keys())))
    for key in all_keys:
        if primacy.get(key) is None and other.get(key) is not None:
            primacy[key] = other[key]
    return primacy


def _inspect_process_ir_param(param, sig):
    """
    Merge signature with param

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param sig: The Signature
    :type sig: ```inspect.Signature```

    :returns: dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    name, _param = param
    del param
    name = name.lstrip("*")
    if name not in sig.parameters:
        return name, _param
    sig_param = sig.parameters[name]
    if sig_param.annotation is not _empty:
        _param["typ"] = lstrip_typings("{!s}".format(sig_param.annotation))
    if sig_param.default is not _empty:
        _param["default"] = sig_param.default
        if _param.get("typ", _empty) is _empty:
            _param["typ"] = type(_param["default"]).__name__
    if name.endswith("kwargs"):
        _param["typ"] = "Optional[dict]"
    return name, _param


def _interpolate_return(function_def, intermediate_repr):
    """
    Interpolate the return value into the IR.

    :param function_def: function definition
    :type function_def: ```FunctionDef```

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :returns: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """
    return_ast = next(
        filter(rpartial(isinstance, Return), function_def.body[::-1]), None
    )
    if return_ast is not None and return_ast.value is not None:
        if intermediate_repr.get("returns") is None:
            intermediate_repr["returns"] = OrderedDict((("return_type", {}),))

        if (
            "typ" in intermediate_repr["returns"]["return_type"]
            and "[" not in intermediate_repr["returns"]["return_type"]["typ"]
        ):
            del intermediate_repr["returns"]["return_type"]["typ"]
        intermediate_repr["returns"]["return_type"]["default"] = (
            lambda default: "({})".format(default)
            if isinstance(return_ast.value, Tuple)
            and (not default.startswith("(") or not default.endswith(")"))
            else (
                lambda default_: default_
                if isinstance(
                    default_, (str, int, float, complex, ast.Num, ast.Str, ast.Constant)
                )
                else "```{}```".format(default)
            )(get_value(get_value(return_ast)))
        )(to_code(return_ast.value).rstrip("\n"))
    if hasattr(function_def, "returns") and function_def.returns is not None:
        intermediate_repr["returns"] = intermediate_repr.get("returns") or OrderedDict(
            (("return_type", {}),)
        )
        intermediate_repr["returns"]["return_type"]["typ"] = to_code(
            function_def.returns
        ).rstrip("\n")

    return intermediate_repr


def column_call_to_param(call):
    """
    Parse column call `Call(func=Name("Column", Load(), …)` into param

    :param call: Column call from SQLAlchemy `Table` construction
    :type call: ```Call```

    :returns: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    assert call.func.id == "Column"
    assert len(call.args) == 2

    _param = dict(
        chain.from_iterable(
            filter(
                None,
                (
                    map(
                        lambda key_word: (key_word.arg, get_value(key_word.value)),
                        call.keywords,
                    ),
                    (("typ", column_type2typ[call.args[1].id]),)
                    if isinstance(call.args[1], Name)
                    else None,
                ),
            )
        )
    )

    if (
        isinstance(call.args[1], Call)
        and call.args[1].func.id.rpartition(".")[2] == "Enum"
    ):
        _param["typ"] = "Literal{}".format(list(map(get_value, call.args[1].args)))

    pk = "primary_key" in _param
    if pk:
        _param["doc"] = "[PK] {}".format(_param["doc"])
        del _param["primary_key"]

    def _handle_null():
        """
        Properly handle null condition
        """
        if not _param["typ"].startswith("Optional["):
            _param["typ"] = "Optional[{}]".format(_param["typ"])

    if "nullable" in _param:
        not _param["nullable"] or _handle_null()
        del _param["nullable"]

    if "default" in _param and not get_value(call.args[0]).endswith("kwargs"):
        _param["doc"] += "."

    return get_value(call.args[0]), _param


def json_schema_property_to_param(param, required):
    """
    Convert a JSON schema property to a param

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param required: Names of all required parameters
    :type required: ```FrozenSet[str]```

    :returns: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    name, _param = param
    del param
    if name.endswith("kwargs"):
        _param["typ"] = "Optional[dict]"
    elif "enum" in _param:
        _param["typ"] = "Literal{}".format(_param.pop("enum"))
        del _param["type"]
    if "description" in _param:
        _param["doc"] = _param.pop("description")

    if _param.get("type"):
        _param["typ"] = json_type2typ[_param.pop("type")]

    if name not in required and _param.get("typ") and "Optional[" not in _param["typ"]:
        _param["typ"] = "Optional[{}]".format(_param["typ"])
        if _param.get("default") in none_types:
            _param["default"] = NoneStr

    return name, _param


def infer(*args, **kwargs):
    """
    Infer the `parse` type

    :param args: The arguments
    :type args: ```Tuple[args]```

    :param kwargs: Keyword arguments
    :type kwargs: ```dict```

    :returns: Name of inferred parser
    :rtype: ```str```
    """
    node = (
        args[0]
        if args
        else kwargs.get(
            "class_def", kwargs.get("function_def", kwargs.get("call_or_name"))
        )
    )
    is_supported_ast_node = isinstance(
        node, (Module, Assign, AnnAssign, Call, ClassDef, FunctionDef)
    )
    if not is_supported_ast_node and (
        isinstance(node, (type, FunctionType)) or type(node).__name__ == "function"
    ):
        return infer(ast.parse(getsource(node)).body[0])

    if not is_supported_ast_node:
        if not isinstance(node, str):
            node = get_value(node)
        if (
            isinstance(node, str)
            and not node.startswith("def ")
            and not node.startswith("class ")
        ):
            return "docstring"
    assert is_supported_ast_node
    if isinstance(node, FunctionDef):
        if next(
            filter(
                partial(eq, "argument_parser"), map(attrgetter("arg"), node.args.args)
            ),
            False,
        ):
            return "argparse_ast"

        return "function"

    elif isinstance(node, ClassDef):
        if any(
            filter(
                partial(eq, "Base"),
                map(attrgetter("id"), filter(rpartial(hasattr, "id"), node.bases)),
            )
        ):
            return "sqlalchemy"
        return "class_"
    elif isinstance(node, (AnnAssign, Assign)):
        return infer(node.value)
    elif isinstance(node, Call):
        if len(node.args) > 2 and node.args[1].id == "metadata":
            return "sqlalchemy_table"


def get_source(obj):
    """
    Call inspect.getsource and raise an error unless class definition could not be found

    :param obj: object to inspect
    :type obj: ```Any```

    :returns: The source
    :rtype: ```str```
    """
    try:
        return getsource(obj)
    except OSError as e:
        if e.args and e.args[0] == "could not find class definition":
            return None
        raise


__all__ = [
    "column_call_to_param",
    "get_source",
    "ir_merge",
    "infer",
    "json_schema_property_to_param",
    "lstrip_typings",
]

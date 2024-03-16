"""
Functions which help the functions within the parser module
"""

import ast
import inspect
from ast import AnnAssign, Assign, Call, ClassDef, FunctionDef, Module
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from importlib import import_module
from inspect import _empty, getdoc, getsource, isfunction, signature
from itertools import chain
from operator import attrgetter, eq, itemgetter
from types import FunctionType
from typing import Optional

import cdd.class_.parse
import cdd.docstring.parse
import cdd.function.parse
import cdd.shared.ast_utils
import cdd.shared.docstring_parsers
import cdd.shared.parse
from cdd.class_.utils.parse_utils import get_source
from cdd.shared.pure_utils import lstrip_namespace, none_types, rpartial, simple_types
from cdd.shared.types import IntermediateRepr

lstrip_typings = partial(lstrip_namespace, namespaces=("typings.", "_extensions."))


def ir_merge(target, other):
    """
    Merge two intermediate_repr (IR) together. It doesn't do a `target.update(other)`,
     instead it carefully merges `params` and `returns`

    :param target: The target IR to modify. These values take precedence. Dict is consistent with `IntermediateRepr`:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :type target: ```dict```

    :param other: Read-only IR to use in update. IR is a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :type other: ```dict```

    :return: IR of updated target. `target` is also updated in-place, and the memory of `other` is used.
    :rtype: ```dict```
    """
    if not target["params"]:
        target["params"] = other["params"]
    elif other["params"]:
        target_params, other_params = map(itemgetter("params"), (target, other))
        target["params"] = merge_params(other_params, target_params)

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


def merge_params(other_params, target_params):
    """
    Merge two ParamVal dicts together. It doesn't do a `target_params.update(other_params)`,
     instead it carefully merges two collections of dicts.

    :param other_params: Read-only params to use in update. Iterable of `dict`s consistent with `ParamVal`, defined as:
        TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
    :type other_params: ```Iterable[dict]```

    :param target_params: The target params to modify. These values take precedence. Iterable of `dict`s consistent
      with `ParamVal`, defined as:
        TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
    :type target_params: ```Iterable[dict]```

    :return: IR of updated target. `target` is also updated in-place, and the memory of `other` is used.
    :rtype: ```dict```
    """
    for name in other_params.keys() & target_params.keys():
        merge_present_params(other_params[name], target_params[name])
    for name in other_params.keys() - target_params.keys():
        target_params[name] = other_params[name]
    return target_params


def merge_present_params(other_param, target_param):
    """
    Merge two ParamVal dicts together. It doesn't do a `target_params.update(other_params)`,
     instead it carefully merges two dicts.

    :param other_param: Read-only param to use in update. Dict consistent with `ParamVal`, defined as:
        TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
    :type other_param: ```dict```

    :param target_param: The target param to modify. These values take precedence. Dict consistent with
      `ParamVal`, defined as:
        TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
    :type target_param: ```dict```

    :return: IR of updated target. `target` is also updated in-place, and the memory of `other` is used.
    :rtype: ```dict```
    """
    if not target_param.get("doc") and other_param.get("doc"):
        target_param["doc"] = other_param["doc"]
    if other_param.get("typ") is not None and (
        target_param.get("typ") is None
        or target_param["typ"] in simple_types
        and other_param["typ"] not in simple_types
    ):
        target_param["typ"] = other_param["typ"]
    if (
        target_param.get("default") in none_types
        and other_param.get("default") is not None
    ):
        target_param["default"] = other_param["default"]


def _join_non_none(primacy, other):
    """
    Set all properties on `primacy` parameter, if not already non None, to the non None property on other

    :param primacy: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type primacy: ```dict```

    :param other: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type other: ```dict```

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    if not primacy:
        return other
    elif not other:
        return primacy
    # & for `dict` keys is only available in newer Python versions
    all_keys = frozenset(chain.from_iterable((primacy.keys(), other.keys())))
    primacy.update(
        {
            key: other[key]
            for key in all_keys
            if primacy.get(key) is None and other.get(key) is not None
        }
    )
    return primacy


def _inspect_process_ir_param(param, sig):
    """
    Merge signature with param

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```tuple[str, dict]```

    :param sig: The Signature
    :type sig: ```inspect.Signature```

    :return: dict with keys: 'typ', 'doc', 'default'
    :rtype: ```tuple[str, dict]```
    """
    name, _param = param
    del param
    name: str = name.lstrip("*")

    if name not in sig.parameters:
        return name, _param
    sig_param: inspect.Parameter = sig.parameters[name]
    if sig_param.annotation is not _empty:
        _param["typ"] = lstrip_typings(
            "{annotation!s}".format(annotation=sig_param.annotation)
        )
    if sig_param.default is not _empty:
        _param["default"] = sig_param.default
        if _param.get("typ", _empty) is _empty:
            _param["typ"] = type(_param["default"]).__name__
    if name.endswith("kwargs"):
        _param["typ"] = "Optional[dict]"
    return name, _param


def infer(*args, **kwargs):
    """
    Infer the `parse` type

    :param args: The arguments
    :type args: ```tuple[args]```

    :param kwargs: Keyword arguments
    :type kwargs: ```dict```

    :return: Name of inferred parser
    :rtype: ```str```
    """
    node = (
        args[0]
        if args
        else kwargs.get(
            "class_def", kwargs.get("function_def", kwargs.get("call_or_name"))
        )
    )
    is_supported_ast_node: bool = isinstance(
        node, (Module, Assign, AnnAssign, Call, ClassDef, FunctionDef)
    )
    if not is_supported_ast_node and (
        isinstance(node, (type, FunctionType)) or type(node).__name__ == "function"
    ):
        return infer(ast.parse(getsource(node)).body[0])

    if not is_supported_ast_node:
        if not isinstance(node, str):
            node = cdd.shared.ast_utils.get_value(node)
        if (
            isinstance(node, str)
            and not node.startswith("def ")
            and not node.startswith("class ")
        ):
            return "docstring"

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
    else:
        raise NotImplementedError(node)


def _inspect(obj, name, parse_original_whitespace, word_wrap):
    """
    Uses the `inspect` module to figure out the IR from the input

    :param obj: Something in memory, like a class, function, variable
    :type obj: ```Any```

    :param name: Name of the object being inspected
    :type name: ```Optional[str]```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :return: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :rtype: ```dict```
    """

    doc: str = getdoc(obj) or ""

    # def is_builtin_class_instance(obj):
    #     builtin_types = tuple(
    #         getattr(builtins, t)
    #         for t in dir(builtins)
    #         if isinstance(getattr(builtins, t), type)
    #     )
    #     return obj.__class__.__module__ == "__builtin__" or isinstance(
    #         obj, builtin_types
    #     )

    is_function: bool = isfunction(obj)
    ir: IntermediateRepr = (
        cdd.docstring.parse.docstring(
            doc,
            emit_default_doc=is_function,
            parse_original_whitespace=parse_original_whitespace,
        )
        if doc
        else {}
    )
    if not is_function and "type" in ir:
        del ir["type"]

    ir["name"] = (
        name or obj.__qualname__ if hasattr(obj, "__qualname__") else obj.__name__
    )
    assert ir["name"], "IR name is empty"

    # if is_builtin_class_instance(obj):
    #    return ir

    sig = None
    with suppress(ValueError):
        sig = signature(obj)
    if sig is not None:
        ir["params"] = OrderedDict(
            filter(
                None,
                map(
                    partial(_inspect_process_ir_param, sig=sig),
                    ir.get("params", {}).items(),
                ),
                # if ir.get("params")
                # else map(_inspect_process_sig, sig.parameters.items()),
            )
        )

    src: Optional[str] = get_source(obj)
    if src is None:
        return ir
    parsed_body = ast.parse(src.lstrip()).body[0]

    if is_function:
        ir["type"] = (
            "static"
            if sig is None
            else {"self": "self", "cls": "cls"}.get(
                next(iter(sig.parameters.values())).name, "static"
            )
        )
        parser = cdd.function.parse.function
    else:
        parser = cdd.class_.parse.class_

    other = parser(parsed_body)
    ir_merge(ir, other)
    if "return_type" in (ir.get("returns") or iter(())):
        ir["returns"] = OrderedDict(
            map(
                partial(
                    cdd.shared.docstring_parsers._set_name_and_type,
                    infer_type=False,
                    word_wrap=word_wrap,
                ),
                ir["returns"].items(),
            )
        )

    return ir


def get_parser(node, parse_name):
    """
    Get parser function specialised for input `node`

    :param node: Node to parse
    :type node: ```AST```

    :param parse_name: Which type to parse.
    :type parse_name: ```Literal["argparse", "class", "function", "json_schema",
                                 "pydantic", "sqlalchemy", "sqlalchemy_table", "sqlalchemy_hybrid","infer"]```

    :return: Function which returns intermediate_repr
    :rtype: ```Callable[[...], dict]````
    """
    if parse_name in (None, "infer"):
        parse_name: str = infer(node)
    parse_name: str = {
        "class": "class_",
        "sqlalchemy_hybrid": "sqlalchemy",
        "sqlalchemy_table": "sqlalchemy",
    }.get(parse_name, parse_name)
    return getattr(import_module(".".join(("cdd", parse_name, "parse"))), parse_name)


__all__ = [
    "_inspect",
    "get_parser",
    "infer",
    "ir_merge",
    "lstrip_typings",
    "merge_present_params",
]  # type: list[str]

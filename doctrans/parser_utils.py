"""
Functions which help the functions within the parser module
"""
import ast
from ast import Call, Name, Return, Tuple
from collections import OrderedDict
from functools import partial
from inspect import _empty
from itertools import chain
from operator import itemgetter

from doctrans.ast_utils import NoneStr, column_type2typ, get_value
from doctrans.pure_utils import lstrip_namespace, none_types, rpartial
from doctrans.source_transformer import to_code

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


def _inspect_process_sig(k_v):
    """
    Postprocess the param

    :param k_v: Key and value from `inspect._parameters` mapping
    :type k_v: ```Tuple[str, inspect.Parameter]```

    :returns: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    # return dict(
    #     name=k_v[0],
    #     **(
    #         {}
    #         if k_v[1].default is _empty
    #         else {
    #             "default": k_v[1].default,
    #             "typ": lstrip_typings(
    #                 type(k_v[1].default).__name__
    #                 if k_v[1].annotation is _empty
    #                 else "{!s}".format(k_v[1].annotation)
    #             ),
    #         }
    #     ),
    # )


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
        if intermediate_repr.get("returns") is None:
            intermediate_repr["returns"] = OrderedDict((("return_type", {}),))
        intermediate_repr["returns"]["return_type"]["typ"] = to_code(
            function_def.returns
        ).rstrip("\n")

    return intermediate_repr


# def strip_docstring_from_body(source, body):
#     """
#     Since we generate the docstring, remove it from the body (to avoid duplicate docstrings)
#
#     :param source: The parsed source code
#     :type source: ```AST```
#
#     :param body: The body
#     :type body: ```List[AST]```
#
#     :returns: Docstring-free body
#     :rtype: ```List[AST]```
#     """
#     return body if ast.get_docstring(source) is not None else body[1:]


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
        # if "default" not in _param:
        #    _param["default"] = NoneStr

    if "nullable" in _param:
        not _param["nullable"] or _handle_null()
        del _param["nullable"]
    # elif not pk:
    #    _handle_null()

    if "default" in _param and not get_value(call.args[0]).endswith("kwargs"):
        _param["doc"] += "."
    #    _param["doc"] += '. Defaults to "{}"'.format(_param.pop("default"))

    return get_value(call.args[0]), _param


__all__ = ["column_call_to_param", "ir_merge", "lstrip_typings"]

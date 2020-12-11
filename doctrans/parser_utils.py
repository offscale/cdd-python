"""
Functions which help the functions within the parser module
"""
import ast
from ast import Return, Tuple
from functools import partial
from inspect import _empty
from itertools import chain
from operator import itemgetter
from typing import Any

from docstring_parser import DocstringParam, DocstringMeta

from doctrans.ast_utils import get_value
from doctrans.defaults_utils import extract_default
from doctrans.pure_utils import assert_equal, lstrip_namespace, rpartial
from doctrans.source_transformer import to_code

lstrip_typings = partial(lstrip_namespace, namespaces=("typings.", "_extensions."))


def ir_merge(target, other):
    """
    Merge two intermediate_repr (IR) together. It doesn't do a `target.update(other)`,
     instead it carefully merges `params` and `returns`

    intermediate_repr is a dict of shape {
            'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}

    :param target: The IR to update
    :type target: ```dict```

    :param other: The IR to use the values of. These values take precedence.
    :type other: ```dict```

    :return: Updated target. `target` is also updated in-place, and the memory of `other` is used.
    :rtype: ```dict```
    """
    if not target["params"]:
        target["params"] = other["params"]
    elif other["params"]:
        target_params_len, other_params_len = len(target["params"]), len(
            other["params"]
        )
        extra_params = []
        if other_params_len > target_params_len:
            assert_equal(target_params_len + 1, other_params_len)
            extra_params.append(other["params"].pop())
            # It's probably a kwargs. Could validate with an `assert last["name"].endswith("kwargs")` and a type check
            # But worried that would cause issues of its own, e.g., missing common idioms like `**config`.
        elif target_params_len > other_params_len:
            extra_params, target["params"] = (
                target["params"][other_params_len:],
                target["params"][:other_params_len],
            )
        else:
            assert_equal(target_params_len, other_params_len)
        target["params"] = (
            list(
                map(
                    lambda idx_param: _join_non_none(
                        idx_param[1], other["params"][idx_param[0]]
                    ),
                    enumerate(target["params"]),
                )
            )
            + extra_params
        )

    if not target["returns"]:
        target["returns"] = other["returns"]
    elif other["returns"]:
        target["returns"] = _join_non_none(target["returns"], other["returns"])
    if target["params"] and target["params"][-1]["name"] == "return_type":
        target["returns"] = _join_non_none(target["returns"], target["params"].pop())

    return target


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
    all_keys = frozenset(chain.from_iterable((primacy.keys(), other.keys())))
    for key in all_keys:
        if primacy.get(key) is None and other.get(key) is not None:
            primacy[key] = other[key]
    return primacy


def _inspect_process_ir_param(param, sig):
    """
    Postprocess the param

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :type param: ```dict``

    :param sig: The Signature
    :type sig: ```inspect.Signature```

    :return: Potentially changed param
    :rtype: ```dict```
    """
    param["name"] = param["name"].lstrip("*")
    if param["name"] not in sig.parameters:
        return param
    sig_param = sig.parameters[param["name"]]
    if sig_param.annotation is not _empty:
        param["typ"] = lstrip_typings("{!s}".format(sig_param.annotation))
    if sig_param.default is not _empty:
        param["default"] = sig_param.default
        if param.get("typ", _empty) is _empty:
            param["typ"] = type(param["default"]).__name__
    if param["name"].endswith("kwargs"):
        param["typ"] = "dict"
    return param


def _inspect_process_sig(k_v):
    """
    Postprocess the param

    :param k_v: Key and value from `inspect._parameters` mapping
    :type k_v: ```Tuple[str, inspect.Parameter]``

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
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
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type intermediate_repr: ```dict```
    """
    return_ast = next(
        filter(rpartial(isinstance, Return), function_def.body[::-1]), None
    )
    if return_ast is not None and return_ast.value is not None:
        # if intermediate_repr["returns"] is None: intermediate_repr["returns"] = {"name": "return_type"}

        intermediate_repr["returns"]["default"] = (
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
        intermediate_repr["returns"]["typ"] = to_code(function_def.returns).rstrip("\n")


def _parse_dict(d):
    """
    Restructure dictionary to match expectations

    :param d: input dictionary
    :type d: ```dict```

    :return: restructured dict
    :rtype: ```dict```
    """
    assert isinstance(d, dict), "Expected 'dict' got `{!r}`".format(type(d).__name__)
    if "args" in d and len(d["args"]) in frozenset((1, 2)):
        d["name"] = d.pop("args")[0]
        if d["name"] == "return":
            d["name"] = "return_type"
    if "type_name" in d:
        d["typ"] = d.pop("type_name")
    if "description" in d:
        d["doc"] = d.pop("description")

    return {k: v for k, v in d.items() if v is not None}


def _evaluate_to_docstring_value(name_value):
    """
    Turn the second element of the tuple into the final representation (e.g., a bool, str, int)

    :param name_value: name value tuple
    :type name_value: ```Tuple[str, Any]```

    :return: Same shape as input
    :rtype: ```Tuple[str, Tuple[Union[str, int, bool, float]]]```
    """
    assert (
        isinstance(name_value, tuple) and len(name_value) == 2
    ), "Expected input of type `Tuple[str, Any]' got value of `{!r}`".format(name_value)
    name: str = name_value[0]
    value: Any = name_value[1]
    if isinstance(value, (list, tuple)):
        value = list(
            map(
                itemgetter(1),
                map(lambda v: _evaluate_to_docstring_value((name, v)), value),
            )
        )
    elif isinstance(value, DocstringParam):
        assert len(value.args) == 2 and value.args[1] == value.arg_name
        value = {
            attr: getattr(value, attr)
            for attr in (
                "type_name",
                "arg_name",
                "is_optional",
                "default",
                "description",
            )
            if getattr(value, attr) is not None
        }
        if "arg_name" in value:
            value["name"] = value.pop("arg_name")
        if "description" in value:
            value["doc"] = extract_default(
                value.pop("description"), emit_default_doc=False
            )[0]
    elif isinstance(value, DocstringMeta):
        value = _parse_dict(
            {
                attr: getattr(value, attr)
                for attr in dir(value)
                if not attr.startswith("_") and getattr(value, attr)
            }
        )
    elif name == "short_description":
        name = "doc"
    # elif not isinstance(value, (str, int, float, bool, type(None))):
    #     raise NotImplementedError(type(value).__name__)
    return name, value

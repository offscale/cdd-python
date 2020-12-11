"""
Transform from string or AST representations of input, to intermediate_repr dict of shape {
            'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}.
"""
import ast
from ast import (
    AnnAssign,
    FunctionDef,
    Return,
    Assign,
    Tuple,
    get_docstring,
    Module,
    ClassDef,
)
from collections import OrderedDict, deque
from copy import deepcopy
from functools import partial
from inspect import signature, getdoc, isfunction, getsource
from itertools import filterfalse, count
from pprint import PrettyPrinter
from types import FunctionType

from docstring_parser import (
    Docstring,
)

from doctrans import get_logger
from doctrans.ast_utils import (
    find_ast_type,
    get_value,
    is_argparse_add_argument,
    is_argparse_description,
    get_function_type,
    argparse_param2param,
)
from doctrans.docstring_parsers import parse_docstring, _set_name_and_type
from doctrans.emitter_utils import parse_out_param, _parse_return
from doctrans.parser_utils import (
    _inspect_process_ir_param,
    _inspect_process_sig,
    _interpolate_return,
    _evaluate_to_docstring_value,
    ir_merge,
)
from doctrans.pure_utils import (
    rpartial,
    assert_equal,
    update_d,
    pp, simple_types,
)
from doctrans.source_transformer import to_code

logger = get_logger("doctrans.parse")


def class_(class_def, class_name=None, merge_inner_function=None, infer_type=False):
    """
    Converts an AST to our IR

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[Module, ClassDef]```

    :param class_name: Name of `class`. If None, gives first found.
    :type class_name: ```Optional[str]```

    :param merge_inner_function: Name of inner function to merge. If None, merge nothing.
    :type merge_inner_function: ```Optional[str]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :return: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    assert not isinstance(class_def, FunctionDef)
    is_supported_ast_node = isinstance(class_def, (Module, ClassDef))
    if not is_supported_ast_node and isinstance(class_def, type):
        ir = _inspect(class_def, class_name)
        parsed_body = ast.parse(getsource(class_def).lstrip()).body[0]

        pp({"parse::class::params:": ir["params"]})

        if merge_inner_function is not None:
            _merge_inner_function(
                parsed_body,
                infer_type=infer_type,
                intermediate_repr=ir,
                merge_inner_function=merge_inner_function,
            )
            return ir

        ir["_internal"] = {
            "body": list(
                filterfalse(
                    rpartial(isinstance, AnnAssign),
                    parsed_body.body,
                )
            ),
            "from_name": class_name,
            "from_type": "cls",
        }
        body_ir = class_(
            class_def=parsed_body,
            class_name=class_name,
            merge_inner_function=merge_inner_function,
        )
        ir_merge(ir, body_ir)
        return ir

    assert (
        is_supported_ast_node
    ), "Expected 'Union[Module, ClassDef]' got `{!r}`".format(type(class_def).__name__)
    class_def = find_ast_type(class_def, class_name)
    intermediate_repr = docstring(get_docstring(class_def).replace(":cvar", ":param"))

    intermediate_repr["params"] = OrderedDict(
        (param.pop("name"), param) for param in intermediate_repr["params"]
    )
    if "return_type" in intermediate_repr["params"]:
        intermediate_repr["returns"] = dict(
            name="return_type", **intermediate_repr["params"].pop("return_type")
        )

    for e in class_def.body:
        if isinstance(e, AnnAssign):
            typ = to_code(e.annotation).rstrip("\n")
            val = (lambda v: {} if v is None else {"default": v if type(v).__name__ in simple_types
                              else to_code(v).rstrip("\n")})(get_value(get_value(e)))
            # if 'str' in typ and val: val["default"] = val["default"].strip("'")  # Unquote?
            typ_default = dict(typ=typ, **val)

            if e.target.id == "return_type":
                intermediate_repr["returns"].update(typ_default)
            else:
                intermediate_repr["params"][e.target.id].update(typ_default)
        elif isinstance(e, Assign):
            val = get_value(e)
            if val is not None:
                val = get_value(val)
                for target in e.targets:
                    if target.id in intermediate_repr["params"]:
                        intermediate_repr["params"][target.id]["default"] = val
                    else:
                        intermediate_repr["params"][target.id] = {"default": val}

    intermediate_repr.update(
        {
            "params": [
                _set_name_and_type(dict(name=k, **v), infer_type=infer_type)
                for k, v in intermediate_repr["params"].items()
            ],
            "_internal": {
                "body": list(
                    filterfalse(rpartial(isinstance, AnnAssign), class_def.body)
                ),
                "from_name": class_def.name,
                "from_type": "cls",
            },
        }
    )

    if merge_inner_function is not None:
        assert isinstance(class_def, ClassDef)
        _merge_inner_function(
            class_def,
            infer_type=infer_type,
            intermediate_repr=intermediate_repr,
            merge_inner_function=merge_inner_function,
        )

    return intermediate_repr


def _merge_inner_function(
    class_def, infer_type, intermediate_repr, merge_inner_function
):
    """
    Merge the inner function if found within the class, with the class IR

    :param class_def: Class AST
    :type class_def: ```ClassDef```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param intermediate_repr: a dictionary of form
              {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
              }
    :type intermediate_repr: ```dict```

    :param merge_inner_function: Name of inner function to merge. If None, merge nothing.
    :type merge_inner_function: ```Optional[str]```

    :return: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    function_def = next(
        filter(
            lambda func: func.name == merge_inner_function,
            filter(rpartial(isinstance, FunctionDef), ast.walk(class_def)),
        ),
        None,
    )

    if function_def is not None:
        function_type = (
            "static" if not function_def.args.args else function_def.args.args[0].arg
        )
        inner_ir = function(
            function_def,
            function_name=merge_inner_function,
            function_type=function_type,
            infer_type=infer_type,
        )
        ir_merge(other=inner_ir, target=intermediate_repr)

    return intermediate_repr


def _inspect(obj, name):
    """
    Uses the `inspect` module to figure out the IR from the input

    :param obj: Something in memory, like a class, function, variable
    :type obj: ```Any```

    :param name: Name of the object being inspected
    :type name: ```str```

    :return: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """

    doc = getdoc(obj) or ""
    ir = docstring(doc) if doc else {}
    sig = signature(obj)
    is_function = isfunction(obj)
    if not is_function and "type" in ir:
        del ir["type"]

    print("_inspect::docstring:", doc, ";\n_inspect::sig:", sig, ";")

    ir.update(
        {
            "name": name or obj.__qualname__
            if hasattr(obj, "__qualname__")
            else obj.__name__,
            "params": list(
                filter(
                    None,
                    map(partial(_inspect_process_ir_param, sig=sig), ir["params"])
                    if ir.get("params")
                    else map(_inspect_process_sig, sig.parameters.items()),
                )
            ),
        }
    )

    parsed_body = ast.parse(getsource(obj).lstrip()).body[0]

    if is_function:
        ir["type"] = {"self": "self", "cls": "cls"}.get(
            next(iter(sig.parameters.values())).name, "static"
        )
        parser = function
    else:
        parser = class_

    was = deepcopy(ir)
    other = parser(parsed_body)
    ir_merge(ir, other)
    PrettyPrinter(indent=4, sort_dicts=False).pprint(
        {"IR was": was, "other": other, "IR now": ir}
    )

    # if ir.get("returns") and "returns" not in ir["returns"]:
    #     if sig.return_annotation is not _empty:
    #         ir["returns"]["typ"] = lstrip_typings("{!s}".format(sig.return_annotation))
    #
    #     return_q = deque(
    #         filter(
    #             rpartial(isinstance, ast.Return),
    #             ast.walk(parsed_body),
    #         ),
    #         maxlen=1,
    #     )
    #     if return_q:
    #         return_val = get_value(return_q.pop())
    #         ir["returns"]["default"] = get_value(return_val)
    #         if not isinstance(
    #             ir["returns"]["default"],
    #             (str, int, float, complex, ast.Num, ast.Str, ast.Constant),
    #         ):
    #             ir["returns"]["default"] = "```{}```".format(
    #                 to_code(ir["returns"]["default"]).rstrip("\n")
    #             )
    return ir


def function(function_def, infer_type=False, function_type=None, function_name=None):
    """
    Converts a method to our IR

    :param function_def: AST node for function definition
    :type function_def: ```Union[FunctionDef, FunctionType]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Literal['self', 'cls', 'static']```

    :param function_name: name of function_def
    :type function_name: ```str```

    :return: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    if isinstance(function_def, FunctionType):
        ir = _inspect(function_def, function_name)
        parsed_source = ast.parse(getsource(function_def).lstrip()).body[0]
        ir["_internal"] = {
            "body": list(
                filterfalse(rpartial(isinstance, AnnAssign), parsed_source.body)
            ),
            "from_name": parsed_source.name,
            "from_type": "cls",
        }
        return ir

    assert isinstance(
        function_def, FunctionDef
    ), "Expected 'FunctionDef' got `{!r}`".format(type(function_def).__name__)
    assert (
        function_name is None or function_def.name == function_name
    ), "Expected {!r} got {!r}".format(function_name, function_def.name)

    found_type = get_function_type(function_def)

    intermediate_repr_docstring = (
        get_docstring(function_def) if isinstance(function_def, FunctionDef) else None
    )
    if intermediate_repr_docstring is None:
        intermediate_repr = {
            "description": "",
            "params": list(
                map(
                    argparse_param2param,
                    function_def.args.args
                    if found_type == "static"
                    else (function_def.args.args[1:]),
                )
            ),
            "returns": None,
        }
        # TODO: `intermediate_repr["returns"]` when no docstring is provided
    else:
        intermediate_repr = docstring(
            intermediate_repr_docstring.replace(":cvar", ":param"),
            infer_type=infer_type,
        )
    # if (
    #     function_type != "static"
    #     and intermediate_repr["params"]
    #     and intermediate_repr["params"][0]["name"] == function_type
    # ):
    #     del intermediate_repr["params"][0]

    intermediate_repr.update(
        {
            "name": function_name or function_def.name,
            "type": function_type or found_type,
        }
    )

    if function_def.body:
        intermediate_repr["_internal"] = {
            "body": function_def.body[0 if found_type == "static" else 1 :],
            "from_name": function_def.name,
            "from_type": found_type,
        }

    params_to_append = []
    if hasattr(function_def.args, "kwarg") and function_def.args.kwarg:
        merge_with = (
            intermediate_repr["params"].pop()
            if intermediate_repr["params"]
            and intermediate_repr["params"][-1]["name"] == function_def.args.kwarg.arg
            else {}
        )
        params_to_append.append(
            update_d(
                {
                    "name": function_def.args.kwarg.arg,
                    "typ": merge_with.get("typ", "dict"),
                },
                merge_with,
            )
        )

    idx = count()

    # Set defaults
    if intermediate_repr["params"]:
        deque(
            map(
                lambda args_defaults: deque(
                    map(
                        lambda idxparam_idx_arg: assert_equal(
                            intermediate_repr["params"][idxparam_idx_arg[0]]["name"],
                            idxparam_idx_arg[2].arg,
                        )
                        and intermediate_repr["params"][idxparam_idx_arg[0]].update(
                            dict(
                                name=idxparam_idx_arg[2].arg,
                                **(
                                    {}
                                    if getattr(idxparam_idx_arg[2], "annotation", None)
                                    is None
                                    else dict(
                                        typ=to_code(
                                            idxparam_idx_arg[2].annotation
                                        ).rstrip("\n")
                                    )
                                ),
                                **(
                                    lambda _defaults: dict(
                                        default=get_value(
                                            _defaults[idxparam_idx_arg[1]]
                                        )
                                    )
                                    if idxparam_idx_arg[1] < len(_defaults)
                                    and _defaults[idxparam_idx_arg[1]] is not None
                                    else {}
                                )(getattr(function_def.args, args_defaults[1])),
                            )
                        ),
                        (
                            lambda _args: map(
                                lambda idx_arg: (next(idx), idx_arg[0], idx_arg[1]),
                                enumerate(
                                    filterfalse(
                                        lambda _arg: _arg.arg[0] == "*",
                                        (
                                            _args
                                            if found_type == "static"
                                            or args_defaults[0] == "kwonlyargs"
                                            else _args[1:]
                                        ),
                                    )
                                ),
                            )
                        )(getattr(function_def.args, args_defaults[0])),
                    ),
                    maxlen=0,
                ),
                (("args", "defaults"), ("kwonlyargs", "kw_defaults")),
            ),
            maxlen=0,
        )

    intermediate_repr["params"] = list(
        map(
            partial(_set_name_and_type, infer_type=infer_type),
            intermediate_repr["params"],
        )
    )
    intermediate_repr["params"] += params_to_append

    # Convention - the final top-level `return` is the default
    _interpolate_return(function_def, intermediate_repr)

    return intermediate_repr


def argparse_ast(function_def, function_type=None, function_name=None):
    """
    Converts an argparse AST to our IR

    :param function_def: AST of argparse function_def
    :type function_def: ```FunctionDef``

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Literal['self', 'cls', 'static']```

    :param function_name: name of function_def
    :type function_name: ```str```

    :return: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    assert isinstance(
        function_def, FunctionDef
    ), "Expected 'FunctionDef' got `{!r}`".format(type(function_def).__name__)

    doc_string = get_docstring(function_def)
    intermediate_repr = {
        "name": function_name,
        "type": function_type or get_function_type(function_def),
        "doc": "",
        "params": [],
    }
    ir = parse_docstring(doc_string, emit_default_doc=True)

    for node in function_def.body[1:]:
        if is_argparse_add_argument(node):
            intermediate_repr["params"].append(
                parse_out_param(node, emit_default_doc=False)
            )
        elif isinstance(node, Assign):
            if is_argparse_description(node):
                intermediate_repr["doc"] = get_value(node.value)
        elif isinstance(node, Return) and isinstance(node.value, Tuple):
            intermediate_repr["returns"] = _parse_return(
                node,
                intermediate_repr=ir,
                function_def=function_def,
                emit_default_doc=False,
            )
    if len(function_def.body) > len(intermediate_repr["params"]) + 3:
        intermediate_repr["_internal"] = {
            "body": list(
                filterfalse(
                    is_argparse_description,
                    filterfalse(is_argparse_add_argument, function_def.body),
                )
            ),
            "from_name": function_def.name,
            "from_type": "static",
        }

    return intermediate_repr


def docstring(doc_string, infer_type=False, return_tuple=False):
    """
    Converts a docstring to an AST

    :param doc_string: docstring portion
    :type doc_string: ```Union[str, Dict]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param return_tuple: Whether to return a tuple, or just the intermediate_repr
    :type return_tuple: ```bool```

    :return: intermediate_repr, whether it returns or not
    :rtype: ```Optional[Union[dict, Tuple[dict, bool]]]```
    """
    assert isinstance(doc_string, str), "Expected 'str' got {!r}".format(
        type(doc_string).__name__
    )
    parsed = (
        doc_string
        if isinstance(doc_string, dict)
        else parse_docstring(doc_string, infer_type=infer_type)
    )

    if return_tuple:
        return parsed, (
            "returns" in parsed
            and parsed["returns"] is not None
            and "name" in parsed["returns"]
        )

    return parsed


def docstring_parser(doc_string):
    """
    Converts Docstring from the docstring_parser library to our internal representation

    :param doc_string: Docstring from the docstring_parser library
    :type doc_string: ```docstring_parser.common.Docstring```

    :return: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    assert isinstance(doc_string, Docstring), "Expected 'Docstring' got `{!r}`".format(
        type(doc_string).__name__
    )

    intermediate_repr = dict(
        map(
            _evaluate_to_docstring_value,
            filter(
                lambda k_v: not isinstance(k_v[1], (type(None), bool)),
                map(
                    lambda attr: (attr, getattr(doc_string, attr)),
                    filter(lambda attr: not attr.startswith("_"), dir(doc_string)),
                ),
            ),
        )
    )

    # def process_param(param):
    #     """
    #     Postprocess the param
    #
    #     :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    #     :type param: ```dict```
    #
    #     :return: Potentially changed param
    #     :rtype: ```dict```
    #     """
    #     if param.get("doc"):
    #         # # param["doc"] = param["doc"].strip()
    #         for term in "Usage:", "Reference:":
    #             idx = param["doc"].rfind(term)
    #             if idx != -1:
    #                 param["doc"] = param["doc"][:idx]
    #     return param

    assert "meta" in intermediate_repr and "params" in intermediate_repr
    meta = {e["name"]: e for e in intermediate_repr.pop("meta")}
    intermediate_repr["params"] = [
        # process_param
        dict(
            **param,
            **{k: v for k, v in meta[param["name"]].items() if k not in param},
        )
        for param in intermediate_repr["params"]
        if " " not in param["name"]
    ]
    # else:
    # intermediate_repr["params"] = list(
    #     map(
    #         process_param,
    #         filterfalse(
    #             lambda param: " " in param["name"], intermediate_repr["params"]
    #         ),
    #     )
    # )
    return intermediate_repr


__all__ = [
    "argparse_ast",
    "class_",
    "docstring",
    "docstring_parser",
    "function",
]

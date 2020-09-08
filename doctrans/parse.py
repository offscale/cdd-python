"""
Transform from string or AST representations of input, to intermediate_repr dict of shape {
            'name': ..., 'platform': ...,
            'module': ..., 'title': ..., 'description': ...,
            'parameters': ..., 'schema': ...,'returns': ...}.
"""

from ast import (
    AnnAssign,
    FunctionDef,
    Return,
    Constant,
    Assign,
    Tuple,
    get_docstring,
    Subscript,
    Str,
    NameConstant,
    Module,
    ClassDef,
    Name,
)
from collections import OrderedDict
from itertools import filterfalse
from operator import itemgetter
from typing import Any

from docstring_parser import DocstringParam, DocstringMeta, Docstring

from doctrans import get_logger
from doctrans.ast_utils import (
    find_ast_type,
    get_value,
    is_argparse_add_argument,
    is_argparse_description,
    get_function_type,
    argparse_param2param,
)
from doctrans.defaults_utils import extract_default
from doctrans.emitter_utils import parse_out_param, _parse_return
from doctrans.pure_utils import rpartial
from doctrans.rest_docstring_parser import parse_docstring
from doctrans.source_transformer import to_code

logger = get_logger("doctrans.parse")


def class_(class_def, config_name=None):
    """
    Converts an AST to our IR

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[Module, ClassDef]```

    :param config_name: Name of `class`. If None, gives first found.
    :type config_name: ```Optional[str]```

    :return: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    assert isinstance(
        class_def, (Module, ClassDef)
    ), "Expected 'Union[Module, ClassDef]' got `{!r}`".format(type(class_def).__name__)
    class_def = find_ast_type(class_def, config_name)
    intermediate_repr = docstring(get_docstring(class_def).replace(":cvar", ":param"))

    intermediate_repr["params"] = OrderedDict(
        (param.pop("name"), param) for param in intermediate_repr["params"]
    )
    if "return_type" in intermediate_repr["params"]:
        intermediate_repr["returns"] = dict(
            name="return_type", **intermediate_repr["params"].pop("return_type")
        )

    for e in filter(rpartial(isinstance, AnnAssign), class_def.body):
        typ = to_code(e.annotation).rstrip("\n")
        if e.target.id == "return_type":
            intermediate_repr["returns"]["typ"] = typ
        else:
            intermediate_repr["params"][e.target.id]["typ"] = typ

    intermediate_repr["params"] = [
        dict(name=k, **v) for k, v in intermediate_repr["params"].items()
    ]

    return intermediate_repr


def function(function_def, function_type=None, function_name=None):
    """
    Converts a method to our IR

    :param function_def: AST node for function definition
    :type function_def: ```FunctionDef```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Literal['self', 'cls', 'static']```

    :param function_name: name of function_def
    :type function_name: ```str```

    :return: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    assert isinstance(
        function_def, FunctionDef
    ), "Expected 'FunctionDef' got `{!r}`".format(type(function_def).__name__)
    assert (
        function_name is None or function_def.name == function_name
    ), "Expected {!r} got {!r}".format(function_name, function_def.name)

    intermediate_repr_docstring = get_docstring(function_def)
    if intermediate_repr_docstring is None:
        intermediate_repr = {"description": "", "params": [], "returns": None}
        for arg in function_def.args.args:
            intermediate_repr["params"].append(argparse_param2param(arg))

        # TODO: Returns when no docstring is provided
        # intermediate_repr["returns"].append(argparse_param2param(function_def.returns))

    else:
        intermediate_repr = docstring(
            intermediate_repr_docstring.replace(":cvar", ":param")
        )

    found_type = get_function_type(function_def)
    intermediate_repr.update(
        {
            "name": function_name or function_def.name,
            "type": function_type or found_type,
        }
    )
    # _function_type = get_function_type(function_def)
    offset = 0 if (intermediate_repr["type"] or "static") == "static" else 1

    if len(function_def.body) > 2:
        intermediate_repr["_internal"] = {
            "body": function_def.body[1:-1],
            "from_name": function_def.name,
            "from_type": found_type,
        }

    for idx, arg in enumerate(function_def.args.args):
        if arg.annotation is not None:
            i = idx - offset
            if i < len(intermediate_repr["params"]):
                intermediate_repr["params"][i]["typ"] = to_code(arg.annotation).rstrip(
                    "\n"
                )
            # else:
            #     logger.warning(
            #         "Ignoring {!r} function argument: {!r}".format(
            #             function_name, to_code(arg.annotation).rstrip("\n")
            #         )
            #     )

    def _get_default():
        """
        Get the default value

        :returns: Default value
        :rtype: ```Optional[Union[Name, str, int, float, bool]]```
        """
        if isinstance(const, Name):
            val = const
        else:
            assert (
                isinstance(const, Constant)
                and (not hasattr(const, "kind") or const.kind is None)
                or isinstance(const, (Str, NameConstant))
            ), type(const).__name__
            val = get_value(const)
        return val

    for idx, const in enumerate(function_def.args.defaults):
        value = _get_default()
        if value is not None:
            if len(intermediate_repr["params"]) > idx:
                intermediate_repr["params"][idx]["default"] = value
            # else: intermediate_repr["params"].append({"default": value})

    offset = len(function_def.args.kw_defaults) - len(function_def.args.kwonlyargs)
    for idx, const in enumerate(function_def.args.kw_defaults):
        value = _get_default()
        if value is not None:
            intermediate_repr["params"][idx + offset]["default"] = value

    if hasattr(function_def.args, "kwarg") and function_def.args.kwarg:
        # if intermediate_repr["params"][-1]["name"] != function_def.args.kwarg.arg:
        #     logger.warning(
        #         "Expected {!r} to be {!r}".format(
        #             intermediate_repr["params"][-1]["name"], function_def.args.kwarg.arg
        #         )
        #     )
        intermediate_repr["params"][-1]["typ"] = "dict"

    # Convention - the final top-level `return` is the default
    return_ast = next(
        filter(rpartial(isinstance, Return), function_def.body[::-1]), None
    )
    if (
        return_ast is not None
        and return_ast.value is not None
        and intermediate_repr.get("returns")
    ):
        intermediate_repr["returns"]["default"] = (
            lambda default: "({})".format(default)
            if isinstance(return_ast.value, Tuple)
            and (not default.startswith("(") or not default.endswith(")"))
            else default
        )(to_code(return_ast.value).rstrip("\n"))

    if hasattr(function_def, "returns") and isinstance(function_def.returns, Subscript):
        intermediate_repr["returns"]["typ"] = to_code(function_def.returns).rstrip("\n")

    if intermediate_repr["params"][-1]["name"] == "returns":
        returns = intermediate_repr["params"].pop()
        del returns["name"]
        intermediate_repr["returns"].update(returns)

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
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
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
        "short_description": "",
        "long_description": "",
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
                intermediate_repr["short_description"] = get_value(node.value)
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
                    filterfalse(is_argparse_add_argument, function_def.body[1:-1]),
                )
            ),
            "from_name": function_def.name,
            "from_type": "static",
        }

    return intermediate_repr


def docstring(doc_string, return_tuple=False):
    """
    Converts a docstring to an AST

    :param doc_string: docstring portion
    :type doc_string: ```Union[str, Dict]```

    :param return_tuple: Whether to return a tuple, or just the intermediate_repr
    :type return_tuple: ```bool```

    :return: intermediate_repr, whether it returns or not
    :rtype: ```Optional[Union[dict, Tuple[dict, bool]]]```
    """
    assert isinstance(doc_string, str), "Expected 'str' got {!r}".format(
        type(doc_string).__name__
    )
    parsed = doc_string if isinstance(doc_string, dict) else parse_docstring(doc_string)
    returns = (
        "returns" in parsed
        and parsed["returns"] is not None
        and "name" in parsed["returns"]
    )

    if returns:
        parsed["returns"]["doc"] = parsed["returns"].get(
            "doc", parsed["returns"]["name"]
        )

    if return_tuple:
        return parsed, returns

    return parsed


def _parse_dict(d):
    """
    Restructure dictionary to match expectations

    :param d: input dictionary
    :type d: ```dict```

    :returns: restructured dict
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
    # elif not isinstance(value, (str, int, float, bool, type(None))):
    #     raise NotImplementedError(type(value).__name__)
    return name, value


def docstring_parser(doc_string):
    """
    Converts Docstring from the docstring_parser library to our internal representation

    :param doc_string: Docstring from the docstring_parser library
    :type doc_string: ```docstring_parser.common.Docstring```

    :return: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
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
    if "meta" in intermediate_repr and "params" in intermediate_repr:
        meta = {e["name"]: e for e in intermediate_repr.pop("meta")}
        intermediate_repr["params"] = [
            dict(
                **param,
                **{k: v for k, v in meta[param["name"]].items() if k not in param}
            )
            for param in intermediate_repr["params"]
        ]

    return intermediate_repr


__all__ = [
    "class_",
    "function",
    "argparse_ast",
    "docstring",
    "docstring_parser",
]

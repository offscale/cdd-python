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
    Expr,
    Module,
    ClassDef,
)
from collections import OrderedDict
from functools import partial
from itertools import filterfalse
from operator import itemgetter
from typing import Any

from docstring_parser import DocstringParam, DocstringMeta, Docstring

from doctrans.ast_utils import (
    find_ast_type,
    get_function_type,
    get_value,
    is_argparse_add_argument,
    is_argparse_description,
)
from doctrans.defaults_utils import extract_default, set_default_doc
from doctrans.emitter_utils import parse_out_param, _parse_return
from doctrans.pure_utils import tab, rpartial
from doctrans.rest_docstring_parser import parse_docstring
from doctrans.source_transformer import to_code


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
        intermediate_repr["returns"] = {
            "return_type": intermediate_repr["params"].pop("return_type")
        }

    for e in filter(rpartial(isinstance, AnnAssign), class_def.body):
        intermediate_repr["returns" if e.target.id == "return_type" else "params"][
            e.target.id
        ]["typ"] = to_code(e.annotation).rstrip("\n")

    intermediate_repr["params"] = [
        dict(name=k, **v) for k, v in intermediate_repr["params"].items()
    ]
    intermediate_repr["returns"] = (
        lambda k: dict(name=k, **intermediate_repr["returns"][k])
    )("return_type")

    return intermediate_repr


def class_with_method(class_def, method_name):
    """
    Converts an AST of a class with a method to our IR

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[Module, ClassDef]```

    :param method_name: Method name
    :type method_name: ```str```

    :return: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :rtype: ```dict```
    """
    assert isinstance(class_def, ClassDef), "Expected 'ClassDef' got {!r}".format(
        type(class_def).__name__
    )
    return function(
        function_def=next(
            node
            for node in find_ast_type(class_def).body
            if isinstance(node, FunctionDef) and node.name == method_name
        )
    )


def function(function_def):
    """
    Converts an AST of a class with a method to our IR

    :param function_def: FunctionDef
    :type function_def: ```FunctionDef```

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

    intermediate_repr = docstring(
        get_docstring(function_def).replace(":cvar", ":param")
    )
    function_type = get_function_type(function_def)
    offset = 0 if function_type is None else 1

    if len(function_def.body) > 2:
        intermediate_repr["_internal"] = {"body": function_def.body[1:-1]}

    for idx, arg in enumerate(function_def.args.args):
        if arg.annotation is not None:
            intermediate_repr["params"][idx - offset]["typ"] = to_code(
                arg.annotation
            ).rstrip("\n")

    for idx, const in enumerate(function_def.args.defaults):
        assert (
            isinstance(const, Constant)
            and const.kind is None
            or isinstance(const, (Str, NameConstant))
        ), type(const).__name__
        value = get_value(const)
        if value is not None:
            intermediate_repr["params"][idx]["default"] = value

    if function_def.args.kwarg:
        assert intermediate_repr["params"][-1]["name"] == function_def.args.kwarg.arg
        intermediate_repr["params"][-1]["typ"] = "dict"

    # Convention - the final top-level `return` is the default
    return_ast = next(
        filter(rpartial(isinstance, Return), function_def.body[::-1]), None
    )
    if return_ast is not None and return_ast.value is not None:
        intermediate_repr["returns"]["default"] = (
            lambda default: "({})".format(default)
            if isinstance(return_ast.value, Tuple)
            and (not default.startswith("(") or not default.endswith(")"))
            else default
        )(to_code(return_ast.value).rstrip("\n"))

    if isinstance(function_def.returns, Subscript):
        intermediate_repr["returns"]["typ"] = to_code(function_def.returns).rstrip("\n")

    return intermediate_repr


def argparse_ast(function_def):
    """
    Converts an AST to our IR

    :param function_def: AST of argparse function
    :type function_def: ```FunctionDef``

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
        "short_description": "",
        "long_description": "",
        "params": [],
    }
    ir = parse_docstring(doc_string, emit_default_doc=True)

    for e in function_def.body:
        if is_argparse_add_argument(e):
            intermediate_repr["params"].append(parse_out_param(e))
        elif isinstance(e, Assign):
            if is_argparse_description(e):
                intermediate_repr["short_description"] = get_value(e.value)
        elif isinstance(e, Return) and isinstance(e.value, Tuple):
            intermediate_repr["returns"] = _parse_return(
                e,
                intermediate_repr=ir,
                function_def=function_def,
                emit_default_doc=True,
            )
    if len(function_def.body) > len(intermediate_repr["params"]) + 3:
        intermediate_repr["_internal"] = {
            "body": list(
                filterfalse(
                    lambda node: (
                        isinstance(node, Expr)
                        and isinstance(get_value(node), (Constant, Str))
                        and doc_string
                        == to_docstring(
                            ir,
                            emit_types=True,
                            emit_separating_tab=False,
                            indent_level=0,
                        )[1:-1]
                    ),
                    filterfalse(
                        is_argparse_description,
                        filterfalse(is_argparse_add_argument, function_def.body),
                    ),
                )
            )[:-1]
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
    assert isinstance(doc_string, str), "Expected 'str' got `{!r}`".format(
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


def to_docstring(
    intermediate_repr,
    emit_default_doc=True,
    docstring_format="rest",
    indent_level=2,
    emit_types=False,
    emit_separating_tab=True,
):
    """
    Converts a docstring to an AST

    :param intermediate_repr: a dictionary of form
          {
              'short_description': ...,
              'long_description': ...,
              'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
              "returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpy', 'google']```

    :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_types: whether to show `:type` lines
    :type emit_types: ```bool```

    :param emit_separating_tab: whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :return: docstring
    :rtype: ```str```
    """
    assert isinstance(intermediate_repr, dict), "Expected 'dict' got `{!r}`".format(
        type(intermediate_repr).__name__
    )
    if docstring_format != "rest":
        raise NotImplementedError(docstring_format)

    def param2docstring_param(
        param,
        docstring_format="rest",
        emit_default_doc=True,
        indent_level=1,
        emit_types=False,
    ):
        """
        Converts param dict from intermediate_repr to the right string representation

        :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
        :type param: ```dict```

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpy', 'google']```

        :param emit_default_doc: Whether help/docstring should include 'With default' text
        :type emit_default_doc: ```bool``

        :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
        :type indent_level: ```int```

        :param emit_types: whether to show `:type` lines
        :type emit_types: ```bool```
        """
        assert isinstance(param, dict), "Expected 'dict' got `{!r}`".format(
            type(param).__name__
        )
        doc, default = extract_default(param["doc"], emit_default_doc=False)
        if default is not None:
            param["default"] = default

        param["typ"] = (
            "**{param[name]}".format(param=param)
            if param.get("typ") == "dict" and param["name"].endswith("kwargs")
            else param.get("typ")
        )

        return "".join(
            filter(
                None,
                (
                    "{tab}:param {param[name]}: {param[doc]}".format(
                        tab=tab * indent_level,
                        param=set_default_doc(param, emit_default_doc=emit_default_doc),
                    ),
                    None
                    if param["typ"] is None or not emit_types
                    else "\n{tab}:type {param[name]}: ```{param[typ]}```".format(
                        tab=tab * indent_level, param=param
                    ),
                ),
            )
        )

    param2docstring_param = partial(
        param2docstring_param,
        emit_default_doc=emit_default_doc,
        docstring_format=docstring_format,
        indent_level=indent_level,
        emit_types=emit_types,
    )
    sep = tab if emit_separating_tab else ""
    return "\n{tab}{description}\n{sep}\n{params}\n{sep}\n{returns}\n{tab}".format(
        sep=sep,
        tab=tab * indent_level,
        description=intermediate_repr.get("long_description")
        or intermediate_repr["short_description"],
        params="\n{sep}\n".format(sep=sep).join(
            map(param2docstring_param, intermediate_repr["params"])
        ),
        returns=(
            param2docstring_param(intermediate_repr["returns"])
            .replace(":param return_type:", ":return:")
            .replace(":type return_type:", ":rtype:")
        )
        if intermediate_repr.get("returns")
        else "",
    )


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
    "class_with_method",
    "function",
    "argparse_ast",
    "docstring",
    "to_docstring",
    "docstring_parser",
]

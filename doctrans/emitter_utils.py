"""
Functions which produce intermediate_repr from various different inputs
"""
import ast
from ast import Constant, Name, Return
from functools import partial
from typing import Any

from doctrans.ast_utils import get_value
from doctrans.defaults_utils import extract_default, set_default_doc
from doctrans.pure_utils import simple_types, identity, tab, quote
from doctrans.source_transformer import to_code


def _handle_value(node):
    """
    Handle keyword.value types, returning the correct one as a `str` or `Any`

    :param node: AST node from keyword.value
    :type node: ```Name```

    :return: `str` or `Any`, representing the type for argparse
    :rtype: ```Union[str, Any]```
    """
    # if isinstance(node, Attribute): return Any
    if isinstance(node, Name):
        return "dict" if node.id == "loads" else node.id
    raise NotImplementedError(type(node).__name__)


def _handle_keyword(keyword, typ):
    """
    Decide which type to wrap the keyword tuples in

    :param keyword: AST keyword
    :type keyword: ```ast.keyword```

    :param typ: string representation of type
    :type typ: ```str```

    :return: string representation of type
    :rtype: ```str``
    """
    quote_f = identity

    type_ = "Union"
    if typ == Any or typ in simple_types:
        if typ == "str" or typ == Any:

            def quote_f(s):
                """
                Wrap the input in quotes

                :param s: Any value
                :type s: ```Any```

                :return: the input value
                :rtype: ```Any```
                """
                return "'{}'".format(s)

        type_ = "Literal"

    return "{type}[{types}]".format(
        type=type_,
        types=", ".join(quote_f(get_value(elt)) for elt in keyword.value.elts),
    )


def parse_out_param(expr, emit_default_doc=True):
    """
    Turns the class_def repr of '--dataset_name', type=str, help='name of dataset.', required=True, default='mnist'
      into
          {'name': 'dataset_name', 'typ': 'str', doc='name of dataset.',
           'required': True, 'default': 'mnist'}

    :param expr: Expr
    :type expr: ```Expr```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    required = next(
        (keyword for keyword in expr.value.keywords if keyword.arg == "required"),
        Constant(value=False, constant_value=None, string=None),
    ).value

    typ = next(
        (
            _handle_value(get_value(keyword))
            for keyword in expr.value.keywords
            if keyword.arg == "type"
        ),
        "str",
    )
    name = get_value(expr.value.args[0])[len("--") :]
    default = next(
        (
            get_value(key_word.value)
            for key_word in expr.value.keywords
            if key_word.arg == "default"
        ),
        None,
    )
    doc = (
        lambda help_: help_
        if help_ is None
        else (
            help_
            if default is None
            or emit_default_doc is False
            or (hasattr(default, "__len__") and len(default) == 0)
            or "defaults to" in help_
            or "Defaults to" in help_
            else "{help} Defaults to {default}".format(
                help=help_ if help_.endswith(".") else "{}.".format(help_),
                default=default,
            )
        )
    )(
        next(
            (
                get_value(key_word.value)
                for key_word in expr.value.keywords
                if key_word.arg == "help"
            ),
            None,
        )
    )
    if default is None:
        doc, default = extract_default(doc, emit_default_doc=emit_default_doc)
    if default is None and typ in simple_types and required:
        default = simple_types[typ]

    # nargs = next(
    #     (
    #         get_value(key_word.value)
    #         for key_word in expr.value.keywords
    #         if key_word.arg == "nargs"
    #     ),
    #     None,
    # )

    action = next(
        (
            get_value(key_word.value)
            for key_word in expr.value.keywords
            if key_word.arg == "action"
        ),
        None,
    )

    typ = next(
        (
            _handle_keyword(keyword, typ)
            for keyword in expr.value.keywords
            if keyword.arg == "choices"
        ),
        typ,
    )
    if action == "append":
        typ = "List[{typ}]".format(typ=typ)

    if not required and not name.endswith("kwargs"):
        typ = "Optional[{typ}]".format(typ=typ)

    if "str" in typ or "Literal" in typ and (typ.count("'") > 1 or typ.count('"') > 1):
        default = quote(default)

    return dict(
        name=name, doc=doc, typ=typ, **({} if default is None else {"default": default})
    )


def interpolate_defaults(param, default_search_announce=None, emit_default_doc=True):
    """
    Correctly set the 'default' and 'doc' parameters

    :param param: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'required': ... }
    :type param: ```dict```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    if "doc" in param:
        doc, default = extract_default(
            param["doc"],
            default_search_announce=default_search_announce,
            emit_default_doc=emit_default_doc,
        )
        param["doc"] = doc
        if default is not None:
            param["default"] = default
    return param


def _parse_return(e, intermediate_repr, function_def, emit_default_doc):
    """
    Parse return into a param dict

    :param e: Return AST node
    :type e: Return

    :param intermediate_repr: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :param function_def: AST node for function definition
    :type function_def: ```FunctionDef```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool``

    :return: dict of shape {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
    :rtype: ```dict```
    """
    assert isinstance(e, Return)

    return set_default_doc(
        {
            "name": "return_type",
            "doc": extract_default(
                next(
                    line.partition(",")[2].lstrip()
                    for line in get_value(function_def.body[0].value).split("\n")
                    if line.lstrip().startswith(":return")
                ),
                emit_default_doc=emit_default_doc,
            )[0],
            "default": to_code(e.value.elts[1]).rstrip("\n"),
            "typ": to_code(
                get_value(
                    ast.parse(intermediate_repr["returns"]["typ"]).body[0].value.slice
                ).elts[1]
            ).rstrip()
            # 'Tuple[ArgumentParser, {typ}]'.format(typ=intermediate_repr['returns']['typ'])
        },
        emit_default_doc=emit_default_doc,
    )


def get_internal_body(target_name, target_type, intermediate_repr):
    """
    Get the internal body from our IR

    :param target_name: name of target. If both `target_name` and `target_type` match internal body extract, then emit
    :type target_name: ```str```

    :param target_type: Type of target, static is static or global method, others just become first arg
    :type target_type: ```Literal['self', 'cls', 'static']```

    :param intermediate_repr: a dictionary of form
          {
                  'name': ...,
                  'type': ...,
                  '_internal': {'body': [...]},
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
          }
    :type intermediate_repr: ```dict```

    :return: Internal body or an empty list
    :rtype: ```list```
    """
    return (
        intermediate_repr["_internal"]["body"]
        if intermediate_repr.get("_internal", {}).get("body")
        and intermediate_repr["_internal"]["from_name"] == target_name
        and intermediate_repr["_internal"]["from_type"] == target_type
        else tuple()
    )


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
                  'name': ...,
                  'type': ...,
                  'doc': ...,
                  'params': [{'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }, ...],
                  'returns': {'name': ..., 'typ': ..., 'doc': ..., 'default': ..., 'required': ... }
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
        assert docstring_format == "rest", docstring_format
        if "doc" in param:
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
    sep = (tab * indent_level) if emit_separating_tab else ""
    return "\n{tab}{description}\n{sep}\n{params}\n{returns}".format(
        sep=sep,
        tab=tab,
        description=intermediate_repr["doc"],
        params="\n{sep}\n".format(sep=sep).join(
            map(param2docstring_param, intermediate_repr["params"])
        ),
        returns=(
            "{sep}\n{returns}\n{tab}".format(
                sep=sep,
                returns=param2docstring_param(intermediate_repr["returns"])
                .replace(":param return_type:", ":return:")
                .replace(":type return_type:", ":rtype:"),
                tab=tab,
            )
            if intermediate_repr.get("returns")
            else ""
        ),
    )


__all__ = [
    "_parse_return",
    "get_internal_body",
    "interpolate_defaults",
    "parse_out_param",
    "to_docstring",
]

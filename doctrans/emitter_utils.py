"""
Functions which produce intermediate_repr from various different inputs
"""
import ast
from ast import Attribute, Expr, FunctionDef, Load, Name, Return, arguments
from functools import partial
from operator import add
from typing import Any

from doctrans.ast_utils import (
    NoneStr,
    code_quoted,
    get_value,
    maybe_type_comment,
    set_arg,
    set_value,
)
from doctrans.defaults_utils import extract_default, set_default_doc
from doctrans.pure_utils import identity, none_types, simple_types, tab, unquote
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
        return "Optional[dict]" if node.id == "loads" else node.id
    raise NotImplementedError(type(node).__name__)


def _handle_keyword(keyword, typ):
    """
    Decide which type to wrap the keyword tuples in

    :param keyword: AST keyword
    :type keyword: ```ast.keyword```

    :param typ: string representation of type
    :type typ: ```str```

    :return: string representation of type
    :rtype: ```str```
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


def parse_out_param(expr, require_default=False, emit_default_doc=True):
    """
    Turns the class_def repr of '--dataset_name', type=str, help='name of dataset.', required=True, default='mnist'
      into
           Tuple[Literal['dataset_name'], {"typ": Literal["str"], "doc": Literal["name of dataset."],
                                           "default": Literal["mnist"]}]

    :param expr: Expr
    :type expr: ```Expr```

    :param require_default: Whether a default is required, if not found in doc, infer the proper default from type
    :type require_default: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    required = get_value(
        get_value(
            next(
                (
                    keyword
                    for keyword in expr.value.keywords
                    if keyword.arg == "required"
                ),
                set_value(False),
            )
        )
    )

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
                if key_word.arg == "help" and key_word.value
            ),
            None,
        )
    )
    if default is None:
        doc, default = extract_default(doc, emit_default_doc=emit_default_doc)
    if default is None:
        if required:
            # if name.endswith("kwargs"):
            #    default = NoneStr
            # else:
            default = simple_types[typ] if typ in simple_types else NoneStr

        elif require_default or typ.startswith("Optional"):
            default = NoneStr

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

    if not required and "Optional" not in typ:
        typ = "Optional[{typ}]".format(typ=typ)

    # if "str" in typ or "Literal" in typ and (typ.count("'") > 1 or typ.count('"') > 1):
    #    default = quote(default)

    return name, dict(
        doc=doc, typ=typ, **({} if default is None else {"default": default})
    )


def interpolate_defaults(
    param, default_search_announce=None, require_default=False, emit_default_doc=True
):
    """
    Correctly set the 'default' and 'doc' parameters

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param default_search_announce: Default text(s) to look for. If None, uses default specified in default_utils.
    :type default_search_announce: ```Optional[Union[str, Iterable[str]]]```

    :param require_default: Whether a default is required, if not found in doc, infer the proper default from type
    :type require_default: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    name, _param = param
    del param
    if "doc" in _param:
        doc, default = extract_default(
            _param["doc"],
            typ=_param.get("typ"),
            default_search_announce=default_search_announce,
            emit_default_doc=emit_default_doc,
        )
        _param["doc"] = doc
        if default is not None:
            _param["default"] = unquote(default)
    if require_default and _param.get("default") is None:
        # if (
        #     "typ" in _param
        #     and _param["typ"] not in frozenset(("Any", "object"))
        #     and not _param["typ"].startswith("Optional")
        # ):
        #     _param["typ"] = "Optional[{}]".format(_param["typ"])
        _param["default"] = (
            simple_types[_param["typ"]]
            if _param.get("typ", memoryview) in simple_types
            else NoneStr
        )
    return name, _param


def _parse_return(e, intermediate_repr, function_def, emit_default_doc):
    """
    Parse return into a param dict

    :param e: Return AST node
    :type e: Return

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param function_def: AST node for function definition
    :type function_def: ```FunctionDef```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Name, dict with keys: 'typ', 'doc', 'default'
    :rtype: ```Tuple[str, dict]```
    """
    assert isinstance(e, Return)

    return set_default_doc(
        (
            "return_type",
            {
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
                        ast.parse(intermediate_repr["returns"]["return_type"]["typ"])
                        .body[0]
                        .value.slice
                    ).elts[1]
                ).rstrip()
                # 'Tuple[ArgumentParser, {typ}]'.format(typ=intermediate_repr['returns']['typ'])
            },
        ),
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
        {  "name": Optional[str],
           "type": Optional[str],
           "_internal": {'body': List[ast.AST]},
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :return: Internal body or an empty tuple
    :rtype: ```Union[list, tuple]```
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
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

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

    def _param2docstring_param(
        param,
        docstring_format="rest",
        emit_default_doc=True,
        indent_level=1,
        emit_types=False,
    ):
        """
        Converts param dict from intermediate_repr to the right string representation

        :param param: Name, dict with keys: 'typ', 'doc', 'default'
        :type param: ```Tuple[str, dict]```

        :param docstring_format: Format of docstring
        :type docstring_format: ```Literal['rest', 'numpy', 'google']```

        :param emit_default_doc: Whether help/docstring should include 'With default' text
        :type emit_default_doc: ```bool```

        :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
        :type indent_level: ```int```

        :param emit_types: whether to show `:type` lines
        :type emit_types: ```bool```
        """
        # if not isinstance(param, tuple):
        assert isinstance(param, tuple), "Expected 'tuple' got `{!r}`".format(
            type(param).__name__
        )
        assert docstring_format == "rest", docstring_format
        name, _param = param
        del param
        if "doc" in _param:
            doc, default = extract_default(
                _param["doc"], emit_default_doc=emit_default_doc
            )
            if default is not None:
                _param["default"] = default

        _param["typ"] = (
            "**{name}".format(name=name)
            if _param.get("typ") == "dict" and name.endswith("kwargs")
            else _param.get("typ")
        )

        return "".join(
            filter(
                None,
                (
                    (
                        lambda name_param: "{tab}:param {name}: {doc}".format(
                            tab=tab * indent_level,
                            name=name_param[0],
                            doc=name_param[1].get("doc"),
                        )
                    )(
                        set_default_doc(
                            (name, _param), emit_default_doc=emit_default_doc
                        )
                    ),
                    None
                    if _param["typ"] is None or not emit_types
                    else "\n{tab}:type {name}: ```{_param[typ]}```".format(
                        tab=tab * indent_level, name=name, _param=_param
                    ),
                ),
            )
        )

    param2docstring_param = partial(
        _param2docstring_param,
        emit_default_doc=emit_default_doc,
        docstring_format=docstring_format,
        indent_level=indent_level,
        emit_types=emit_types,
    )
    sep = (tab * indent_level) if emit_separating_tab else ""
    return "\n{description}\n{sep}\n{params}\n{returns}".format(
        sep=sep,
        description="\n".join(
            map(partial(add, tab), intermediate_repr["doc"].split("\n"))
        ),
        params="\n{sep}\n".format(sep=sep).join(
            map(
                partial(param2docstring_param, emit_default_doc=emit_default_doc),
                intermediate_repr["params"].items(),
            )
        ),
        returns=(
            "{sep}\n{returns}\n{tab}".format(
                sep=sep,
                returns=param2docstring_param(
                    next(iter(intermediate_repr["returns"].items())),
                    emit_default_doc=emit_default_doc,
                )
                .replace(":param return_type:", ":return:")
                .replace(":type return_type:", ":rtype:"),
                tab=tab,
            )
            if intermediate_repr.get("returns")
            else ""
        ),
    )


class RewriteName(ast.NodeTransformer):
    """
    A :class:`NodeTransformer` subclass that walks the abstract syntax tree and
    allows modification of nodes. Here it modifies parameter names to be `self.param_name`
    """

    def __init__(self, node_ids):
        """
        Set parameter

        :param node_ids: Container of AST `id`s to match for rename
        :type node_ids: ```Optional[Iterator[str]]```
        """
        self.node_ids = node_ids

    def visit_Name(self, node):
        """
        Rename parameter name with a `self.` attribute prefix

        :param node: The AST node
        :type node: ```Name```

        :return: `Name` iff `Name` is not a parameter else `Attribute`
        :rtype: ```Union[Name, Attribute]```
        """
        # print("loc:", getattr(node, "_location", None), ";")
        return (
            Attribute(Name("self", Load()), node.id, Load())
            if not self.node_ids or node.id in self.node_ids
            else ast.NodeTransformer.generic_visit(self, node)
        )


def _make_call_meth(body, return_type, param_names):
    """
    Construct a `__call__` method from the provided `body`

    :param body: The body, probably from a FunctionDef.body
    :type body: ```List[AST]```

    :param return_type: The return type of the parent symbol (probably class). Used to fill in `__call__` return.
    :type return_type: ```Optional[str]```

    :param param_names: Container of AST `id`s to match for rename
    :type param_names: ```Optional[Iterator[str]]```

    :return: Internal function for `__call__`
    :rtype: ```FunctionDef```
    """
    body_len = len(body)
    # return_ = (
    #     ast.fix_missing_locations(
    #         RewriteName(param_names).visit(
    #             Return(get_value(ast.parse(return_type.strip("`")).body[0]), expr=None)
    #         )
    #     )
    #     if return_type is not None and code_quoted(return_type)
    #     else None
    # )
    if body_len:
        if isinstance(body, dict):
            body = list(
                filter(
                    None,
                    (
                        None
                        if body["doc"] in none_types
                        else Expr(
                            set_value("\n:return: {doc}\n\n".format(doc=body["doc"]))
                        ),
                        RewriteName(param_names).visit(
                            Return(
                                get_value(ast.parse(return_type.strip("`")).body[0]),
                                expr=None,
                            )
                        )
                        if code_quoted(body["default"])
                        else Return(set_value(body["default"]), expr=None),
                    ),
                )
            )

        # elif isinstance(body[0], Expr):
        #     doc_str = get_value(body[0].value)
        #     if isinstance(doc_str, str) and body_len > 0:
        #         body = (
        #             body[1:]
        #             if body_len > 1
        #             else (
        #                 [
        #                     set_value(doc_str.replace(":cvar", ":param"))
        #                     if return_ is None
        #                     else return_
        #                 ]
        #                 if body_len == 1
        #                 else body
        #             )
        #         )
        # elif not isinstance(body[0], Return) and return_ is not None:
        #     body.append(return_)
    # elif return_ is not None:
    #    body = [return_]

    return (
        ast.fix_missing_locations(
            FunctionDef(
                args=arguments(
                    args=[set_arg("self")],
                    defaults=[],
                    kw_defaults=[],
                    kwarg=None,
                    kwonlyargs=[],
                    posonlyargs=[],
                    vararg=None,
                    arg=None,
                ),
                body=body,
                decorator_list=[],
                name="__call__",
                returns=None,
                arguments_args=None,
                identifier_name=None,
                stmt=None,
                lineno=None,
                **maybe_type_comment
            )
        )
        if body
        else None
    )


def ast_parse_fix(s):
    """
    Hack to resolve unbalanced parentheses SyntaxError acquired from PyTorch parsing
    TODO: remove

    :param s: String to parse
    :type s: ```str```

    :return: Value
    """
    balanced = (s.count("[") + s.count("]")) & 1 == 0
    return ast.parse(s if balanced else "{}]".format(s)).body[0].value


__all__ = [
    "_parse_return",
    "ast_parse_fix",
    "get_internal_body",
    "interpolate_defaults",
    "parse_out_param",
    "to_docstring",
]

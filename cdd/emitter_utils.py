"""
Functions which produce intermediate_repr from various different inputs
"""
import ast
from ast import (
    Attribute,
    Call,
    Expr,
    FunctionDef,
    Load,
    Name,
    Return,
    arg,
    arguments,
    keyword,
)
from functools import partial
from platform import system
from textwrap import indent
from typing import Any

from cdd.ast_utils import (
    NoneStr,
    code_quoted,
    get_value,
    maybe_type_comment,
    set_arg,
    set_value,
    typ2column_type,
    typ2json_type,
)
from cdd.defaults_utils import extract_default, set_default_doc
from cdd.docstring_utils import emit_param_str
from cdd.pure_utils import (
    fill,
    identity,
    indent_all_but_first,
    line_length,
    multiline,
    none_types,
    simple_types,
    tab,
    unquote,
    update_d,
)
from cdd.source_transformer import to_code
from cdd.tests.mocks.docstrings import docstring_repr_google_str, docstring_repr_str


def _handle_value(node):
    """
    Handle keyword.value types, returning the correct one as a `str` or `Any`

    :param node: AST node from keyword.value
    :type node: ```Name```

    :returns: `str` or `Any`, representing the type for argparse
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

    :returns: string representation of type
    :rtype: ```str```
    """
    quote_f = identity

    type_ = "Union"
    if typ == Any or typ in simple_types:
        if typ in ("str", Any):

            def quote_f(s):
                """
                Wrap the input in quotes

                :param s: Any value
                :type s: ```Any```

                :returns: the input value
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

    :returns: Name, dict with keys: 'typ', 'doc', 'default'
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
            _handle_value(get_value(key_word))
            for key_word in expr.value.keywords
            if key_word.arg == "type"
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

    :returns: Name, dict with keys: 'typ', 'doc', 'default'
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

    :returns: Name, dict with keys: 'typ', 'doc', 'default'
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

    :returns: Internal body or an empty tuple
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
    word_wrap=True,
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
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_types: whether to show `:type` lines
    :type emit_types: ```bool```

    :param emit_separating_tab: whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :returns: docstring
    :rtype: ```str```
    """
    assert isinstance(intermediate_repr, dict), "Expected 'dict' got `{!r}`".format(
        type(intermediate_repr).__name__
    )
    if docstring_format != "rest":
        raise NotImplementedError(docstring_format)

    def _fill(s):
        """
        Word wrap if length suggests

        :param s: Input string
        :type s: ```str```

        :returns: Potentially word wrapped + 1+ indented output
        :rtype: ```str```
        """
        return (
            indent_all_but_first(fill(s), indent_level + 1, wipe_indents=True)
            if word_wrap and any(len(line) > line_length for line in s.splitlines())
            else s
        )

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
        :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

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
        name, _param = param
        del param
        if "doc" in _param:
            doc, default = extract_default(
                _param["doc"], emit_default_doc=emit_default_doc
            )
            if default is not None:
                _param["default"] = default

        _sep = abs(indent_level) * tab

        def _joiner(__param, param_type):
            """
            Internal function to join new lines

            :param __param: The `:param`
            :type __param: ```Optional[str]```

            :param param_type: The `:type`
            :type param_type: ```Optional[str]```

            :returns: Newline joined string
            :rtype: ```str```
            """
            if param_type is None and __param is not None:
                return "{}\n{}".format(_fill(__param), _sep)
            elif __param is None:
                return __param
            return "".join(
                (
                    _fill(__param.replace("\n", "\n{sep}".format(sep=_sep))),
                    "\n",
                    _sep,
                    _fill(param_type.replace("\n", "\n{sep}".format(sep=_sep))),
                    "\n",
                    _sep,
                )
            )

        return _joiner(
            (
                emit_param_str(
                    (
                        name,
                        update_d(
                            _param,
                            doc=multiline(
                                indent_all_but_first(
                                    set_default_doc(
                                        (name, _param),
                                        emit_default_doc=emit_default_doc,
                                    )[1]["doc"],
                                    indent_level=indent_level - 1,
                                ),
                                quote_with=("", ""),
                            ),
                        ),
                    ),
                    emit_type=False,
                    emit_default_doc=emit_default_doc,
                    style=docstring_format,
                    word_wrap=word_wrap,
                )
            )
            if _param.get("doc")
            else None,
            (
                None
                if _param.get("typ") is None or not emit_types
                else emit_param_str(
                    (name, _param),
                    emit_doc=False,
                    emit_default_doc=emit_default_doc,
                    style=docstring_format,
                    word_wrap=word_wrap,
                )
            ),
        )

    param2docstring_param = partial(
        _param2docstring_param,
        emit_default_doc=emit_default_doc,
        docstring_format=docstring_format,
        indent_level=indent_level,
        emit_types=emit_types,
    )

    sep = (tab * abs(indent_level)) if emit_separating_tab else ""

    return "{header}{params}{returns}".format(
        header="\n{description}{afterward}{sep}".format(
            sep=sep,
            description=indent(_fill(intermediate_repr["doc"]), sep),
            afterward=""
            if intermediate_repr["doc"].rstrip(" \t").endswith("\n")
            else "\n",
        )
        if intermediate_repr.get("doc")
        else "",
        params="\n{sep}{s}\n{sep}".format(
            sep=sep,
            s="\n{sep}".format(sep=sep).join(
                filter(
                    None,
                    map(
                        partial(
                            param2docstring_param, emit_default_doc=emit_default_doc
                        ),
                        intermediate_repr["params"].items(),
                    ),
                ),
            ),
        )
        if intermediate_repr.get("params")
        else "",
        returns=(
            "{returns}\n{sep}".format(
                returns=param2docstring_param(
                    next(iter(intermediate_repr["returns"].items())),
                    emit_default_doc=emit_default_doc,
                ).rstrip(),
                sep=sep,
            )
            if (intermediate_repr.get("returns") or {"return_type": {}})["return_type"]
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

        :returns: `Name` iff `Name` is not a parameter else `Attribute`
        :rtype: ```Union[Name, Attribute]```
        """
        # print("loc:", getattr(node, "_location", None), ";")
        return (
            Attribute(Name("self", Load()), node.id, Load())
            if not self.node_ids or node.id in self.node_ids
            else ast.NodeTransformer.generic_visit(self, node)
        )


def _make_call_meth(body, return_type, param_names, docstring_format, word_wrap):
    """
    Construct a `__call__` method from the provided `body`

    :param body: The body, probably from a FunctionDef.body
    :type body: ```List[AST]```

    :param return_type: The return type of the parent symbol (probably class). Used to fill in `__call__` return.
    :type return_type: ```Optional[str]```

    :param param_names: Container of AST `id`s to match for rename
    :type param_names: ```Optional[Iterator[str]]```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :returns: Internal function for `__call__`
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
                        if body.get("doc") in none_types
                        else Expr(
                            set_value(
                                emit_param_str(
                                    (
                                        "return_type",
                                        {
                                            "doc": multiline(
                                                indent_all_but_first(body["doc"])
                                            )
                                        },
                                    ),
                                    style=docstring_format,
                                    word_wrap=word_wrap,
                                )
                            )
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

    :returns: Value
    """
    balanced = (s.count("[") + s.count("]")) & 1 == 0
    return ast.parse(s if balanced else "{}]".format(s)).body[0].value


def param2json_schema_property(param, required):
    """
    Turn a param into a JSON schema property

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param required: Required parameters. This function may push to the list.
    :type required: ```List[str]```

    :returns: JSON schema property. Also may push to `required`.
    :rtype: ```dict```
    """
    name, _param = param
    del param

    if _param.get("doc"):
        _param["description"] = _param.pop("doc")
    if _param.get("typ", ast) is not ast:
        _param["type"] = _param.pop("typ")
        if _param["type"].startswith("Optional["):
            _param["type"] = _param["type"][len("Optional[") : -1]
        else:
            required.append(name)

        if _param["type"].startswith("Literal["):
            parsed_typ = get_value(ast.parse(_param["type"]).body[0])
            assert (
                parsed_typ.value.id == "Literal"
            ), "Only basic Literal support is implemented, not {}".format(
                parsed_typ.value.id
            )
            _param["enum"] = list(map(get_value, get_value(parsed_typ.slice).elts))
            _param["type"] = typ2json_type[type(_param["enum"][0]).__name__]
        else:
            _param["type"] = typ2json_type[_param["type"]]
    if _param.get("default", False) in none_types:
        del _param["default"]  # Will be inferred as `null` from the type
    return name, _param


def param_to_sqlalchemy_column_call(param, include_name):
    """
    Turn a param into a `Column(…)`

    :param param: Name, dict with keys: 'typ', 'doc', 'default'
    :type param: ```Tuple[str, dict]```

    :param include_name: Whether to include the name (exclude in declarative base)
    :type include_name: ```bool```

    :returns: Form of: `Column(…)`
    :rtype: ```Call```
    """
    if system() == "Darwin":
        print("param_to_sqlalchemy_column_call::include_name:", include_name, ";")
    name, _param = param
    del param

    args, keywords, nullable = [], [], None

    if _param["typ"].startswith("Optional["):
        _param["typ"] = _param["typ"][len("Optional[") : -1]
        nullable = True

    if include_name:
        args.append(set_value(name))

    if "Literal[" in _param["typ"]:
        parsed_typ = get_value(ast.parse(_param["typ"]).body[0])
        assert (
            parsed_typ.value.id == "Literal"
        ), "Only basic Literal support is implemented, not {}".format(
            parsed_typ.value.id
        )
        args.append(
            Call(
                func=Name("Enum", Load()),
                args=get_value(parsed_typ.slice).elts,
                keywords=[keyword(arg="name", value=set_value(name), identifier=None)],
                expr=None,
                expr_func=None,
            )
        )

    else:
        args.append(Name(typ2column_type[_param["typ"]], Load()))

    has_default = _param.get("default", ast) is not ast
    pk = _param.get("doc", "").startswith("[PK]")
    if pk:
        _param["doc"] = _param["doc"][4:].lstrip()
    elif has_default and _param["default"] not in none_types:
        nullable = False

    keywords.append(
        keyword(arg="doc", value=set_value(_param["doc"].rstrip(".")), identifier=None)
    )

    if has_default:
        if _param["default"] == NoneStr:
            _param["default"] = None
        keywords.append(
            keyword(
                arg="default",
                value=set_value(_param["default"]),
                identifier=None,
            )
        )

    # Sorting :\
    if pk:
        keywords.append(
            keyword(arg="primary_key", value=set_value(True), identifier=None),
        )

    if isinstance(nullable, bool):
        keywords.append(
            keyword(arg="nullable", value=set_value(nullable), identifier=None)
        )

    return Call(
        func=Name("Column", Load()),
        args=args,
        keywords=keywords,
        expr=None,
        expr_func=None,
    )


def generate_repr_method(params, cls_name, docstring_format):
    """
    Generate a `__repr__` method with all params, using `str.format` syntax

    :param params: an `OrderedDict` of form
        OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
    :type params: ```OrderedDict```

    :param cls_name: Name of class
    :type cls_name: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :returns: `__repr__` method
    :rtype: ```FunctionDef```
    """
    keys = tuple(params.keys())
    return FunctionDef(
        name="__repr__",
        args=arguments(
            posonlyargs=[],
            arg=None,
            args=[
                arg(
                    arg="self",
                    annotation=None,
                    expr=None,
                    identifier_arg=None,
                    **maybe_type_comment
                )
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        ),
        body=[
            Expr(
                set_value(
                    """\n{sep}{_repr_docstring}""".format(
                        sep=tab * 2,
                        _repr_docstring=(
                            docstring_repr_str
                            if docstring_format == "rest"
                            else docstring_repr_google_str
                        ).lstrip(),
                    )
                )
            ),
            Return(
                value=Call(
                    func=Attribute(
                        set_value(
                            "{cls_name}({format_args})".format(
                                cls_name=cls_name,
                                format_args=", ".join(
                                    map("{0}={{{0}!r}}".format, keys)
                                ),
                            )
                        ),
                        "format",
                        Load(),
                    ),
                    args=[],
                    keywords=list(
                        map(
                            lambda key: keyword(
                                arg=key,
                                value=Attribute(Name("self", Load()), key, Load()),
                                identifier=None,
                            ),
                            keys,
                        )
                    ),
                    expr=None,
                    expr_func=None,
                ),
                expr=None,
            ),
        ],
        decorator_list=[],
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        lineno=None,
        returns=None,
        **maybe_type_comment
    )


__all__ = [
    "_parse_return",
    "ast_parse_fix",
    "get_internal_body",
    "interpolate_defaults",
    "parse_out_param",
    "param_to_sqlalchemy_column_call",
    "to_docstring",
]

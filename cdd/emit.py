"""
Transform from string or AST representations of input, to AST, file, or str input_str.
"""

import ast
from ast import (
    Assign,
    Attribute,
    Call,
    ClassDef,
    Expr,
    FunctionDef,
    Load,
    Module,
    Name,
    Return,
    Store,
    Tuple,
    arguments,
    keyword,
)
from collections import OrderedDict
from functools import partial
from importlib import import_module
from itertools import chain
from operator import add
from sys import modules

from cdd.ast_utils import (
    get_value,
    maybe_type_comment,
    param2argparse_param,
    param2ast,
    set_arg,
    set_value,
)
from cdd.docstring_utils import (
    ARG_TOKENS,
    RETURN_TOKENS,
    emit_param_str,
    header_args_footer_to_str,
    parse_docstring_into_header_args_footer,
)
from cdd.emitter_utils import (
    RewriteName,
    _make_call_meth,
    generate_repr_method,
    get_internal_body,
    param2json_schema_property,
    param_to_sqlalchemy_column_call,
)
from cdd.pure_utils import (
    PY3_8,
    code_quoted,
    deindent,
    fill,
    identity,
    indent_all_but_first,
    none_types,
    num_of_nls,
    rpartial,
    simple_types,
    tab,
)
from cdd.source_transformer import to_code

black = (
    import_module("black")
    if "black" in modules
    else type(
        "black",
        tuple(),
        {
            "format_str": lambda src_contents, mode: src_contents,
            "Mode": lambda target_versions, line_length, is_pyi, string_normalization: None,
        },
    )
)


def argparse_function(
    intermediate_repr,
    emit_default_doc=False,
    function_name="set_cli_args",
    function_type="static",
    wrap_description=False,
    word_wrap=True,
    docstring_format="rest",
):
    """
    Convert to an argparse FunctionDef

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

    :param function_name: name of function_def
    :type function_name: ```str```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Literal['self', 'cls', 'static']```

    :param wrap_description: Whether to word-wrap the description. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type wrap_description: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :return:  AST node for function definition which constructs argparse
    :rtype: ```FunctionDef```
    """
    function_name = function_name or intermediate_repr["name"]
    function_type = function_type or intermediate_repr["type"]
    internal_body = get_internal_body(
        target_name=function_name,
        target_type=function_type,
        intermediate_repr=intermediate_repr,
    )

    return FunctionDef(
        args=arguments(
            args=[set_arg("argument_parser")],
            # None if function_type in frozenset((None, "static"))
            # else set_arg(function_type),
            defaults=[],
            kw_defaults=[],
            kwarg=None,
            kwonlyargs=[],
            posonlyargs=[],
            vararg=None,
            arg=None,
        ),
        body=list(
            chain.from_iterable(
                (
                    iter(
                        (
                            Expr(
                                set_value(
                                    docstring(
                                        {
                                            "doc": "Set CLI arguments",
                                            "params": OrderedDict(
                                                (
                                                    (
                                                        "argument_parser",
                                                        {
                                                            "doc": "argument parser",
                                                            "typ": "ArgumentParser",
                                                        },
                                                    ),
                                                )
                                            ),
                                            "returns": OrderedDict(
                                                (
                                                    (
                                                        "return_type",
                                                        {
                                                            "doc": "argument_parser, {returns_doc}".format(
                                                                returns_doc=intermediate_repr[
                                                                    "returns"
                                                                ][
                                                                    "return_type"
                                                                ][
                                                                    "doc"
                                                                ]
                                                            )
                                                            if intermediate_repr[
                                                                "returns"
                                                            ]["return_type"].get("doc")
                                                            else "argument_parser",
                                                            "typ": "Tuple[ArgumentParser, {typ}]".format(
                                                                typ=intermediate_repr[
                                                                    "returns"
                                                                ]["return_type"]["typ"]
                                                            ),
                                                        }
                                                        if "return_type"
                                                        in (
                                                            (
                                                                intermediate_repr or {}
                                                            ).get("returns")
                                                            or iter(())
                                                        )
                                                        and intermediate_repr[
                                                            "returns"
                                                        ]["return_type"].get("typ")
                                                        not in none_types
                                                        else {
                                                            "doc": "argument_parser",
                                                            "typ": "ArgumentParser",
                                                        },
                                                    ),
                                                ),
                                            ),
                                        },
                                        docstring_format=docstring_format,
                                        word_wrap=word_wrap,
                                        indent_level=1,
                                    )
                                )
                            ),
                            Assign(
                                targets=[
                                    Attribute(
                                        Name("argument_parser", Load()),
                                        "description",
                                        Store(),
                                    )
                                ],
                                value=set_value(
                                    (fill if wrap_description else identity)(
                                        intermediate_repr["doc"]
                                    )
                                ),
                                lineno=None,
                                expr=None,
                                **maybe_type_comment
                            ),
                        )
                    ),
                    filter(
                        None,
                        (
                            *(
                                (
                                    map(
                                        partial(
                                            param2argparse_param,
                                            word_wrap=word_wrap,
                                            emit_default_doc=emit_default_doc,
                                        ),
                                        intermediate_repr["params"].items(),
                                    )
                                )
                                if "params" in intermediate_repr
                                else ()
                            ),
                            *(
                                internal_body[
                                    2
                                    if len(internal_body) > 1
                                    and isinstance(internal_body[1], Assign)
                                    and internal_body[1].targets[0].id
                                    == "argument_parser"
                                    else 1 :
                                ]
                                if internal_body
                                and isinstance(internal_body[0], Expr)
                                and isinstance(get_value(internal_body[0].value), str)
                                else internal_body
                            ),
                            None
                            if internal_body and isinstance(internal_body[-1], Return)
                            else (
                                Return(
                                    value=Tuple(
                                        ctx=Load(),
                                        elts=[
                                            Name("argument_parser", Load()),
                                            set_value(
                                                intermediate_repr["returns"][
                                                    "return_type"
                                                ]["default"]
                                            )
                                            if code_quoted(
                                                intermediate_repr["returns"][
                                                    "return_type"
                                                ]["default"]
                                            )
                                            else ast.parse(
                                                intermediate_repr["returns"][
                                                    "return_type"
                                                ]["default"]
                                            )
                                            .body[0]
                                            .value,
                                        ],
                                        expr=None,
                                    ),
                                    expr=None,
                                )
                                if "default"
                                in (
                                    intermediate_repr.get("returns")
                                    or {"return_type": iter(())}
                                )["return_type"]
                                else Return(
                                    value=Name("argument_parser", Load()), expr=None
                                )
                            ),
                        ),
                    ),
                )
            )
        ),
        decorator_list=[],
        name=function_name,
        returns=None,
        lineno=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        **maybe_type_comment
    )


def class_(
    intermediate_repr,
    emit_call=False,
    class_name="ConfigClass",
    class_bases=("object",),
    decorator_list=None,
    word_wrap=True,
    docstring_format="rest",
    emit_original_whitespace=False,
    emit_default_doc=False,
):
    """
    Construct a class

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param emit_call: Whether to emit a `__call__` method from the `_internal` IR subdict
    :type emit_call: ```bool```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param emit_original_whitespace: Whether to emit original whitespace or strip it out (in docstring)
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: Class AST
    :rtype: ```ClassDef```
    """
    assert isinstance(
        intermediate_repr, dict
    ), "{intermediate_repr_type_name} != dict".format(
        intermediate_repr_type_name=type(intermediate_repr).__name__
    )

    returns = (
        intermediate_repr["returns"]
        if "return_type" in ((intermediate_repr or {}).get("returns") or iter(()))
        else OrderedDict()
    )
    if returns:
        intermediate_repr["params"].update(returns)
        del intermediate_repr["returns"]

    internal_body = intermediate_repr.get("_internal", {}).get("body", [])
    # TODO: Add correct classmethod/staticmethod to decorate function using `annotate_ancestry` and first-field checks
    # Such that the `self.` or `cls.` rewrite only applies to non-staticmethods
    # assert internal_body, "Expected `internal_body` to have contents"
    param_names = (
        frozenset(intermediate_repr["params"].keys())
        if "params" in intermediate_repr
        else None
    )
    if param_names:
        if internal_body:
            internal_body = list(
                map(
                    ast.fix_missing_locations,
                    map(RewriteName(param_names).visit, internal_body),
                )
            )
        elif (returns or {"return_type": None}).get("return_type") is not None:
            internal_body = returns["return_type"]

    indent_level = 1

    _emit_docstring = partial(
        docstring,
        docstring_format=docstring_format,
        indent_level=indent_level,
        emit_default_doc=emit_default_doc,
        emit_separating_tab=True,
        emit_types=False,
        word_wrap=word_wrap,
    )
    return ClassDef(
        bases=list(map(rpartial(Name, Load()), class_bases)),
        body=list(
            chain.from_iterable(
                filter(
                    None,
                    (
                        (
                            (lambda ds: Expr(set_value(ds.rstrip())))(
                                _emit_docstring(
                                    {
                                        k: intermediate_repr[k]
                                        for k in intermediate_repr
                                        if k != "_internal"
                                    },
                                    emit_original_whitespace=emit_original_whitespace,
                                    purpose="class",
                                )
                            ),
                        ),
                        map(
                            param2ast,
                            (intermediate_repr.get("params") or OrderedDict()).items(),
                        ),
                        iter(
                            (
                                (
                                    internal_body[0]
                                    if len(internal_body) == 1
                                    and isinstance(internal_body[0], FunctionDef)
                                    and internal_body[0].name == "__call__"
                                    else _make_call_meth(
                                        internal_body,
                                        returns["return_type"]["default"]
                                        if "default"
                                        in (
                                            (returns or {"return_type": iter(())}).get(
                                                "return_type"
                                            )
                                            or iter(())
                                        )
                                        else None,
                                        param_names,
                                        docstring_format=docstring_format,
                                        word_wrap=word_wrap,
                                    ),
                                )
                                or iter(())
                            )
                            if emit_call and internal_body
                            else iter(())
                        ),
                    ),
                )
            )
        )
        or Expr(set_value("")),  # empty body will cause syntax error
        decorator_list=list(map(rpartial(Name, Load()), decorator_list))
        if decorator_list
        else [],
        keywords=[],
        name=class_name,
        expr=None,
        identifier_name=None,
    )


def docstring(
    intermediate_repr,
    docstring_format="rest",
    purpose="function",
    word_wrap=True,
    indent_level=0,
    emit_separating_tab=True,
    emit_types=True,
    emit_original_whitespace=False,
    emit_default_doc=True,
):
    """
    Converts an IR to a docstring

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param purpose: Emit `:param` if purpose == 'function' elif purpose == 'class' then `:cvar`
    :type purpose: ```Literal['class', 'function']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param indent_level: indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_separating_tab: Whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :param emit_types: Whether to show `:type` lines
    :type emit_types: ```bool```

    :param emit_original_whitespace: Whether to emit original whitespace or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: docstring
    :rtype: ```str```
    """
    # _sep = tab * indent_level
    params = "\n{maybe_nl}".format(
        maybe_nl="\n" if docstring_format == "rest" and purpose != "class" else ""
    ).join(
        (
            lambda param_lines: [getattr(ARG_TOKENS, docstring_format)[0]] + param_lines
            if param_lines and docstring_format != "rest"
            else param_lines
        )(
            list(
                map(
                    partial(
                        emit_param_str,
                        style=docstring_format,
                        purpose=purpose,
                        emit_type=emit_types,
                        emit_default_doc=emit_default_doc,
                        word_wrap=word_wrap,
                    ),
                    (intermediate_repr["params"] or OrderedDict()).items(),
                ),
            )
        )
    )

    returns = (
        (
            lambda l: "".join(
                "{maybe_nl0_and_token}{maybe_nl1}{returns_doc}".format(
                    maybe_nl0_and_token=""
                    if docstring_format == "rest"
                    else "\n{return_token}".format(
                        return_token=getattr(RETURN_TOKENS, docstring_format)[0]
                    ),
                    maybe_nl1="" if not params or params[-1] == "\n" else "\n",
                    returns_doc=l,
                )
            )
            if l
            else ""
        )(
            next(
                map(
                    partial(
                        emit_param_str,
                        style=docstring_format,
                        purpose=purpose,
                        emit_type=emit_types,
                        emit_default_doc=emit_default_doc,
                        word_wrap=word_wrap,
                    ),
                    intermediate_repr["returns"].items(),
                ),
                None,
            )
        )
        if "return_type" in (intermediate_repr.get("returns") or iter(()))
        else ""
    )

    params_end_nls = num_of_nls(params, end=True)
    returns_end_nls = num_of_nls(returns, end=True)

    candidate_args_returns = "{params}{maybe_nl0}{returns}{maybe_nl1}".format(
        params=params,
        maybe_nl0="\n" if params_end_nls < 2 and returns else "",
        returns=returns,
        maybe_nl1="\n"
        if not returns and params_end_nls > 0 or returns and returns_end_nls == 0
        else "",
    )

    original_doc_str = intermediate_repr.get("_internal", {}).get(
        "original_doc_str", ""
    )
    if original_doc_str:
        header, _, footer = parse_docstring_into_header_args_footer(
            candidate_args_returns, original_doc_str
        )
        header = (
            intermediate_repr.get("doc", "") if not header and not footer else header
        )
    else:
        header, footer = intermediate_repr.get("doc", ""), ""

    candidate_doc_str = header_args_footer_to_str(
        header=header,
        args_returns="" if candidate_args_returns.isspace() else candidate_args_returns,
        footer=footer,
    )

    if not candidate_doc_str or candidate_doc_str.isspace():
        return ""

    prev_nl, next_nl = 0, candidate_doc_str.find("\n")
    current_indent, line = 0, None

    # One line only
    if next_nl == -1:
        # current_indent = count_iter_items(takewhile(str.isspace, candidate_doc_str))
        # _sep = (indent_level - current_indent) * tab
        return (
            candidate_doc_str
            if candidate_doc_str[0] == "\n"
            else "\n{_sep}{candidate_doc_str}".format(
                _sep="", candidate_doc_str=candidate_doc_str
            )
        )
    else:
        # Ignore starting newlines/whitespace only lines, keep munching until last line
        while next_nl > -1:
            line = candidate_doc_str[prev_nl:next_nl]
            if not line.isspace():
                break
            # prev_nl = next_nl
            # current_indent = count_iter_items(takewhile(str.isspace, line))

    if indent_level > current_indent:
        _tab = (indent_level - current_indent) * tab
        lines = ([line] if line else []) + candidate_doc_str[
            next_nl
            if len(candidate_doc_str) == next_nl
            or next_nl + 1 < len(candidate_doc_str)
            and candidate_doc_str[next_nl + 1] != "\n"
            else next_nl + 1 :
        ].splitlines()
        candidate_doc_str = "\n".join(
            map(
                lambda _line: "{_tab}{_line}".format(_tab=_tab, _line=_line)
                if _line or emit_separating_tab
                # and not _line.startswith(_tab)
                else _line,
                lines,
            )
        )
        if len(lines) > 1:
            candidate_doc_str = "{maybe_nl}{candidate_doc_str}{maybe_nl_tab}".format(
                maybe_nl="\n" if candidate_doc_str.startswith(_tab) else "",
                candidate_doc_str=candidate_doc_str,
                maybe_nl_tab=""
                if candidate_doc_str[-1] == "\n"
                else "\n{_tab}".format(_tab=_tab),
            )

    return candidate_doc_str


def file(node, filename, mode="a", skip_black=False):
    """
    Convert AST to a file

    :param node: AST node
    :type node: ```Union[Module, ClassDef, FunctionDef]```

    :param filename: emit to this file
    :type filename: ```str```

    :param mode: Mode to open the file in, defaults to append
    :type mode: ```str```

    :param skip_black: Whether to skip formatting with black
    :type skip_black: ```bool```

    :return: None
    :rtype: ```NoneType```
    """
    if not isinstance(node, Module):
        node = Module(body=[node], type_ignores=[], stmt=None)
    src = to_code(node)
    if not skip_black:
        src = black.format_str(
            src,
            mode=black.Mode(
                target_versions=set(),
                line_length=119,
                is_pyi=False,
                string_normalization=False,
            ),
        )
    with open(filename, mode) as f:
        f.write(src)


def function(
    intermediate_repr,
    function_name,
    function_type,
    word_wrap=True,
    emit_default_doc=False,
    docstring_format="rest",
    indent_level=2,
    emit_separating_tab=PY3_8,
    type_annotations=True,
    emit_as_kwonlyargs=True,
    emit_original_whitespace=False,
):
    """
    Construct a function from our IR

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param function_name: name of function_def
    :type function_name: ```Optional[str]```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Optional[Literal['self', 'cls', 'static']]```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :param indent_level: docstring indentation level whence: 0=no_tabs, 1=one tab; 2=two tabs
    :type indent_level: ```int```

    :param emit_separating_tab: docstring decider for whether to put a tab between :param and return and desc
    :type emit_separating_tab: ```bool```

    :param type_annotations: True to have type annotations (3.6+), False to place in docstring
    :type type_annotations: ```bool```

    :param emit_as_kwonlyargs: Whether argument(s) emitted must be keyword only
    :type emit_as_kwonlyargs: ```bool```

    :param emit_original_whitespace: Whether to emit an original whitespace (in docstring) or strip it out
    :type emit_original_whitespace: ```bool```

    :return: AST node for function definition
    :rtype: ```FunctionDef```
    """
    params_no_kwargs = tuple(
        filter(
            lambda param: not param[0].endswith("kwargs"),
            intermediate_repr["params"].items(),
        )
    )

    function_name = function_name or intermediate_repr["name"]
    function_type = function_type or intermediate_repr["type"]

    args = (
        [] if function_type in frozenset((None, "static")) else [set_arg(function_type)]
    )
    from cdd.emitter_utils import ast_parse_fix

    args_from_params = list(
        map(
            lambda param: set_arg(
                annotation=(
                    Name(param[1]["typ"], Load())
                    if param[1]["typ"] in simple_types
                    else ast_parse_fix(param[1]["typ"])
                )
                if type_annotations and "typ" in param[1]
                else None,
                arg=param[0],
            ),
            params_no_kwargs,
        ),
    )
    defaults_from_params = list(
        map(
            lambda param: set_value(None)
            if param[1].get("default") in none_types
            else set_value(param[1].get("default")),
            params_no_kwargs,
        )
    )
    if emit_as_kwonlyargs:
        kwonlyargs, kw_defaults, defaults = args_from_params, defaults_from_params, []
    else:
        kwonlyargs, kw_defaults, defaults = [], [], defaults_from_params
        args += args_from_params

    internal_body = get_internal_body(
        target_name=function_name,
        target_type=function_type,
        intermediate_repr=intermediate_repr,
    )
    return_val = (
        Return(
            value=ast.parse(
                intermediate_repr["returns"]["return_type"]["default"].strip("`")
            )
            .body[0]
            .value,
            expr=None,
        )
        if (intermediate_repr.get("returns") or {"return_type": {}})["return_type"].get(
            "default"
        )
        else None
    )

    return FunctionDef(
        args=arguments(
            args=args,
            defaults=defaults,
            kw_defaults=kw_defaults,
            kwarg=next(
                map(
                    lambda param: set_arg(param[0]),
                    filter(
                        lambda param: param[0].endswith("kwargs"),
                        intermediate_repr["params"].items(),
                    ),
                ),
                None,
            ),
            kwonlyargs=kwonlyargs,
            posonlyargs=[],
            vararg=None,
            arg=None,
        ),
        body=list(
            filter(
                None,
                (
                    Expr(
                        set_value(
                            docstring(
                                intermediate_repr,
                                docstring_format=docstring_format,
                                emit_default_doc=emit_default_doc,
                                emit_original_whitespace=emit_original_whitespace,
                                emit_separating_tab=emit_separating_tab,
                                emit_types=not type_annotations,
                                indent_level=indent_level,
                                word_wrap=word_wrap,
                            )
                        )
                    ),
                    *(
                        internal_body[:-1]
                        if internal_body
                        and isinstance(internal_body[-1], Return)
                        and return_val
                        else internal_body
                    ),
                    return_val,
                ),
            )
        ),
        decorator_list=[],
        name=function_name,
        returns=(
            ast.parse(intermediate_repr["returns"]["return_type"]["typ"]).body[0].value
            if type_annotations
            and (intermediate_repr.get("returns") or {"return_type": {}})[
                "return_type"
            ].get("typ")
            else None
        ),
        lineno=None,
        arguments_args=None,
        identifier_name=None,
        stmt=None,
        **maybe_type_comment
    )


def json_schema(
    intermediate_repr,
    identifier="https://offscale.io/json.schema.json",
    emit_original_whitespace=False,
):
    """
    Construct a JSON schema dict

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param identifier: The `$id` of the schema
    :type identifier: ```str```

    :param emit_original_whitespace: Whether to emit original whitespace (in top-level `description`) or strip it out
    :type emit_original_whitespace: ```bool```

    :return: JSON Schema dict
    :rtype: ```dict```
    """
    required = []
    _param2json_schema_property = partial(param2json_schema_property, required=required)
    properties = dict(
        map(_param2json_schema_property, intermediate_repr["params"].items())
    )

    return {
        "$id": identifier,
        "$schema": "http://json-schema.org/draft-07/schema#",
        "description": deindent(
            add(
                *map(
                    partial(
                        docstring,
                        emit_default_doc=True,
                        emit_original_whitespace=emit_original_whitespace,
                        emit_types=True,
                    ),
                    (
                        {
                            "doc": intermediate_repr["doc"],
                            "params": OrderedDict(),
                            "returns": None,
                        },
                        {
                            "doc": "",
                            "params": OrderedDict(),
                            "returns": intermediate_repr["returns"],
                        },
                    ),
                )
            )
        ).lstrip("\n"),
        "type": "object",
        "properties": properties,
        "required": required,
    }


def sqlalchemy_table(
    intermediate_repr,
    name="config_tbl",
    docstring_format="rest",
    word_wrap=True,
    emit_original_whitespace=False,
    emit_default_doc=True,
):
    """
    Construct an `name = sqlalchemy.Table(name, metadata, Column(…), …)`

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param name: name of binding + table
    :type name: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param emit_original_whitespace: Whether to emit an original whitespace (in docstring) or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_original_whitespace: Whether to emit original whitespace or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: AST of the Table expression + assignment
    :rtype: ```ClassDef```
    """
    return Assign(
        targets=[Name(name, Store())],
        value=Call(
            func=Name("Table", Load()),
            args=list(
                chain.from_iterable(
                    (
                        iter(
                            (
                                set_value(name),
                                Name("metadata", Load()),
                            )
                        ),
                        map(
                            partial(param_to_sqlalchemy_column_call, include_name=True),
                            intermediate_repr["params"].items(),
                        ),
                    )
                )
            ),
            keywords=[
                keyword(
                    arg="comment",
                    value=set_value(
                        deindent(
                            add(
                                *map(
                                    partial(
                                        docstring,
                                        emit_default_doc=emit_default_doc,
                                        docstring_format=docstring_format,
                                        word_wrap=word_wrap,
                                        emit_original_whitespace=emit_original_whitespace,
                                        emit_types=True,
                                    ),
                                    (
                                        {
                                            "doc": intermediate_repr["doc"].lstrip()
                                            + "\n\n"
                                            if intermediate_repr["returns"]
                                            else "",
                                            "params": OrderedDict(),
                                            "returns": None,
                                        },
                                        {
                                            "doc": "",
                                            "params": OrderedDict(),
                                            "returns": intermediate_repr["returns"],
                                        },
                                    ),
                                )
                            ).strip()
                        )
                    ),
                    identifier=None,
                )
            ]
            if intermediate_repr["doc"]
            else [],
            expr=None,
            expr_func=None,
        ),
        lineno=None,
        expr=None,
        **maybe_type_comment
    )


def sqlalchemy(
    intermediate_repr,
    emit_repr=True,
    class_name="Config",
    class_bases=("Base",),
    decorator_list=None,
    table_name=None,
    docstring_format="rest",
    word_wrap=True,
    emit_original_whitespace=False,
    emit_default_doc=True,
):
    """
    Construct an SQLAlchemy declarative class

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param emit_repr: Whether to generate a `__repr__` method
    :type emit_repr: ```bool```

    :param class_name: name of class
    :type class_name: ```str```

    :param class_bases: bases of class (the generated class will inherit these)
    :type class_bases: ```Iterable[str]```

    :param decorator_list: List of decorators
    :type decorator_list: ```Optional[Union[List[Str], List[]]]```

    :param table_name: Table name, defaults to `class_name`
    :type table_name: ```str```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param docstring_format: Format of docstring
    :type docstring_format: ```Literal['rest', 'numpydoc', 'google']```

    :param emit_original_whitespace: Whether to emit an original whitespace (in docstring) or strip it out
    :type emit_original_whitespace: ```bool```

    :param emit_default_doc: Whether help/docstring should include 'With default' text
    :type emit_default_doc: ```bool```

    :return: SQLalchemy declarative class AST
    :rtype: ```ClassDef```
    """

    def _add(a, b):
        """
        Concatenate a with b with correct whitespace around and within each

        :param a: first string
        :type a: ```str```

        :param b: second string
        :type b: ```str```

        :return: combined strings with correct whitespace around and within
        :rtype: ```str```
        """
        b_splits = b.split("\n{tab}".format(tab=tab))
        res = "{a}{snd}\n{tab}{end}{tab}".format(
            a=a, tab=tab, snd=b_splits[0], end="\n".join(b_splits[1:])
        )
        return indent_all_but_first(res, indent_level=1, sep=tab)

    return ClassDef(
        name=class_name,
        bases=list(map(lambda class_base: Name(class_base, Load()), class_bases)),
        decorator_list=decorator_list or [],
        keywords=[],
        body=list(
            filter(
                None,
                (
                    Expr(
                        set_value(
                            _add(
                                *map(
                                    partial(
                                        docstring,
                                        docstring_format=docstring_format,
                                        emit_default_doc=emit_default_doc,
                                        emit_original_whitespace=emit_original_whitespace,
                                        emit_separating_tab=True,
                                        emit_types=True,
                                        indent_level=1,
                                        word_wrap=word_wrap,
                                    ),
                                    (
                                        {
                                            "doc": intermediate_repr["doc"],
                                            "params": OrderedDict(),
                                            "returns": None,
                                        },
                                        {
                                            "doc": "",
                                            "params": OrderedDict(),
                                            "returns": intermediate_repr["returns"],
                                        },
                                    ),
                                )
                            )
                        )
                    )
                    if intermediate_repr["doc"]
                    or intermediate_repr["returns"].get("return_type", {}).get("doc")
                    else None,
                    Assign(
                        targets=[Name("__tablename__", Store())],
                        value=set_value(table_name or class_name),
                        expr=None,
                        lineno=None,
                        **maybe_type_comment
                    ),
                    *map(
                        lambda param: Assign(
                            targets=[Name(param[0], Store())],
                            value=param_to_sqlalchemy_column_call(
                                param, include_name=False
                            ),
                            expr=None,
                            lineno=None,
                            **maybe_type_comment
                        ),
                        intermediate_repr["params"].items(),
                    ),
                    generate_repr_method(
                        intermediate_repr["params"], class_name, docstring_format
                    )
                    if emit_repr
                    else None,
                ),
            )
        ),
        expr=None,
        identifier_name=None,
    )


__all__ = [
    "argparse_function",
    "class_",
    "docstring",
    "file",
    "function",
    "sqlalchemy_table",
    "sqlalchemy",
]

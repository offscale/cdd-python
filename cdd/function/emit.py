"""
Function/method emitter
"""

import ast
from ast import Expr, FunctionDef, Load, Name, Return, arguments

from cdd.docstring.emit import docstring
from cdd.shared.ast_utils import maybe_type_comment, set_arg, set_value
from cdd.shared.emit.utils.emitter_utils import get_internal_body
from cdd.shared.pure_utils import PY3_8, none_types, simple_types


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
    from cdd.shared.emit.utils.emitter_utils import ast_parse_fix

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


__all__ = ["function"]

"""
Argparse emitter
"""
import ast
from ast import (
    Assign,
    Attribute,
    Expr,
    FunctionDef,
    Load,
    Name,
    Return,
    Store,
    Tuple,
    arguments,
)
from collections import OrderedDict
from functools import partial
from itertools import chain

from cdd.ast_utils import (
    get_value,
    maybe_type_comment,
    param2argparse_param,
    set_arg,
    set_value,
)
from cdd.emit.docstring import docstring
from cdd.emit.emitter_utils import get_internal_body
from cdd.pure_utils import code_quoted, fill, identity, none_types


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


__all__ = ["argparse_function"]

"""
Argparse function parser
"""

from ast import Assign, Call, FunctionDef, Return, Tuple, get_docstring
from collections import OrderedDict
from functools import partial
from itertools import filterfalse
from operator import setitem
from typing import List, Optional, cast

from cdd.argparse_function.utils.emit_utils import _parse_return, parse_out_param
from cdd.shared.ast_utils import (
    get_function_type,
    get_value,
    is_argparse_add_argument,
    is_argparse_description,
)
from cdd.shared.docstring_parsers import parse_docstring
from cdd.shared.types import IntermediateRepr


def argparse_ast(
    function_def,
    function_type=None,
    function_name=None,
    parse_original_whitespace=False,
    word_wrap=False,
):
    """
    Converts an argparse AST to our IR

    :param function_def: AST of argparse function_def
    :type function_def: ```FunctionDef```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Literal['self', 'cls', 'static']```

    :param function_name: name of function_def
    :type function_name: ```str```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :return: a dictionary consistent with `IntermediateRepr`, defined as:
        ParamVal = TypedDict("ParamVal", {"typ": str, "doc": Optional[str], "default": Any})
        IntermediateRepr = TypedDict("IntermediateRepr", {
            "name": Optional[str],
            "type": Optional[str],
            "doc": Optional[str],
            "params": OrderedDict[str, ParamVal],
            "returns": Optional[OrderedDict[Literal["return_type"], ParamVal]],
        })
    :rtype: ```dict```
    """
    assert isinstance(
        function_def, FunctionDef
    ), "Expected `FunctionDef` got `{node_name!r}`".format(
        node_name=type(function_def).__name__
    )

    doc_string: Optional[str] = get_docstring(
        function_def, clean=parse_original_whitespace
    )
    intermediate_repr: IntermediateRepr = {
        "name": function_name or function_def.name,
        "type": function_type or get_function_type(function_def),
        "doc": "",
        "params": OrderedDict(),
    }
    ir: IntermediateRepr = parse_docstring(
        doc_string,
        word_wrap=word_wrap,
        emit_default_doc=True,
        parse_original_whitespace=parse_original_whitespace,
    )

    # Whether a default is required, if not found in doc, infer the proper default from type
    require_default = False

    # Parse all relevant nodes from function body
    body: FunctionDef.body = (
        function_def.body if doc_string is None else function_def.body[1:]
    )
    for node in body:
        if is_argparse_add_argument(node):
            name, _param = parse_out_param(
                node,
                emit_default_doc=False,  # require_default=require_default
            )
            (
                intermediate_repr["params"][name].update
                if name in intermediate_repr["params"]
                else partial(setitem, intermediate_repr["params"], name)
            )(_param)
            if not require_default and _param.get("default") is not None:
                require_default: bool = True
        elif isinstance(node, Assign) and is_argparse_description(node):
            intermediate_repr["doc"] = get_value(node.value)
        elif isinstance(node, Return) and isinstance(node.value, Tuple):
            intermediate_repr["returns"] = OrderedDict(
                (
                    _parse_return(
                        node,
                        intermediate_repr=ir,
                        function_def=function_def,
                        emit_default_doc=False,
                    ),
                )
            )

    inner_body: List[Call] = cast(
        List[Call],
        list(
            filterfalse(
                is_argparse_description,
                filterfalse(is_argparse_add_argument, body),
            )
        ),
    )
    if inner_body:
        intermediate_repr["_internal"] = {
            "original_doc_str": (
                doc_string
                if parse_original_whitespace
                else get_docstring(function_def, clean=False)
            ),
            "body": inner_body,
            "from_name": function_def.name,
            "from_type": "static",
        }

    return intermediate_repr


__all__ = ["argparse_ast"]  # type: list[str]

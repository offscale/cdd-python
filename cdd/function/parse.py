"""
Function parser
"""

import ast
from ast import AST, AnnAssign, Assign, FunctionDef, get_docstring
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from inspect import getsource
from itertools import cycle, filterfalse, islice
from types import FunctionType
from typing import List, Optional, cast

import cdd.docstring.parse
import cdd.shared.ast_utils
import cdd.shared.docstring_parsers
import cdd.shared.parse.utils.parser_utils
from cdd.function.utils.parse_utils import _interpolate_return
from cdd.shared.pure_utils import rpartial
from cdd.shared.types import IntermediateRepr


def function(
    function_def,
    infer_type=False,
    parse_original_whitespace=False,
    word_wrap=True,
    function_type=None,
    function_name=None,
):
    """
    Converts a method to our IR

    :param function_def: AST node for function definition
    :type function_def: ```Union[FunctionDef, FunctionType]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :param function_type: Type of function, static is static or global method, others just become first arg
    :type function_type: ```Literal['self', 'cls', 'static']```

    :param function_name: name of function_def
    :type function_name: ```Optional[str]```

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
    if isinstance(function_def, FunctionType):
        # Dynamic function, i.e., this isn't source code; and is in your memory
        ir: IntermediateRepr = cdd.shared.parse.utils.parser_utils._inspect(
            function_def,
            function_name,
            parse_original_whitespace=parse_original_whitespace,
            word_wrap=word_wrap,
        )
        parsed_source: FunctionDef = cast(
            FunctionDef, ast.parse(getsource(function_def).lstrip()).body[0]
        )
        original_doc_str: Optional[str] = ast.get_docstring(
            parsed_source, clean=parse_original_whitespace
        )
        body: FunctionDef.body = (
            parsed_source.body if original_doc_str is None else parsed_source.body[1:]
        )
        ir["_internal"] = {
            "original_doc_str": (
                original_doc_str
                if parse_original_whitespace
                else ast.get_docstring(parsed_source, clean=False)
            ),
            "body": cast(
                List[AST],
                list(filterfalse(rpartial(isinstance, (AnnAssign, Assign)), body)),
            ),
            "from_name": parsed_source.name,
            "from_type": "cls",
        }
        return ir

    assert isinstance(
        function_def, FunctionDef
    ), "Expected `FunctionDef` got `{node_name!r}`".format(
        node_name=type(function_def).__name__
    )
    assert (
        function_name is None or function_def.name == function_name
    ), "Expected {function_name!r} got {function_def_name!r}".format(
        function_name=function_name, function_def_name=function_def.name
    )

    found_type = cdd.shared.ast_utils.get_function_type(function_def)

    # Read docstring
    doc_str: Optional[str] = (
        get_docstring(function_def, clean=parse_original_whitespace)
        if isinstance(function_def, FunctionDef)
        else None
    )

    function_def: FunctionDef = deepcopy(function_def)
    function_def.args.args = (
        function_def.args.args if found_type == "static" else function_def.args.args[1:]
    )

    if doc_str is None:
        intermediate_repr: IntermediateRepr = {
            "name": function_name or function_def.name,
            "params": OrderedDict(),
            "returns": None,
            "_internal": {},
        }
    else:
        intermediate_repr: IntermediateRepr = cdd.docstring.parse.docstring(
            doc_str.replace(":cvar", ":param"),
            parse_original_whitespace=parse_original_whitespace,
            infer_type=infer_type,
        )
        intermediate_repr["_internal"] = {
            "original_doc_str": (
                doc_str
                if parse_original_whitespace
                else (
                    get_docstring(function_def, clean=False)
                    if isinstance(function_def, FunctionDef)
                    else None
                )
            )
        }

    intermediate_repr.update(
        {
            "name": function_name or function_def.name,
            "type": function_type or found_type,
        }
    )

    intermediate_repr["_internal"].update(
        {
            "from_name": function_def.name,
            "from_type": found_type,
        }
    )
    function_def.body = function_def.body if doc_str is None else function_def.body[1:]
    if function_def.body:
        intermediate_repr["_internal"]["body"] = function_def.body

    params_to_append = OrderedDict()
    if (
        hasattr(function_def.args, "kwarg")
        and function_def.args.kwarg
        and function_def.args.kwarg.arg in intermediate_repr["params"]
    ):
        _param = intermediate_repr["params"].pop(function_def.args.kwarg.arg)
        assert "typ" in _param
        _param["default"] = cdd.shared.ast_utils.NoneStr
        params_to_append[function_def.args.kwarg.arg] = _param
        del _param

    # Set defaults

    # Fill with `None`s when no default is given to make the `zip` below it work cleanly
    for args, defaults in (
        ("args", "defaults"),
        ("kwonlyargs", "kw_defaults"),
    ):
        diff = abs(
            len(getattr(function_def.args, args))
            - len(getattr(function_def.args, defaults))
        )
        if diff:
            setattr(
                function_def.args,
                defaults,
                list(islice(cycle((None,)), diff))
                + getattr(function_def.args, defaults),
            )
    cdd.shared.parse.utils.parser_utils.ir_merge(
        intermediate_repr,
        {
            "params": OrderedDict(
                (
                    cdd.shared.ast_utils.func_arg2param(
                        getattr(function_def.args, args)[idx],
                        default=getattr(function_def.args, defaults)[idx],
                    )
                    for args, defaults in (
                        ("args", "defaults"),
                        ("kwonlyargs", "kw_defaults"),
                    )
                    for idx in range(len(getattr(function_def.args, args)))
                )
            ),
            "returns": None,
        },
    )

    intermediate_repr["params"].update(params_to_append)
    intermediate_repr["params"] = OrderedDict(
        map(
            partial(
                cdd.shared.docstring_parsers._set_name_and_type,
                infer_type=infer_type,
                word_wrap=word_wrap,
            ),
            intermediate_repr["params"].items(),
        )
    )

    # Convention - the final top-level `return` is the default
    intermediate_repr: IntermediateRepr = _interpolate_return(
        function_def, intermediate_repr
    )
    if "return_type" in (intermediate_repr.get("returns") or iter(())):
        intermediate_repr["returns"] = OrderedDict(
            map(
                partial(
                    cdd.shared.docstring_parsers._set_name_and_type,
                    infer_type=infer_type,
                    word_wrap=word_wrap,
                ),
                intermediate_repr["returns"].items(),
            )
        )
    return intermediate_repr


__all__ = ["function"]  # type: list[str]

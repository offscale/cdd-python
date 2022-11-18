"""
_inspect is a shared a function betwixt a couple of parse modules
"""
import ast
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from inspect import getdoc, isfunction, signature

import cdd.parse.class_
import cdd.parse.docstring
import cdd.parse.function
from cdd.docstring_parsers import _set_name_and_type
from cdd.parse.parser_utils import _inspect_process_ir_param, get_source, ir_merge


def _inspect(obj, name, parse_original_whitespace, word_wrap):
    """
    Uses the `inspect` module to figure out the IR from the input

    :param obj: Something in memory, like a class, function, variable
    :type obj: ```Any```

    :param name: Name of the object being inspected
    :type name: ```str```

    :param parse_original_whitespace: Whether to parse original whitespace or strip it out
    :type parse_original_whitespace: ```bool```

    :param word_wrap: Whether to word-wrap. Set `DOCTRANS_LINE_LENGTH` to configure length.
    :type word_wrap: ```bool```

    :return: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """

    doc = getdoc(obj) or ""

    # def is_builtin_class_instance(obj):
    #     builtin_types = tuple(
    #         getattr(builtins, t)
    #         for t in dir(builtins)
    #         if isinstance(getattr(builtins, t), type)
    #     )
    #     return obj.__class__.__module__ == "__builtin__" or isinstance(
    #         obj, builtin_types
    #     )

    is_function = isfunction(obj)
    ir = (
        cdd.parse.docstring.docstring(
            doc,
            emit_default_doc=is_function,
            parse_original_whitespace=parse_original_whitespace,
        )
        if doc
        else {}
    )
    if not is_function and "type" in ir:
        del ir["type"]

    ir["name"] = (
        name or obj.__qualname__ if hasattr(obj, "__qualname__") else obj.__name__
    )
    assert ir["name"], "IR name is empty"

    # if is_builtin_class_instance(obj):
    #    return ir

    sig = None
    with suppress(ValueError):
        sig = signature(obj)
    if sig is not None:
        ir["params"] = OrderedDict(
            filter(
                None,
                map(
                    partial(_inspect_process_ir_param, sig=sig),
                    ir.get("params", {}).items(),
                )
                # if ir.get("params")
                # else map(_inspect_process_sig, sig.parameters.items()),
            )
        )

    src = get_source(obj)
    if src is None:
        return ir
    parsed_body = ast.parse(src.lstrip()).body[0]

    if is_function:
        ir["type"] = (
            "static"
            if sig is None
            else {"self": "self", "cls": "cls"}.get(
                next(iter(sig.parameters.values())).name, "static"
            )
        )
        parser = cdd.parse.function.function
    else:
        parser = cdd.parse.class_.class_

    other = parser(parsed_body)
    ir_merge(ir, other)
    if "return_type" in (ir.get("returns") or iter(())):
        ir["returns"] = OrderedDict(
            map(
                partial(_set_name_and_type, infer_type=False, word_wrap=word_wrap),
                ir["returns"].items(),
            )
        )

    return ir

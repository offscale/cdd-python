"""
`class` parser
"""

import ast
from ast import (
    AnnAssign,
    Assign,
    ClassDef,
    Dict,
    FunctionDef,
    Module,
    Name,
    get_docstring,
)
from collections import OrderedDict, deque
from functools import partial
from itertools import filterfalse
from operator import setitem

import cdd.docstring.parse
import cdd.function.parse
import cdd.shared.parse.utils.parser_utils
from cdd.class_.utils.parse_utils import get_source
from cdd.shared.ast_utils import NoneStr, find_ast_type, get_value, parse_to_scalar
from cdd.shared.docstring_parsers import _set_name_and_type
from cdd.shared.pure_utils import rpartial, simple_types
from cdd.shared.source_transformer import to_code


def class_(
    class_def,
    class_name=None,
    merge_inner_function=None,
    infer_type=False,
    parse_original_whitespace=False,
    word_wrap=True,
):
    """
    Converts an AST to our IR

    :param class_def: Class AST or Module AST with a ClassDef inside
    :type class_def: ```Union[Module, ClassDef]```

    :param class_name: Name of `class`. If None, gives first found.
    :type class_name: ```Optional[str]```

    :param merge_inner_function: Name of inner function to merge. If None, merge nothing.
    :type merge_inner_function: ```Optional[str]```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

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
    assert not isinstance(class_def, FunctionDef), "Expected not `FunctionDef`"
    is_supported_ast_node = isinstance(class_def, (Module, ClassDef))
    if not is_supported_ast_node and isinstance(class_def, type):
        return _class_from_memory(
            class_def=class_def,
            class_name=class_name,
            infer_type=infer_type,
            merge_inner_function=merge_inner_function,
            parse_original_whitespace=parse_original_whitespace,
            word_wrap=word_wrap,
        )

    assert (
        is_supported_ast_node
    ), "Expected 'Union[Module, ClassDef]' got `{node_name!r}`".format(
        node_name=type(class_def).__name__
    )
    class_def = find_ast_type(class_def, class_name)
    doc_str = get_docstring(class_def, clean=parse_original_whitespace)
    intermediate_repr = (
        {
            "name": class_name,
            "type": "static",
            "doc": "",
            "params": OrderedDict(),
            "returns": None,
        }
        if doc_str is None
        else cdd.docstring.parse.docstring(
            doc_str,
            emit_default_doc=False,
            parse_original_whitespace=parse_original_whitespace,
        )
    )

    if "return_type" in intermediate_repr["params"]:
        intermediate_repr["returns"] = OrderedDict(
            (("return_type", intermediate_repr["params"].pop("return_type")),)
        )

    body = class_def.body if doc_str is None else class_def.body[1:]
    for e in body:
        if isinstance(e, AnnAssign):
            typ = to_code(e.annotation).rstrip("\n")
            # print(ast.dump(e, indent=4))
            val = (
                (
                    lambda v: {"default": NoneStr}
                    if v is None
                    else {
                        "default": v
                        if type(v).__name__ in simple_types
                        else (
                            lambda value: {
                                "{}": {} if isinstance(v, Dict) else set(),
                                "[]": [],
                                "()": (),
                            }.get(value, parse_to_scalar(value))
                        )(to_code(v).rstrip("\n"))
                    }
                )(get_value(get_value(e)))
                if hasattr(e, "value") and e.value is not None
                else {}
            )

            # if 'str' in typ and val: val["default"] = val["default"].strip("'")  # Unquote?
            typ_default = {"typ": typ} if val is None else dict(typ=typ, **val)

            target_id = e.target.id.lstrip("*")

            for key in "params", "returns":
                if target_id in (intermediate_repr[key] or iter(())):
                    intermediate_repr[key][target_id].update(typ_default)
                    typ_default = False
                    break

            if typ_default:
                k = "returns" if target_id == "return_type" else "params"
                if intermediate_repr.get(k) is None:
                    intermediate_repr[k] = OrderedDict()
                intermediate_repr[k][target_id] = typ_default
        elif isinstance(e, Assign):
            val = get_value(e)

            if val is not None:
                val = get_value(val)
                deque(
                    map(
                        lambda target: setitem(
                            *(
                                (
                                    lambda _target_id: (
                                        intermediate_repr["params"][_target_id],
                                        "default",
                                        val,
                                    )
                                    if isinstance(target, Name)
                                    and _target_id in intermediate_repr["params"]
                                    else (
                                        intermediate_repr["params"],
                                        _target_id
                                        if isinstance(target, Name)
                                        else get_value(get_value(target)),
                                        {"default": val},
                                    )
                                )(
                                    target.id.lstrip("*")
                                    if hasattr(target, "id")
                                    else target.value.id
                                )
                            )
                        ),
                        e.targets,
                    ),
                    maxlen=0,
                )

    intermediate_repr.update(
        {
            "name": class_name or class_def.name,
            "params": OrderedDict(
                map(
                    partial(
                        _set_name_and_type, infer_type=infer_type, word_wrap=word_wrap
                    ),
                    intermediate_repr["params"].items(),
                )
            ),
            "_internal": {
                "original_doc_str": doc_str
                if parse_original_whitespace
                else get_docstring(class_def, clean=False),
                "body": list(
                    filterfalse(rpartial(isinstance, (AnnAssign, Assign)), body)
                ),
                "from_name": class_def.name,
                "from_type": "cls",
            },
        }
    )

    if merge_inner_function is not None:
        assert isinstance(
            class_def, ClassDef
        ), "Expected `ClassDef` got `{node_name!r}`".format(
            node_name=type(class_def).__name__
        )

        _merge_inner_function(
            class_def,
            infer_type=infer_type,
            intermediate_repr=intermediate_repr,
            merge_inner_function=merge_inner_function,
        )

    return intermediate_repr


def _class_from_memory(
    class_def,
    class_name,
    infer_type,
    merge_inner_function,
    parse_original_whitespace,
    word_wrap,
):
    """
    Merge the inner function if found within the class, with the class IR.
    Internal func just for internal memory. Uses `inspect`.

    :param class_def: Class AST
    :type class_def: ```ClassDef```

    :param class_name: Class name
    :type class_name: ```str```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param merge_inner_function: Name of inner function to merge. If None, merge nothing.
    :type merge_inner_function: ```Optional[str]```

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
    ir = cdd.shared.parse.utils.parser_utils._inspect(
        class_def,
        class_name,
        parse_original_whitespace=parse_original_whitespace,
        word_wrap=word_wrap,
    )
    src = get_source(class_def)
    if src is None:
        return ir
    parsed_body = ast.parse(src.lstrip()).body[0]
    original_doc_str = get_docstring(parsed_body, clean=parse_original_whitespace)
    parsed_body.body = (
        parsed_body.body if original_doc_str is None else parsed_body.body[1:]
    )
    if merge_inner_function is not None:
        _merge_inner_function(
            parsed_body,
            infer_type=infer_type,
            intermediate_repr=ir,
            merge_inner_function=merge_inner_function,
        )
        return ir
    ir["_internal"] = {
        "original_doc_str": original_doc_str
        if parse_original_whitespace
        else get_docstring(parsed_body, clean=False),
        "body": list(
            filterfalse(
                rpartial(isinstance, (AnnAssign, Assign)),
                parsed_body.body,
            )
        ),
        "from_name": class_name,
        "from_type": "cls",
    }
    if class_name is None:
        class_name = parsed_body.name
    body_ir = class_(
        class_def=parsed_body,
        class_name=class_name,
        merge_inner_function=merge_inner_function,
    )

    cdd.shared.parse.utils.parser_utils.ir_merge(ir, body_ir)
    return ir


def _merge_inner_function(
    class_def, infer_type, intermediate_repr, merge_inner_function
):
    """
    Merge the inner function if found within the class, with the class IR

    :param class_def: Class AST
    :type class_def: ```ClassDef```

    :param infer_type: Whether to try inferring the typ (from the default)
    :type infer_type: ```bool```

    :param intermediate_repr: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :type intermediate_repr: ```dict```

    :param merge_inner_function: Name of inner function to merge. If None, merge nothing.
    :type merge_inner_function: ```Optional[str]```

    :return: a dictionary of form
        {  "name": Optional[str],
           "type": Optional[str],
           "doc": Optional[str],
           "params": OrderedDict[str, {'typ': str, 'doc': Optional[str], 'default': Any}]
           "returns": Optional[OrderedDict[Literal['return_type'],
                                           {'typ': str, 'doc': Optional[str], 'default': Any}),)]] }
    :rtype: ```dict```
    """
    function_def = next(
        filter(
            lambda func: func.name == merge_inner_function,
            filter(rpartial(isinstance, FunctionDef), ast.walk(class_def)),
        ),
        None,
    )

    if function_def is not None:
        function_type = (
            "static" if not function_def.args.args else function_def.args.args[0].arg
        )
        inner_ir = cdd.function.parse.function(
            function_def,
            function_name=merge_inner_function,
            function_type=function_type,
            infer_type=infer_type,
        )
        cdd.shared.parse.utils.parser_utils.ir_merge(
            other=inner_ir, target=intermediate_repr
        )

    return intermediate_repr


__all__ = ["class_"]
